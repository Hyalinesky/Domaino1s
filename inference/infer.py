import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Model inference arguments')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='checkpoint-merge',
                        help='Path to the model')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run the model on')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of new tokens to generate')
    
    # Generation control parameters  
    parser.add_argument('--num_attempts', type=int, default=2,
                        help='Number of generation attempts (K)')
    parser.add_argument('--perplexity_threshold', type=float, default=1.2,
                        help='Perplexity threshold (theta)')

    return parser.parse_args()

def get_model_response_with_perplexity(model, tokenizer, question, args):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    generated_tokens = []
    step_perplexities = []
    current_step_logits = []
    current_step_tokens = []
    current_step_text = ""
    all_steps = []
    
    past_key_values = None
    input_ids = model_inputs.input_ids
    
    def calculate_step_perplexity(step_logits, step_target_tokens):
        if len(step_logits) == 0:
            return 0.0
            
        step_logits = torch.cat(step_logits, dim=1)
        step_target_tokens = torch.cat(step_target_tokens, dim=1)
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            step_logits.reshape(-1, step_logits.size(-1)),
            step_target_tokens.reshape(-1)
        )
        return torch.exp(loss).item()

    with torch.no_grad():
        for i in range(args.max_new_tokens):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            next_token_logits = outputs.logits[:, -1:, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            past_key_values = outputs.past_key_values
            input_ids = next_token
            
            generated_tokens.append(next_token)
            if len(generated_tokens) > 1:
                current_step_logits.append(next_token_logits)
                current_step_tokens.append(next_token)
            
            new_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            current_step_text += new_text
            
            if "\n\n" in current_step_text:
                step_text = current_step_text.split("\n\n")[0].strip()
                if step_text:
                    if len(current_step_logits) > 0:
                        perplexity = calculate_step_perplexity(
                            current_step_logits[:-1], 
                            current_step_tokens[:-1]
                        )
                        step_perplexities.append(perplexity)
                        all_steps.append(step_text)
                
                current_step_text = current_step_text.split("\n\n", 1)[1]
                current_step_logits = []
                current_step_tokens = []
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            del outputs.logits
            torch.cuda.empty_cache()
    
    if current_step_text.strip():
        if len(current_step_logits) > 0:
            perplexity = calculate_step_perplexity(
                current_step_logits[:-1], 
                current_step_tokens[:-1]
            )
            step_perplexities.append(perplexity)
            all_steps.append(current_step_text.strip())
    
    response = "\n\n".join(all_steps)
    return response, step_perplexities, time.time() - start_time

def generate_and_check_response(model, tokenizer, question, args, previous_steps=""):
    best_steps = []
    current_question = question
    total_start_time = time.time()
    
    while True:
        response, perplexities, inference_time = get_model_response_with_perplexity(
            model, tokenizer, current_question, args
        )
        
        steps = [s.strip() for s in response.split("\n\n") if s.strip()]
        
        i = len(best_steps)
        while i < len(steps):
            current_perplexity = perplexities[i]
            best_perplexity = current_perplexity
            best_step = steps[i]
            best_remaining_steps = steps[i:]
            found_better = False
            
            if current_perplexity >= args.perplexity_threshold:                
                for attempt in range(args.num_attempts-1):
                    new_question = current_question
                    if previous_steps:
                        new_question = current_question + "\n\n" + previous_steps
                    if best_steps:
                        new_question += "\n\n" + "\n\n".join(best_steps)
                        
                    new_response, new_perplexities, inference_time = get_model_response_with_perplexity(
                        model, tokenizer, new_question, args
                    )
                    
                    new_steps = [s.strip() for s in new_response.split("\n\n") if s.strip()]
                    
                    if i < len(new_steps):
                        new_perplexity = new_perplexities[i]
                        
                        if new_perplexity < best_perplexity:
                            best_perplexity = new_perplexity
                            best_step = new_steps[i]
                            best_remaining_steps = new_steps[i:]
                            found_better = True
                            if new_perplexity < args.perplexity_threshold:
                                break
                
                if found_better:
                    steps = best_steps + best_remaining_steps
            
            best_steps.append(best_step)
            i += 1
            
            if best_perplexity >= args.perplexity_threshold and len(best_steps) == len(steps):
                continue
                
        if len(best_steps) == len(steps):
            break
    
    return "\n\n".join(best_steps)

def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16, 
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    try:
        question = """Here is a service agreement:
Your content
Some of our services give you the opportunity to make your content publicly available for example, you might post a product or restaurant review that you wrote, or you might upload a blog post that you created.
See the Permission to use your content section for more about your rights in your content, and how your content is used in our services
See the Removing your content section to learn why and how we might remove user-generated content from our services
If you think that someone is infringing your intellectual property rights, you can send us notice of the infringement and well take appropriate action. For example, we suspend or close the Google Accounts of repeat copyright infringers as described in our Copyright Help Centre.
Here is a question about it:
Will Google help me if I think someone has taken and used content Ive created without my permission?
Answer with Yes/No."""
        final_response = generate_and_check_response(model, tokenizer, question, args)
        print(final_response)
    except Exception as e:
        print(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()