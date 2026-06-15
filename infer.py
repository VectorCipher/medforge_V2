import os
import argparse
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)

def main():
    parser = argparse.ArgumentParser(description="MedForge CT Inference")
    parser.add_argument("--model_id_or_path", type=str, required=True, help="Path to your fine-tuned LoRA checkpoint or base model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the CT scan image")
    args = parser.parse_args()

    print(f"Loading model from {args.model_id_or_path}...")
    
    # Load model and tokenizer
    # Using the standard Qwen3-VL-8B configuration
    model_type = ModelType.qwen3_vl_8b_instruct
    template_type = get_default_template_type(model_type)
    
    model, tokenizer = get_model_tokenizer(
        model_type, 
        model_id_or_path=args.model_id_or_path,
        model_kwargs={'device_map': 'auto', 'torch_dtype': 'bfloat16'}
    )
    
    # Get the chat template
    template = get_template(template_type, tokenizer)
    
    # Construct the prompt exactly as used during training
    system_prompt = (
        "<image>\n"
        "Please act as a medical forensics expert. Examine this CT lung scan carefully "
        "and detect if there is any deepfake manipulation (e.g., lesion implant or removal). "
        "Provide your step-by-step reasoning, localize the manipulated region with a bounding box if any, "
        "and state your final conclusion."
    )
    
    query = system_prompt
    
    # Provide the image
    images = [args.image_path]

    print(f"Running inference on {args.image_path}...")
    
    # Generate the response
    # We set temperature=0.1 to keep the reasoning highly deterministic and factual
    response, history = inference(
        model, template, query, images=images,
        temperature=0.1, max_new_tokens=1024
    )
    
    print("\n" + "="*50)
    print("MODEL OUTPUT:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    main()
