import json
import argparse
import os
import re

def reformat_annotation(item):
    """
    Transforms the user's raw JSON item into the exact structure
    expected by MedForge-Reasoner (ms-swift multimodal format).
    """
    # 1. Image Path
    # Make sure this points to wherever your images are actually stored on Kaggle.
    raw_image_path = item.get("image", "")
    
    # Remove 'images/' prefix if it exists in the JSON path to match your Kaggle structure
    if raw_image_path.startswith("images/"):
        raw_image_path = raw_image_path[len("images/"):]
        
    # Append the Kaggle dataset base path
    kaggle_base_path = "/kaggle/input/datasets/naitikpal/ct-v2-medforge"
    image_path = os.path.join(kaggle_base_path, raw_image_path).replace("\\", "/")
    
    # 2. Extract components
    label = item.get("label", "real").lower()
    evidence = item.get("evidence", "")
    conclusion = item.get("conclusion", "")
    
    # Clean the raw annotation to become pure 'think' process
    raw_anno = item.get("raw_annotation", "")
    
    # Remove the <evidence> and <conclusion> tags from inside the raw thought,
    # as MedForge expects them sequentially AFTER the <think> block.
    think_content = raw_anno
    think_content = re.sub(r'<evidence>.*?</evidence>\.?\s*', '', think_content, flags=re.DOTALL)
    think_content = re.sub(r'<conclusion>.*?</conclusion>\.?\s*', '', think_content, flags=re.DOTALL)
    
    # Clean up lingering <think> tags if they exist inside the text
    think_content = think_content.replace("<think>", "").replace("</think>", "").strip()

    # 3. Build the Target String (Ground Truth)
    target_output = f"<think>\n{think_content}\n</think>\n"
    
    # Bounding Box Logic:
    # If the image is a deepfake, append the bounding box.
    # We normalize coordinates to 0-1000 if Qwen3 requires it, but standard pixel coords
    # are often fine if the prompt specifies it. Assuming Qwen-VL standard 0-1000:
    # However, if your coordinates are raw pixels, we output them as is.
    if label != "real" and "bbox" in item:
        bbox = item["bbox"]
        # Format: <box class="deepfake" x1="197" y1="318" x2="237" y2="358" />
        target_output += f'<box class="deepfake" x1="{bbox["x1"]}" y1="{bbox["y1"]}" x2="{bbox["x2"]}" y2="{bbox["y2"]}" />\n'
    
    # Append Evidence and Conclusion
    target_output += f"<evidence>\n{evidence}\n</evidence>\n"
    target_output += f"<conclusion>\n{conclusion}\n</conclusion>"
    
    # 4. Construct MS-Swift Message Array
    # This is the prompt the model sees during training.
    system_prompt = (
        "<image>\n"
        "Please act as a medical forensics expert. Examine this CT lung scan carefully "
        "and detect if there is any deepfake manipulation (e.g., lesion implant or removal). "
        "Provide your step-by-step reasoning, localize the manipulated region with a bounding box if any, "
        "and state your final conclusion."
    )
    
    ms_swift_format = {
        "images": [image_path],
        "messages": [
            {
                "role": "user",
                "content": system_prompt
            },
            {
                "role": "assistant",
                "content": target_output
            }
        ]
    }
    
    return ms_swift_format

def main():
    parser = argparse.ArgumentParser(description="Convert custom CT dataset to MS-Swift format")
    parser.add_argument("--input", type=str, required=True, help="Path to your input JSON or JSONL file")
    parser.add_argument("--output", type=str, default="sft_train_ct.json", help="Path to output JSON file")
    args = parser.parse_args()
    
    formatted_dataset = []
    
    # Read the data (handling both standard JSON array and JSONL)
    print(f"Reading from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f) # Try standard JSON list
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()] # Fallback to JSONL
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    # Process each item
    for idx, item in enumerate(data):
        try:
            swift_item = reformat_annotation(item)
            formatted_dataset.append(swift_item)
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            
    # Save the output
    print(f"Saving {len(formatted_dataset)} samples to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(formatted_dataset, f, indent=2, ensure_ascii=False)
        
    print("Done! Your dataset is ready for ms-swift SFT.")

if __name__ == "__main__":
    main()
