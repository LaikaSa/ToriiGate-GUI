import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from pathlib import Path
from os.path import join as opj
from os import listdir

class ImageProcessor:
    def __init__(self, quantization_mode=None):
        self.model_name = "Minthy/ToriiGate-v0.3"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Configure model loading based on quantization mode
        if quantization_mode == "4bit":
            print("Loading model in 4-bit mode...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quantization_config,
            )
        elif quantization_mode == "8bit":
            print("Loading model in 8-bit mode...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            print("Loading model in default mode...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        
    def process_single_image(self, image_path, prompt_type="json", use_tags=False, tags_path=None):
        image = load_image(image_path)
        
        if prompt_type == "json":
            user_prompt = "Describe the picture in structuted json-like format."
        elif prompt_type == "detailed":
            user_prompt = "Give a long and detailed description of the picture."
        else:
            user_prompt = "Describe the picture briefly."
                
        if use_tags and tags_path and Path(tags_path).exists():
            try:
                with open(tags_path, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()
                print(f"Using tags from: {Path(tags_path).name}")
                user_prompt += ' Also here are booru tags for better understanding of the picture, you can use them as reference.'
                user_prompt += f' <tags>\n{tags}\n</tags>'
            except Exception as e:
                print(f"✗ Warning: Could not read tags from {tags_path}: {e}")

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensored. Help user with his task."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        
        # Get the correct device for inputs
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts[0].split('Assistant: ')[1]

    def process_batch(self, folder_path, prompt_type="json", use_tags=False):
        image_extensions = ['.jpg', '.png', '.webp', '.jpeg']
        results = {}
        
        files = [f for f in listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]
        print(f"\nFound {len(files)} images to process")
        
        for file in files:
            image_path = opj(folder_path, file)
            print(f"\nProcessing: {file}")
            
            # Get base name without extension and look for matching .txt file
            base_name = Path(file).stem
            tags_path = opj(folder_path, f"{base_name}.txt")
            
            if use_tags:
                if Path(tags_path).exists():
                    try:
                        with open(tags_path, 'r', encoding='utf-8') as f:
                            tags = f.read().strip()
                        print(f"✓ Found tags file: {base_name}.txt")
                    except Exception as e:
                        print(f"✗ Error reading tags file: {e}")
                        tags = None
                else:
                    print(f"✗ No tags file found for: {base_name}.txt")
                    tags = None
            else:
                tags = None
            
            try:
                caption = self.process_single_image(
                    image_path, 
                    prompt_type, 
                    use_tags and tags is not None,  # Only use tags if they were successfully read
                    tags_path if tags is not None else None
                )
                results[file] = caption
                
                # Save caption to file
                output_path = opj(folder_path, f"{base_name}_caption.txt")
                with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(caption)
                print(f"✓ Saved caption to: {Path(output_path).name}")
                    
            except Exception as e:
                error_msg = f"Error processing {file}: {str(e)}"
                print(f"✗ {error_msg}")
                results[file] = f"Error: {str(e)}"
                continue
                
        return results