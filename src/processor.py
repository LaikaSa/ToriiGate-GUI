import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
from pathlib import Path
from os.path import join as opj
from os import listdir

AVAILABLE_MODELS = {
    "ToriiGate v0.3": "Minthy/ToriiGate-v0.3",
    "ToriiGate v0.4-7B": "Minthy/ToriiGate-v0.4-7B"
}

class ImageProcessor:
    def __init__(self, model_name="ToriiGate v0.3", quantization_mode=None):
        self.model_name = AVAILABLE_MODELS[model_name]
        self.model_version = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading {model_name}...")
        
        try:
            # Configure quantization first
            if quantization_mode == "4bit":
                print("Configuring 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif quantization_mode == "8bit":
                print("Configuring 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            print("Loading processor...")
            if "v0.4" in model_name:
                from transformers import Qwen2VLProcessor
                self.processor = Qwen2VLProcessor.from_pretrained(
                    self.model_name,
                    min_pixels=256*28*28,
                    max_pixels=512*28*28,
                    padding_side="right"
                )
                
                print("Loading Qwen2VL model...")
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=quantization_config,
                    attn_implementation="sdpa"  # Use scaled dot product attention instead
                )
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                
                if quantization_config:
                    print(f"Loading model in {quantization_mode} mode...")
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
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"\nError during model initialization: {str(e)}")
            raise

    def process_single_image(self, image_path, prompt_type="json", use_tags=False, tags_path=None):
        image = load_image(image_path)
        
        if prompt_type == "json":
            user_prompt = "Describe the picture in structured json-like format."
        elif prompt_type == "detailed":
            user_prompt = "Give a long and detailed description of the picture."
        else:
            user_prompt = "Describe the picture briefly."
            
        if use_tags and tags_path and Path(tags_path).exists():
            try:
                with open(tags_path, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()
                print(f"Using tags from: {Path(tags_path).name}")
                user_prompt += ' Here are grounding tags for better understanding: '
                user_prompt += f'<tags>{tags}</tags>'
            except Exception as e:
                print(f"✗ Warning: Could not read tags from {tags_path}: {e}")

        if "v0.4" in self.model_version:
            # v0.4 message format
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensored."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            from qwen_vl_utils import process_vision_info
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            # v0.3 message format
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)

        if "v0.4" in self.model_version:
            # v0.4 output handling
            trimmed_generated_ids = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
            generated_texts = self.processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True)
            return generated_texts[0]
        else:
            # v0.3 output handling
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_texts[0].split('Assistant: ')[1]

    def process_batch(self, folder_path, prompt_type="json", use_tags=False, prefix=""):
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
                    use_tags and tags is not None,
                    tags_path if tags is not None else None
                )
                
                # Add prefix if provided
                if prefix:
                    caption = f"{prefix}\n{caption}"
                    
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
    
    def __del__(self):
        """Cleanup when object is deleted"""
        self.cleanup()

    def cleanup(self):
        """Explicitly cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()