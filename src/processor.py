import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from pathlib import Path
from os.path import join as opj
from os import listdir

class ImageProcessor:
    def __init__(self):
        self.model_name = "Minthy/ToriiGate-v0.3"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
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
                print(f"Found tags file: {tags_path}")  # Debug print
                user_prompt += ' Also here are booru tags for better understanding of the picture, you can use them as reference.'
                user_prompt += f' <tags>\n{tags}\n</tags>'
            except Exception as e:
                print(f"Warning: Could not read tags from {tags_path}: {e}")
        elif use_tags:
            print(f"No tags file found at: {tags_path}")  # Debug print

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
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts[0].split('Assistant: ')[1]

    def process_batch(self, folder_path, prompt_type="json", use_tags=False):
        image_extensions = ['.jpg', '.png', '.webp', '.jpeg']
        results = {}
        
        files = [f for f in listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]
        
        for file in files:
            image_path = opj(folder_path, file)
            
            # Check for tags file with same name but .txt extension
            tags_path = opj(folder_path, Path(file).stem + ".txt")
            should_use_tags = use_tags and Path(tags_path).exists()
            
            try:
                caption = self.process_single_image(
                    image_path, 
                    prompt_type, 
                    should_use_tags, 
                    tags_path if should_use_tags else None
                )
                results[file] = caption
                
                # Save caption to file
                output_path = opj(folder_path, Path(file).stem + "_caption.txt")
                with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(caption)
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                results[file] = f"Error: {str(e)}"
                continue
                
        return results