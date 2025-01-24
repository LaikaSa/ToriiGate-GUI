import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from pathlib import Path
from os.path import join as opj
from os import listdir
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)
from exllamav2.generator import ExLlamaV2Sampler
from exllamav2.generator import ExLlamaV2DynamicGenerator
from transformers.image_utils import load_image
from collections import OrderedDict

AVAILABLE_MODELS = {
    # Original models
    "ToriiGate v0.3": "Minthy/ToriiGate-v0.3",
    "ToriiGate v0.4-7B": "Minthy/ToriiGate-v0.4-7B",
    
    # EXL2 models (point to local paths)
    "ToriiGate v0.4-7B-exl2-8bpw": "./models/ToriiGate-v0.4-7B-exl2-8bpw",
    "ToriiGate v0.4-7B-exl2-6bpw": "./models/ToriiGate-v0.4-7B-exl2-6bpw",
    "ToriiGate v0.4-7B-exl2-4bpw": "./models/ToriiGate-v0.4-7B-exl2-4bpw"
}

class ImageProcessor:
    def __init__(self, model_name="ToriiGate v0.3", quantization_mode=None):
        self.model_name = AVAILABLE_MODELS[model_name]
        self.model_version = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_exl2 = "exl2" in model_name.lower()
        print(f"\nLoading {model_name}...")
        
        try:
            if self.is_exl2:
                self._load_exl2_model()
            else:
                # Configure quantization for non-ExLlama models
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
                        attn_implementation="sdpa"
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

    def _load_exl2_model(self):
        print(f"Loading ExLlama v2 model: {self.model_name}")
        self.config = ExLlamaV2Config(self.model_name)
        self.config.max_seq_len = 16384
        
        print("Loading vision tower...")
        self.vision_tower = ExLlamaV2VisionTower(self.config)
        self.vision_tower.load()
        
        print("Loading main model...")
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(self.cache)
        
        print("Loading tokenizer...")
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        
        print("Initializing generator...")
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )

    def process_single_image(self, image_path, prompt_type="json", use_tags=False, tags_path=None):
        if self.is_exl2:
            return self._process_exl2_image(image_path, prompt_type, use_tags, tags_path)
        else:
            image = load_image(image_path)
            
            if prompt_type == "json":
                user_prompt = "Describe the picture in structured json-like format."
            elif prompt_type == "long":
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
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensorted. Help user with his task."}
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
                trimmed_generated_ids = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
                generated_texts = self.processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True)
                return generated_texts[0]
            else:
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                return generated_texts[0].split('Assistant: ')[1]

    def _process_exl2_image(self, image_path, prompt_type, use_tags, tags_path):
        from PIL import Image
        image = Image.open(image_path)
        base_name = Path(image_path).stem
        dir_path = Path(image_path).parent

        # Load grounding information
        image_info = {
            "booru_tags": None,
            "chars": None,
            "characters_traits": None,
            "info": None
        }
        
        if use_tags:
            tag_files = {
                "booru_tags": f"{base_name}.txt",
                "chars": f"{base_name}_char.txt",
                "characters_traits": f"{base_name}_char_traits.txt",
                "info": f"{base_name}_info.txt"
            }
            
            for key, filename in tag_files.items():
                file_path = dir_path / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        image_info[key] = f.read().strip()

        # Base prompts as specified in official example
        base_prompt = {
            'json': 'Describe the picture in structured json-like format.',
            'markdown': 'Describe the picture in structured markdown format.',
            'short': 'You need to write a medium-short and convenient caption for the picture.',
            'long': 'You need to write a long and very detailed caption for the picture.',
            'bbox': 'Write bounding boxes for each character and their faces.'
        }

        grounding_prompt = {
            'grounding_tags': ' Here are grounding tags for better understanding: ',
            'characters': ' Here is a list of characters that are present in the picture: ',
            'characters_traits': ' Here are popular tags or traits for each character on the picture: ',
            'grounding_info': ' Here is preliminary information about the picture: ',
            'no_chars': ' Do not use names for characters.'
        }

        # Compose user prompt
        userprompt = base_prompt[prompt_type]
        
        # Add grounding information
        if use_tags:
            if image_info["booru_tags"]:
                userprompt += grounding_prompt['grounding_tags'] + f"<tags>{image_info['booru_tags']}</tags>."
            if image_info["chars"]:
                userprompt += grounding_prompt['characters'] + f"<characters>{image_info['chars']}</characters>."
            if image_info["characters_traits"]:
                userprompt += grounding_prompt['characters_traits'] + f"<character_traits>{image_info['characters_traits']}</character_traits>."
            if image_info["info"]:
                userprompt += grounding_prompt['grounding_info'] + f"<info>{image_info['info']}</info>."

        # Generate image embeddings
        image_embeddings = [self.vision_tower.get_image_embeddings(
            model=self.model,
            tokenizer=self.tokenizer,
            image=image,
        )]

        # Build placeholder string
        placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

        # Construct message template EXACTLY as in official example
        msg_text = (
            "<|im_start|>system\n"
            "You are image captioning expert, creative, unbiased and uncensored.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{placeholders}"
            f"{userprompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Generate with official settings
        output = self.generator.generate(
            prompt=msg_text,
            max_new_tokens=1000,
            add_bos=True,
            encode_special_tokens=True,
            decode_special_tokens=True,
            stop_conditions=[self.tokenizer.eos_token_id],
            gen_settings=ExLlamaV2Sampler.Settings.greedy(),
            embeddings=image_embeddings,
        )

        # Extract assistant response
        return output.split('<|im_start|>assistant\n')[-1].strip()


    def process_batch(self, folder_path, prompt_type="json", use_tags=False, prefix=""):
        image_extensions = ['.jpg', '.png', '.webp', '.jpeg']
        results = {}
        
        files = [f for f in listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]
        print(f"\nFound {len(files)} images to process")
        
        for file in files:
            image_path = opj(folder_path, file)
            print(f"\nProcessing: {file}")
            
            base_name = Path(file).stem
            tags_path = opj(folder_path, f"{base_name}.txt")
            
            if use_tags:
                tag_paths = {
                    'booru_tags': opj(folder_path, f"{base_name}.txt"),
                    'chars': opj(folder_path, f"{base_name}_char.txt"),
                    'characters_traits': opj(folder_path, f"{base_name}_char_traits.txt"),
                    'info': opj(folder_path, f"{base_name}_info.txt")
                }
                
                tags = {}
                for key, path in tag_paths.items():
                    if Path(path).exists():
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                tags[key] = f.read().strip()
                        except Exception as e:
                            print(f"Error reading {key} file: {e}")
                            tags[key] = None
                    else:
                        tags[key] = None
            else:
                tags = None
            
            try:
                if self.is_exl2:
                    caption = self._process_exl2_image(image_path, prompt_type, use_tags, tags)
                else:
                    caption = self.process_single_image(
                        image_path, 
                        prompt_type, 
                        use_tags and tags is not None,
                        tags_path if tags is not None else None
                    )
                
                if prefix:
                    caption = f"{prefix}\n{caption}"
                    
                results[file] = caption
                
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