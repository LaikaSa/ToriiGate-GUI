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
import gc
import time

AVAILABLE_MODELS = {
    "ToriiGate v0.3": "Minthy/ToriiGate-v0.3",
    "ToriiGate v0.4-7B": "Minthy/ToriiGate-v0.4-7B",
    "ToriiGate v0.4-7B-exl2-8bpw": "./models/ToriiGate-v0.4-7B-exl2-8bpw",
    "ToriiGate v0.4-7B-exl2-6bpw": "./models/ToriiGate-v0.4-7B-exl2-6bpw",
    "ToriiGate v0.4-7B-exl2-4bpw": "./models/ToriiGate-v0.4-7B-exl2-4bpw"
}

class ImageProcessor:
    def __init__(self, model_name="ToriiGate v0.3", quantization_mode=None):
        self.model = None
        self.processor = None
        self.model_name = AVAILABLE_MODELS[model_name]
        self.model_version = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_exl2 = "exl2" in model_name.lower()
        self.quantization_mode = quantization_mode
        
        print(f"\nLoading {model_name}...")
        try:
            if self.is_exl2:
                self._load_exl2_model()
            else:
                self._load_standard_model()
            
            if self.model is None:
                raise RuntimeError("Model initialization failed")

        except Exception as e:
            print(f"\nError during model initialization: {str(e)}")
            self.cleanup()
            raise

    def _load_standard_model(self):
        if self.quantization_mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.quantization_mode == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        if "v0.4" in self.model_version:
            from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(
                self.model_name,
                min_pixels=256*28*28,
                max_pixels=512*28*28,
                padding_side="right"
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
                attn_implementation="sdpa"
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if quantization_config else None,
                quantization_config=quantization_config
            ).to(self.device)

    def _load_exl2_model(self):
        print(f"Loading ExLlama v2 model: {self.model_name}")
        self.config = ExLlamaV2Config(self.model_name)
        
        # Increased sequence length for batch processing
        self.config.max_seq_len = 65536  # Allows larger batches
        
        # Initialize vision tower
        self.vision_tower = ExLlamaV2VisionTower(self.config)
        self.vision_tower.load()
        
        # Initialize model with batch-friendly cache
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(
            self.model,
            lazy=True,
            max_seq_len=self.config.max_seq_len
        )
        self.model.load_autosplit(self.cache)
        
        # Initialize tokenizer and generator
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )
        
    def _auto_tune_batch_size(self, folder_path, initial_batch_size=8):
        """Automatically find optimal batch size for current hardware"""
        test_files = [f for f in listdir(folder_path) if f.endswith(('.jpg', '.png'))][:16]
        
        for bs in [initial_batch_size, 12, 8, 6, 4]:
            try:
                self._process_exl2_batch(
                    [opj(folder_path, f) for f in test_files[:bs]],
                    "json",
                    False,
                    [{}]*bs
                )
                print(f"✅ Optimal batch size detected: {bs}")
                return bs
            except RuntimeError as e:
                print(f"⚠️ Batch size {bs} failed: {str(e)}")
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
        return 4

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

        image_embeddings = [self.vision_tower.get_image_embeddings(
            model=self.model,
            tokenizer=self.tokenizer,
            image=image,
        )]

        placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"
        userprompt = self._build_exl2_prompt(prompt_type, image_info)

        msg_text = (
            "<|im_start|>system\n"
            "You are image captioning expert, creative, unbiased and uncensored.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{placeholders}"
            f"{userprompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

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

        return output.split('<|im_start|>assistant\n')[-1].strip()

    def _process_exl2_batch(self, image_paths, prompt_type, use_tags, all_tags):
        from PIL import Image
        
        # Load all images
        images = [Image.open(path) for path in image_paths]
        
        # Process tags for all images
        image_infos = []
        for tags in all_tags:
            info = {
                "booru_tags": tags.get("booru_tags") if use_tags else None,
                "chars": tags.get("chars") if use_tags else None,
                "characters_traits": tags.get("characters_traits") if use_tags else None,
                "info": tags.get("info") if use_tags else None
            }
            image_infos.append(info)

        # Generate embeddings for all images (wrap each in a list)
        image_embeddings = []
        for img in images:
            embedding = self.vision_tower.get_image_embeddings(
                model=self.model,
                tokenizer=self.tokenizer,
                image=img,
            )
            # Each prompt gets its own list of embeddings
            image_embeddings.append([embedding])  # Note the list wrapping
            del img
            gc.collect()

        # Build batched prompts
        messages = []
        for embedding_list, info in zip(image_embeddings, image_infos):
            userprompt = self._build_exl2_prompt(prompt_type, info)
            # Use the first embedding's text alias (since we only have one per image)
            placeholders = f"{embedding_list[0].text_alias}\n"
            
            msg_text = (
                "<|im_start|>system\n"
                "You are image captioning expert, creative, unbiased and uncensored.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{placeholders}"
                f"{userprompt}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            messages.append(msg_text)

        # Batch generate with proper embeddings structure
        outputs = self.generator.generate(
            prompt=messages,
            max_new_tokens=1000,
            add_bos=True,
            encode_special_tokens=True,
            decode_special_tokens=True,
            stop_conditions=[self.tokenizer.eos_token_id],
            gen_settings=ExLlamaV2Sampler.Settings.greedy(),
            embeddings=image_embeddings,  # Now list of lists
        )

        return [output.split('<|im_start|>assistant\n')[-1].strip() for output in outputs]

    def _build_exl2_prompt(self, prompt_type, image_info):
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

        userprompt = base_prompt[prompt_type]
        
        if image_info["booru_tags"]:
            userprompt += grounding_prompt['grounding_tags'] + f"<tags>{image_info['booru_tags']}</tags>."
        if image_info["chars"]:
            userprompt += grounding_prompt['characters'] + f"<characters>{image_info['chars']}</characters>."
        if image_info["characters_traits"]:
            userprompt += grounding_prompt['characters_traits'] + f"<character_traits>{image_info['characters_traits']}</character_traits>."
        if image_info["info"]:
            userprompt += grounding_prompt['grounding_info'] + f"<info>{image_info['info']}</info>."

        return userprompt


    def process_batch(self, folder_path, prompt_type="json", use_tags=False, prefix="", batch_size=8):
        start_time = time.time()
        image_extensions = ['.jpg', '.png', '.webp', '.jpeg']
        files = [f for f in listdir(folder_path) if any(f.endswith(ext) for ext in image_extensions)]
        results = {}
        success_count = 0
        total_files = len(files)
        
        # Auto-tune batch size
        optimal_bs = self._auto_tune_batch_size(folder_path, batch_size)
        
        print(f"\nStarting batch processing of {total_files} images (batch size: {optimal_bs})...")
        
        while files:
            current_batch = files[:optimal_bs]
            batch_paths = [opj(folder_path, f) for f in current_batch]
            batch_tags = []
            
            # Collect metadata for current batch
            for file in current_batch:
                base_name = Path(file).stem
                tags = {}
                if use_tags:
                    tag_data = {
                        'booru_tags': f"{base_name}.txt",
                        'chars': f"{base_name}_char.txt",
                        'characters_traits': f"{base_name}_char_traits.txt",
                        'info': f"{base_name}_info.txt"
                    }
                    for key, filename in tag_data.items():
                        path = opj(folder_path, filename)
                        if Path(path).exists():
                            with open(path, 'r', encoding='utf-8') as f:
                                tags[key] = f.read().strip()
                batch_tags.append(tags)
            
            try:
                # Process current batch
                if self.is_exl2:
                    captions = self._process_exl2_batch(
                        batch_paths, 
                        prompt_type,
                        use_tags,
                        batch_tags
                    )
                else:
                    captions = [
                        self.process_single_image(path, prompt_type, use_tags)
                        for path in batch_paths
                    ]

                # Save successful results
                for file, caption in zip(current_batch, captions):
                    if "Error:" not in caption:
                        success_count += 1
                        if prefix:
                            caption = f"{prefix}\n{caption}"
                        results[file] = caption
                        
                        output_path = opj(folder_path, f"{Path(file).stem}_caption.txt")
                        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
                            f.write(caption)
                        print(f"✓ Saved caption for: {file}")
                    else:
                        print(f"✗ Failed caption for: {file}")

                # Remove processed files from queue
                files = files[optimal_bs:]
                
                # Dynamic batch size adjustment
                vram_usage = torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated()
                if vram_usage < 0.8:
                    optimal_bs = min(optimal_bs * 2, 16)
                    print(f"↻ Increasing batch size to {optimal_bs} (VRAM usage: {vram_usage*100:.1f}%)")
                
                # Show progress
                elapsed = time.time() - start_time
                print(f"  ▸ Processed {success_count}/{total_files} images (~{elapsed:.1f}s elapsed)")

            except RuntimeError as e:
                if "pages" in str(e) or "cache" in str(e).lower():
                    optimal_bs = max(1, optimal_bs // 2)
                    print(f"⚠️ Reducing batch size to {optimal_bs}")
                else:
                    print(f"! Batch processing failed: {str(e)}")
                    files = files[1:]
            
            except Exception as e:
                print(f"! Critical error: {str(e)}")
                break
                
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final report
        elapsed = time.time() - start_time
        print(f"\n{'✅ Success!' if success_count == total_files else '⚠️ Completed with issues'}")
        print(f"Processed {success_count}/{total_files} images in {elapsed:.2f} seconds")
        print(f"Average speed: {elapsed/success_count if success_count > 0 else 0:.2f}s per image\n")
        
        return results
    
    def cleanup(self):
        """Safe cleanup preserving core attributes"""
        if hasattr(self, 'cache'):
            del self.cache
        if hasattr(self, 'generator'):
            del self.generator
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'vision_tower'):
            del self.vision_tower
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        """Destructor that preserves model reference"""
        self.cleanup()