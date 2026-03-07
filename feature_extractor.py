import os
import json
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import Union, List, Dict
from tqdm import tqdm

class Dinov2FeatureExtractor:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on {self.device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 model: {e}")

    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return np.array([])
            
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return cls_features

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_features_with_checkpointing(data_list: List[Dict], output_jsonl: str = "features.jsonl"):
    """
    Process images to extract features with resume and checkpointing logic.
    Saves state every 100 images.
    """
    processed_ids = set()
    
    # 1. Load existing progress
    if os.path.exists(output_jsonl):
        print(f"Found existing progress in {output_jsonl}. Loading...")
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # We use image_path or generate an ID as unique identifier
                    img_id = item.get("image_id", item.get("image_path"))
                    processed_ids.add(img_id)
        print(f"Resuming after {len(processed_ids)} processed items.")

    # 2. Filter remaining data
    def get_id(item):
        return item.get("image_id", item.get("image_path"))
        
    remaining_data = [item for item in data_list if get_id(item) not in processed_ids]
    if not remaining_data:
        print("All items already processed.")
        return
        
    print(f"Remaining items to process: {len(remaining_data)}")
    
    extractor = Dinov2FeatureExtractor()
    buffer = []
    
    # 3. Process with checkpointing
    # Open file in append mode
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(remaining_data, desc="Extracting Features")):
            img_path = item['image_path']
            img_id = get_id(item)
            
            features = extractor.extract_features(img_path)
            
            result = {
                "image_id": img_id,
                "image_path": img_path,
                "dinov2_cls_feature": features.tolist() if features.size > 0 else []
            }
            buffer.append(json.dumps(result))
            
            # Save every 100 images
            if (idx + 1) % 100 == 0:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer = []
                print(f"  Checkpoint saved at {idx + 1} items.")
                
        # Save remaining in buffer
        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()
            print("  Final checkpoint saved.")
            
    extractor.cleanup()
    print("Feature extraction complete.")

if __name__ == "__main__":
    # Example usage:
    # from data_loader import get_dataloader
    # data = get_dataloader('local', './test_images')
    # process_features_with_checkpointing(data)
    pass
