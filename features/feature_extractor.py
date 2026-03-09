import os
import json
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel, CLIPTokenizer, CLIPTextModel
from PIL import Image
from typing import Union, List, Dict
from tqdm import tqdm


# ========================================================================
# DINOv2 Feature Extractor
# ========================================================================

class Dinov2FeatureExtractor:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DINOv2] Loading {model_name} on {self.device}...")
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
            print(f"[DINOv2] Error opening {image_path}: {e}")
            return np.array([])

        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return cls_features

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ========================================================================
# CLIP Text Feature Extractor
# ========================================================================

class ClipTextFeatureExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[CLIP-Text] Loading {model_name} on {self.device}...")
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP text model: {e}")

    @torch.no_grad()
    def extract_text_features(self, text: str) -> np.ndarray:
        if not text:
            return np.array([])

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            outputs = self.model(**inputs)
            # Use the pooled output (EOS token representation)
            text_features = outputs.pooler_output.cpu().numpy()[0]
            return text_features
        except Exception as e:
            print(f"[CLIP-Text] Error extracting features: {e}")
            return np.array([])

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ========================================================================
# Combined processing with checkpointing
# ========================================================================

def process_features_with_checkpointing(
    data_list: List[Dict],
    output_jsonl: str = "features.jsonl",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    dinov2_model_name: str = "facebook/dinov2-base",
):
    """
    Extract DINOv2 CLS features (image) AND CLIP text features (prompt)
    with checkpointing every 100 items.
    """
    processed_ids = set()

    # 1. Load existing progress
    if os.path.exists(output_jsonl):
        print(f"Found existing progress in {output_jsonl}. Loading...")
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_ids.add(item.get("image_id", item.get("image_path")))
        print(f"Resuming after {len(processed_ids)} processed items.")

    def get_id(item):
        return item.get("image_id", item.get("image_path"))

    remaining = [item for item in data_list if get_id(item) not in processed_ids]
    if not remaining:
        print("All items already processed.")
        return

    print(f"Remaining items to process: {len(remaining)}")

    # 2. Load both models
    dinov2 = Dinov2FeatureExtractor(model_name=dinov2_model_name)
    clip_text = ClipTextFeatureExtractor(model_name=clip_model_name)

    buffer = []

    # 3. Process
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(remaining, desc="Extracting Features")):
            img_path = item["image_path"]
            img_id = get_id(item)
            prompt = item.get("prompt", "")

            # DINOv2 image feature
            dinov2_feat = dinov2.extract_features(img_path)

            # CLIP text feature
            clip_feat = clip_text.extract_text_features(prompt) if prompt else np.array([])

            result = {
                "image_id": img_id,
                "image_path": img_path,
                "dinov2_cls_feature": dinov2_feat.tolist() if dinov2_feat.size > 0 else [],
                "clip_text_feature": clip_feat.tolist() if clip_feat.size > 0 else [],
            }
            buffer.append(json.dumps(result, ensure_ascii=False))

            # Checkpoint every 100
            if (idx + 1) % 100 == 0:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer = []
                print(f"  Checkpoint saved at {idx + 1} items.")

        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()
            print("  Final checkpoint saved.")

    # 4. Cleanup models
    dinov2.cleanup()
    clip_text.cleanup()
    print("Feature extraction complete (DINOv2 + CLIP-Text).")


if __name__ == "__main__":
    pass
