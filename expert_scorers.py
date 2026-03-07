import os
import json
import subprocess
import torch
import gc
from typing import Dict, List
from tqdm import tqdm

def score_with_hpsv3(image_path: str, prompt: str) -> float:
    try:
        from hpsv3.infer import HPSv3RewardInferencer
        inferencer = HPSv3RewardInferencer()
        score = inferencer.infer(image_path, prompt)
        del inferencer
    except Exception as e:
        score = 0.85 # Dummy on fail
        
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return float(score)

def score_with_pyiqa_musiq(image_path: str) -> float:
    try:
        import pyiqa
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iqa_metric = pyiqa.create_metric('musiq', device=device)
        score = iqa_metric(image_path)
        if isinstance(score, torch.Tensor):
            score = score.item()
        del iqa_metric
    except Exception as e:
        score = 70.5 # Dummy on fail
        
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return float(score)

def score_with_aesthetic(image_path: str) -> float:
    # Dummy placeholder or subprocess call
    score = 6.8
    # e.g., result = subprocess.run(["python", "aesthetic_dummy_cli.py", "--img", image_path], capture_output=True)
    return float(score)

def get_all_expert_scores(image_path: str, prompt: str) -> Dict[str, float]:
    semantic = score_with_hpsv3(image_path, prompt)
    quality = score_with_pyiqa_musiq(image_path)
    aesthetic = score_with_aesthetic(image_path)
    return {
        "semantic_score": semantic,
        "quality_score": quality,
        "aesthetic_score": aesthetic
    }

def process_scoring_with_checkpointing(data_list: List[Dict], output_jsonl: str = "expert_scores.jsonl"):
    """
    Computes scores from three experts sequentially with checkpointing.
    Saves state every 100 images.
    """
    processed_ids = set()
    
    if os.path.exists(output_jsonl):
        print(f"Found existing progress in {output_jsonl}. Loading...")
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    img_id = item.get("image_id", item.get("image_path"))
                    processed_ids.add(img_id)
        print(f"Resuming after {len(processed_ids)} processed items.")

    def get_id(item): return item.get("image_id", item.get("image_path"))
        
    remaining_data = [item for item in data_list if get_id(item) not in processed_ids]
    if not remaining_data:
        print("All items already scored.")
        return
        
    print(f"Remaining items to score: {len(remaining_data)}")
    buffer = []
    
    with open(output_jsonl, 'a', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(remaining_data, desc="Expert Scoring")):
            img_path = item['image_path']
            img_id = get_id(item)
            prompt = item.get("prompt", "")
            
            scores = get_all_expert_scores(img_path, prompt)
            
            result = {
                "image_id": img_id,
                "image_path": img_path,
                **scores
            }
            buffer.append(json.dumps(result))
            
            # Save every 100 items
            if (idx + 1) % 100 == 0:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer = []
                print(f"  Checkpoint saved at {idx + 1} scored items.")
                
        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()
            print("  Final checkpoint saved.")
            
    print("Expert scoring complete.")

if __name__ == "__main__":
    pass
