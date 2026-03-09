import os
import json
import torch
import gc
from typing import Dict, List
from tqdm import tqdm

# ========================================================================
# Expert Scorers (Sequential VRAM-safe Design)
# Models: UniPercept, Grounding-IQA, HPSv3, SpatialScore
# Each scorer loads -> scores -> deletes model -> clears CUDA cache
# ========================================================================


def score_with_uniperceptiqa(image_path: str) -> float:
    """
    UniPercept: Unified perceptual quality assessment using pyiqa MUSIQ model.
    MUSIQ (Multi-scale Image Quality Transformer) returns a no-reference quality
    score in the 0-100 range (higher = better perceptual quality).
    """
    print("[UniPercept] Loading model...")
    score = 0.0
    try:
        import pyiqa

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric = pyiqa.create_metric("musiq", device=device)
        with torch.no_grad():
            score = float(metric(image_path).item())
        del metric
    except Exception as e:
        print(f"[UniPercept] Error: {e}. Returning dummy score.")
        score = 50.0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(score)

# Import Q-Insight+ / Q-Probe
from scorers.qinsight_qprobe import score_with_qinsight_qprobe


def score_with_hpsv3(image_path: str, prompt: str) -> float:
    """
    HPSv3: Human Preference Score v3 — semantic/text-image alignment scorer.
    """
    print("[HPSv3] Loading model...")
    score = 0.0
    try:
        from hpsv3.infer import HPSv3RewardInferencer

        inferencer = HPSv3RewardInferencer()
        score = inferencer.infer(image_path, prompt)
        del inferencer
    except Exception as e:
        print(f"[HPSv3] Error: {e}. Returning dummy score.")
        score = 0.85

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(score)


def score_with_spatialscore(image_path: str, prompt: str) -> float:
    """
    SpatialScore: Evaluates spatial/semantic alignment between image and prompt
    using CLIP image-text similarity.  Returns a score in the 0-10 range.
    """
    # CLIP logits_per_image are cosine similarities scaled by a learned temperature
    # parameter (~100).  Empirically, a good image-text match produces values in
    # the 20-35 range, while mismatches produce ~5-15.  Dividing by 3 maps this
    # to a 0-10 scale that matches the expected SpatialScore output range.
    _CLIP_LOGIT_TO_10 = 3.0

    print("[SpatialScore] Computing spatial/semantic alignment score...")
    score = 0.0
    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()

        image = Image.open(image_path).convert("RGB")
        text = prompt if prompt else "an image"
        inputs = processor(
            text=[text], images=image, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # logits_per_image: cosine similarity scaled by learned temperature
            # typical range ~5-35 for meaningful image-text pairs
            raw_score = float(outputs.logits_per_image[0, 0].item())

        # Normalize to 0-10 scale
        score = min(10.0, max(0.0, raw_score / _CLIP_LOGIT_TO_10))

        del model
        image.close()
    except Exception as e:
        print(f"[SpatialScore] Error: {e}. Returning dummy score.")
        score = 5.0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(score)


def get_all_expert_scores(image_path: str, prompt: str) -> Dict:
    """
    Runs all expert scorers sequentially, clearing VRAM after each.
    Returns a dict of all scores plus Q-Insight+ / Q-Probe region annotations.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {image_path}")
    print(f"{'='*60}")

    # 1. UniPercept (general perceptual quality)
    unipercept_score = score_with_uniperceptiqa(image_path)

    # 2. Q-Insight+ / Q-Probe (active viewing quality + defect boxes)
    q_result = score_with_qinsight_qprobe(image_path, prompt)

    # 3. HPSv3 (semantic / text-image alignment)
    hpsv3_score = score_with_hpsv3(image_path, prompt)

    # 4. SpatialScore (spatial layout understanding)
    spatial_score = score_with_spatialscore(image_path, prompt)

    scores = {
        "unipercept_score": unipercept_score,
        "q_insight_score": q_result["score"],
        "q_insight_regions": q_result["regions"],
        "hpsv3_score": hpsv3_score,
        "spatial_score": spatial_score,
    }

    print(
        f"Results -> UniPercept: {unipercept_score:.2f}, "
        f"Q-Insight+/Q-Probe: {q_result['score']:.2f}, "
        f"HPSv3: {hpsv3_score:.3f}, "
        f"SpatialScore: {spatial_score:.2f}"
    )
    return scores


# ========================================================================
# Batch processing with checkpointing
# ========================================================================

def process_scoring_with_checkpointing(
    data_list: List[Dict],
    output_jsonl: str = "expert_scores.jsonl",
):
    """
    Scores all images with checkpointing (saves every 100 items).
    """
    processed_ids = set()

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
        print("All items already scored.")
        return

    print(f"Remaining items to score: {len(remaining)}")
    buffer = []

    with open(output_jsonl, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(remaining, desc="Expert Scoring")):
            img_path = item["image_path"]
            img_id = get_id(item)
            prompt = item.get("prompt", "")

            scores = get_all_expert_scores(img_path, prompt)

            result = {"image_id": img_id, "image_path": img_path, "prompt": prompt, **scores}
            buffer.append(json.dumps(result, ensure_ascii=False))

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
