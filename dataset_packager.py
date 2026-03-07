import os
import json
import numpy as np
from typing import Dict, List, Optional


def _load_jsonl(path: str) -> List[Dict]:
    data = []
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_lookup(data: List[Dict], key: str = "image_id") -> Dict[str, Dict]:
    lookup = {}
    for item in data:
        k = item.get(key, item.get("image_path", ""))
        if k:
            lookup[k] = item
    return lookup


def compute_variance(item: Dict) -> float:
    """Compute score variance across expert dimensions."""
    scores = []
    for key in ("unipercept_score", "grounding_iqa_score", "hpsv3_score", "spatial_score"):
        val = item.get(key)
        if val is not None:
            scores.append(float(val))
    if len(scores) < 2:
        return 0.0
    return float(np.var(scores))


def package_final_dataset(
    features_jsonl: str = "features.jsonl",
    scores_jsonl: str = "expert_scores.jsonl",
    cot_jsonl: str = "train_reasoning_data.jsonl",
    mos_json: str = "mos_calibration_data.json",
    metadata_jsonl: Optional[str] = None,
    output_basic: str = "Dataset_basic.jsonl",
    output_reasoning: str = "Dataset_Reasoning.jsonl",
    variance_threshold: float = 1.0,
):
    """
    Aligns all data sources by image_id and splits into TWO output files:
      - Dataset_basic.jsonl      → low-conflict samples (variance <= threshold)
      - Dataset_Reasoning.jsonl  → high-conflict samples that have CoT analysis
    """
    print("=== Dataset Packager (Split Mode) ===")

    # 1. Load all sources
    features_data = _load_jsonl(features_jsonl)
    scores_data = _load_jsonl(scores_jsonl)
    cot_data = _load_jsonl(cot_jsonl)
    mos_data = _load_json(mos_json)

    metadata_data = _load_jsonl(metadata_jsonl) if metadata_jsonl else []

    # 2. Build lookup tables
    feat_lookup = _build_lookup(features_data)
    score_lookup = _build_lookup(scores_data)
    cot_lookup = _build_lookup(cot_data)
    meta_lookup = _build_lookup(metadata_data) if metadata_data else {}

    # MOS lookup (keyed by image_path or image_id)
    mos_lookup = {}
    for path_key, val in mos_data.items():
        mos_lookup[path_key] = val
        candidate_id = val.get("image_id", path_key)
        mos_lookup[candidate_id] = val

    # 3. Canonical ID set (union of features + scores)
    all_ids = set()
    all_ids.update(feat_lookup.keys())
    all_ids.update(score_lookup.keys())
    for path_key in mos_data:
        all_ids.add(path_key)

    print(f"Total unique image IDs across all sources: {len(all_ids)}")

    # 4. Merge & split
    basic_count = 0
    reasoning_count = 0

    with open(output_basic, "w", encoding="utf-8") as f_basic, \
         open(output_reasoning, "w", encoding="utf-8") as f_reason:

        for img_id in sorted(all_ids):
            feat_item = feat_lookup.get(img_id, {})
            score_item = score_lookup.get(img_id, {})
            cot_item = cot_lookup.get(img_id, {})
            meta_item = meta_lookup.get(img_id, {})
            mos_item = mos_lookup.get(img_id, {})

            record = {
                "image_id": img_id,
                "image_path": (
                    score_item.get("image_path")
                    or feat_item.get("image_path")
                    or meta_item.get("image_path")
                    or img_id
                ),
                "prompt": meta_item.get("prompt", score_item.get("prompt", "")),
                # Features
                "dinov2_cls_feature": feat_item.get("dinov2_cls_feature", []),
                "clip_text_feature": feat_item.get("clip_text_feature", []),
                # Expert scores
                "unipercept_score": score_item.get("unipercept_score", None),
                "grounding_iqa_score": score_item.get("grounding_iqa_score", None),
                "grounding_iqa_regions": score_item.get("grounding_iqa_regions", []),
                "hpsv3_score": score_item.get("hpsv3_score", None),
                "spatial_score": score_item.get("spatial_score", None),
                # Human MOS
                "human_mos": mos_item.get("mos", None),
            }

            # Decide: basic vs reasoning
            var = compute_variance(score_item)
            has_cot = bool(cot_item.get("cot_analysis", "").strip())

            if var > variance_threshold and has_cot:
                # High-conflict WITH CoT → Reasoning dataset
                record["cot_analysis"] = cot_item["cot_analysis"]
                record["score_variance"] = var
                f_reason.write(json.dumps(record, ensure_ascii=False) + "\n")
                reasoning_count += 1
            else:
                # Low-conflict OR no CoT → Basic dataset
                record["score_variance"] = var
                f_basic.write(json.dumps(record, ensure_ascii=False) + "\n")
                basic_count += 1

    print(f"Dataset_basic.jsonl    : {basic_count} records")
    print(f"Dataset_Reasoning.jsonl: {reasoning_count} records")
    print("Packaging complete.")


if __name__ == "__main__":
    pass
