import os
import json
from typing import Dict, List, Optional

def _load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def _load_json(path: str) -> Dict:
    """Load a JSON file into a dict."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _build_lookup(data: List[Dict], key: str = "image_id") -> Dict[str, Dict]:
    """Index a list of dicts by a key for O(1) lookup."""
    lookup = {}
    for item in data:
        k = item.get(key, item.get("image_path", ""))
        if k:
            lookup[k] = item
    return lookup

def package_final_dataset(
    features_jsonl: str = "features.jsonl",
    scores_jsonl: str = "expert_scores.jsonl",
    cot_jsonl: str = "train_reasoning_data.jsonl",
    mos_json: str = "mos_calibration_data.json",
    metadata_jsonl: Optional[str] = None,
    output_jsonl: str = "final_training_dataset.jsonl"
):
    """
    Aligns all data sources by image_id and exports a unified JSONL.

    For each record the final JSONL contains:
      - image_id
      - image_path
      - prompt                     (from metadata / original data)
      - dinov2_cls_feature         (from feature extractor)
      - semantic_score             (from expert scorer)
      - quality_score              (from expert scorer)
      - aesthetic_score            (from expert scorer)
      - cot_analysis               (from CoT generator, if exists)
      - human_mos                  (from MOS calibrator, if exists)
    """
    print("=== Dataset Packager ===")

    # 1. Load each source
    features_data = _load_jsonl(features_jsonl)
    scores_data = _load_jsonl(scores_jsonl)
    cot_data = _load_jsonl(cot_jsonl)
    mos_data = _load_json(mos_json)  # keyed by image_path

    metadata_data = []
    if metadata_jsonl and os.path.exists(metadata_jsonl):
        metadata_data = _load_jsonl(metadata_jsonl)

    # 2. Build quick lookup tables
    feat_lookup = _build_lookup(features_data)
    score_lookup = _build_lookup(scores_data)
    cot_lookup = _build_lookup(cot_data)
    meta_lookup = _build_lookup(metadata_data) if metadata_data else {}

    # 3. Determine the canonical set of image IDs (union of all sources)
    all_ids = set()
    all_ids.update(feat_lookup.keys())
    all_ids.update(score_lookup.keys())
    # mos_data is keyed by image_path; map them too
    mos_lookup = {}
    for path_key, val in mos_data.items():
        # Try to derive an image_id that matches the others
        candidate_id = val.get("image_id", path_key)
        mos_lookup[path_key] = val
        mos_lookup[candidate_id] = val
        all_ids.add(path_key)

    print(f"Total unique image IDs across all sources: {len(all_ids)}")

    # 4. Merge & write
    count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
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
                "dinov2_cls_feature": feat_item.get("dinov2_cls_feature", []),
                "semantic_score": score_item.get("semantic_score", None),
                "quality_score": score_item.get("quality_score", None),
                "aesthetic_score": score_item.get("aesthetic_score", None),
                "cot_analysis": cot_item.get("cot_analysis", ""),
                "human_mos": mos_item.get("mos", None),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Packaged {count} records -> {output_jsonl}")

if __name__ == "__main__":
    # package_final_dataset()
    pass
