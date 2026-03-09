import os
import json
import numpy as np
from typing import List, Dict

def calculate_composite_score(items: List[Dict]) -> List[Dict]:
    """
    Normalizes specific expert scores (Min-Max) and calculates a simple 
    average to represent the composite score for sorting.
    """
    if not items:
        return items

    # Extract all scores
    sem_scores = [item.get("semantic_score", 0.0) for item in items]
    qual_scores = [item.get("quality_score", 0.0) for item in items]
    aes_scores = [item.get("aesthetic_score", 0.0) for item in items]

    # Helper for min-max normalization
    def normalize(scores):
        s_min = min(scores)
        s_max = max(scores)
        if s_max == s_min:
            return [0.5 for _ in scores]
        return [(s - s_min) / (s_max - s_min) for s in scores]

    sem_norm = normalize(sem_scores)
    qual_norm = normalize(qual_scores)
    aes_norm = normalize(aes_scores)

    # Compute composite
    for i in range(len(items)):
        items[i]["composite_score"] = (sem_norm[i] + qual_norm[i] + aes_norm[i]) / 3.0

    return items

def filter_data_middle_60(input_jsonl: str = "expert_scores.jsonl", output_jsonl: str = "filtered_scores.jsonl"):
    """
    Reads the expert scores, sorts by composite score, and keeps the middle 60%.
    (Trims top 20% and bottom 20%).
    """
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")

    data = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        print("Empty dataset. Nothing to filter.")
        return

    print(f"Total samples loaded: {len(data)}")

    # Calculate composite score for each item
    data = calculate_composite_score(data)

    # Sort descending based on composite score (best to worst)
    data.sort(key=lambda x: x["composite_score"], reverse=True)

    # Filter top 20% and bottom 20%
    total_len = len(data)
    trim_count = int(total_len * 0.20)
    
    start_idx = trim_count
    end_idx = total_len - trim_count
    
    filtered_data = data[start_idx:end_idx]

    print(f"Filtering complete.")
    print(f"Removed top 20% ({trim_count} samples) and bottom 20% ({trim_count} samples).")
    print(f"Remaining (middle 60%): {len(filtered_data)} samples.")

    # Save to a new JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Filtered data saved to {output_jsonl}")

if __name__ == "__main__":
    # Example usage:
    # filter_data_middle_60("expert_scores.jsonl", "filtered_scores.jsonl")
    pass
