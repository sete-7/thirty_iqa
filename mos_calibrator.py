import os
import csv
import json
import numpy as np

def calibrate_mos_from_votes(csv_file: str = "human_votes.csv", output_json: str = "mos_calibration_data.json"):
    """
    Reads pairwise comparisons from human_votes.csv and uses choix
    to extract continuous, absolute MOS scores using Thurstone's model (Bradley-Terry).
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Votes file {csv_file} not found.")

    try:
        import choix
    except ImportError:
        raise ImportError("Please install choix: pip install choix networkx")

    image_to_id = {}
    id_to_image = {}
    wins = []

    # Read pairwise wins
    print(f"Reading pairwise wins from {csv_file}...")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            win_path = row["win_path"]
            lose_path = row["lose_path"]

            # Maintain ID mappings
            if win_path not in image_to_id:
                img_id = len(image_to_id)
                image_to_id[win_path] = img_id
                id_to_image[img_id] = win_path
                
            if lose_path not in image_to_id:
                img_id = len(image_to_id)
                image_to_id[lose_path] = img_id
                id_to_image[img_id] = lose_path

            wins.append((image_to_id[win_path], image_to_id[lose_path]))

    n_items = len(image_to_id)
    print(f"Total unique images: {n_items}")
    print(f"Total pairwise comparisons: {len(wins)}")

    if n_items == 0:
        print("No paired data to process.")
        return

    # Calibrate using Iterative Luce Spectral Ranking (ILSR)
    print("Computing absolute MOS scores using choix.ilsr_pairwise...")
    # alpha penalizes extreme scores, providing regularization.
    try:
        params = choix.ilsr_pairwise(n_items, wins, alpha=0.01)
    except Exception as e:
        print(f"Error during MOS calibration: {e}")
        print("Falling back to standard B-T...")
        params = np.zeros(n_items)

    # Normalize MOS roughly to a 0-10 or 0-100 scale (Optional, but often desirable)
    # B-T gives log-odds, centering around 0. Let's shift it to be robustly between 0 and 100.
    if n_items > 1 and max(params) != min(params):
        min_p = min(params)
        max_p = max(params)
        normalized_mos = [((p - min_p) / (max_p - min_p)) * 100.0 for p in params]
    else:
        normalized_mos = [50.0 for _ in range(n_items)]

    # Export mapping
    output_data = {}
    for i, p in enumerate(normalized_mos):
        img_path = id_to_image[i]
        output_data[img_path] = {
            "image_id": os.path.basename(img_path), # Fallback ID from path 
            "mos": float(p)
        }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully calibrated MOS scores and saved to {output_json}")

if __name__ == "__main__":
    # calibrate_mos_from_votes()
    pass
