import math
import os
import tempfile
import torch
import gc
from typing import Dict


def score_with_qinsight_qprobe(image_path: str, prompt: str = "") -> Dict:
    """
    Q-Insight+ / Q-Probe: Active viewing quality assessment.

    Simulates human active viewing in two steps:
      1. Q-Insight+ (global scan): scores the full image with MUSIQ to obtain
         an overall perceptual quality score (0-100, higher = better).
      2. Q-Probe (local scan): divides the image into a 3x3 grid and evaluates
         each cell with BRISQUE (lower BRISQUE = better quality).  Cells whose
         BRISQUE score exceeds *defect_threshold* are reported as defect regions.

    Returns a dict with:
      - "score"   : float, overall quality (0-100)
      - "regions" : list of dicts describing defect regions (may be empty)
    """
    print("[Q-Insight+/Q-Probe] Loading active viewing models...")
    result = {"score": 0.0, "regions": []}

    try:
        import pyiqa
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Step 1: Q-Insight+ — global quality scan (MUSIQ, higher = better) ---
        quality_metric = pyiqa.create_metric("musiq", device=device)
        with torch.no_grad():
            overall_score = float(quality_metric(image_path).item())
        del quality_metric

        # --- Step 2: Q-Probe — regional defect scan (BRISQUE, lower = better) ---
        scan_metric = pyiqa.create_metric("brisque", device=device)
        defect_threshold = 50.0  # BRISQUE > 50 indicates degraded quality

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        grid_size = 3
        regions = []

        for row in range(grid_size):
            for col in range(grid_size):
                x1 = col / grid_size
                y1 = row / grid_size
                x2 = (col + 1) / grid_size
                y2 = (row + 1) / grid_size

                crop = img.crop(
                    (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
                )
                if crop.width < 32 or crop.height < 32:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    crop.save(tmp_path)
                    with torch.no_grad():
                        local_brisque = float(scan_metric(tmp_path).item())
                    if math.isnan(local_brisque):
                        local_brisque = 0.0
                finally:
                    os.remove(tmp_path)

                local_quality = max(0.0, 100.0 - local_brisque)

                if local_brisque > defect_threshold:
                    regions.append(
                        {
                            "action": (
                                f"name: Image crop, bbox: "
                                f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"
                            ),
                            "bbox": [
                                round(x1, 2),
                                round(y1, 2),
                                round(x2, 2),
                                round(y2, 2),
                            ],
                            "label": "quality_defect",
                            "local_score": round(local_quality, 2),
                        }
                    )

        del scan_metric
        img.close()

        result = {"score": overall_score, "regions": regions}

    except Exception as e:
        print(f"[Q-Insight+/Q-Probe] Error: {e}. Returning dummy.")
        result = {"score": 50.0, "regions": []}

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
