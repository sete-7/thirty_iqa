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
        
        # Context-Aware Cropping: Generate a pseudo-Saliency Map 
        # to find top regions instead of a fixed 3x3 grid.
        # ？？？？？？？
        # In a real scenario, a dedicated lightweight saliency network (e.g. U-2-Net, BASNet)
        # or a fast variance/edge filter is used to find salient object/noise regions.
        # Here we simulate this by detecting high-frequency variance boxes.
        
        import torchvision.transforms.functional as TF
        from torchvision.transforms import Grayscale
        
        img_tensor = TF.to_tensor(img).unsqueeze(0) # (1, 3, H, W)
        gray_tensor = Grayscale()(img_tensor) # (1, 1, H, W)
        
        # Simple pseudo-saliency: local variance (using average pooling)
        kernel_size = min(w, h) // 10
        if kernel_size % 2 == 0: kernel_size += 1
        
        local_mean = torch.nn.functional.avg_pool2d(gray_tensor, kernel_size, stride=kernel_size//2)
        local_mean_sq = torch.nn.functional.avg_pool2d(gray_tensor**2, kernel_size, stride=kernel_size//2)
        local_variance = torch.clamp(local_mean_sq - local_mean**2, min=0)
        
        # Find top N salient (high variance/texture/noise) regions
        N_regions = 5
        b, c, ph, pw = local_variance.shape
        flat_var = local_variance.view(-1)
        topk_vals, topk_idx = torch.topk(flat_var, min(N_regions, flat_var.size(0)))
        
        stride = kernel_size // 2
        regions = []
        
        # Crop context-aware regions based on saliency peaks
        for idx in topk_idx:
            row = (idx.item() // pw) * stride
            col = (idx.item() % pw) * stride
            
            # Context window around the peak
            x1 = max(0, col - kernel_size//2)
            y1 = max(0, row - kernel_size//2)
            x2 = min(w, col + kernel_size + kernel_size//2)
            y2 = min(h, row + kernel_size + kernel_size//2)
            
            crop = img.crop((x1, y1, x2, y2))
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
                            f"name: Context Aware Crop, bbox: "
                            f"[{x1/w:.2f}, {y1/h:.2f}, {x2/w:.2f}, {y2/h:.2f}]"
                        ),
                        "bbox": [
                            round(x1/w, 2),
                            round(y1/h, 2),
                            round(x2/w, 2),
                            round(y2/h, 2),
                        ],
                        "label": "quality_defect_salient",
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
