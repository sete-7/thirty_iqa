# Multi-source Image Evaluation Dataset: Automated Construction & Calibration Pipeline

An end-to-end Python pipeline that automates the building and calibration of a high-quality image evaluation training dataset. It combines **machine expert scoring**, **LLM-based reasoning**, and **human preference annotation** into a unified workflow.

## Project Structure

```
thirty_iqa/
├── data_loader.py            # Load images from local folder or HuggingFace JSONL
├── feature_extractor.py      # DINOv2 CLS-token global feature extraction (with checkpointing)
├── expert_scorers.py         # Sequential expert scoring: HPSv3 / MUSIQ / Aesthetic (with checkpointing)
├── cot_generator.py          # Variance-based controversy detection + LLM CoT generation (with checkpointing)
├── data_filter.py            # Trim top/bottom 20%, keep middle 60% hard examples
├── app_gradio.py             # Gradio blind-test preference annotation UI
├── mos_calibrator.py         # Thurstone (Bradley-Terry) MOS calibration via choix
├── dataset_packager.py       # Align all sources by Image ID → final_training_dataset.jsonl
├── visualization.py          # Histogram & boxplot of score distributions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `hpsv3` may require installation from source. The aesthetic scorer is a placeholder; replace with your own CLI tool.

### 2. Prepare Data

Place test images in a folder, or prepare a JSONL file with `image_path` and `prompt` fields:

```jsonl
{"image_path": "images/001.jpg", "prompt": "A cat sitting on a sofa"}
{"image_path": "images/002.jpg", "prompt": "Sunset over mountains"}
```

### 3. Run the Pipeline Step-by-Step

#### Step 1 — Extract DINOv2 Features

```python
from data_loader import get_dataloader
from feature_extractor import process_features_with_checkpointing

data = get_dataloader('local', './test_images')   # or ('jsonl', 'metadata.jsonl')
process_features_with_checkpointing(data, output_jsonl="features.jsonl")
```

#### Step 2 — Expert Scoring (Sequential, VRAM-safe)

```python
from expert_scorers import process_scoring_with_checkpointing

process_scoring_with_checkpointing(data, output_jsonl="expert_scores.jsonl")
```

#### Step 3 — Filter Middle 60% Hard Examples

```python
from data_filter import filter_data_middle_60

filter_data_middle_60("expert_scores.jsonl", "filtered_scores.jsonl")
```

#### Step 4 — Generate CoT for Controversial Images

Set your API key first:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your provider
export LLM_MODEL_NAME="qwen-vl-max"                  # or gpt-4o
```

```python
from cot_generator import process_cot_generation

process_cot_generation("expert_scores.jsonl", "train_reasoning_data.jsonl", variance_threshold=1.0)
```

#### Step 5 — Human Preference Annotation (Gradio)

Prepare an `app_pairs.jsonl` file:

```jsonl
{"prompt": "A futuristic city", "img_left": "img_a.jpg", "img_right": "img_b.jpg"}
```

Then launch:

```bash
python app_gradio.py
```

Open `http://localhost:7860` in your browser. Votes are saved to `human_votes.csv`.

#### Step 6 — MOS Calibration

```python
from mos_calibrator import calibrate_mos_from_votes

calibrate_mos_from_votes("human_votes.csv", "mos_calibration_data.json")
```

#### Step 7 — Package Final Dataset

```python
from dataset_packager import package_final_dataset

package_final_dataset(
    features_jsonl="features.jsonl",
    scores_jsonl="expert_scores.jsonl",
    cot_jsonl="train_reasoning_data.jsonl",
    mos_json="mos_calibration_data.json",
    output_jsonl="final_training_dataset.jsonl"
)
```

#### Step 8 — Visualize Dataset Health

```bash
python visualization.py
```

Plots are saved to `plots/score_histograms.png` and `plots/score_boxplots.png`.

## Checkpointing

`feature_extractor.py`, `expert_scorers.py`, and `cot_generator.py` all support **automatic checkpointing every 100 images**. If a run is interrupted, simply re-run the same command — already-processed items are skipped automatically.

## Output Files

| File | Description |
|---|---|
| `features.jsonl` | DINOv2 CLS features per image |
| `expert_scores.jsonl` | Three expert scores per image |
| `filtered_scores.jsonl` | Middle 60% after trimming extremes |
| `train_reasoning_data.jsonl` | CoT analysis for controversial images |
| `human_votes.csv` | Pairwise human preference votes |
| `mos_calibration_data.json` | Calibrated absolute MOS scores |
| `final_training_dataset.jsonl` | **Unified final dataset** |
| `plots/` | Distribution histograms & boxplots |
