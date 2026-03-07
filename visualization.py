import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

def load_final_dataset(jsonl_path: str = "final_training_dataset.jsonl") -> List[Dict]:
    """Load the final packaged JSONL."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def _extract_scores(data: List[Dict], key: str) -> List[float]:
    """Extract a list of non-None float scores for a given key."""
    return [item[key] for item in data if item.get(key) is not None]

def plot_histograms(data: List[Dict], output_dir: str = "plots"):
    """
    Plot distribution histograms of human MOS and expert scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    score_keys = [
        ("human_mos", "Human MOS Score"),
        ("semantic_score", "Semantic Score (HPSv3)"),
        ("quality_score", "Quality Score (MUSIQ)"),
        ("aesthetic_score", "Aesthetic Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Score Distribution Histograms", fontsize=16, fontweight="bold")

    for ax, (key, title) in zip(axes.flatten(), score_keys):
        scores = _extract_scores(data, key)
        if scores:
            sns.histplot(scores, bins=30, kde=True, ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean={np.mean(scores):.2f}')
            ax.axvline(np.median(scores), color='orange', linestyle='-.', label=f'Median={np.median(scores):.2f}')
            ax.legend(fontsize=9)
        else:
            ax.set_title(f"{title} (No Data)", fontsize=12)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hist_path = os.path.join(output_dir, "score_histograms.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Histogram saved to {hist_path}")

def plot_boxplots(data: List[Dict], output_dir: str = "plots"):
    """
    Plot box plots for human MOS and expert scores on a single figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    score_keys = [
        ("human_mos", "Human MOS"),
        ("semantic_score", "Semantic"),
        ("quality_score", "Quality"),
        ("aesthetic_score", "Aesthetic"),
    ]

    all_scores = []
    labels = []
    for key, label in score_keys:
        scores = _extract_scores(data, key)
        if scores:
            # Normalize to 0-1 for visual comparison (optional)
            s_min, s_max = min(scores), max(scores)
            if s_max != s_min:
                norm = [(s - s_min) / (s_max - s_min) for s in scores]
            else:
                norm = [0.5 for _ in scores]
            all_scores.append(norm)
            labels.append(label)

    if not all_scores:
        print("No score data to plot boxplots.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Normalized Score Box Plots", fontsize=16, fontweight="bold")

    bp = ax.boxplot(all_scores, patch_artist=True, labels=labels, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6})

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Normalized Score (0–1)")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    box_path = os.path.join(output_dir, "score_boxplots.png")
    plt.savefig(box_path, dpi=150)
    plt.close()
    print(f"Boxplot saved to {box_path}")

def generate_all_plots(jsonl_path: str = "final_training_dataset.jsonl", output_dir: str = "plots"):
    """Entry point: load data and generate all visualizations."""
    print(f"Loading data from {jsonl_path}...")
    data = load_final_dataset(jsonl_path)
    print(f"Loaded {len(data)} records.")

    plot_histograms(data, output_dir)
    plot_boxplots(data, output_dir)
    print("All plots generated successfully.")

if __name__ == "__main__":
    generate_all_plots()
