import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# Use a Chinese-friendly font if available, fallback gracefully
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_final_dataset(jsonl_path: str) -> List[Dict]:
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _extract_scores(data: List[Dict], key: str) -> List[float]:
    return [item[key] for item in data if item.get(key) is not None]


def plot_histograms(data: List[Dict], output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    score_keys = [
        ("human_mos", "Human MOS"),
        ("unipercept_score", "UniPercept"),
        ("grounding_iqa_score", "Grounding-IQA"),
        ("hpsv3_score", "HPSv3"),
        ("spatial_score", "SpatialScore"),
    ]

    n = len(score_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle("Score Distribution Histograms", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, score_keys):
        scores = _extract_scores(data, key)
        if scores:
            sns.histplot(scores, bins=30, kde=True, ax=ax, color="steelblue", edgecolor="white")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.axvline(np.mean(scores), color="red", linestyle="--", label=f"Mean={np.mean(scores):.2f}")
            ax.axvline(np.median(scores), color="orange", linestyle="-.", label=f"Median={np.median(scores):.2f}")
            ax.legend(fontsize=9)
        else:
            ax.set_title(f"{title} (No Data)")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "score_histograms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Histogram saved to {path}")


def plot_boxplots(data: List[Dict], output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    score_keys = [
        ("human_mos", "Human MOS"),
        ("unipercept_score", "UniPercept"),
        ("grounding_iqa_score", "G-IQA"),
        ("hpsv3_score", "HPSv3"),
        ("spatial_score", "Spatial"),
    ]

    all_scores = []
    labels = []
    for key, label in score_keys:
        scores = _extract_scores(data, key)
        if scores:
            s_min, s_max = min(scores), max(scores)
            norm = [(s - s_min) / (s_max - s_min) for s in scores] if s_max != s_min else [0.5] * len(scores)
            all_scores.append(norm)
            labels.append(label)

    if not all_scores:
        print("No data for boxplots.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Normalized Score Box Plots", fontsize=16, fontweight="bold")

    bp = ax.boxplot(all_scores, patch_artist=True, labels=labels, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6})

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Normalized Score (0-1)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "score_boxplots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Boxplot saved to {path}")


def generate_all_plots(jsonl_path: str = "final_training_dataset.jsonl", output_dir: str = "plots"):
    """
    Entry point. Also accepts Dataset_basic.jsonl or Dataset_Reasoning.jsonl.
    """
    print(f"Loading data from {jsonl_path}...")
    data = load_final_dataset(jsonl_path)
    print(f"Loaded {len(data)} records.")
    plot_histograms(data, output_dir)
    plot_boxplots(data, output_dir)
    print("All plots generated.")


if __name__ == "__main__":
    # By default visualize the basic dataset, can also pass reasoning dataset
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Dataset_basic.jsonl"
    generate_all_plots(path)
