import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import sys
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parents[0]  # assumes notebooks are in a subfolder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_paths



import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your existing imports
import datetime

sns.set(style="whitegrid", font_scale=1.2)

paths = load_paths()

# SYNTHETIC_DATA_DIR = paths['SYNTHETIC_DATA_DIR']
# PROCESSED_DATA_DIR = paths['PROCESSED_DATA_DIR']
FIG_DIR = paths['FIG_DIR']


# --------------------------------------------------
# Creating folder automatically
# --------------------------------------------------
def save_fig(name, folder="visualization"):
    # os.makedirs(folder, exist_ok=True)
    # Create full path using FIG_DIR
    save_dir = FIG_DIR / folder

    # Create the directory (including parents if needed)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_dir / f"{timestamp}_{name}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    print(f"[SAVED] {filename}")
    plt.close()

# --------------------------------------------------
# 1. Bar plot: Precision / Recall / F1
# --------------------------------------------------
def plot_metrics(results_df, show=False):
    metrics = ["precision", "recall", "f1"]
    df = results_df.melt(
        id_vars="experiment",
        value_vars=metrics,
        var_name="metric",
        value_name="score"
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="experiment", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.title("Performance Comparison Across PPRL Methods")
    plt.tight_layout()

    # save_fig("metrics_overview")
    
    if show:
        plt.show()
    else:
        plt.close()



# --------------------------------------------------
# 2. Distribution of similarity scores
# --------------------------------------------------
def plot_similarity_distribution(scores, method_name, show=False):
    sims = [s for (_, _, s) in scores if s is not None]

    plt.figure(figsize=(10, 4))
    sns.histplot(sims, bins=50, kde=True)
    plt.title(f"Similarity Score Distribution — {method_name}")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")

    # save_fig(f"{method_name}_similarity_distribution")

    if show:
        plt.show()
    else:
        plt.close()



# --------------------------------------------------
# 3. Confusion matrix-style bar plot
# --------------------------------------------------
def plot_confusion_bars(tp, fp, fn, method_name, show=False):
    data = pd.DataFrame({
        "Type": ["True Positives", "False Positives", "False Negatives"],
        "Count": [tp, fp, fn]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(data=data, x="Type", y="Count", palette="viridis")
    plt.title(f"Match Outcome Distribution — {method_name}")


    # save_fig(f"{method_name}_confusion_bars")

    if show:
        plt.show()
    else:
        plt.close()


# --------------------------------------------------
# 4. Threshold sweep plot
# --------------------------------------------------
def plot_threshold_sweep(thresholds, precisions, recalls, f1s, method_name, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    
    plt.title(f"Threshold Sensitivity — {method_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    # save_fig(f"{method_name}_threshold_sweep")

    if show:
        plt.show()
    else:
        plt.close()