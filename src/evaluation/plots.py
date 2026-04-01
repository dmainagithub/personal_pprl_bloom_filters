
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# Ensure folder exists
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Convert results → labels + scores
def prepare_labels(results, df_A, df_B, true_matches):
    df = pd.DataFrame(results, columns=["i", "j", "sim"])

    df["id_A"] = df["i"].apply(lambda x: df_A.loc[x, "id"])
    df["id_B"] = df["j"].apply(lambda x: df_B.loc[x, "id"])

    df["pair"] = df["id_A"].astype(str) + "_" + df["id_B"].astype(str)

    true_pairs = set(
        true_matches["id_A"].astype(str) + "_" +
        true_matches["id_B"].astype(str)
    )

    df["label"] = df["pair"].apply(lambda x: 1 if x in true_pairs else 0)

    return df["label"].values, df["sim"].values


# 🔷 ROC
def plot_roc(y_true, y_scores, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()


# 🔷 PR
def plot_pr(y_true, y_scores, save_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()


# 🔷 Threshold curves
def plot_threshold_metrics(y_true, y_scores, save_path):
    thresholds = np.linspace(0, 1, 50)

    precision_list, recall_list, f1_list = [], [], []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)

        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    plt.figure()
    plt.plot(thresholds, precision_list, label="Precision")
    plt.plot(thresholds, recall_list, label="Recall")
    plt.plot(thresholds, f1_list, label="F1")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Performance")
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()


# 🔷 Distribution
def plot_score_distribution(y_true, y_scores, save_path):
    plt.figure()

    plt.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label="Matches")
    plt.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label="Non-Matches")

    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()