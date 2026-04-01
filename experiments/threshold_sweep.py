# experiments/threshold_sweep.py

import pandas as pd

from src.pipeline import run_pipeline
from src.blocking.rule_based import rule_blocking
from src.encoding.clk import clk_encode
from src.matching.smpc import smpc_dice_similarity

from visualization.visualization import plot_threshold_sweep

def run_threshold_sweep(df_A, df_B, true_matches, encoder, blocker, sim_func, thresholds):

    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        print(f"\n=== Running threshold {t} ===")
        result = run_pipeline(df_A, df_B, true_matches,
                              encoder, blocker, sim_func, t)

        precisions.append(result["precision"])
        recalls.append(result["recall"])
        f1s.append(result["f1"])

    # PLOT + SAVE automatically
    plot_threshold_sweep(thresholds, precisions, recalls, f1s, sim_func.__name__)

    print("\nThreshold sweep finished.")
