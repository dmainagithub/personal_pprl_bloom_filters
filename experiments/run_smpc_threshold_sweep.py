import pandas as pd

from src.encoding.clk import clk_encode
from src.blocking.rule_based import rule_blocking
from src.matching.smpc import smpc_dice_similarity

from threshold_sweep import run_threshold_sweep

# Load your data
df_A = pd.read_csv("data/df_A.csv")
df_B = pd.read_csv("data/df_B.csv")
true_matches = pd.read_csv("data/true_matches.csv")

thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]  # adjust for SMPC range

run_threshold_sweep(df_A, df_B, true_matches,
                    clk_encode, rule_blocking, smpc_dice_similarity,
                    thresholds)