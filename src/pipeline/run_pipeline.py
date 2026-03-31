from src.matching.matcher import match_pairs
from src.evaluation.evaluate import evaluate
from src.matching.smpc import smpc_dice_similarity
from src.matching.similarity import dice_similarity
from src.blocking.rule_based import rule_blocking
from src.encoding.bloom import bloom_encode
from src.encoding.clk import clk_encode
from src.encoding.hybrid import hybrid_encode

import pandas as pd

def run_pipeline(df_A, df_B, true_matches,
                 encoder,
                 blocker,
                 sim_func,
                 threshold):

    # ---------------------------------
    # 1. Encoding
    # ---------------------------------
    df_A["encoded"] = df_A.apply(encoder, axis=1)
    df_B["encoded"] = df_B.apply(encoder, axis=1)

    # ---------------------------------
    # 2. Blocking
    # ---------------------------------
    pairs = blocker(df_A, df_B)

    # ---------------------------------
    # 3. Matching
    # ---------------------------------
    matches, all_scores = match_pairs(
        pairs, df_A, df_B, sim_func, threshold
    )

    matches_df = pd.DataFrame(matches, columns=["i", "j", "sim"])  # New # 

    # Removing any remaining bad rows
    matches_df = matches_df.dropna(subset=["i", "j"])

    # # Ensuring integer indices
    # matches_df["i"] = matches_df["i"].astype(int)
    # matches_df["j"] = matches_df["j"].astype(int)

    # # Mapping IDs
    # matches_df["id_A"] = matches_df["i"].apply(lambda x: df_A.loc[x, "id"])
    # matches_df["id_B"] = matches_df["j"].apply(lambda x: df_B.loc[x, "id"])

    if len(matches_df) == 0:
        # Creating empty structure safely
        matches_df = pd.DataFrame(columns=["id_A", "id_B", "sim"])
    else:
        matches_df["id_A"] = matches_df["i"].apply(lambda x: df_A.loc[x, "id"])
        matches_df["id_B"] = matches_df["j"].apply(lambda x: df_B.loc[x, "id"])
        matches_df = matches_df[["id_A", "id_B", "sim"]]

    # Debugging
    pred_pairs = set(matches_df["id_A"].astype(str) + "_" + matches_df["id_B"].astype(str))
    true_pairs = set(true_matches["id_A"].astype(str) + "_" + true_matches["id_B"].astype(str))

    print("Predicted pairs:", len(pred_pairs))
    print("True pairs:", len(true_pairs))
    print("Overlap:", len(pred_pairs & true_pairs))

    # ---------------------------------
    # 4. Evaluation
    # ---------------------------------
    precision, recall, f1 = evaluate(matches_df, true_matches)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pairs": len(pairs)
    }