def evaluate(matches_df, true_matches):

    matches_df["pair"] = matches_df["id_A"].astype(str) + "_" + matches_df["id_B"].astype(str)
    # # Debugging: print("Matches DataFrame with pair column:\n", matches_df.head())    
    # print("EVALUATE:matches_df.columns", matches_df.columns)
    # print("EVALUATE: matches_df.head()", matches_df.head())
    # print("EVALUATE: len(matches_df)", len(matches_df))

    true_matches["pair"] = true_matches["id_A"].astype(str) + "_" + true_matches["id_B"].astype(str)

    pred = set(matches_df["pair"])
    true = set(true_matches["pair"])

    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1