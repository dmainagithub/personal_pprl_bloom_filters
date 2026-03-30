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

    matches_df = pd.DataFrame(matches, columns=["i","j","sim"])

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