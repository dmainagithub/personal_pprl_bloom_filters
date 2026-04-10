def prepare_plot_data(matches_df, true_matches_df):
     
    # Ensuring 'pair' column exists
    if 'pair' not in matches_df.columns:
        matches_df['pair'] = matches_df['id_A'].astype(str) + "_" + matches_df['id_B'].astype(str)
    if 'pair' not in true_matches_df.columns:
        true_matches_df['pair'] = true_matches_df['id_A'].astype(str) + "_" + true_matches_df['id_B'].astype(str)
    
    # Creating sets for fast lookup
    true_pairs_set = set(true_matches_df['pair'])
    
    y_true = []
    y_scores = []
    
    # Building arrays
    for _, row in matches_df.iterrows():
        pair = row['pair']
        score = row['sim']
        y_scores.append(score)
        y_true.append(1 if pair in true_pairs_set else 0)
    
    return y_true, y_scores

