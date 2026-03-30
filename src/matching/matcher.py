from joblib import Parallel, delayed

def compare_pair(i, j, df_A, df_B, sim_func):
    bf1 = df_A.loc[i, "encoded"]
    bf2 = df_B.loc[j, "encoded"]

    sim = sim_func(bf1, bf2)

    return (i, j, sim)


def match_pairs(pairs, df_A, df_B, sim_func, threshold):

    results = Parallel(n_jobs=-1)(
        delayed(compare_pair)(i, j, df_A, df_B, sim_func)
        for i, j in pairs
    )

    matches = [
        (i, j, sim)
        for i, j, sim in results
        if sim >= threshold
    ]

    return matches, results  # return both filtered + all scores