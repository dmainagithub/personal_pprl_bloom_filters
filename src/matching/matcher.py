from joblib import Parallel, delayed
from bitarray import bitarray
import numpy as np


def to_bitarray(x):
    if isinstance(x, bitarray):
        return x
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return bitarray(x)
    if isinstance(x, str):
        return bitarray(x)
    raise ValueError("Unknown bitarray type: " + str(type(x)))


def compare_pair(i, j, df_A, df_B, sim_func):
    bf1 = to_bitarray(df_A.loc[i, "encoded"])
    bf2 = to_bitarray(df_B.loc[j, "encoded"])

    sim = sim_func(bf1, bf2)

    return (i, j, sim)


def match_pairs(pairs, df_A, df_B, sim_func, threshold):

    results = Parallel(n_jobs=-1)(
        delayed(compare_pair)(i, j, df_A, df_B, sim_func)
        for i, j in pairs
    )
    matches = [
        {
            "id_A": df_A.loc[i, "id"],
            "id_B": df_B.loc[j, "id"],
            "similarity": sim
        }
        for i, j, sim in results
        if sim >= threshold
    ]

    return matches, results  # return both filtered + all scores

# Old version
    # matches = [
    #     (i, j, sim)
    #     for i, j, sim in results
    #     if sim >= threshold
    # ]
    
