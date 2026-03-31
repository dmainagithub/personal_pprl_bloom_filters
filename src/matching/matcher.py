from joblib import Parallel, delayed
from bitarray import bitarray
import numpy as np
import math


def to_bitarray(x):
    if isinstance(x, bitarray):
        return x
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return bitarray(x)
    if isinstance(x, str):
        return bitarray(x)
    raise ValueError("Unknown bitarray type: " + str(type(x)))


def compare_pair(i, j, df_A, df_B, sim_func):
    try:
        # Get encoded values
        bf1 = df_A.loc[i, "encoded"]
        bf2 = df_B.loc[j, "encoded"]

        # Check for missing values
        if bf1 is None or bf2 is None:
            return None

        # Convert to bit arrays
        try:
            bf1 = to_bitarray(bf1)
            bf2 = to_bitarray(bf2)
        except Exception as e:
            print("COMPAIR-PAIRS: ARBITARRAY ERROR:", e, bf1, bf2)
            
            return None

        # Compute similarity
        sim = sim_func(bf1, bf2)
        
        # # ---- Debug SMPC similarity ----
        # if "smpc" in sim_func.__name__.lower():
        #     print(f"SMPC DEBUG → i:{i}, j:{j}, sim:{sim}")

        # Catch NaN similarity
        if sim is None or sim != sim:  # sim != sim detects NaN
            return None

        return (i, j, sim)

    except Exception as e:
        print("COMPAIR-PAIRS: ", e, "i:", i, "j:", j)
        return None

# def auto_threshold(scores):
#     sims = [s for (_, _, s) in scores if s is not None and s > 0]
#     if len(sims) == 0:
#         return 0.0
#     return np.percentile(sims, 60)    # top 10% of scores

def match_pairs(pairs, df_A, df_B, sim_func, threshold):

    # Running all comparisons in parallel
    results = Parallel(n_jobs=-1)(
        delayed(compare_pair)(i, j, df_A, df_B, sim_func)
        for i, j in pairs
    )

    # Removing failed comparisons
    clean_results = [r for r in results if r is not None]

    
    # Auto-threshold for SMPC   
    is_smpc = "smpc" in sim_func.__name__.lower()

    
    if is_smpc:
        sims = [sim for (_, _, sim) in clean_results if sim > 0]
        if len(sims) > 0:
            threshold = np.percentile(sims, 50)   # median, not 90th
        else:
            threshold = 0.0

        print("\n[SMPC] Auto threshold set to:", threshold)

    matches = []

    for r in clean_results:
        i, j, sim = r

        # Skip invalid similarity
        if sim is None or math.isnan(sim):
            continue

        # Apply threshold
        if sim >= threshold:
            try:
                matches.append({
                    "i": i,
                    "j": j,
                    "sim": sim
                })
            except Exception:
                # Skip bad indices safely
                continue
    # # Debugging prints        
    # print("MATCH-PAIRS: Total pairs:", len(pairs))
    # print("MATCH-PAIRS: Raw results:", len(results))
    # print("MATCH-PAIRS: Clean results:", len(clean_results))
    # print("MATCH-PAIRS: Final matches:", len(matches))            
    
    return matches, clean_results  # filtered + all scores

