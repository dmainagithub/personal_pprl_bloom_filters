# ===============================================================================================
# libraries and pipeline components
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

import sys
from pathlib import Path

# pipeline components
from src.pipeline.run_pipeline import run_pipeline
from src.encoding.bloom import bloom_encode                # Bloom filter encoding
from src.encoding.clk import clk_encode                    # cryptographic long-term key encoding
from src.encoding.hybrid import hybrid_encode              # Combining encoding strategies          
from src.blocking.lsh import lsh_blocking                  # LSH - Locality Sensitive Hashing
from src.blocking.rule_based import rule_blocking                            # Rule-based blocking  
from src.matching.similarity import dice_similarity        # Similarity function
# from src.evaluation.evaluation import evaluate_matches     # Evaluation metrics
from src.matching.smpc import smpc_dice_similarity
# from src.matching.similarity import dice_similarity
# from src.encoding.hybrid import hybrid_encode



# ===============================================================================================

# ===============================================================================================
# 2. Loading data
df_A = pd.read_csv("data/synthetic/dataset_A.csv")
df_B = pd.read_csv("data/synthetic/dataset_B.csv")
true_matches = pd.read_csv("data/synthetic/true_matches.csv")

# ===============================================================================================
df_A["block_key"] = df_A["last_name"].str[0]
df_B["block_key"] = df_B["last_name"].str[0]

# 3. Defining different experiments
experiments = [
    {
		"name": "Bloom + Rule", 
		"encoder": bloom_encode, 
		"blocker": lambda A, B: rule_blocking(A, B, col="block_key"),
        "sim_func": dice_similarity
	},
    {
		"name": "CLK + Rule", 
		"encoder": clk_encode, 
		"blocker": lambda A, B: rule_blocking(A, B, col="block_key"),
        "sim_func": dice_similarity
	},
    {
        "name": "CLK + Rule + SMPC",
        "encoder": clk_encode,
        "blocker": lambda A, B: rule_blocking(A, B, col="block_key"),
        "sim_func": smpc_dice_similarity
    },
    # {                             # Taking so much time
	# 	"name": "Hybrid + LSH", 
	# 	"encoder": hybrid_encode, 
	# 	"blocker": lsh_blocking
	# },
]
# ===============================================================================================
# 4. Running the experiments
results = []

for exp in experiments:

    print(f"\nRunning: {exp['name']}")
	
    res = run_pipeline(
        df_A.copy(),
        df_B.copy(),
        true_matches,
        encoder=exp["encoder"],
        blocker=exp["blocker"],
        sim_func=exp["sim_func"], 
        threshold=0.85
    )

    results.append({"experiment": exp["name"], **res})


# ===============================================================================================
# Saving the results
results_df = pd.DataFrame(results)

results_df.to_csv("results/experiment_results.csv", index=False)

print(results_df)

# ===============================================================================================

# ===============================================================================================

