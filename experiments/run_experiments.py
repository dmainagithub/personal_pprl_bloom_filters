# ===============================================================================================
# libraries and pipeline components
# ===============================================================================================
# These lines of code help resolve the issue of folder paths.
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================================================================================

from pathlib import Path
from src.evaluation.plots import (
    prepare_labels,
    plot_roc,
    plot_pr,
    plot_threshold_metrics,
    plot_score_distribution,
    ensure_dir
)
from src.evaluation.plot_helpers import prepare_plot_data


BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "results" / "plots"
ensure_dir(PLOT_DIR)
# ===============================================================================================

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
from src.encoding.clk import clk_encode, clk_encode_enhanced                    # cryptographic long-term key encoding
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
		"name": "CLK-basic + Rule", 
		"encoder": clk_encode, 
		"blocker": lambda A, B: rule_blocking(A, B, col="block_key"),
        "sim_func": dice_similarity
	},
    {
		"name": "CLK + Rule", 
		"encoder": clk_encode_enhanced, 
		"blocker": lambda A, B: rule_blocking(A, B, col="block_key"),
        "sim_func": dice_similarity
	},
    {
        "name": "CLK + Rule + SMPC",
        "encoder": clk_encode_enhanced,
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
plot_data = {} # This one will be used to plot PPRL techniques comparison plots at the end.

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

    # Visualization
    matches_df = res['matches_df']  
    true_matches_df = res['true_matches']  

    # Get y_true and y_scores for plots
    y_true, y_scores = prepare_plot_data(matches_df, true_matches_df)

    
    # ---- CLEAN experiment name BEFORE plotting ----
    exp_name = exp["name"].replace(" ", "_").replace("+", "_")

    # exp_name = exp["name"].replace(" ", "_")
    plot_data[exp_name] = (y_true, y_scores)  # Storing for later comparison plots


    plot_roc(
        y_true, 
        y_scores, 
        save_path=PLOT_DIR / f"{exp_name}_roc.png",
        title=f"{exp['name']} ROC"
        )
    plot_pr(
        y_true, 
        y_scores, 
        save_path=PLOT_DIR / f"{exp_name}_pr.png",
        title=f"{exp['name']} PR"
    )
# ===============================================================================================
    plot_threshold_metrics(
        y_true,
        y_scores,
        save_path=PLOT_DIR / f"{exp_name}_threshold.png"
    )

    plot_score_distribution(
        y_true,
        y_scores,
        save_path=PLOT_DIR / f"{exp_name}_distribution.png"
    )

    print(f"[SAVED PLOTS] {exp['name']}")
    # End of visualzation

# ===============================================================================================
# ===============================================================================================
    # Comparison plots (all techniques in one plot)
# ===============================================================================================
# ===============================================================================================
    # ROC curves  156 - 173  #         
# ===============================================================================================
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(8,6))

    for name, (y_true, y_scores) in plot_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curves of PPRL Techniques')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(PLOT_DIR / "comparative_roc.png")
    plt.show()
# ===============================================================================================
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(8,6))

    for name, (y_true, y_scores) in plot_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, label=name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparative Precision-Recall Curves of PPRL Techniques')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(PLOT_DIR / "comparative_pr.png")
    plt.show() 
# ===============================================================================================
# Saving the results
results_df = pd.DataFrame(results)

results_df.to_csv("results/experiment_results.csv", index=False)

print(results_df)

# ===============================================================================================

# ===============================================================================================

