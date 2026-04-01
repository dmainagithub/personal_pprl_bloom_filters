warning: in the working copy of 'experiments/run_experiments.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'pprl_notebooks/01_data_preparation.ipynb', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/pipeline/run_pipeline.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/experiments/run_experiments.py b/experiments/run_experiments.py[m
[1mindex 7a93759..5673a7d 100644[m
[1m--- a/experiments/run_experiments.py[m
[1m+++ b/experiments/run_experiments.py[m
[36m@@ -1,10 +1,31 @@[m
 # ===============================================================================================[m
[32m+[m[32m# import matplotlib[m
[32m+[m[32m# matplotlib.use("Agg")  # For headless environments (like servers or CI)[m
[32m+[m
 # libraries and pipeline components[m
 import sys[m
[32m+[m[32mfrom pathlib import Path[m
[32m+[m
[32m+[m[32mPROJECT_ROOT = Path.cwd().parents[0]  # assumes notebooks are in a subfolder[m
[32m+[m[32mif str(PROJECT_ROOT) not in sys.path:[m
[32m+[m[32m    sys.path.insert(0, str(PROJECT_ROOT))[m
[32m+[m
[32m+[m[32mfrom src.utils.helpers import load_paths[m
[32m+[m
[32m+[m[32m# paths = load_paths()[m
[32m+[m
[32m+[m[32m# for key, value in paths.items():[m
[32m+[m[32m#     print(f"{key}: {value}")[m
[32m+[m[41m	[m
[32m+[m
[32m+[m[32m# SYNTHETIC_DATA_DIR = paths['SYNTHETIC_DATA_DIR'][m
[32m+[m[32m# PROCESSED_DATA_DIR = paths['PROCESSED_DATA_DIR'][m
[32m+[m[32m# FIG_DIR = paths['FIG_DIR'][m
[32m+[m
 import os[m
 [m
 [m
[31m-sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))[m
[32m+[m
 [m
 import pandas as pd[m
 import seaborn as sns[m
[1mdiff --git a/pprl_notebooks/01_data_preparation.ipynb b/pprl_notebooks/01_data_preparation.ipynb[m
[1mindex de222f3..d370926 100644[m
[1m--- a/pprl_notebooks/01_data_preparation.ipynb[m
[1m+++ b/pprl_notebooks/01_data_preparation.ipynb[m
[36m@@ -32,7 +32,6 @@[m
     "\n",[m
     "# ========================================================\n",[m
     "import os\n",[m
[31m-    "from pathlib import Path\n",[m
     "import numpy as np\n",[m
     "import pandas as pd\n",[m
     "import matplotlib.pyplot as plt\n",[m
[36m@@ -50,11 +49,11 @@[m
     "from pathlib import Path\n",[m
     "\n",[m
     "# ========================================================\n",[m
[31m-    "# 1️⃣ Ensure project root is in Python path\n",[m
[31m-    "# Adjust this if your notebooks are nested deeper\n",[m
[31m-    "PROJECT_ROOT = Path.cwd().parents[0]  # assumes notebooks are in a subfolder\n",[m
[31m-    "if str(PROJECT_ROOT) not in sys.path:\n",[m
[31m-    "    sys.path.insert(0, str(PROJECT_ROOT))\n",[m
[32m+[m[32m    "# # 1️⃣ Ensure project root is in Python path\n",[m
[32m+[m[32m    "# # Adjust this if your notebooks are nested deeper\n",[m
[32m+[m[32m    "# PROJECT_ROOT = Path.cwd().parents[0]  # assumes notebooks are in a subfolder\n",[m
[32m+[m[32m    "# if str(PROJECT_ROOT) not in sys.path:\n",[m
[32m+[m[32m    "#     sys.path.insert(0, str(PROJECT_ROOT))\n",[m
     "\n",[m
     "# ========================================================\n",[m
     "# 2️⃣ Import helper to load paths\n",[m
[1mdiff --git a/results/experiment_results.csv b/results/experiment_results.csv[m
[1mindex 6aaf8ae..b503be5 100644[m
[1m--- a/results/experiment_results.csv[m
[1m+++ b/results/experiment_results.csv[m
[36m@@ -1,4 +1,4 @@[m
 experiment,precision,recall,f1,pairs[m
[31m-Bloom + Rule,0.07442489851150136,0.8249999999999175,0.13653289199971222,198791[m
[31m-CLK + Rule,0.7841823056299216,0.5849999999999415,0.6701030927344864,198791[m
[31m-CLK + Rule + SMPC,0.004316025634576491,0.4289999999999571,0.008546072091815396,198791[m
[32m+[m[32mbloom_encode + dice_similarity,0.07442489851150136,0.8249999999999175,0.13653289199971222,198791[m
[32m+[m[32mclk_encode + dice_similarity,0.7841823056299216,0.5849999999999415,0.6701030927344864,198791[m
[32m+[m[32mclk_encode + smpc_dice_similarity,0.004114853716447338,0.40899999999995906,0.008147734967548413,198791[m
[1mdiff --git a/src/pipeline/run_pipeline.py b/src/pipeline/run_pipeline.py[m
[1mindex eeaefd3..83b06c5 100644[m
[1m--- a/src/pipeline/run_pipeline.py[m
[1m+++ b/src/pipeline/run_pipeline.py[m
[36m@@ -1,3 +1,23 @@[m
[32m+[m[32mimport sys[m
[32m+[m[32mfrom pathlib import Path[m
[32m+[m
[32m+[m[32mPROJECT_ROOT = Path.cwd().parents[0]  # assumes notebooks are in a subfolder[m
[32m+[m[32mif str(PROJECT_ROOT) not in sys.path:[m
[32m+[m[32m    sys.path.insert(0, str(PROJECT_ROOT))[m
[32m+[m
[32m+[m[32mfrom src.utils.helpers import load_paths[m
[32m+[m
[32m+[m[32m# paths = load_paths()[m
[32m+[m
[32m+[m[32m# for key, value in paths.items():[m
[32m+[m[32m#     print(f"{key}: {value}")[m
[32m+[m[41m	[m
[32m+[m
[32m+[m[32m# SYNTHETIC_DATA_DIR = paths['SYNTHETIC_DATA_DIR'][m
[32m+[m[32m# PROCESSED_DATA_DIR = paths['PROCESSED_DATA_DIR'][m
[32m+[m[32m# FIG_DIR = paths['FIG_DIR'][m
[32m+[m
[32m+[m
 from src.matching.matcher import match_pairs[m
 from src.evaluation.evaluate import evaluate[m
 from src.matching.smpc import smpc_dice_similarity[m
[36m@@ -6,6 +26,11 @@[m [mfrom src.blocking.rule_based import rule_blocking[m
 from src.encoding.bloom import bloom_encode[m
 from src.encoding.clk import clk_encode[m
 from src.encoding.hybrid import hybrid_encode[m
[32m+[m[32mfrom visualization.visualization import ([m
[32m+[m[32m    plot_metrics,[m
[32m+[m[32m    plot_similarity_distribution,[m
[32m+[m[32m    plot_confusion_bars,[m
[32m+[m[32m)[m
 [m
 import pandas as pd[m
 [m
[36m@@ -38,14 +63,6 @@[m [mdef run_pipeline(df_A, df_B, true_matches,[m
     # Removing any remaining bad rows[m
     matches_df = matches_df.dropna(subset=["i", "j"])[m
 [m
[31m-    # # Ensuring integer indices[m
[31m-    # matches_df["i"] = matches_df["i"].astype(int)[m
[31m-    # matches_df["j"] = matches_df["j"].astype(int)[m
[31m-[m
[31m-    # # Mapping IDs[m
[31m-    # matches_df["id_A"] = matches_df["i"].apply(lambda x: df_A.loc[x, "id"])[m
[31m-    # matches_df["id_B"] = matches_df["j"].apply(lambda x: df_B.loc[x, "id"])[m
[31m-[m
     if len(matches_df) == 0:[m
         # Creating empty structure safely[m
         matches_df = pd.DataFrame(columns=["id_A", "id_B", "sim"])[m
[36m@@ -67,10 +84,28 @@[m [mdef run_pipeline(df_A, df_B, true_matches,[m
     # ---------------------------------[m
     precision, recall, f1 = evaluate(matches_df, true_matches)[m
 [m
[32m+[m[41m    [m
[32m+[m[32m    # ---------------------------------[m
[32m+[m[32m    # 5. Visualization (per‑experiment)[m
[32m+[m[32m    # ---------------------------------[m
[32m+[m[32m    # 5A. Similarity distribution[m
[32m+[m[32m    # plot_similarity_distribution(all_scores, sim_func.__name__) # This[m
[32m+[m
[32m+[m[32m    # 5B. Confusion bars[m
[32m+[m[32m    tp = len(pred_pairs & true_pairs)[m
[32m+[m[32m    fp = len(pred_pairs) - tp[m
[32m+[m[32m    fn = len(true_pairs) - tp[m
[32m+[m[32m    # plot_confusion_bars(tp, fp, fn, sim_func.__name__)  # This[m
[32m+[m
[32m+[m[32m     # ---------------------------------[m
[32m+[m[32m      # ---------------------------------[m
     return {[m
[32m+[m[32m        "experiment": f"{encoder.__name__} + {sim_func.__name__}",[m
         "precision": precision,[m
         "recall": recall,[m
         "f1": f1,[m
         "pairs": len(pairs)[m
     }[m
 [m
[32m+[m[32m    # plot_metrics(results_df)[m
[32m+[m
