import papermill as pm
from datetime import datetime
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Making src importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_paths

# ---------------------------------------------------------------------
# Load paths from config.yaml
# ---------------------------------------------------------------------
paths = load_paths()

NOTEBOOK_DIR = paths["NOTEBOOKS_DIR"]
OUTPUT_DIR = paths["NOTEBOOKS_EXECUTED_DIR"]

# ---------------------------------------------------------------------
# Notebook execution order
# ---------------------------------------------------------------------


NOTEBOOKS = [
    "00_synthetic_data_generation.ipynb",
    "01_data_preparation.ipynb",
    "02_bloom_filter_encoding.ipynb",
    "03_blocking_strategies.ipynb",
    "04_similarity_and_linkage.ipynb",
    "05_scalable_pipeline.ipynb"
]

# ---------------------------------------------------------------------
# Executing notebooks
# ---------------------------------------------------------------------
print("Starting Bloom Filter Based PPRL Pipeline...")
start_time = datetime.now()

for nb in NOTEBOOKS:
    input_nb = NOTEBOOK_DIR / nb
    output_nb = OUTPUT_DIR / nb

    print(f"\nRunning: {nb}")
    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        kernel_name="python3",
    )

end_time = datetime.now()

print("\nPipeline completed successfully!")
print(f"Start time: {start_time}")
print(f"End time  : {end_time}")
print(f"Duration  : {end_time - start_time}")
print(f"Executed notebooks saved to: {OUTPUT_DIR}")