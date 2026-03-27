import yaml
from pathlib import Path

def load_paths(config_filename="config.yaml"):
    """
    Load project paths from config.yaml regardless of where the notebook is run.
    """

    # Search upwards for config.yaml
    current_dir = Path.cwd()

    for parent in [current_dir] + list(current_dir.parents):
        config_path = parent / config_filename
        if config_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"{config_filename} not found in current directory or any parent directories."
        )

    # Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = Path(config["paths"]["base_dir"])
    data_dir_str = config["paths"]["data_dir"].format(base_dir=base_dir)
    out_dir_str = config["paths"]["out_dir"].format(base_dir=base_dir)


    # --------------------------------------------------
    # Build paths dictionary
    # --------------------------------------------------
    paths = {
        "BASE_DIR": base_dir,

        # Main directories
        "DATA_DIR": Path(config["paths"]["data_dir"].format(base_dir=base_dir)),
        "OUT_DIR":  Path(config["paths"]["out_dir"].format(base_dir=base_dir)),
        "FIG_DIR":  Path(config["paths"]["fig_dir"].format(base_dir=base_dir)),
        # "MODEL_DIR":  Path(config["paths"]["statsmodels"].format(base_dir=base_dir)),
        
        # Subfolder inside OUT_DIR
        "MODEL_DIR": Path(
            config["paths"]["statsmodels"].format(
                base_dir=base_dir,
                out_dir = out_dir_str,
            )
        ),      

        
        # Notebooks directories
        "NOTEBOOKS_DIR": Path(config["paths"]["notebooks_dir"].format(base_dir=base_dir)),
        "NOTEBOOKS_EXECUTED_DIR": Path(
            config["paths"]["notebooks_executed_dir"].format(base_dir=base_dir)
        ),

        
        # Subdirectories inside DATA_DIR
        "RAW_DATA_DIR": Path(
            config["paths"]["raw_data_dir"].format(
                base_dir=base_dir,
                data_dir=data_dir_str,
            )
        ),
        "PROCESSED_DATA_DIR": Path(
            config["paths"]["processed_data_dir"].format(
                base_dir=base_dir,
                data_dir=data_dir_str,
            )
        ),
        "SYNTHETIC_DATA_DIR": Path(
            config["paths"]["synthetic_data_dir"].format(
                base_dir=base_dir,
                data_dir=data_dir_str,
            )
        ),

    }

    # --------------------------------------------------
    # Ensure directories exist
    # --------------------------------------------------
    for key, p in paths.items():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)

            
    # # --------------------------------------------------
    # # Ensure output directories exist
    # # --------------------------------------------------
    # #paths["OUT_DIR"].mkdir(parents=True, exist_ok=True)
    # #paths["FIG_DIR"].mkdir(parents=True, exist_ok=True)
    # # Ensure directories exist
    # for p in paths.values():
    #     if isinstance(p, Path):
    #         p.mkdir(parents=True, exist_ok=True)
    # # --------------------------------------------------

    return paths
    
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

def shap_dependence_safe(var_name, shap_values, X, save_dir, suffix="png"):
    """
    Plots a SHAP dependence plot safely, even if var_name was one-hot encoded.
    
    Parameters
    ----------
    var_name : str
        Base variable name (e.g., "hpt_status_derived")
    shap_values : np.array or shap.Explanation
        SHAP values from TreeExplainer or KernelExplainer
    X : pd.DataFrame
        Feature matrix
    save_dir : Path
        Directory to save the plot
    suffix : str
        File format for saving ("png", "pdf", etc.)
    """

    # Check if exact column exists
    if var_name in X.columns:
        col_to_plot = var_name
    else:
        # Try to find first dummy/encoded column starting with var_name
        matching_cols = [c for c in X.columns if c.startswith(var_name)]
        if not matching_cols:
            raise ValueError(f"No column found for variable '{var_name}' in X.")
        col_to_plot = matching_cols[0]  # pick first match

    import matplotlib.pyplot as plt
    import shap

    shap.dependence_plot(
        col_to_plot,
        shap_values,
        X,
        show=False
    )

    save_path = save_dir / f"shap_dependence_{col_to_plot}.{suffix}"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"SHAP dependence plot saved to: {save_path}")
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
