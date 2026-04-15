# libraries and pipeline components
# ===============================================================================================
# These lines of code help resolve the issue of folder paths.
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================================================================================


import pandas as pd
from typing import Callable, Dict, Any
from src.evaluation.evaluate import evaluate

class SMPCPipeline:
    """
    Privacy-preserving pipeline with SMPC integration
    """
    
    def __init__(self, encoder, blocker, sim_func, threshold: float = 0.85):
        self.encoder = encoder
        self.blocker = blocker
        self.sim_func = sim_func
        self.threshold = threshold
    
    def run(self, df_A: pd.DataFrame, df_B: pd.DataFrame, 
            true_matches: pd.DataFrame) -> Dict[str, Any]:
        
        # Phase 1: Secure encoding (with party-specific blinding)
        df_A["encoded"] = df_A.apply(lambda row: self.encoder.encode(row), axis=1)
        df_B["encoded"] = df_B.apply(lambda row: self.encoder.encode(row), axis=1)
        
        # Phase 2: Blocking (can be done on public attributes)
        pairs = self.blocker.block(df_A, df_B)
        
        df_A = df_A.reset_index(drop=True)
        df_B = df_B.reset_index(drop=True)
        
        # Phase 3: Secure matching
        matches = []
        all_scores = []
        
        for i, j in pairs:
            sim = self.sim_func.compute(df_A.loc[i, "encoded"], 
                                       df_B.loc[j, "encoded"])
            all_scores.append(sim)
            
            if sim >= self.threshold:
                matches.append((i, j, sim))
        
        # Phase 4: Format results
        matches_df = pd.DataFrame(matches, columns=["i", "j", "sim"])
        matches_df = matches_df.dropna(subset=["i", "j"])
        
        if len(matches_df) > 0:
            matches_df["id_A"] = matches_df["i"].apply(lambda x: df_A.loc[x, "id"])
            matches_df["id_B"] = matches_df["j"].apply(lambda x: df_B.loc[x, "id"])
            matches_df = matches_df[["id_A", "id_B", "sim"]]
        else:
            matches_df = pd.DataFrame(columns=["id_A", "id_B", "sim"])
        
        # Phase 5: Evaluation
        precision, recall, f1 = evaluate(matches_df, true_matches)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pairs": len(pairs),
            "matches_df": matches_df,
            "true_matches": true_matches
        }