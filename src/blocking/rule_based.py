import hashlib
import numpy as np
import pandas as pd
from bitarray import bitarray
from abc import ABC, abstractmethod
from typing import Set, Tuple  


def rule_blocking(df_A, df_B, col):
    pairs = []

    for val in df_A[col].dropna().unique():
        subA = df_A[df_A[col] == val]
        subB = df_B[df_B[col] == val]

        for i in subA.index:
            for j in subB.index:
                pairs.append((i, j))

    return pairs

# A class for rule-based blocking that can be used in the experiments. It simply blocks records that share the same value in a specified column.
class BaseBlocker(ABC):
    """Abstract base class for all blocking strategies"""
    
    @abstractmethod
    def block(self, df_A: pd.DataFrame, df_B: pd.DataFrame) -> Set[Tuple[int, int]]:
        """
        Generate candidate record pairs for comparison
        
        Args:
            df_A: First dataframe with records
            df_B: Second dataframe with records
            
        Returns:
            Set of tuples (index_in_A, index_in_B) representing candidate pairs
        """
        pass
    
    def __call__(self, df_A: pd.DataFrame, df_B: pd.DataFrame) -> Set[Tuple[int, int]]:
        """Make blocker callable like a function"""
        return self.block(df_A, df_B)

class RuleBasedBlocker(BaseBlocker):
    """Rule-based blocking using a blocking key column"""
    
    def __init__(self, blocking_col: str = "block_key"):
        self.blocking_col = blocking_col
    
    def block(self, df_A: pd.DataFrame, df_B: pd.DataFrame) -> set:
        pairs = set()
        
        # Get unique blocking keys from both dataframes
        keys_A = df_A[self.blocking_col].unique()
        keys_B = df_B[self.blocking_col].unique()
        
        # Only block on keys that exist in both
        common_keys = set(keys_A) & set(keys_B)
        
        for key in common_keys:
            indices_A = df_A[df_A[self.blocking_col] == key].index
            indices_B = df_B[df_B[self.blocking_col] == key].index
            
            for i in indices_A:
                for j in indices_B:
                    pairs.add((i, j))
        
        return pairs