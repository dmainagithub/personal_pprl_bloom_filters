import hashlib
import numpy as np


def rule_blocking(df_A, df_B, col):
    pairs = []

    for val in df_A[col].dropna().unique():
        subA = df_A[df_A[col] == val]
        subB = df_B[df_B[col] == val]

        for i in subA.index:
            for j in subB.index:
                pairs.append((i, j))

    return pairs