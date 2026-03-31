# src.matching.smpc

import numpy as np

def secret_share(vec):
    """
    Split vector into 2 random shares
    """
    share1 = np.random.randint(0, 2, size=len(vec))
    share2 = (vec - share1) % 2
    return share1, share2


def secure_dot_product(a1, a2, b1, b2):
    """
    Simulated secure dot product
    """
    # In real SMPC, this happens without revealing data
    return np.dot(a1 + a2, b1 + b2)


def smpc_dice_similarity(bf1, bf2):
    """
    Privacy-preserving Dice similarity
    """

    bf1 = np.array(bf1)
    bf2 = np.array(bf2)

    # Secret share both vectors
    a1, a2 = secret_share(bf1)
    b1, b2 = secret_share(bf2)

    # Secure intersection
    intersection = secure_dot_product(a1, a2, b1, b2)

    # Secure sums
    sum1 = bf1.sum()
    sum2 = bf2.sum()

    dice = (2 * intersection) / (sum1 + sum2 + 1e-10)

    return float(dice)