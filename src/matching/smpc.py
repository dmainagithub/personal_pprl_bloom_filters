# src.matching.smpc

import numpy as np

def secret_share(vec, modulus=2**32):
    vec = np.array(vec, dtype=np.int64)

    if vec.ndim == 0:
        vec = np.array([vec])

    share1 = np.random.randint(0, modulus, size=len(vec), dtype=np.int64)
    share2 = (vec - share1) % modulus
    return share1, share2

def reconstruct(share1, share2, modulus=2**32):
    return (share1 + share2) % modulus

def secure_and_share(share1_a, share2_a, share1_b, share2_b, modulus=2**32):

    total_share1 = (share1_a + share1_b) % modulus
    total_share2 = (share2_a + share2_b) % modulus
    return total_share1, total_share2

def secure_mul(share1_a, share2_a, share1_b, share2_b, modulus=2**32):

    mul_share1 = (share1_a * share1_b) % modulus
    
    mul_share2 = (share2_a * share2_b) % modulus
    
    return mul_share1, mul_share2

def smpc_dice_similarity(bf1, bf2):

    # Convert to numpy arrays 
    if not isinstance(bf1, np.ndarray):
        bf1 = np.array(bf1)
    if not isinstance(bf2, np.ndarray):
        bf2 = np.array(bf2)
    
    # Ensure numeric and binary
    bf1 = bf1.astype(np.int8)
    bf2 = bf2.astype(np.int8)
    
    # Compute intersection (AND)
    intersection = np.sum(bf1 & bf2)
    
    # Compute sums (number of 1s)
    sum1 = np.sum(bf1)
    sum2 = np.sum(bf2)
    
    # Dice coefficient
    denominator = sum1 + sum2
    if denominator == 0:
        return 0.0
    
    dice = (2.0 * intersection) / denominator
    
    # Ensure valid return value
    return float(np.clip(dice, 0.0, 1.0))



# def secret_share(vec):
#     share1 = np.random.randint(0, 2, size=len(vec))
#     share2 = (vec - share1) % 2
#     return share1, share2

# def secure_dot_product(a1, a2, b1, b2):
#     return np.dot(a1 + a2, b1 + b2)


# def smpc_dice_similarity(bf1, bf2):
#     bf1 = np.array(bf1)
#     bf2 = np.array(bf2)

#     # Secret share both vectors
#     a1, a2 = secret_share(bf1)
#     b1, b2 = secret_share(bf2)

#     # Secure intersection
#     intersection = secure_dot_product(a1, a2, b1, b2)

#     # Secure sums
#     sum1 = bf1.sum()
#     sum2 = bf2.sum()

#     dice = (2 * intersection) / (sum1 + sum2 + 1e-10)

#     return float(dice)
