# libraries and pipeline components
# ===============================================================================================
# These lines of code help resolve the issue of folder paths.
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================================================================================


import numpy as np
from bitarray import bitarray
from src.matching.base_similarity import BaseSimilarity

class SMPCDiceSimilarity(BaseSimilarity):
    
    def __init__(self, epsilon: float = 0.5, use_noise: bool = True):
        self.epsilon = epsilon  # Privacy budget
        self.use_noise = use_noise
    
    def compute(self, bf1: bitarray, bf2: bitarray) -> float:
        # Convert to bitarray if needed
        bf1 = bitarray(bf1) if not isinstance(bf1, bitarray) else bf1
        bf2 = bitarray(bf2) if not isinstance(bf2, bitarray) else bf2
        
        # Standard Dice computation
        intersection = (bf1 & bf2).count()
        total_set_bits = bf1.count() + bf2.count()
        
        if total_set_bits == 0:
            dice = 0.0
        else:
            dice = (2.0 * intersection) / total_set_bits
        
        # Add Laplace noise for differential privacy
        if self.use_noise:
            sensitivity = 2.0 / max(total_set_bits, 1)
            noise = np.random.laplace(0, sensitivity / self.epsilon)
            dice = max(0.0, min(1.0, dice + noise))
        
        return dice

class PSIBasedSimilarity(BaseSimilarity):
    
    def __init__(self, psi_matcher):
        self.psi_matcher = psi_matcher
    
    def compute(self, enc1, enc2) -> float:
        # This would use the PSI matcher to compute secure intersection
        # Simplified for now
        return self.psi_matcher.compute_similarity(enc1, enc2)