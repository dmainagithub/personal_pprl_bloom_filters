# libraries and pipeline components
# ===============================================================================================
# These lines of code help resolve the issue of folder paths.
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================================================================================

from bitarray import bitarray
from src.matching.base_similarity import BaseSimilarity


class DiceSimilarity(BaseSimilarity):
    """Dice coefficient similarity for Bloom filters"""
    
    def compute(self, bf1: bitarray, bf2: bitarray) -> float:
        if not isinstance(bf1, bitarray):
            bf1 = bitarray(bf1) if isinstance(bf1, str) else bf1
        if not isinstance(bf2, bitarray):
            bf2 = bitarray(bf2) if isinstance(bf2, str) else bf2
        
        intersection = (bf1 & bf2).count()
        total_set_bits = bf1.count() + bf2.count()
        
        if total_set_bits == 0:
            return 0.0
        
        return (2.0 * intersection) / total_set_bits
    