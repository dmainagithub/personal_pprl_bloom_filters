# ===============================================================================================
# libraries and pipeline components
# ===============================================================================================
# These lines of code help resolve the issue of folder paths.
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================================================================================


from bitarray import bitarray
import hashlib
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.utils.helpers import get_qgrams

class SMPCBloomEncoder:

    def __init__(self, size=1024, num_hashes=5, party_id=None, shared_secret=None):
        self.size = size
        self.num_hashes = num_hashes
        self.party_id = party_id  # 'A' or 'B'
        self.shared_secret = shared_secret  # Pre-shared key (in practice, use DH key exchange)
        
    def __call__(self, row):
        # Extract relevant fields (adjust based on your data)
        record_str = f"{row.get('first_name', '')}|{row.get('last_name', '')}|{row.get('dob', '')}|{row.get('zip', '')}"
        
        # Generate CLK first
        clk = self._generate_clk(record_str)
        
        # Create Bloom filter
        bf = bitarray(self.size)
        bf.setall(0)
        
        # Get q-grams
        # qgrams = self._get_qgrams(clk, q=2) # Using the helper function instead of the method to avoid confusion
        qgrams = get_qgrams(clk)
        
        # For SMPC, we need to blind the Bloom filter if this is for comparison
        for qg in qgrams:
            for i in range(self.num_hashes):
                # Use keyed hash with party-specific salt
                if self.shared_secret:
                    hash_input = f"{qg}{i}{self.party_id}{self.shared_secret}".encode()
                else:
                    hash_input = f"{qg}{i}".encode()
                    
                h = hashlib.sha256(hash_input).hexdigest()
                idx = int(h, 16) % self.size
                bf[idx] = 1
        
        # If party B, we need to encrypt the Bloom filter for SMPC
        if self.party_id == 'B' and self.shared_secret:
            bf = self._blind_bloom_filter(bf)
            
        return bf
    
    def _generate_clk(self, record_str):
        """Generate Composite Link Key"""
        # Normalize: lowercase, remove spaces, handle missing values
        normalized = record_str.lower().replace(" ", "")
        # Add phonetic encoding for names (simplified version)
        return normalized
    
    # def _get_qgrams(self, text, q=2):
    #     """Generate q-grams"""
    #     if len(text) < q:
    #         return [text]
    #     return [text[i:i+q] for i in range(len(text) - q + 1)]
    
    def _blind_bloom_filter(self, bf):
        """
        Blind the Bloom filter using XOR with a random mask
        This is a simplified blinding - in production, use homomorphic encryption
        """
        # Generate random mask
        mask = bitarray(self.size)
        mask.setall(0)
        rng = secrets.randbits(self.size)
        for i in range(self.size):
            mask[i] = (rng >> i) & 1
            
        # Blind: bf XOR mask
        blinded = bf ^ mask
        return blinded