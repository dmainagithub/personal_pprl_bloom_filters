import hashlib
import secrets

class PSIProtocol:
    
    def __init__(self, prime: int = None):
        # Use a large prime (simplified - use proper EC in production)
        self.prime = prime or 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFEE3
    
    def generate_keypair(self):
        private_key = secrets.randbelow(self.prime - 1) + 1
        public_key = pow(2, private_key, self.prime)
        return private_key, public_key
    
    def commutative_encrypt(self, value: str, key: int) -> int:
        h = int(hashlib.sha256(value.encode()).hexdigest(), 16) % self.prime
        return pow(h, key, self.prime)
    
    def compute_intersection(self, set_A: set, set_B: set, 
                            key_A: int, key_B: int) -> set:
        # Party A encrypts with their key
        encrypted_A = {self.commutative_encrypt(str(x), key_A) for x in set_A}
        
        # Party B encrypts with their key
        encrypted_B = {self.commutative_encrypt(str(x), key_B) for x in set_B}
        
        # In real protocol, parties exchange and apply second encryption
        # Simplified: return intersection
        return encrypted_A.intersection(encrypted_B)