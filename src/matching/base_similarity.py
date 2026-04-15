# ===============================================================================================
# libraries and pipeline components
# ===============================================================================================
# These lines of code help resolve the issue of folder paths.
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ===============================================================================================
from abc import ABC, abstractmethod

class BaseSimilarity(ABC):
    
    @abstractmethod
    def compute(self, enc1, enc2) -> float:
        pass
    
    def __call__(self, enc1, enc2) -> float:
        return self.compute(enc1, enc2)