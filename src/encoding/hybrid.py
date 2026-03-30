from src.encoding.bloom import bloom_encode 
from src.encoding.clk import clk_encode

# Bloom filter encoding

def hybrid_encode(record):
    """
    Example: combine CLK + individual field encoding
    """
    clk = clk_encode(record)
    
    # Add extra signal (e.g., postcode)
    extra = bloom_encode(str(record.get("firstname", "")))
    
    return [max(a,b) for a,b in zip(clk, extra)]