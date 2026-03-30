from src.encoding.bloom import bloom_encode              # Bloom filter encoding

# CLK = Cryptographic Long-term Key
def clk_encode(record):
    """
    CLK = combine multiple identifiers into one Bloom filter
    """
    fields = [
        record["first_name"],
        record["last_name"],
        record["dob"]
    ]
    
    combined = " ".join([str(f) for f in fields if f])
    
    return bloom_encode(combined)