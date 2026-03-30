def hybrid_encode(record):
    """
    Example: combine CLK + individual field encoding
    """
    clk = clk_encode(record)
    
    # Add extra signal (e.g., postcode)
    extra = bloom_encode(str(record.get("postcode", "")))
    
    return [max(a,b) for a,b in zip(clk, extra)]