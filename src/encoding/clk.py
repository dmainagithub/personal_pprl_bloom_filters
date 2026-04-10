from src.encoding.bloom import bloom_encode              # Bloom filter encoding
import hashlib
import secrets
import re


# Normalization function
def normalize_clk_field(value, field_type="text"):
    if not value:
        return ""
    value = str(value).strip().lower()

    if field_type == "name":
        # Removing titles, special characters, noirmalize spaces, and convert to lowercase
        titles = ["mr", "mrs", "ms", "dr", "prof", "miss", "sir", "madam", "lord", "lady", "rev", "fr", "sr", "jr"]
        for title in titles:
            if value.startswith(title + " "):
                value = value[len(title):].strip()
             
        value = re.sub(r'[^a-z\s]', '', value)  # Remove special characters
        value = re.sub(r'\s+', ' ', value)  # Normalize spaces
        value = ' '.join(value.split())
            
    elif field_type == "dob":
        # Simple date normalization (YYYY-MM-DD)
        value = value.replace("/", "-").replace(".", "-")
        parts = value.split("-")
        if len(parts) == 3:
            year, month, day = parts
            return f"{year.zfill(4)}-{month.zfill(2)}-{day.zfill(2)}"
    else:   
        return str(value).strip().lower()

    return value


def clk_encode(record):
        
    fields = [
        record["first_name"],
        record["last_name"],
        record["dob"]
    ]
    
    combined = " ".join([str(f) for f in fields if f])
    
    return bloom_encode(combined)


def clk_encode_enhanced(record, secret=None, size=1024, num_hashes=5):
    
    # if not isinstance(record, dict):
    #     raise ValueError("Record must be a dictionary")
    
    # Normalize fields
    first_name = normalize_clk_field(record.get("first_name", ""), "name")
    last_name = normalize_clk_field(record.get("last_name", ""), "name")
    dob = normalize_clk_field(record.get("dob", ""), "dob")
    
    if not any([first_name, last_name, dob]):
        raise ValueError("Insufficient identifying information")
    
    # Create weighted combined string (q-grams work better on longer strings)
    # Adding separators that aren't removed in get_qgrams
    combined_parts = []
    
    if first_name:
        combined_parts.append(f"FN:{first_name}")
    if last_name:
        combined_parts.append(f"LN:{last_name}")
    if dob:
        combined_parts.append(f"DOB:{dob}")
    
    combined = "|".join(combined_parts)
    
    # Add secret if provided
    if secret:
        combined = f"{combined}|S:{secret}"
    
    return bloom_encode(combined, size, num_hashes)