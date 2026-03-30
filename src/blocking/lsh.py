import hashlib
import numpy as np

def bf_to_array(bf):
    if isinstance(bf, str):
        return np.array([int(x) for x in bf])
    return np.array(bf)

def lsh_blocking(df, bands=20, rows_per_band=5):

    blocks = {}

    for idx, row in df.iterrows():

        bf = bf_to_array(row["bloom"])

        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band

            band_slice = bf[start:end]

            # Convert to string
            band_str = ''.join(map(str, band_slice))

            # Hash band
            band_hash = hashlib.md5(band_str.encode()).hexdigest()

            # Create block key
            block_key = f"{b}_{band_hash}"

            if block_key not in blocks:
                blocks[block_key] = []

            blocks[block_key].append(idx)

    return blocks