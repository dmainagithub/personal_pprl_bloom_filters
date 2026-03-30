import hashlib
import numpy as np

def bf_to_array(bf):
    if isinstance(bf, str):
        return np.array([int(x) for x in bf])
    return np.array(bf)


import hashlib
import numpy as np

def bf_to_array(bf):
    """Convert Bloom filter or bitstring into a NumPy array."""
    if isinstance(bf, str):
        return np.array([int(x) for x in bf])
    return np.array(bf)

def compute_lsh_blocks(df, bands=20, rows_per_band=5, column="encoded"):
    """
    Compute LSH blocks for a single dataframe.
    Returns a dictionary:
        { block_key : [row_indices] }
    """
    blocks = {}

    for idx, row in df.iterrows():
        bf = bf_to_array(row[column])

        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_slice = bf[start:end]

            # Convert bit slice to a string
            band_str = ''.join(map(str, band_slice))

            # Hash the band
            band_hash = hashlib.md5(band_str.encode()).hexdigest()

            block_key = f"{b}_{band_hash}"

            if block_key not in blocks:
                blocks[block_key] = []

            blocks[block_key].append(idx)

    return blocks


def lsh_blocking(df_A, df_B, bands=20, rows_per_band=5, column="encoded"):
    """
    LSH blocking across two datasets.
    Returns list of (i, j) index pairs.
    """

    A_blocks = compute_lsh_blocks(df_A, bands, rows_per_band, column)
    B_blocks = compute_lsh_blocks(df_B, bands, rows_per_band, column)

    pairs = []

    # Match blocks
    for block_key in A_blocks:
        if block_key in B_blocks:
            for i in A_blocks[block_key]:
                for j in B_blocks[block_key]:
                    pairs.append((i, j))

    return pairs


# # Original
# def lsh_blocking(df_A, df_B, bands=20, rows_per_band=5):
    
#     blocks_A = compute_lsh_blocks(df_A, bands, rows_per_band)
#     blocks_B = compute_lsh_blocks(df_B, bands, rows_per_band)


#     blocks = {}

#     for idx, row in df.iterrows():

#         bf = bf_to_array(row["encoded"])  # Initially bloom but converting it to encoded

#         for b in range(bands):
#             start = b * rows_per_band
#             end = start + rows_per_band

#             band_slice = bf[start:end]

#             # Convert to string
#             band_str = ''.join(map(str, band_slice))

#             # Hash band
#             band_hash = hashlib.md5(band_str.encode()).hexdigest()

#             # Create block key
#             block_key = f"{b}_{band_hash}"

#             if block_key not in blocks:
#                 blocks[block_key] = []

#             blocks[block_key].append(idx)

#     return blocks