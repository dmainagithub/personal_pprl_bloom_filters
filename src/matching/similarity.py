# Calculate similarity score (bitwise - fast similarity)
def dice_similarity(bf1, bf2):
    intersection = (bf1 & bf2).count()
    return 2 * intersection / (bf1.count() + bf2.count() + 1e-10) # 0.0000000001 to prevent division by zero
