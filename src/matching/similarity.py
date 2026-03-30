# Calculate similarity score (bitwise - fast similarity)
def dice_similarity(bf1, bf2):
    intersection = (bf1 & bf2).count()
    return 2 * intersection / (bf1.count() + bf2.count() + 1e-10) # 0.0000000001 to prevent division by zero


# This version is a bit slow	
# def dice_similarity(bf1, bf2):
#     bf1 = set([i for i,v in enumerate(bf1) if v == 1])
#     bf2 = set([i for i,v in enumerate(bf2) if v == 1])

#     return 2 * len(bf1 & bf2) / (len(bf1) + len(bf2) + 1e-10)