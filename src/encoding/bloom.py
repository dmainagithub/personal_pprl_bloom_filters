from bitarray import bitarray
import hashlib

def get_qgrams(text, q=2):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.replace(" ", "")
    return [text[i:i+q] for i in range(len(text) - q + 1)]

def bloom_encode(text, size=1024, num_hashes=5):
    # Creating a blank memory
    bf = bitarray(size)
    bf.setall(0)

    qgrams = get_qgrams(text) # Breaking the word into pieces (overlapping pieces) - like tokenization

    for qg in qgrams:
        for i in range(num_hashes):  # Create multiple versions of the same qgram
            h = int(hashlib.sha1((qg + str(i)).encode()).hexdigest(), 16) # Turn each version into a number
            bf[h % size] = 1         # Pick a spot in the list and mark the spot

    return bf