import xxhash


def array_hash(arr):
    return xxhash.xxh64(arr.tobytes()).hexdigest()

