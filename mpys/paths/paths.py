from itertools import product
import numpy as np

def random_path(dimensions):
    ranges = [range(dim) for dim in dimensions]
    coordinate_pairs = list(product(*ranges))
    linear_path = np.array(coordinate_pairs, dtype=np.int32)
    np.random.shuffle(linear_path)
    return linear_path