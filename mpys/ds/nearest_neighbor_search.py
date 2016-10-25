from numba import jit
import numpy as np
import bottleneck as btn


@jit(nopython=True, nogil=True)
def compute_distances_brute_force_2d(current_node, traced_path, distances):
    for i in range(len(traced_path)):
        distances[i] = (traced_path[i][0] - current_node[0]) * (traced_path[i][0] - current_node[0]) \
                       + (traced_path[i][1] - current_node[1]) * (traced_path[i][1] - current_node[1])
    return distances


@jit(nopython=True, nogil=True)
def compute_distances_brute_force_3d(current_node, traced_path, distances):
    for i in range(len(traced_path)):
        distances[i] = (traced_path[i][0] - current_node[0]) * (traced_path[i][0] - current_node[0]) \
                       + (traced_path[i][1] - current_node[1]) * (traced_path[i][1] - current_node[1]) \
                       + (traced_path[i][2] - current_node[2]) * (traced_path[i][2] - current_node[2])
    return distances


class NearestNeighborSearch(object):
    def __init__(self, nearest_neighbor_method=compute_distances_brute_force_2d, n_neighbors=30):
        self.n_neighbors = n_neighbors

        self.nearest_neighbor_search = nearest_neighbor_method

        self.distances = None

        self.nearest_neighbor_ids = None

    def find(self, current_node, traced_path):

        self.distances = np.zeros(len(traced_path), dtype=np.int64)
        self.nearest_neighbor_search(current_node, traced_path, self.distances)

        self.nearest_neighbor_ids = btn.argpartition(self.distances, kth=self.n_neighbors)

        return self.nearest_neighbor_ids
