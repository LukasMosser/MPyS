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

from scipy.spatial import cKDTree

def compute_distances_kd_tree(current_node, traced_path, n_neighbors):
    tree = cKDTree(traced_path)
    return tree.query(current_node, k=n_neighbors)

@jit(nopython=True, nogil=True)
def shell_search(current_node, n_neighbors, shape, image):
    nx, ny = current_node
    shapea, shapeb = shape
    max_shape = None
    shell = None
    if shapea > shapeb:
        max_shape = shapea
    else:
        max_shape = shapeb
    for i in xrange(max_shape):
        low_x = nx-i-1
        if low_x <= 0:
            low_x = 0

        high_x = nx+i+2
        if high_x >= shapea:
            high_x = shapea

        low_y = ny-i-1
        if low_y <= 0:
            low_y = 0

        high_y = ny+i+2
        if high_y >= shapeb:
            high_y = shapeb

        shell = image[low_x:high_x, low_y:high_y]
        #flat_shell = shell.flatten()
        count = 0
        node_ids = np.ones((n_neighbors+1, 2))
        for k in xrange(shell.shape[0]):
            for l in xrange(shell.shape[1]):
                val = shell[k, l]
                if val == 0 or val == 1:
                    count += 1
                    node_ids[count-1][0] = low_x+k
                    node_ids[count-1][1] = low_y+l
#                    node_ids.append([int(), int(low_y+l)])
                if count >= n_neighbors:
                    break
            if count >= n_neighbors:
                break
        if count >= n_neighbors:
            break
    return node_ids


class NearestNeighborSearch(object):
    def __init__(self, nearest_neighbor_method=shell_search, n_neighbors=30):
        self.n_neighbors = n_neighbors

        self.nearest_neighbor_search = nearest_neighbor_method

        self.distances = None

        self.nearest_neighbor_ids = None

    def find(self, current_node, traced_path, simulation_grid):

        #self.distances = np.zeros(len(traced_path), dtype=np.int64)
        self.nodes = self.nearest_neighbor_search(current_node, self.n_neighbors,
                                     simulation_grid.shape, simulation_grid)#, traced_path, self.distances)
        return self.nodes
        #self.nearest_neighbor_ids = btn.argpartition(self.distances, kth=self.n_neighbors)
        #
        #return self.nearest_neighbor_ids
