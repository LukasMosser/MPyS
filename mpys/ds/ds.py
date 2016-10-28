import numpy as np

from ..paths import random_path
from .nearest_neighbor_search import NearestNeighborSearch
from .event_scan import DirectSampler
from tqdm import tqdm

class DirectSampling(object):
    def __init__(self, training_image, simulation_grid, simulation_path, traced_path,
                 n_neighbors=30, threshold=0.1, sampling_fraction=0.2):
        from nearest_neighbor_search import shell_search, brute_force_search
        self.training_image = training_image
        self.simulation_grid = simulation_grid
        self.simulation_path = simulation_path
        self.traced_path = traced_path

        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.sampling_fraction = sampling_fraction

        self.nearest_neighbor_search = NearestNeighborSearch(nearest_neighbor_method=shell_search, n_neighbors=self.n_neighbors)

        self.event_scanner = DirectSampler(self.training_image, self.sampling_fraction)

    def get_node_value(self, current_node):

        nearest_point_indices = self.nearest_neighbor_search.find(current_node, self.traced_path, self.simulation_grid)

        if len(nearest_point_indices) < self.n_neighbors:
            return 1

        nearest_points = nearest_point_indices

        lag_vectors = self.compute_lag_vectors(current_node, nearest_points).astype(np.int64)

        nearest_points = np.array(nearest_points, dtype=np.int64)

        simulation_grid_event = self.get_values_at_node_indices(self.simulation_grid, nearest_points)

        node_value = self.event_scanner.find_match(simulation_grid_event, lag_vectors, self.threshold)

        return node_value

    def run(self):

        for i, path_index in enumerate(tqdm(self.simulation_path, desc="Performing direct sampling operation: ")):

            if len(self.traced_path) <= self.n_neighbors:
                self.simulation_grid[tuple(path_index)] = self.get_random_training_image_value()
            else:
                self.simulation_grid[tuple(path_index)] = self.get_node_value(path_index)

            self.traced_path.append(tuple(path_index))

        return self.simulation_grid

    def get_random_training_image_value(self):
        indices = tuple(int(np.random.uniform(low=0, high=dimension)) for dimension in self.training_image.shape)

        return self.training_image[indices]

    @staticmethod
    def compute_lag_vectors(current_node, nearest_points):
        lag_vectors = np.subtract(nearest_points, current_node)
        return lag_vectors

    @staticmethod
    def get_values_at_node_indices(grid, nodes):
        if len(grid.shape) == 2:
            x_cords, y_cords = nodes.T
            return grid[[x_cords, y_cords]]
        elif len(grid.shape) == 3:
            x_cords, y_cords, z_cords = nodes.T
            return grid[[x_cords, y_cords, z_cords]]

