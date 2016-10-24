import numpy as np
from numba import jit


class DirectSampler(object):
    def __init__(self, training_image, sampling_fraction):

        self.training_image = training_image

        self._max_iterations = int(sampling_fraction*self.training_image.shape[0]*self.training_image.shape[1])

        self.sampling_function = find_event_in_ti

    def find_match(self, simulation_grid_event, lag_vectors, threshold):
        bounds = self.get_search_window_bounds(lag_vectors)
        abs_bounds = np.abs(bounds)
        nx, ny = self.training_image.shape

        node_value = self.sampling_function(self.training_image, lag_vectors, simulation_grid_event,
                                            threshold, abs_bounds, nx, ny, self._max_iterations)

        return node_value

    @staticmethod
    def get_search_window_bounds(lag_vectors):
        x, y = lag_vectors.T
        a, b, c, d = x.min(), x.max(), y.min(), y.max()

        return a, b, c, d


@jit(nopython=True, nogil=True)
def find_event_in_ti(training_image, lag_vectors, simulation_grid_event,
                               threshold, abs_bounds, nx, ny, max_iter):

    d_min = 999
    event_min = 1

    a, b, c, d = abs_bounds
    count = 0
    event_length = len(simulation_grid_event)
    for i in xrange(a, nx - b):
        for j in xrange(c, ny - d):

            # training_image_event = []
            sum = 0.0
            for k in range(len(lag_vectors)):
                if simulation_grid_event[k] != training_image[lag_vectors[k][0] + i, lag_vectors[k][1] + j]:
                    sum += 1

            event_distance = sum / event_length

            # if we find a perfect match, take it straight away
            if event_distance == 0.0 or event_distance < threshold:
                return training_image[i, j]

            # if the current distance is between threshold and d_min set as new d_min, continue search
            elif threshold < event_distance < d_min:
                d_min = event_distance
                event_min = training_image[i, j]

            # if the distance is greater than d_min, continue on our search

            if count >= max_iter:
                # return the event with smallest found distance in max_iter iterations
                # print "MaxIter Reached", d_min
                return event_min

            count += 1

    # print "MaxIter Reached", d_min
    if d_min == 999:
        return 1  # get_random_training_image_value(training_image)
    else:
        return event_min
