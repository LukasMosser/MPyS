import numpy as np
from scipy import spatial
from itertools import product
import tqdm
import matplotlib.pyplot as plt
from operator import itemgetter
from numba import jit, autojit, cuda
import bottleneck as bn
import threading

from mpys.utils import output_to_png, import_sgems_dat_file
from mpys.paths import random_path

def create_new_empty_simulation_grid(nx, ny):
    return np.ones((nx, ny)).astype(np.int)-2

def compute_distance_unweighted(training_image_event, simulation_grid_event):
    sum = 0
    n = len(simulation_grid_event)
    for i, j in zip(simulation_grid_event, training_image_event):
        if i != j:
            sum += 1
    return float(sum)/n


def compute_distance_hamming(training_image_event, simulation_grid_event):
    return np.count_nonzero(training_image_event!=simulation_grid_event)/float(len(training_image_event))

def get_n_informed_neighbors(simulation_grid, current_node_indices):
    """
    #################
    #001000^00000000#
    #000000|00000000#
    #000000c00000000#
    #000000|00000000#
    #<--a--x-----b->#
    #000000|00000000#
    #000000d00000000#
    #000000_00000000#
    #################
    :param simulation_grid:
    :param current_node_indices:
    :return:
    """
    shape_x, shape_y = simulation_grid.shape
    nx = current_node_indices[0]
    ny = current_node_indices[1]

    a = nx
    b = shape_x-nx

    c = ny
    d = shape_y-ny


def find_nearest_points(traced_path, current_node, n_points):
    """
    informed_grid_nodes_x = []
    informed_grid_nodes_y = []
    #print simulation_grid.shape
    for i in range(simulation_grid.shape[0]-1):
        for j in range(simulation_grid.shape[1]-1):
            if simulation_grid[i, j] != -1:
                informed_grid_nodes_x.append(i)
                informed_grid_nodes_y.append(j)

    point_list = list(zip(informed_grid_nodes_x, informed_grid_nodes_y))
    """

    tree = spatial.KDTree(traced_path)
    distance, nearest_points_ids = tree.query(current_node, n_points, distance_upper_bound=50.)
    #print distance, nearest_points_ids

    if np.isinf(distance).any():
        return None

    nearest_points = tree.data[nearest_points_ids]

    return nearest_points, distance

def get_search_window_bounds(lag_vectors):
    x, y = lag_vectors.T
    a = x.min()
    b = x.max()
    c = y.min()
    d = y.max()

    return a, b, c, d

def compute_lag_vectors(current_node, nearest_points):
    lag_vectors = np.subtract(nearest_points, current_node)
    return lag_vectors

def get_absolute_image_positions(current_node, lag_vectors):
    return np.add(current_node, lag_vectors)

def get_values_relative_to_lag_vector(grid, current_node, lag_vectors):
    absolute_node_indices = get_absolute_image_positions(current_node, lag_vectors)
    return get_values_at_node_indices(grid, absolute_node_indices)


def get_values_at_node_indices(grid, nodes):
    X_cords, Y_cords = nodes.T
    return grid[[X_cords, Y_cords]]


def get_random_node_in_search_window(training_image, search_window_bounds):
    nx, ny = training_image.shape
    search_window_bounds = np.abs(search_window_bounds)

    lower_x_bound = search_window_bounds[0]
    upper_x_bound = search_window_bounds[1]

    lower_y_bound = search_window_bounds[2]
    upper_y_bound = search_window_bounds[3]

    lower_x = lower_x_bound
    upper_x = nx-upper_x_bound

    lower_y = lower_y_bound
    upper_y = ny - upper_y_bound

    return (int(np.random.uniform(low=lower_x, high=upper_x)), int(np.random.uniform(low=lower_y, high=upper_y)))

def get_random_training_image_value(training_image):
    return training_image[(int(np.random.uniform(low=0, high=training_image.shape[0])),
                           int(np.random.uniform(low=0, high=training_image.shape[1])))]


import itertools

def lcs_lens(xs, ys):
    curr = list(itertools.repeat(0, 1 + len(ys)))
    for x in xs:
        prev = list(curr)
        for i, y in enumerate(ys):
            if x == y:
                curr[i + 1] = prev[i] + 1
            else:
                curr[i + 1] = max(curr[i], prev[i + 1])
    return curr

def lcs(xs, ys):
    nx, ny = len(xs), len(ys)
    if nx == 0:
        return []
    elif nx == 1:
        return [xs[0]] if xs[0] in ys else []
    else:
        i = nx // 2
        xb, xe = xs[:i], xs[i:]
        ll_b = lcs_lens(xb, ys)
        ll_e = lcs_lens(xe[::-1], ys[::-1])
        _, k = max((ll_b[j] + ll_e[ny - j], j)
                    for j in range(ny + 1))
        yb, ye = ys[:k], ys[k:]
        return lcs(xb, yb) + lcs(xe, ye)


def resolve_nearest_points(nearest_points, traced_path):
    return itemgetter(*nearest_points)(traced_path)

@autojit
def find_n_closest_points_matrix(image, node, n_neighbors):
    a, b, c, d = 1, 1, 1, 1
    if node[0] == 0:
        a = 0
    if node[1] == 0:
        c = 0
    temp_image = image+1
    while True:
        sub_template = temp_image[node[0] - a:node[0] + b + 1, node[1] - c:node[1] + d + 1]
        neighbor_count = np.argwhere(sub_template)

        if len(neighbor_count) >= n_neighbors:
            return neighbor_count

        if node[0]-a > 0:
            a += 1

        if node[1]-c > 0:
            c += 1

        if b < image.shape[0]-node[0]:
            b += 1

        if d < image.shape[1]-node[1]:
            d += 1

        if node[0] - a == 0 and node[1]-c == 0 and node[0]+b == image.shape[0] and node[1]+d == image.shape[1]:
            return neighbor_count


def create_tree(traced_path):
    return spatial.cKDTree(traced_path)


def find_closest_tree(current_node, traced_path, n_neighbors):
    tree = create_tree(traced_path)
    ndx = tree.query(current_node, n_neighbors)
    return ndx


@jit(nopython=True, nogil=True)
def find_closest_points_brute_force(current_node, traced_path, distances):
    for i in range(len(traced_path)):
        distances[i] = (traced_path[i][0]-current_node[0])*(traced_path[i][0]-current_node[0]) \
                       + (traced_path[i][1]-current_node[1])*(traced_path[i][1]-current_node[1])
    return distances

@jit(nopython=True)
def compute_distances(current_node, traced_path, distances):
    for i in range(len(traced_path)):
        a = traced_path[i][0]-current_node[0]
        b = traced_path[i][1]-current_node[1]
        distances[i] = a*a+b*b
    return distances


@cuda.jit#(nopython=True)
def compute_distances_cuda(current_node, traced_path, distances):
    for i in range(len(traced_path)):
        a = traced_path[i][0]-current_node[0]
        b = traced_path[i][1]-current_node[1]
        distances[i] = a*a+b*b
    return distances


def convert_to_list(traced_path, nearest_point_ids):
    return [traced_path[i] for i in nearest_point_ids]

def get_values_relative_to_lag_vector_rework(training_image, node, lag_vectors):
    values = np.zeros(lag_vectors.shape[0]).astype(np.int32)

    for i, (a, b) in enumerate(lag_vectors):
        values[i] = training_image[node[0]+a, node[1]+b]
    return values

def subwindow_linear_search(training_image, bounds):
    nx, ny = training_image.shape
    a, b, c, d = bounds
    for i in xrange(a, nx-b):
        for j in xrange(c, ny-d):
            yield (i, j)

def get_random_node_in_search_window(training_image, search_window_bounds):
    nx, ny = training_image.shape
    search_window_bounds = np.abs(search_window_bounds)

    lower_x_bound = search_window_bounds[0]
    upper_x_bound = search_window_bounds[1]

    lower_y_bound = search_window_bounds[2]
    upper_y_bound = search_window_bounds[3]

    lower_x = lower_x_bound
    upper_x = nx - upper_x_bound

    lower_y = lower_y_bound
    upper_y = ny - upper_y_bound

    return (int(np.random.uniform(low=lower_x, high=upper_x)), int(np.random.uniform(low=lower_y, high=upper_y)))

def create_predicate(x_min, x_max, y_min, y_max):
    def predicate(x, y):
        if x_min <= x < x_max and y_min <= y < y_max:
            return True
        else:
            False
    return predicate

def get_absolute_image_positions(current_node, lag_vectors):
    return np.add(current_node, lag_vectors)

def get_values_relative_to_lag_vector(grid, current_node, lag_vectors):
    absolute_node_indices = get_absolute_image_positions(current_node, lag_vectors)
    return get_values_at_node_indices(grid, absolute_node_indices)


def get_values_at_node_indices(grid, nodes):
    X_cords, Y_cords = nodes.T
    return grid[[X_cords, Y_cords]]


@jit(nopython=True)
def get_event_jit(training_image, x_cords, y_cords):
    event = []
    for i in range(len(x_cords)):
        event.append(training_image[x_cords[i], y_cords[i]])
    return event

@jit(nopython=True)#("i4[:, :](i4, i4, i4[:, :])", nopython=True)
def sum_lag_vectors_jit(a, b, lags):
    x_nodes = []
    y_nodes = []
    for i in range(len(lags)):
        x_nodes.append(lags[i][0] + a)
        y_nodes.append(lags[i][1] + b)
    return x_nodes, y_nodes

@jit("f4(i4[:], i4[:])", nopython=True)
def compute_distance_unweighted_jit(training_image_event, simulation_grid_event):
    sum = 0
    n = len(simulation_grid_event)
    for i in range(len(simulation_grid_event)):
        if simulation_grid_event[i] != training_image_event[i]:
            sum += 1
    return float(sum)/n

@jit(nopython=True, nogil=True) #"i4(i4[:,:], i4[:, :], i4[:], f4, i4[:], i4, i4, i4)"
def find_event_in_ti_for_numba(training_image, lag_vectors, simulation_grid_event,
                               threshold, abs_bounds, nx, ny, max_iter):

    d_min = 999
    event_min = 1

    a, b, c, d = abs_bounds
    count = 0
    event_length = len(simulation_grid_event)
    for i in xrange(a, nx-b):
        for j in xrange(c, ny-d):

            #training_image_event = []
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
                #print "MaxIter Reached", d_min
                return event_min

            count += 1

    # print "MaxIter Reached", d_min
    if d_min == 999:
        return 1  # get_random_training_image_value(training_image)
    else:
        return event_min

@jit(nopython=True, nogil=True) #"i4(i4[:,:], i4[:, :], i4[:], f4, i4[:], i4, i4, i4)"
def find_event_in_ti_for_numba_parallel(node_val, training_image, lag_vectors, simulation_grid_event,
                                        threshold, abs_bounds, nx, ny, max_iter):

    d_min = 999

    a, b, c, d = abs_bounds
    count = 0
    event_length = len(simulation_grid_event)
    find = False
    for i in xrange(a, nx-b):
        for j in xrange(c, ny-d):

            #training_image_event = []
            sum = 0.0
            for k in range(len(lag_vectors)):
                if simulation_grid_event[k] != training_image[lag_vectors[k][0] + i, lag_vectors[k][1] + j]:
                    sum += 1

            event_distance = sum / event_length

            # if we find a perfect match, take it straight away
            if event_distance == 0.0 or event_distance < threshold:
                node_val = training_image[i, j]
                find = True
                break

            # if the current distance is between threshold and d_min set as new d_min, continue search
            elif threshold < event_distance < d_min:
                d_min = event_distance
                event_min = training_image[i, j]

            # if the distance is greater than d_min, continue on our search

            if count >= max_iter:
                # return the event with smallest found distance in max_iter iterations
                #print "MaxIter Reached", d_min
                find = True
                break

            count += 1
        if find:
            break

    # print "MaxIter Reached", d_min
    if d_min == 999:
        node_val = 1

def call_numba_search(training_image, lag_vectors, simulation_grid_event, threshold, factor=0.05):
    bounds = get_search_window_bounds(lag_vectors)
    abs_bounds = np.abs(bounds)
    nx, ny = training_image.shape

    max_iter = int(factor*nx*ny)
    node_val = 1

    #find_event = make_multithreaded_node_find(node_val)

    return find_event_in_ti_for_numba(training_image, lag_vectors, simulation_grid_event,
                                      threshold, abs_bounds, nx, ny, max_iter)

def find_corresponding_event_in_ti(training_image, lag_vectors, simulation_grid_event, threshold, factor=0.1):

    bounds = get_search_window_bounds(lag_vectors)
    abs_bounds = np.abs(bounds)
    d_min, event_min = np.inf, np.inf

    linear_search_space = subwindow_linear_search(training_image, abs_bounds)

    nx, ny = training_image.shape

    max_iter = int(factor * nx * ny)
    for i, node in enumerate(linear_search_space):

        training_image_event = get_values_relative_to_lag_vector(training_image,
                                                                 node,
                                                                 lag_vectors)

        event_distance = compute_distance_hamming(training_image_event, simulation_grid_event)

        # if we find a perfect match, take it straight away
        if event_distance == 0.0 or event_distance < threshold:
            return training_image[node]

        # if the current distance is between threshold and d_min set as new d_min, continue search
        elif threshold < event_distance < d_min:
            d_min = event_distance
            event_min = training_image[node]

        # if the distance is greater than d_min, continue on our search

        if i >= max_iter:
            # return the event with smallest found distance in max_iter iterations
            # print "MaxIter Reached", d_min
            return event_min

    else:
        # print "MaxIter Reached", d_min
        if d_min == np.inf:
            return 1 #get_random_training_image_value(training_image)
        else:
            return event_min

def find_n_smallest_distances(distances, n_neighbors):
    return bn.argpartition(distances, kth=n_neighbors)

def find_n_nearest_neighbors(current_node, traced_path, n_neighbors):
    distances = np.zeros(len(traced_path), dtype=np.int64)

    find_closest_points_brute_force(current_node, traced_path, distances)

    nearest_point_ids = find_n_smallest_distances(distances, n_neighbors)
    return nearest_point_ids

def nearest_neighbor_circle_scane(current_node, simulation_grid, max_range):
    flip = False
    factor = -1
    a = -1
    b = 1
    count = 0 #simulation_grid.shape[0]*simulation_grid[1] = max_range
    for i in xrange(max_range):
        if not flip:
            pass
        next_node = (current_node[0]+a, current_node[1]+1)


def spiral(current_node, X, Y):
    x, y = 0,0#current_node[0], current_node[1]
    dx = 0
    dy = -1
    for i in range(X*Y):#max(X, Y)**2):
        print (current_node[0]+x, current_node[1]+y)
        #if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
        #    print (x, y)
        # DO STUFF...
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy



def determine_value_at_node(current_node, training_image, simulation_grid, traced_path, threshold=0.1, n_neighbors=10):

    nearest_point_ids = find_n_nearest_neighbors(current_node, traced_path, n_neighbors)

    #get_closest_points_brute_force_revised_jit(current_node, traced_path, n_neighbors)#
    if len(nearest_point_ids) < n_neighbors:
        return 1
    #if nearest_point_ids is None:
    #    return 1

    nearest_point_ids = nearest_point_ids[0:n_neighbors]

    nearest_points = convert_to_list(traced_path, nearest_point_ids)

    lag_vectors = compute_lag_vectors(current_node, nearest_points)
    nearest_points = np.array(nearest_points)
    simulation_grid_event = get_values_at_node_indices(simulation_grid, nearest_points)

    node_value = call_numba_search(training_image, lag_vectors, simulation_grid_event, threshold)
    return node_value

def create_random_path(simulation_grid):
    x_range = product(range(simulation_grid.shape[0]), range(simulation_grid.shape[1]))
    path = list(x_range)
    np.random.shuffle(path)
    return path

@jit(nopython=True, nogil=True)
def inner_func_nb(result, traced_path, current_node):
    """
    Function under test.
    """
    for i in range(len(result)):
        a = current_node[0]
        b = current_node[1]
        result[i] = (traced_path[i][0]-a)*(traced_path[i][0]-a) \
                       + (traced_path[i][1]-b)*(traced_path[i][1]-b)

def make_multithread(inner_func, traced_path, distances, numthreads=4):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(traced_path)

        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk

        chunks = []
        for i in range(numthreads):
            result_chunk = distances[i * chunklen:(i + 1) * chunklen]
            distances_chunk = traced_path[i * chunklen:(i + 1) * chunklen]
            chunks.append([result_chunk, distances_chunk, args[0]])


        threads = [threading.Thread(target=inner_func, args=chunk) for chunk in chunks]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return distances
    return func_mt

def make_multithreaded_node_find(node_value):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(*args):
        args = (node_value, ) + args
        thread = threading.Thread(target=find_event_in_ti_for_numba_parallel, args=args)
        thread.start()
        thread.join()
        return node_value
    return func_mt

def direct_sampling_algorithm(simulation_grid, training_image, path, traced_path, n_neighbors=30):
    for i, path_index in enumerate(tqdm.tqdm(path, desc='Direct Sampling Operation', miniters=1)):

        if len(traced_path) <= n_neighbors:
            simulation_grid[path_index] = get_random_training_image_value(training_image)
        else:
            simulation_grid[path_index] = determine_value_at_node(path_index, training_image,
                                                                  simulation_grid, traced_path,
                                                                  n_neighbors=n_neighbors)

        traced_path.append(path_index)
        if i % 10000 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(13, 13))
            output_to_png("output/direct_sampling_strebelle_"+str(i)+".png", simulation_grid, ax, fig)
            output_to_png("output/input.png", training_image)
    return simulation_grid


from mpys.ds import DirectSampling

def main():
    np.random.seed(43)
    training_img = import_sgems_dat_file("data/bangladesh.sgems.txt").astype(np.int32)
    simulation_grid = create_new_empty_simulation_grid(100, 100).astype(np.int32)
    directed_path = random_path(simulation_grid.shape)

    support = tuple(directed_path[0])
    simulation_grid[support] = 1

    directed_path = directed_path[1::]

    traced_path = [support]


    direct_sampling = DirectSampling(training_img, simulation_grid, directed_path, traced_path,
                                     n_neighbors=30, threshold=0.1, sampling_fraction=0.1)

    simulated_grid = direct_sampling.run()

    #simulated_grid = direct_sampling_algorithm(simulation_grid, training_img, directed_path, traced_path, n_neighbors=50)

    fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    output_to_png("output/direct_sampling_bangladesh_final.png", simulated_grid, ax, fig)


if __name__ == "__main__":
    import platform

    print platform.architecture()
    main()