import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from cvxopt import matrix, solvers

def select_3_clients(M, all_selected_clients, tolerance=1e-6):

    _selected_clients = []

    M /= M.sum(axis=1, keepdims=True)

    index = np.random.choice(M.shape[1], p=M[0])
    _selected_clients.append(index)

    with np.errstate(divide='ignore', invalid='ignore'):
        M /= np.linalg.norm(M, axis=0)
        M = np.nan_to_num(M, nan=0.0)
    client1 = np.copy(M[:, index])

    dot_products = np.dot(M.T, client1)
    for i in all_selected_clients + _selected_clients:
        dot_products[i] = 1.0

    min_value = np.min(dot_products)
    min_indices = np.where(np.abs(dot_products - min_value) <= tolerance)[0]
    min_dot_product_index = np.random.choice(min_indices)

    _selected_clients.append(min_dot_product_index)
    client2 = np.copy(M[:, min_dot_product_index])

    M[:, index] = np.zeros(3)
    M[:, min_dot_product_index] = np.zeros(3)

    orthogonal_vector = np.cross(client1, client2)
    orthonormal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    dot_products = np.dot(M.T, orthonormal_vector)
    for i in all_selected_clients + _selected_clients:
        dot_products[i] = -1.0

    max_value = np.max(dot_products)
    max_indices = np.where(np.abs(dot_products - max_value) <= tolerance)[0]
    max_dot_product_index = np.random.choice(max_indices)

    _selected_clients.append(max_dot_product_index)
    M[:, max_dot_product_index] = np.zeros(3)

    M = M[[2, 0, 1], :]

    return _selected_clients, M


def select_noreplacement(original_M, p):
    """Given stochastic matrix generated from the client triplets, selects p clients without replacement"""
    indices = np.random.permutation(original_M.shape[0])
    original_M = original_M[indices]

    _selected_clients = []
    np.random.shuffle(original_M)
    M = original_M.copy()

    while len(_selected_clients) - p < 0:
        three_new_clients, M = select_3_clients(M, _selected_clients)
        _selected_clients += three_new_clients

    return _selected_clients


def client_weights_IDA(results):
    """IDA weights from paper: https://arxiv.org/pdf/2008.07665"""
    # Convert results
    numpy_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]
    w_flats = [w[-1] for w,_ in numpy_results]
    # w_flats = [np.concatenate([l.flatten() for l in w]) for w,_ in numpy_results]
    w_avg = np.average(w_flats)
    l1_norms = [np.linalg.norm(w-w_avg) for w in w_flats]
    l1_sum = sum(l1_norms)
    client_weights = [l1/l1_sum for l1 in l1_norms]
    return client_weights


def apply_smoothing(client_weights, eps=0.0001):
    """Replace 0s with eps and reduce non0s by eps"""
    binary_arr = (client_weights > 0).astype(int)
    smoothed_labels = client_weights - binary_arr + binary_arr * (1 - eps) + (1 - binary_arr) * eps
    return smoothed_labels


def client_weights_known_groups(metric_list, conf):
    """Calculate client weights if we know the N matrix of all clients"""
    client_weights = []
    n_matrix_list = []
    for m in metric_list:
        group_keys = [k for k in m.keys() if k.startswith("groupsize_")]
        n_dict = {k:v for k,v in m.items() if k in group_keys}
        max_index = max(int(key.split('_')[1]) for key in n_dict)
        n_list = [0] * (max_index + 1)
        for key, value in n_dict.items():
            index = int(key.split('_')[1])  # Extract the index part from the key
            n_list[index] = value
        n_matrix = np.array(n_list)
        n_matrix = np.resize(n_matrix, (conf["dataset_options"]["num_targets"],conf["dataset_options"]["num_groups"]))
        n_matrix_list.append(n_matrix)
    client_weights = weights_from_n_matrix_list(n_matrix_list)
    return client_weights


def apply_softmax(client_weights):
    """Put weights between 0 and 1 with softmax"""
    client_weights = np.array(client_weights)
    client_weights = np.exp(client_weights)/sum(np.exp(client_weights))
    return client_weights


def temperature_weighted_values(list1, list2, temperature):
    """Weighted sum of the two list by temperature"""
    temperature1 = 0+temperature
    temperature2 = 2-temperature
    client_weights1 = np.array(list1)
    client_weights2 = np.array(list2)
    cw1sum = sum(np.exp(client_weights1/temperature1))
    cw2sum = sum(np.exp(client_weights2/temperature2))
    client_weights = [np.exp(cw1/temperature1)/cw1sum+np.exp(cw2/temperature2)/cw2sum for cw1, cw2 in zip(client_weights1, client_weights2)]
    return client_weights


def upscale(client_weights, factor):
    cw_sum = np.sum(client_weights)
    client_weights = [cw/cw_sum*factor for cw in client_weights]
    return client_weights


def weights_from_n_matrix_list(matrix_list):

    A = np.array(matrix_list)

    n = len(A)
    m, p = A[0].shape

    # Compute means of each matrix
    a_bar = [np.mean(A[i]) for i in range(n)]

    # Compute deviations
    deviations = [A[i] - a_bar[i] for i in range(n)]

    # Flatten deviations and stack
    flattened_devs = [deviations[i].flatten() for i in range(n)]
    D = np.stack(flattened_devs, axis=1)  # Shape: (m*p, n)

    # Quadratic term
    P = (1.0 / (m * p)) * np.dot(D.T, D)
    P = matrix(P)

    # Linear term
    q = np.zeros(n)
    q = matrix(q)

    # Inequality constraints: w_i >= 0
    G = -np.eye(n)
    h = np.zeros(n)
    G = matrix(G)
    h = matrix(h)

    # Equality constraints: sum(w) = 1
    A_eq = np.ones((1, n))
    A_eq = matrix(A_eq)
    b_eq = matrix(1.0)

    # Solve QP
    solution = solvers.qp(P, q, G, h, A_eq, b_eq)

    # Extract weights
    weights = np.array(solution['x']).flatten()
    #up_weights = weights/min(weights)
    #up_weights = [int(w) for w in up_weights]
    return weights