import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays


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


def client_weights_known_groups(metric_list, conf, shared_copt_params):
    """Calculate client weights if we know the N matrix of all clients"""
    client_weights = []
    for m in metric_list:
        group_keys = [k for k in m.keys() if k.startswith("groupsize_")]
        w = np.sum([shared_copt_params[k]/m[k] for k in group_keys if m[k]>0])
        client_weights.append(w)
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