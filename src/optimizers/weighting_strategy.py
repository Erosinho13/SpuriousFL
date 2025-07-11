from typing import List
import numpy as np
from src.utils import adjust_array, get_device
import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from cvxopt import matrix, solvers
from sklearn.cluster import KMeans

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

def client_weights_nova(results):
    """FedNova client weights (use together with FedAvgM optimizer): 
    https://github.com/adap/flower/blob/main/baselines/fednova/fednova/strategy.py"""
    local_tau = np.array([float(res.metrics["tau"]) for _, res in results])
    tau_eff = np.sum(local_tau)
    local_norm = np.array([float(res.metrics["local_norm"]) for _, res in results])
    datasize = np.array([float(res.metrics["weights"]) for _, res in results])
    client_weights = tau_eff / local_norm * datasize
    return client_weights
    


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

def client_weights_fairfed(metric_list, omega_list, conf):
    """Weight clients according to FairFed: https://ojs.aaai.org/index.php/AAAI/article/view/25911"""
    train_acc_list = []
    size_list = []
    group_f_list = []
    for m in metric_list:
        datasize = 0
        group_keys = [k for k in m.keys() if k.startswith("groupsize_")]
        for k in group_keys:
            datasize += m[k]
        train_acc = m["groupacc_train_accuracy"]
        train_acc_list.append(train_acc)
        gacc_keys = [k for k in m.keys() if k.startswith("groupacc_y")]
        acc_dict = {k:v for k,v in m.items() if k in gacc_keys}
        max_index = max(int(key.split('_')[1]) for key in acc_dict)
        acc_list = [0] * (max_index + 1)
        for key, value in acc_dict.items():
            index = int(key.split('_')[1])  # Extract the index part from the key
            acc_dict[index] = value
        acc_matrix = np.array(acc_list)
        acc_matrix = np.resize(acc_matrix, (conf["dataset_options"]["num_targets"],conf["dataset_options"]["num_groups"]))
        acc_diff = acc_matrix.max(axis=1) - acc_matrix.min(axis=1)
        fairness_scalar = np.average(acc_diff)
        group_f_list.append(fairness_scalar)
    group_f_list = np.array(group_f_list)
    train_acc_list = np.array(train_acc_list)
    avg_acc = np.average(train_acc_list)
    avg_f = np.average(group_f_list)
    delta_list = np.abs(group_f_list - avg_f)
    avg_delta = np.average(delta_list)
    omega_list = omega_list - conf["server_opt"]["fairfed_beta"]*(delta_list-avg_delta)
    omega_list = omega_list/np.sum(omega_list)
    return omega_list

def select_clients_with_uniform_distribution(metric_list, conf, server_round):
    """
    Select k clients whose sum matrix has the most uniform distribution using a greedy approach.
    
    :param client_matrices: np.ndarray of shape (n, C, G) representing n clients with CxG matrices.
    :param k: Number of clients to select.
    :return: A tuple (selected_indices, sum_matrix), where selected_indices is a list of k client indices,
             and sum_matrix is the sum of their matrices.
    """
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
    
    client_matrices = np.array(n_matrix_list)
    k = conf["server_opt"]["num_active_clients"]

    n, C, G = client_matrices.shape
    first_client = np.random.default_rng(server_round).integers(0,n,1)[0]
    selected_indices = [first_client]
    remaining_indices = set(range(n))
    remaining_indices.remove(first_client)
    sum_matrix = np.zeros((C, G))
    
    for _ in range(k-1):
        best_client = None
        min_variance = float('inf')
        
        for i in remaining_indices:
            temp_sum_matrix = sum_matrix + client_matrices[i]
            variance = np.var(temp_sum_matrix)
            
            if variance < min_variance:
                min_variance = variance
                best_client = i
        
        selected_indices.append(best_client)
        remaining_indices.remove(best_client)
        sum_matrix += client_matrices[best_client]
    
    return selected_indices, sum_matrix

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

def top_k_binary_list(float_list, k):
    # Get the indices of the sorted list in descending order
    sorted_indices = sorted(range(len(float_list)), key=lambda i: float_list[i], reverse=True)
    
    # Initialize a list of zeros
    binary_list = [0] * len(float_list)
    
    # Set the top k indices to 1
    for i in sorted_indices[:k]:
        binary_list[i] = 1
    
    return binary_list


def get_relation(grad_list, avg_grad, idxs_users, conf={}):
    """Given list of gradients and an avg_grad, calculates the avg_grad's relation to the others
    Adapted from FedPNS code"""
    innnr_value = {}

    avg_grad_tensor = torch.tensor(avg_grad, dtype=torch.float32).to(get_device(conf))
    for i in range(len(idxs_users)):
        user_grad_tensor = torch.tensor(grad_list[idxs_users[i]], dtype=torch.float32).to(get_device(conf))
        innnr_value[idxs_users[i]] = dot_sum_torch(user_grad_tensor, avg_grad_tensor)
    print(innnr_value)
    return sum(list(innnr_value.values()))

def dot_sum(K, L):
    return sum(a * b for a,b in zip(K, L))

def dot_sum_torch(K_tensor, L_tensor):
    return torch.dot(K_tensor, L_tensor).cpu().item()

def flatten_weights(weights):
    """Get one long list of weights from layer weight list"""
    ini = []
    for w in weights:
        ini += list(w.flatten())
    return ini


def node_deleting(expect_list, expect_value, worker_ind, grads, conf={}):
    """Calculates expectation for each client based on the rest of the grads
    Adapted from FedPNS code"""
    # expect_list.pop("all")
    for i in range(len(worker_ind)):
        worker_ind_del  = [n for n in worker_ind if n != worker_ind[i]]
        grad_del = grads.copy()
        grad_del.pop(worker_ind[i])
        avg_grad_del = np.mean(list(grad_del.values()), axis=0)
        expect_value_del = get_relation(grad_del, avg_grad_del, worker_ind_del, conf=conf)
        expect_list[worker_ind[i]] = expect_value_del
    expect_list["all"] = expect_value
    return expect_list

def probabilistic_selection(node_prob, node_count, labeled, alpha, beta):
    """update FedPNS p for each client (does not select). Name kept for compatibility with orig code"""
    all_ids = list(node_prob.keys())
    rest_nodes = [i for i in all_ids if i not in labeled]
    weight = 0
 
        
    ratio = {}
    for i in labeled:
        ratio[i] = node_count[i][1]/ node_count[i][0]
        
    for i in labeled:
        prob_change =  node_prob[i] * min( (ratio[i] + beta)**alpha, 0.99)
        weight += prob_change
        node_prob[i] =  node_prob[i] - prob_change
 
    for i in rest_nodes:
            node_prob[i] = node_prob[i] + weight / (len(rest_nodes))

    return node_prob

def clients_clustering(compressed_gradients: np.array, num_clusters: int, tolerance: float = 1e-10) -> List[int]:
    """
    Client clustering according to their compressed gradients.

    :param compressed_gradients: numpy array of compressed gradients per each client.
    :param num_clusters: desired number of clusters.
    :param tolerance: tolerance parameter for computing the centers.
    :return: list of cluster indices associated with each client.
    """

    row_indices = np.random.choice(compressed_gradients.shape[0], size=num_clusters, replace=False)
    centers = compressed_gradients[row_indices]
    clusters = (KMeans(n_clusters=num_clusters, random_state=0, init=centers, tol=tolerance)
                .fit(compressed_gradients).labels_)
    return clusters

def HCSFed(num_clients_per_round: int, num_clients: int, clusters: List[int], num_clusters: int,
           compressed_gradients: np.array, cluster_clients: List[List]) -> List[int]:

    """
    The HCSFed algorithm.

    :param num_clients_per_round: number of clients per round.
    :param num_clients: number of clients in total.
    :param clusters: list l such that client at index k belongs to cluster with index l[k].
    :param num_clusters: desired number of clusters.
    :param compressed_gradients: compressed gradients per each client, computed using compress_gradients.
    :param cluster_clients: inverse of the clusters array. It is a list of lists, where at each index i (cluster) there
                            is the list of indices of the clients belonging to cluster i.

    :return: indices of selected clients for the current round.
    https://arxiv.org/pdf/2208.05135
    """

    q = num_clients_per_round / num_clients
    tot_m = max(q * num_clients, 1)
    N = np.bincount(clusters, minlength=num_clusters)

    eps = 1e-6  # needed for numerical stability in case N[h] = 1, no need to change this value
    S = []
    X = []
    for h in range(num_clusters):
        Xh = compressed_gradients[[i for i, j in enumerate(clusters) if j == h]]
        X.append(Xh)
        if len(Xh) == 0:
            diff = 0
        else:
            diff = np.sum(np.linalg.norm(Xh[:, np.newaxis] - Xh, ord=2, axis=2) ** 2)
        S.append(1 / (N[h] - 1 + eps) * diff)
    m = np.array([(N[h] * S[h]) / (sum(N * S)) * tot_m for h in range(num_clusters)])
    m = adjust_array(m, num_clients_per_round, [len(i) for i in cluster_clients])

    clusters_norm = [sum([np.linalg.norm(k, ord=2) for k in compressed_gradients[clusters == h]]) for h in
                     range(num_clusters)]
    all_p = np.array(
        [np.linalg.norm(compressed_gradients[k], ord=2) / clusters_norm[clusters[k]] for k in range(num_clients)])
    p = [[j for i, j in enumerate(all_p) if h == clusters[i]] for h in range(num_clusters)]

    sampled_clients = []
    for h in range(num_clusters):
        if m[h] == 0:
            continue
        p[h] /= sum(p[h])  # probabilities already sum to 1, however this is needed for numerical errors
        try:
            cluster_clients_idx = np.random.choice(np.arange(len(p[h])), size=m[h], replace=False, p=p[h])
        except Exception as e:
            print(e)
            pass
        sampled_clients += list(np.array(cluster_clients[h])[cluster_clients_idx])
    sampled_clients = list(map(int, sampled_clients))
    sampled_clients.sort()

    return sampled_clients