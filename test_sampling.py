import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from src.corr import CI, AI, SC

def select_3_clients(M, all_selected_clients, tolerance=1e-6):

    selected_clients = []

    M /= M.sum(axis=1, keepdims=True)

    index = np.random.choice(M.shape[1], p=M[0])
    selected_clients.append(index)

    with np.errstate(divide='ignore', invalid='ignore'):
        M /= np.linalg.norm(M, axis=0)
        M = np.nan_to_num(M, nan=0.0)
    client1 = np.copy(M[:, index])

    dot_products = np.dot(M.T, client1)
    for i in all_selected_clients + selected_clients:
        dot_products[i] = 1.0

    min_value = np.min(dot_products)
    min_indices = np.where(np.abs(dot_products - min_value) <= tolerance)[0]
    min_dot_product_index = np.random.choice(min_indices)

    selected_clients.append(min_dot_product_index)
    client2 = np.copy(M[:, min_dot_product_index])

    M[:, index] = np.zeros(3)
    M[:, min_dot_product_index] = np.zeros(3)

    orthogonal_vector = np.cross(client1, client2)
    orthonormal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    dot_products = np.dot(M.T, orthonormal_vector)
    for i in all_selected_clients + selected_clients:
        dot_products[i] = -1.0

    max_value = np.max(dot_products)
    max_indices = np.where(np.abs(dot_products - max_value) <= tolerance)[0]
    max_dot_product_index = np.random.choice(max_indices)

    selected_clients.append(max_dot_product_index)
    M[:, max_dot_product_index] = np.zeros(3)

    M = M[[2, 0, 1], :]

    return selected_clients, M

def select_clients(original_M, p, types_names):

    indices = np.random.permutation(original_M.shape[0])
    original_M = original_M[indices]
    types_names = [types_names[i] for i in indices]
    print(f"Types order: {types_names}. ", end='')

    selected_clients = []
    np.random.shuffle(original_M)
    M = original_M.copy()

    while len(selected_clients) - p < 0:
        three_new_clients, M = select_3_clients(M, selected_clients)
        selected_clients += three_new_clients

    return selected_clients


def plot_selected_clients(sorted_client_ids, all_selected_clients, client_types, types_names):

    fig, ax = plt.subplots(figsize=(10, 6))

    unique_types = sorted(set(client_types))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_types)))
    colors_dots = ['r', 'g', 'b']

    handles = []
    for client_type, color in zip(unique_types, colors):
        client_ids = [client_id for client_id in sorted_client_ids if client_types[client_id] == client_type]
        ax.axhspan(min(client_ids) - 0.5, max(client_ids) + 0.5, color=color, alpha=0.2)
        handles.append((client_type, Rectangle((0, 0), 1, 1, color=color)))

    dot_handles = []
    for i, color in enumerate(colors_dots):
        dot_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
        dot_handles.append((f"{i+1} client selected", dot_handle))

    for round_num, sampled_clients in enumerate(all_selected_clients):
        for i, client_id in enumerate(sampled_clients):
            ax.scatter(round_num + 1, client_id, color=colors_dots[i % len(colors_dots)])

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Client IDs")
    ax.set_title(f"Sampled Clients Over Rounds, {len(sorted_client_ids)} tot clients")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    legend_labels = [types_names[h[0]] for h in handles] + [h[0] for h in dot_handles]
    legend_handles = [h[1] for h in handles] + [h[1] for h in dot_handles]
    ax.legend(legend_handles, legend_labels, title='Client Types', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=(0, 0, 0.85, 1))
    plt.xlim(left=0.5)

    plt.show()


def get_original_M(client_samples):
    original_M = np.zeros((3, len(client_samples)))
    for i, client in enumerate(client_samples):
        original_M[0][i] = CI(client)
        original_M[1][i] = AI(client)
        original_M[2][i] = SC(client)
    types_names = ['CI', 'AI', 'SC']
    return original_M, types_names


def main():

    client_samples = []
    client_samples += [[[28, 28], [0, 0]]] * 3  # Mostly waterbirds, Type CI (0)
    client_samples += [[[0, 0], [28, 28]]] * 3  # Mostly landbirds, Type CI (0)
    client_samples += [[[28, 0], [28, 0]]] * 3  # Birds on land, Type AI (1)
    client_samples += [[[0, 28], [0, 28]]] * 3  # Birds on water, Type AI (1)
    client_samples += [[[28, 0], [0, 28]]] * 5  # Expected background, Type SC (2)
    client_samples += [[[0, 28], [28, 0]]] * 1  # Unexpected background, Type SC (2)

    p = 6  # 3, 6, 9, ...
    num_rounds = 10

    original_M, types_names = get_original_M(client_samples)
    client_types = [np.argmax(original_M[:, c]) for c in range(len(client_samples))]

    combined = list(zip(client_samples, client_types))
    sorted_combined = sorted(combined, key=lambda x: x[1])
    client_samples = [client[0] for client in sorted_combined]
    client_types = [client[1] for client in sorted_combined]
    original_M, types_names = get_original_M(client_samples)

    all_selected_clients = []
    for r in range(num_rounds):
        selected_clients = select_clients(original_M, p, types_names=types_names)
        print(f"The selected clients are {selected_clients}")
        all_selected_clients.append(selected_clients)

    plot_selected_clients(list(range(len(client_samples))), all_selected_clients, client_types, types_names)


if __name__ == '__main__':
    np.random.seed(13)
    main()
