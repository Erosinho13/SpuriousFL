import numpy as np
import random
import flwr as fl
import threading
from typing import Optional
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from src.optimizers.weighting_strategy import HCSFed, client_weights_known_groups, clients_clustering, select_noreplacement, select_clients_with_uniform_distribution
from src.utils import log
from logging import ERROR, INFO, DEBUG
from logging import WARNING

class MyManager(fl.server.SimpleClientManager):
    """Provides a pool of available clients."""

    def __init__(self, conf) -> None:
        self.clients: dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.conf = conf
        self.past_sampled_ids = np.zeros(self.conf['dataset_options']['num_clients'])  # needed for roundrobin sampling strategy
        self.clusters = None

    def sample(
        self,
        num_clients: int,
        client_info:Optional[dict] = None, 
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        evaluate: Optional[bool] = False,
        server_round: Optional[int] = None
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        if num_clients == len(available_cids) or evaluate:
            sampled_cids = random.sample(available_cids, num_clients)
            return [self.clients[cid] for cid in sampled_cids]

        print(client_info)

        if self.conf["server_opt"]["selection_method"] == "random":
            sampled_cids = random.sample(available_cids, num_clients)
        elif self.conf["server_opt"]["selection_method"] == "groupweights":
            metric_list = [client_info[int(cid)] for cid in available_cids]
            client_weights = client_weights_known_groups(metric_list, self.conf)
            client_weights = np.array(client_weights)/sum(client_weights)
            elements = list(range(len(client_weights)))
            sampled_ids = np.random.choice(elements, size=num_clients, p=client_weights, replace=False)
            sampled_cids = [str(cid) for cid in sampled_ids]
        elif self.conf["server_opt"]["selection_method"] == "uniformsample":
            metric_list = [client_info[int(cid)] for cid in available_cids]
            sampled_ids, _ = select_clients_with_uniform_distribution(metric_list, self.conf, server_round)
            sampled_cids = [str(cid) for cid in sampled_ids]
        elif self.conf["server_opt"]["selection_method"].startswith("triplets_stochasticmatrix"):
            metric_list = [client_info[int(cid)] for cid in available_cids]
            M = np.array([[a['SC'],a['AI'],a['CI']] for a in metric_list])
            selected_clients = select_noreplacement(M.T, num_clients)
            sampled_cids = [str(cid) for cid in selected_clients]
        elif self.conf["server_opt"]["selection_method"] == "roundrobin":
            sampled_ids = []
            zero_indices = np.where(self.past_sampled_ids == 0)[0]
            if len(zero_indices)>num_clients:
                sampled_ids = np.random.choice(zero_indices,num_clients, replace=False)
                self.past_sampled_ids[zero_indices] = 1
            else:
                sampled_ids = zero_indices
                remaining = num_clients - len(sampled_ids)
                one_indices = np.where(self.past_sampled_ids != 0)[0]
                sampled_ids_more = np.random.choice(one_indices,remaining, replace=False)
                sampled_ids = np.concatenate((sampled_ids, sampled_ids_more))
                self.past_sampled_ids[:] = 0
                self.past_sampled_ids[sampled_ids_more] = 1
            sampled_cids = [str(cid) for cid in sampled_ids]
        elif self.conf["server_opt"]["selection_method"] == "fedpns":
            pns_probs = [client_info[int(cid)]["fedpns_p"] for cid in available_cids]
            sampled_cids = np.random.choice(available_cids,num_clients,replace=False,p=pns_probs)
        elif self.conf["server_opt"]["selection_method"] == "oort":
            utils = [client_info[int(cid)]["oort"]+np.sqrt(0.1*np.log(server_round)/client_info[int(cid)]["last_round"]) for cid in available_cids]
            sampled_utils = np.argpartition(utils, num_clients)[:num_clients]
            sampled_cids = np.array(available_cids)[sampled_utils]

        elif self.conf["server_opt"]["selection_method"].startswith("embbalance"):
            available_client_info = {int(k): client_info[int(k)] for k in available_cids}
            sampled_ids = get_furthest_points(available_client_info, num_clients, server_round, mode=self.conf["server_opt"]["selection_method"])
            sampled_cids = [str(k) for k in sampled_ids.keys()]
        elif self.conf["server_opt"]["selection_method"] == "hcsfed":

            dimension_keys = [key for key in sorted(client_info[0].keys()) if key.startswith('compgrad_')]
            compressed_gradients = np.array([[client_info[int(c)][k] for k in dimension_keys] for c in available_cids])

            if self.clusters is None or((self.conf["server_opt"]["update_static_info_rounds"] != 0) and (server_round % self.conf["server_opt"]["update_static_info_rounds"]== 1)):
                self.clusters = clients_clustering(compressed_gradients=compressed_gradients, num_clusters=self.conf["client_opt"]["hcsfed_num_clusters"],
                                  tolerance=self.conf["client_opt"]["hcsfed_tolerance_cl"])
            cluster_clients = [list(np.where(np.array(self.clusters) == i)[0]) for i in range(self.conf["client_opt"]["hcsfed_num_clusters"])]    
            sampled_clients = HCSFed(num_clients, len(available_cids), self.clusters,
                             self.conf["client_opt"]["hcsfed_num_clusters"], compressed_gradients, cluster_clients)
            sampled_cids = [available_cids[i] for i in sampled_clients]
        return [self.clients[cid] for cid in sampled_cids]
    



def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two d-dimensional points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_furthest_points(data, K, server_round, mode="embbalance"):
    """Return K furthest points from the center, then iteratively from the last chosen point."""
    # Extract all dimension keys dynamically
    sample_entry = next(iter(data.values()))
    dimension_keys = [key for key in sample_entry if key.startswith('netemb_')]

    # Extract points in the form {id: (netemb_0, netemb_1, ..., netemb_d)}
    points = {idx: tuple(entry[key] for key in dimension_keys) for idx, entry in data.items()}

    # Compute the center (mean of all points in d-dimensional space)
    coords = np.array(list(points.values()))
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords = 2 * (coords - coords_min) / (coords_max - coords_min) - 1
    if mode=="embbalance2":
        delta = np.array(list({idx: server_round-entry['last_round'] for idx, entry in data.items()}))
        coords = coords * (1 / (1 + np.exp(-1 * delta))).reshape(-1, 1)
    if mode=="embbalance3":
        pass

    points = {idx: tuple(coord) for idx, coord in zip(points.keys(), coords)}

    center = np.mean(coords, axis=0) 

    # Start with the point furthest from the center
    first_id = max(points, key=lambda idx: euclidean_distance(points[idx], center))
    furthest_ids = [first_id]

    # Iteratively select the point furthest from the last chosen one
    while len(furthest_ids) < K:
        last_id = furthest_ids[-1]
        next_id = max(set(points.keys()) - set(furthest_ids), 
                      key=lambda idx: euclidean_distance(points[idx], points[last_id]))
        furthest_ids.append(next_id)

    # Return the selected points in {id: {netemb_0, ..., netemb_d}} format
    return {idx: {dim: points[idx][i] for i, dim in enumerate(dimension_keys)} for idx in furthest_ids}