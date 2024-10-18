import numpy as np
import random
import flwr as fl
import threading
from typing import Optional
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from src.optimizers.weighting_strategy import client_weights_known_groups, select_noreplacement
from src.utils import log
from logging import ERROR, INFO, DEBUG
from logging import WARNING

class MyManager(fl.server.SimpleClientManager):
    """Provides a pool of available clients."""

    def __init__(self, conf) -> None:
        self.clients: dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.conf = conf

    def sample(
        self,
        num_clients: int,
        client_info:Optional[dict] = None, 
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        eval: Optional[bool] = False
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
        if num_clients == len(available_cids) or eval:
            sampled_cids = random.sample(available_cids, num_clients)
            return [self.clients[cid] for cid in sampled_cids]
        

        if self.conf["server_opt"]["selection_method"] == "random":
            sampled_cids = random.sample(available_cids, num_clients)
        elif self.conf["server_opt"]["selection_method"] == "groupweights":
            metric_list = [client_info[int(cid)] for cid in available_cids]
            client_weights = client_weights_known_groups(metric_list, self.conf)
            client_weights = np.array(client_weights)/sum(client_weights)
            elements = list(range(len(client_weights)))
            sampled_ids = np.random.choice(elements, size=num_clients, p=client_weights, replace=False)
            sampled_cids = [str(cid) for cid in sampled_ids]
        elif self.conf["server_opt"]["selection_method"].startswith("triplets_stochasticmatrix"):
            metric_list = [client_info[int(cid)] for cid in available_cids]
            M = np.array([[a['SC'],a['AI'],a['CI']] for a in metric_list])
            selected_clients = select_noreplacement(M.T, num_clients)
            sampled_cids = [str(cid) for cid in selected_clients]
        return [self.clients[cid] for cid in sampled_cids]