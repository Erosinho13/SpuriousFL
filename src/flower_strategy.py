import flwr as fl
from logging import ERROR, INFO, DEBUG
from logging import WARNING
import os
import numpy as np
import copy
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from sklearn.cluster import kmeans_plusplus
from src.optimizers import subpop_federated 
from src.utils import log
from src.models import model_utils


def fit_metrics_aggregation_fn(fit_metrics):
    losses = [a[1]["loss"] for a in fit_metrics]
    return {"train_loss": np.mean(losses), "train_lossStd": np.std(losses)}


def evaluate_metrics_aggregation_fn(eval_metrics):
    keys = list(eval_metrics[0][1].keys())
    eval_res = {}

    for k in keys:
        eval_res[k] = sum([e[1][k] for e in eval_metrics])
    
    return eval_res


class MyStrategy(fl.server.strategy.FedOpt):
    """FedAdam, FedAvgM and saving and logging with WandB

    Implementation based on https://arxiv.org/abs/2003.00295v5

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    eta : float, optional
        Server-side learning rate. Defaults to 1e-1.
    eta_l : float, optional
        Client-side learning rate. Defaults to 1e-1.
    beta_1 : float, optional
        Momentum parameter. Defaults to 0.9.
    beta_2 : float, optional
        Second moment parameter. Defaults to 0.99.
    tau : float, optional
        Controls the algorithm's degree of adaptability. Defaults to 1e-9.
    """

    def __init__(self, conf, *args, **kwargs):
        self.conf = conf
        eta = self.conf["server_opt"]["learning_rate"]
        eta_l = self.conf["client_opt"]["learning_rate"]
        beta_1 = self.conf["server_opt"]["beta_1"]
        beta_2 = self.conf["server_opt"]["beta_2"]
        tau = self.conf["server_opt"]["tau"]
        self.shared_copt_params = subpop_federated.init_shared_opt_params(self.conf)

        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         eta=eta,
                         eta_l=eta_l,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         tau=tau,
                         *args, **kwargs)

    def aggregate_fit(
            self,
            server_round,
            results,
            failures,
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            self.train_metrics_aggregated = metrics_aggregated
            log(INFO, "aggregated fit results %s", str(metrics_aggregated))
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Calculate global client opt params from shared metrics
        self.aggregate_client_opt_params([res.metrics for _, res in results])

        # Calculate client weights with post-training methods
        if self.conf["server_opt"]["weight_clients"].startswith("server_post_"):
            results = self.post_calculate_weights(results, server_round=server_round)
        log(DEBUG, "Client weights: %s", [(res.num_examples, res.metrics["cid"]) for _, res in results])

        # Aggregate weights
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        if self.conf["server_opt"]["optimizer"]=="FedAvg":
            self.current_weights = fedavg_weights_aggregate
            return ndarrays_to_parameters(self.current_weights), metrics_aggregated
        
        if self.conf["server_opt"]["optimizer"]=="FedAvgM":
            # Following https://flower.ai/docs/framework/_modules/flwr/server/strategy/fedavgm.html#FedAvgM
            delta_t: NDArrays = [
                x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
            ]
            # m_t
            if not self.m_t:
                self.m_t = [np.zeros_like(x) for x in delta_t]
            self.m_t = [
                np.multiply(self.beta_1, x) + y
                for x, y in zip(self.m_t, delta_t)
            ]

            new_weights = [
                x + self.eta * y
                for x, y in zip(self.current_weights, self.m_t)
            ]
            self.current_weights = new_weights
            return ndarrays_to_parameters(self.current_weights), metrics_aggregated

        if self.conf["server_opt"]["optimizer"]=="FedAdam":
            # Following https://flower.ai/docs/framework/_modules/flwr/server/strategy/fedadam.html#FedAdam
            delta_t: NDArrays = [
                x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
            ]

            # m_t
            if not self.m_t:
                self.m_t = [np.zeros_like(x) for x in delta_t]
            self.m_t = [
                np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
                for x, y in zip(self.m_t, delta_t)
            ]

            # v_t
            if not self.v_t:
                self.v_t = [np.zeros_like(x) for x in delta_t]
            self.v_t = [
                self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
                for x, y in zip(self.v_t, delta_t)
            ]

            new_weights = [
                x + self.eta * y / (np.sqrt(z) + self.tau)
                for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
            ]

            self.current_weights = new_weights
            return ndarrays_to_parameters(self.current_weights), metrics_aggregated
        raise NotImplementedError(f'Server optimizer not recognized: {self.conf["server_opt"]["optimizer"]}')

    def aggregate_evaluate(
            self,
            rnd,
            results,
            failures,
    ):
        """Save final model"""
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        if rnd == self.conf["rounds"]:
            # end of training calls
            save_path = os.path.join(
                "checkpoints",
                self.conf["exp_id"],
                "final"
            )
            log(INFO, "Saving model to %s", save_path)
            model = model_utils.init_model(
                conf=self.conf, weights=self.current_weights
            )
            model_utils.save_model(model, save_path)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            log(INFO, "aggregated eval results %s", str(metrics_aggregated))
        elif rnd == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        if self.conf["wandb"]:
            import wandb
            wandb_log = aggregated_result[1]
            wandb_log["round"] = rnd
            wandb_log.update(self.train_metrics_aggregated)
            wandb.log(wandb_log)

        return aggregated_result

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        fit_configurations = []


        for client in clients:
            client_config = copy.deepcopy(self.shared_copt_params)     # {}

            if self.conf["server_opt"]["weight_clients"].startswith("server_pre_"):
                c_w = self.pre_calculate_weights(client)
                client_config['client_weight'] = c_w

            fit_configurations.append((client, FitIns(parameters, client_config)))
            del client_config
                
        return fit_configurations
    
    def pre_calculate_weights(self, client):
        """Override num_examples with weights defined before training round
        to achieve weighted federated average using the prewritten code"""
        #!TODO: I think because of the ClientProxy something is not good here
        raise NotImplementedError("Unrecognized weighting")
    
    def post_calculate_weights(self, results, server_round):
        """Override num_examples with weights defined based on training results
        to achieve weighted federated average using the prewritten code"""
        metric_list = [res.metrics for _, res in results]
        # Convert results
        numpy_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        #import pdb
        #pdb.set_trace()
        if self.conf["server_opt"]["weight_clients"] == "server_post_loss":
            losses = [res.metrics["loss"] for _, res in results]
            client_weights = losses
        elif self.conf["server_opt"]["weight_clients"] == "server_post_IDA" or self.conf["server_opt"]["weight_clients"] == "server_post_IDA_softmax":
            # https://arxiv.org/pdf/2008.07665
            w_flats = [w[-1] for w,_ in numpy_results]
            # w_flats = [np.concatenate([l.flatten() for l in w]) for w,_ in numpy_results]
            w_avg = np.average(w_flats)
            l1_norms = [np.linalg.norm(w-w_avg) for w in w_flats]
            l1_sum = sum(l1_norms)
            client_weights = [l1/l1_sum for l1 in l1_norms]
            if self.conf["server_opt"]["weight_clients"] == "server_post_IDA_softmax":
                client_weights = np.array(client_weights)
                client_weights = np.exp(client_weights)/sum(np.exp(client_weights))
        elif self.conf["server_opt"]["weight_clients"] == "server_post_groupweights" or self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_softmax":
            # Weighting with the known groups in mind
            metric_list = [res.metrics for _, res in results]
            client_weights = []
            for m in metric_list:
                group_keys = [k for k in m.keys() if k.startswith("groupsize_")]
                w = np.sum([self.shared_copt_params[k]/m[k] for k in group_keys if m[k]>0])
                client_weights.append(w)
            if self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_softmax":
                client_weights = np.array(client_weights)
                client_weights = np.exp(client_weights)/sum(np.exp(client_weights))
        elif self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_IDA" or self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_IDA_softmax":
            # Previous 2 combined
            w_flats = [w[-1] for w,_ in numpy_results]
            w_avg = np.average(w_flats)
            l1_norms = [np.linalg.norm(w-w_avg) for w in w_flats]
            l1_sum = sum(l1_norms)
            client_weights1 = [l1/l1_sum for l1 in l1_norms]

            metric_list = [res.metrics for _, res in results]
            client_weights2 = []
            for m in metric_list:
                group_keys = [k for k in m.keys() if k.startswith("groupsize_")]
                w = np.sum([self.shared_copt_params[k]/m[k] for k in group_keys if m[k]>0])
                client_weights2.append(w)
            client_weights = [cw1*cw2 for cw1, cw2 in zip(client_weights1, client_weights2)]
            if self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_IDA_softmax":
                temperature = (server_round)/self.conf["rounds"]*2
                temperature1 = 0+temperature
                temperature2 = 2-temperature
                client_weights1 = np.array(client_weights1)
                client_weights2 = np.array(client_weights2)
                cw1sum = sum(np.exp(client_weights1/temperature1))
                cw2sum = sum(np.exp(client_weights2/temperature2))
                client_weights = [np.exp(cw1/temperature1)/cw1sum+np.exp(cw2/temperature2)/cw2sum for cw1, cw2 in zip(client_weights1, client_weights2)]
        elif self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets"):
            log(DEBUG, "client metrics %s", str([res.metrics for _, res in results]))
            if self.conf["server_opt"]["weight_clients"] == "server_post_triplets_importanceclusters":
                
                clusters = {"SC":[],"CI":[],"AI":[]}
                for _, res in results:
                    keys_to_check = ["SC", "CI", "AI"]
                    max_key = max(keys_to_check, key=res.metrics.get)
                    clusters[max_key].append(res.metrics["cid"])
                # Sample one element from each list if the list is not empty
                weighted_clients = {key: random.sample(value, 1) if value else [] for key, value in clusters.items()}
                weighted_clients = [elem for sublist in weighted_clients.values() for elem in sublist]
                client_weights = [1 if res.metrics["cid"] in weighted_clients else 0 for _, res in results]
                
            if self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix"):
                M = [res.metrics for _, res in results]
                M = np.array([[a['SC'],a['AI'],a['CI']] for a in M])
                
                if self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix_replacement"):
                    column_sums = M.sum(axis=0)  # Sum of each column
                    M_norm = M / column_sums 
                    sampled_row_ids = np.apply_along_axis(lambda col: np.random.choice(len(col), p=col), axis=0, arr=M_norm)
                    counts = np.bincount(sampled_row_ids, minlength=len(results))
                    binary_arr = (counts > 0).astype(int)
                    client_weights = counts
                if self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix_kmeans"):
                    if "num_active_clients" in self.conf["server_opt"]:
                        active_clients = self.conf["server_opt"]["num_active_clients"]
                    else:
                        active_clients = 3
                    column_sums = M.sum(axis=0)  # Sum of each column
                    M_norm = M / column_sums 
                    _, sampled_row_ids = kmeans_plusplus(M_norm, active_clients)
                    counts = np.bincount(sampled_row_ids, minlength=len(results))
                    binary_arr = (counts > 0).astype(int)
                    client_weights = counts
                if self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix_noreplacement"):
                    if "num_active_clients" in self.conf["server_opt"]:
                        active_clients = self.conf["server_opt"]["num_active_clients"]
                    else:
                        active_clients = 3
                    M = M.T
                    M_original = copy.deepcopy(M)
                    selected_clients = []
                    num_clients = len(results)
                    while len(selected_clients) - active_clients < 0:
                        M /= M.sum(axis=1, keepdims=True)

                        index = np.random.choice(num_clients, p=M[0])
                        selected_clients.append(index)

                        with np.errstate(divide='ignore', invalid='ignore'):
                            M /= np.linalg.norm(M, axis=0)
                            M = np.nan_to_num(M, nan=0.0)
                        client1 = np.copy(M[:, index])

                        dot_products = np.dot(M.T, client1)
                        for i in selected_clients:
                            dot_products[i] = 1.0
                        min_dot_product_index = np.argmin(dot_products)

                        selected_clients.append(min_dot_product_index)
                        client2 = np.copy(M[:, min_dot_product_index])

                        M[:, index] = np.zeros(3)
                        M[:, min_dot_product_index] = np.zeros(3)

                        orthogonal_vector = np.cross(client1, client2)
                        orthonormal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
                        dot_products = np.dot(M.T, orthonormal_vector)
                        for i in selected_clients:
                            dot_products[i] = -1.0
                        max_dot_product_index = np.argmax(dot_products)
                        selected_clients.append(max_dot_product_index)
                        M[:, max_dot_product_index] = np.zeros(3)
                    counts = np.bincount(selected_clients, minlength=num_clients)
                    client_weights = counts
                    binary_arr = (counts > 0).astype(int)
                if "smoothing" in self.conf["server_opt"]["weight_clients"]:
                    smoothing_value = 0.0001
                    if "weight_smoothing" in self.conf["server_opt"]:
                        smoothing_value = self.conf["server_opt"]["weight_smoothing"]
                    smoothed_labels = counts - binary_arr + binary_arr * (1 - smoothing_value) + (1 - binary_arr) * smoothing_value
                    client_weights = smoothed_labels
            else:
                raise NotImplementedError("method not implemented:", self.conf["server_opt"]["weight_clients"])
            for i in range(len(results)):
                    results[i][1].num_examples = client_weights[i]  # FitRes of the i-th client
            return results
        else:
            raise NotImplementedError("Client weights not set!", self.conf["server_opt"]["weight_clients"])
        cw_sum = np.sum(client_weights)
        client_weights = [cw/cw_sum*self.conf['len_total_data'] for cw in client_weights]
        for i in range(len(results)):
            results[i][1].num_examples = client_weights[i]  # FitRes of the i-th client
        return results
    
    def aggregate_client_opt_params(self, metric_list):
        """Update shared client optimizer parameters for subpopbench optimizers"""
        self.shared_copt_params = subpop_federated.aggregate_metrics(self.shared_copt_params, metric_list)
