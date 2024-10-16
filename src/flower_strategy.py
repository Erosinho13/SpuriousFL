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
from src.optimizers.weighting_strategy import apply_smoothing, apply_softmax, client_weights_IDA, client_weights_known_groups, select_noreplacement, temperature_weighted_values, upscale
from src.utils import log
from src.models import model_utils
import json


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
        self.stored_client_data = {}
        self.client_update_requested = []

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
        # Update stored client info with new data
        self.update_stored_client_info(results)
        # Get stored client info to look like client sent it
        results = self.update_results_from_cache(results)

        # Calculate client weights with post-training methods
        if self.conf["server_opt"]["weight_clients"].startswith("server_post_"):
            results = self.post_calculate_weights(results, server_round=server_round)

        self.log_client_weights(results)
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
        if rnd == self.conf["server_opt"]["rounds"]:
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
        self.client_update_requested = []

        for client in clients:
            client_config = copy.deepcopy(self.shared_copt_params)     # {}
            client_config["round"] = server_round

            if "pretrain_rounds" in self.conf["server_opt"].keys() and server_round<=self.conf["server_opt"]["pretrain_rounds"]:
                    client_config["update_info"] = False
            else:
                if int(client.cid) not in self.stored_client_data.keys():
                    client_config["update_info"] = True
                    self.client_update_requested.append(int(client.cid))
                else:
                    client_config["update_info"] = False

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
        
        #import pdb
        #pdb.set_trace()
        if "pretrain_rounds" in self.conf["server_opt"].keys():
            if server_round<=self.conf["server_opt"]["pretrain_rounds"]:
                for i in range(len(results)):
                    results[i][1].num_examples = 1  # FitRes of the i-th client
                return results
        if self.conf["server_opt"]["weight_clients"] == "server_post_loss":
            losses = [res.metrics["loss"] for _, res in results]
            client_weights = losses
            client_weights = upscale(client_weights, self.conf['len_total_data'])
        elif self.conf["server_opt"]["weight_clients"] == "server_post_IDA" or self.conf["server_opt"]["weight_clients"] == "server_post_IDA_softmax":
            # https://arxiv.org/pdf/2008.07665
            client_weights = client_weights_IDA(results)
            if self.conf["server_opt"]["weight_clients"] == "server_post_IDA_softmax":
                client_weights = apply_softmax(client_weights)
            client_weights = upscale(client_weights, self.conf['len_total_data'])
        elif self.conf["server_opt"]["weight_clients"] == "server_post_groupweights" or self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_softmax":
            # Weighting with the known groups in mind
            client_weights = client_weights_known_groups(metric_list, self.conf, self.shared_copt_params)
            if self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_softmax":
                client_weights = apply_softmax(client_weights)
        elif self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_IDA" or self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_IDA_softmax":
            # Previous 2 combined
            client_weights1 = client_weights_IDA(results)
            client_weights2 = client_weights_known_groups(metric_list, self.conf, self.shared_copt_params)
            client_weights2 = apply_softmax(client_weights2)

            client_weights = [cw1*cw2 for cw1, cw2 in zip(client_weights1, client_weights2)]
            if self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_IDA_softmax":
                temperature = (server_round)/self.conf["server_opt"]["rounds"]*2
                client_weights = temperature_weighted_values(client_weights1, client_weights2, temperature)
            client_weights = upscale(client_weights, self.conf['len_total_data'])

        elif self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets"):
            # log(DEBUG, "client metrics %s", str([res.metrics for _, res in results]))
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
                if "num_active_clients" in self.conf["server_opt"]:
                    active_clients = self.conf["server_opt"]["num_active_clients"]
                else:
                    active_clients = 3
                M = [res.metrics for _, res in results]
                M = np.array([[a['SC'],a['AI'],a['CI']] for a in M])
                column_sums = M.sum(axis=0)  # Sum of each column
                M_norm = M / column_sums 
                
                if self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix_replacement"):
                    sampled_row_ids = np.apply_along_axis(lambda col: np.random.choice(len(col), p=col), axis=0, arr=M_norm)
                    client_weights = np.bincount(sampled_row_ids, minlength=len(results))

                elif self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix_kmeans"):
                    _, sampled_row_ids = kmeans_plusplus(M_norm, active_clients)
                    client_weights = np.bincount(sampled_row_ids, minlength=len(results))

                elif self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets_stochasticmatrix_noreplacement"):
                    M_original = copy.deepcopy(M.T)
                    selected_clients = select_noreplacement(M_original, active_clients)
                    num_clients = len(results)
                    client_weights = np.bincount(selected_clients, minlength=num_clients)


                if "smoothing" in self.conf["server_opt"]["weight_clients"]:
                    smoothing_value = 0.0001
                    if "weight_smoothing" in self.conf["server_opt"]:
                        smoothing_value = self.conf["server_opt"]["weight_smoothing"]
                    client_weights = apply_smoothing(client_weights, smoothing_value)
            else:
                raise NotImplementedError("method not implemented:", self.conf["server_opt"]["weight_clients"])
        else:
            raise NotImplementedError("Client weights not set!", self.conf["server_opt"]["weight_clients"])
        
        for i in range(len(results)):
            results[i][1].num_examples = client_weights[i]  # FitRes of the i-th client
        return results
    
    def aggregate_client_opt_params(self, metric_list):
        """Update shared client optimizer parameters for subpopbench optimizers"""
        self.shared_copt_params = subpop_federated.aggregate_metrics(self.shared_copt_params, metric_list)


    def update_results_from_cache(self, results):
        """Add stored data for client from server cache if available"""
        for _, res in results:
            cid = res.metrics["cid"]
            if cid in self.stored_client_data.keys():
                for k,v in self.stored_client_data[cid].items():
                    res.metrics[k] = v
        return results
    
    def update_stored_client_info(self, results):
        """Save data from clients so they don't have to compute again"""
        metrics = [res.metrics for _, res in results]
        for client_metric in metrics:
            if client_metric["cid"] in self.client_update_requested:
                store_dict = copy.deepcopy(client_metric)
                del store_dict['cid']
                del store_dict['loss']
                self.stored_client_data[client_metric["cid"]] = store_dict
        if len(self.client_update_requested)>0:
            log(DEBUG, "Updated client info %s", str(self.stored_client_data))
            self.log_client_info()
    
    def log_client_weights(self, results=None):
        """Log client weights into a csv file"""
        save_path = os.path.join(
                "checkpoints",
                self.conf["exp_id"],
                "client_weights.csv"
            )
        num_clients = self.conf["dataset_options"]["num_clients"]
        client_num_examples = [None] * num_clients
        for _, res in results:
            cid = int(res.metrics["cid"])  # Get the client id and convert to integer
            client_num_examples[cid] = res.num_examples
        
        with open(save_path,'a') as file:
            file.write(",".join(map(str, client_num_examples)) + "\n")


    def log_client_info(self):
        """Log latest info of clients to file"""
        save_path = os.path.join(
                "checkpoints",
                self.conf["exp_id"],
                "client_info.json"
            )
        with open(save_path, 'w') as file:
            json.dump(self.stored_client_data, file)