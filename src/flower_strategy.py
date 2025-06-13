import flwr as fl
from logging import ERROR, INFO, DEBUG
from logging import WARNING
import os
import numpy as np
import copy
import random
import operator
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
from src.models.afed_generator import AFedGenerator
from src.optimizers import subpop_federated 
from src.optimizers.optim_utils import train_generator
from src.optimizers.weighting_strategy import apply_smoothing, apply_softmax, client_weights_IDA, client_weights_known_groups, client_weights_nova, flatten_weights, get_relation, node_deleting, probabilistic_selection, select_noreplacement, temperature_weighted_values, top_k_binary_list, upscale
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

    def __init__(self, conf, initial_parameters=None, *args, **kwargs):
        self.conf = conf
        eta = self.conf["server_opt"]["learning_rate"]
        eta_l = self.conf["client_opt"]["learning_rate"]
        beta_1 = self.conf["server_opt"]["beta_1"]
        beta_2 = self.conf["server_opt"]["beta_2"]
        tau = self.conf["server_opt"]["tau"]
        if initial_parameters is not None:
            self.current_weights: NDArrays = parameters_to_ndarrays(
                initial_parameters
            )
            if self.conf["server_opt"]["optimizer"] in ["FedAvgM", "FedAdam", "FedNova"]:
                self.m_t = [np.zeros_like(x) for x in self.current_weights]  # Momentum vector
            if self.conf["server_opt"]["optimizer"] in ["FedAdam"]:
                self.v_t = [np.zeros_like(x) for x in self.current_weights]
        self.stored_client_data = {}
        if "fedpns" in self.conf["server_opt"]["weight_clients"] or "fedpns" in self.conf["server_opt"]["selection_method"]:
            self.fedpnslog={i:[0,0] for i in range(self.conf["dataset_options"]["num_clients"])}
            for i in range(self.conf["dataset_options"]["num_clients"]):
                self.stored_client_data[i] = {"update_needed":True}
                self.stored_client_data[i]["fedpns_p"] = 1/self.conf["dataset_options"]["num_clients"]
        if self.conf["server_opt"]["afed_generator"]:
            self.afed_generator, self.generator_optimizer, self.generator_lr_scheduler = AFed_generator_init(self.conf)
        self.shared_copt_params = subpop_federated.init_shared_opt_params(self.conf)
        self.client_update_requested = []
        self.round_interaction_matrix = {} # Stores relative interaction matrix each round

        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         initial_parameters=initial_parameters,
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
        self.update_stored_client_info(results, server_round)
        # Get stored client info to look like client sent it
        results = self.update_results_from_cache(results)
        # Calculate round interaction matrix (data from cache)
        self.track_group_dist(results)

        # Calculate client weights with post-training methods
        if self.conf["server_opt"]["weight_clients"].startswith("server_post_"):
            results = self.post_calculate_weights(results, server_round=server_round)

        # AFed generator training https://arxiv.org/pdf/2501.02732
        if self.conf["server_opt"]["afed_generator"]:
            #!TODO: train generator
            local_model_lst = [w for w, res in results]
            train_generator(self.afed_generator, self.generator_optimizer, self.generator_lr_scheduler, local_model_lst, self.conf)

            model_path = os.path.join(
            "checkpoints",
            self.conf["exp_id"],
            "afed_generator"
            )
            model_utils.save_model(self.afed_generator, model_path)

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

        if self.conf["server_opt"]["optimizer"] in ["FedAvgM", "FedNova"]:
            # Following https://flower.ai/docs/framework/_modules/flwr/server/strategy/fedavgm.html#FedAvgM
            # Following: https://github.com/adap/flower/blob/main/baselines/fednova/fednova/strategy.py
            # Pseudo gradients
            delta_t: NDArrays = [
                x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
            ]
            # m_t
            if not self.m_t:
                self.m_t = [np.zeros_like(x) for x in delta_t]
            # Applying Nesterov
            self.m_t = [
                np.multiply(self.beta_1, x) + y
                for x, y in zip(self.m_t, delta_t)
            ]
            # Federated Averaging with Server Momentum
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
            wandb_log.update(self.round_interaction_matrix)
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
        if "pretrain_rounds" in self.conf["server_opt"].keys() and server_round>self.conf["server_opt"]["pretrain_rounds"]:
            if self.conf["server_opt"]["participation"] == "selection":
                if "num_active_clients" in self.conf["server_opt"] and isinstance(self.conf["server_opt"]["num_active_clients"],int):
                    active_clients = self.conf["server_opt"]["num_active_clients"]
                else:
                    active_clients = 3
                sample_size = active_clients
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, client_info=self.stored_client_data, server_round=server_round
        )

        # Create custom configs
        fit_configurations = []
        self.client_update_requested = []

        for client in clients:
            client_config = copy.deepcopy(self.shared_copt_params)     # {}
            client_config["round"] = server_round

            if "pretrain_rounds" in self.conf["server_opt"].keys() and server_round<self.conf["server_opt"]["pretrain_rounds"]:
                client_config["update_info"] = False
            else:
                if int(client.cid) not in self.stored_client_data.keys():
                    client_config["update_info"] = True
                    self.client_update_requested.append(int(client.cid))
                else:
                    if self.stored_client_data[int(client.cid)]["update_needed"]:
                        client_config["update_info"] = True
                        self.client_update_requested.append(int(client.cid))
                    else:
                        if (
                            self.conf["server_opt"]["update_static_info_rounds"] != 0
                        ) and (
                            server_round
                            % self.conf["server_opt"]["update_static_info_rounds"]
                            == 1
                        ):
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

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, evaluate=True
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def pre_calculate_weights(self, client):
        """Override num_examples with weights defined before training round
        to achieve weighted federated average using the prewritten code"""
        #!TODO: I think because of the ClientProxy something is not good here
        raise NotImplementedError("Unrecognized weighting")

    def post_calculate_weights(self, results, server_round):
        """Override num_examples with weights defined based on training results
        to achieve weighted federated average using the prewritten code"""
        metric_list = [res.metrics for _, res in results]
        if "num_active_clients" in self.conf["server_opt"] and isinstance(self.conf["server_opt"]["num_active_clients"], int):
            active_clients = self.conf["server_opt"]["num_active_clients"]
        else:
            active_clients = 3
        smoothing_value = 0.0001
        if "weight_smoothing" in self.conf["server_opt"]:
            smoothing_value = self.conf["server_opt"]["weight_smoothing"]
        # import pdb
        # pdb.set_trace()

        if "pretrain_rounds" in self.conf["server_opt"].keys():
            if server_round<=self.conf["server_opt"]["pretrain_rounds"]:
                for i in range(len(results)):
                    results[i][1].num_examples = 1  # FitRes of the i-th client
                return results
        if self.conf["server_opt"]["weight_clients"] == "server_post_random":
            client_weights = [1]*len(results)
            elements = list(range(len(results)))
            sampled_ids = np.random.choice(elements, size=active_clients, p=client_weights, replace=False)
            client_weights = [np.sum(sampled_ids == element) for element in elements]
        if self.conf["server_opt"]["weight_clients"] == "server_post_loss":
            losses = [res.metrics["loss"] for _, res in results]
            client_weights = losses
            client_weights = upscale(client_weights, self.conf['len_total_data'])
        elif self.conf["server_opt"]["weight_clients"] == "server_post_nova":
            # https://flower.ai/docs/baselines/fednova.html
            client_weights = client_weights_nova(results)
        elif self.conf["server_opt"]["weight_clients"] == "server_post_IDA" or self.conf["server_opt"]["weight_clients"] == "server_post_IDA_softmax":
            # https://arxiv.org/pdf/2008.07665
            client_weights = client_weights_IDA(results)
            if self.conf["server_opt"]["weight_clients"] == "server_post_IDA_softmax":
                client_weights = apply_softmax(client_weights)
            client_weights = upscale(client_weights, self.conf['len_total_data'])
        elif self.conf["server_opt"]["weight_clients"].startswith("server_post_groupweights"):
            # Weighting with the known groups in mind
            client_weights = client_weights_known_groups(metric_list, self.conf)
            if self.conf["server_opt"]["weight_clients"] == "server_post_groupweights":
                client_weights = np.array(client_weights)/min(client_weights)
                client_weights = [int(w) for w in client_weights]
            if self.conf["server_opt"]["weight_clients"] == "server_post_groupweights_softmax":
                pass
            if self.conf["server_opt"]["weight_clients"].startswith("server_post_groupweights_choice"):
                client_weights = np.array(client_weights)/sum(client_weights)
                elements = list(range(len(results)))
                if "noreplacement" in self.conf["server_opt"]["weight_clients"]:
                    sampled_ids = np.random.choice(elements, size=active_clients, p=client_weights, replace=False)
                else:
                    sampled_ids = np.random.choice(elements, size=active_clients, p=client_weights, replace=True)
                client_weights = [np.sum(sampled_ids == element) for element in elements]
                if "smoothing" in self.conf["server_opt"]["weight_clients"]:
                    client_weights = apply_smoothing(client_weights, smoothing_value)

        elif self.conf["server_opt"]["weight_clients"].startswith("server_post_powd"):
            # https://proceedings.mlr.press/v151/jee-cho22a/jee-cho22a.pdf
            glosses = [res.metrics["gloss"] for _, res in results]
            client_weights = top_k_binary_list(glosses, active_clients)
        elif self.conf["server_opt"]["weight_clients"].startswith("server_post_fedpns"):
            cids = [res.metrics["cid"] for _,res in results]
            client_weights = [self.stored_client_data[cid]["fedpns_p"] for cid in cids]
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

    def update_stored_client_info(self, results, server_round):
        """Save data from clients so they don't have to compute again"""
        update_something = False
        # Calculate FedPNS scores
        if "fedpns" in self.conf["server_opt"]["weight_clients"] or "fedpns" in self.conf["server_opt"]["selection_method"]:
            self.calculate_pns_scores(results)

        metrics = [res.metrics for _, res in results]
        for client_metric in metrics:
            if client_metric["cid"] not in self.stored_client_data.keys():
                self.stored_client_data[client_metric["cid"]] = {"update_needed":True}
            self.stored_client_data[client_metric["cid"]]["last_round"] = server_round
            always_update_list = ["gloss", "weights", "tau", "local_norm", "oort", "netemb"]
            for k in always_update_list:
                if k in client_metric.keys():
                    update_something = True
                    self.stored_client_data[client_metric["cid"]][k] = client_metric[k]
                else:
                    for kk in client_metric.keys():
                        if kk.startswith(k):
                            update_something = True
                            self.stored_client_data[client_metric["cid"]][kk] = client_metric[kk]
            if client_metric["cid"] in self.client_update_requested:
                update_something = True
                self.stored_client_data[client_metric["cid"]]["update_needed"] = False
                store_dict = copy.deepcopy(client_metric)
                del store_dict['cid']
                del store_dict['loss']
                for k in always_update_list:
                    if k in store_dict.keys():
                        del store_dict[k]
                for k,v in store_dict.items():
                    self.stored_client_data[client_metric["cid"]][k] = v
        if update_something:
            log(DEBUG, "Updated client info %s", str(self.stored_client_data))
            self.log_client_info(server_round)

    def log_client_weights(self, results=None):
        """Log client weights into a csv file"""
        save_path = os.path.join(
                "checkpoints",
                self.conf["exp_id"],
                "client_weights.csv"
            )
        num_clients = self.conf["dataset_options"]["num_clients"]
        client_num_examples = [0] * num_clients
        for _, res in results:
            cid = int(res.metrics["cid"])  # Get the client id and convert to integer
            client_num_examples[cid] = res.num_examples

        with open(save_path,'a') as file:
            file.write(",".join(map(str, client_num_examples)) + "\n")

    def log_client_info(self,server_round):
        """Log latest info of clients to file"""
        if ((
            self.conf["server_opt"]["update_static_info_rounds"] != 0
        ) and (
            server_round
            % self.conf["server_opt"]["update_static_info_rounds"]
            == 1
        )) or server_round==1:
            save_path = os.path.join(
                "checkpoints",
                self.conf["exp_id"],
                f"client_info-{server_round}.json"
            )
            with open(save_path, 'w') as file:
                json.dump(self.stored_client_data, file)
        save_path = os.path.join(
                "checkpoints",
                self.conf["exp_id"],
                "client_info.json"
            )
        with open(save_path, 'w') as file:
            json.dump(self.stored_client_data, file)

    def calculate_pns_scores(self, results):
        """Based on self.current_weights and weights in results, calculates pi probabilities for FedPNS
        following https://arxiv.org/pdf/2105.07066
        Updates self.stored_client_data[cid]['fedpns_p'] for each client"""
        client_weights = [parameters_to_ndarrays(res.parameters) for _,res in results]
        idxs_users = [res.metrics["cid"] for _,res in results]
        for idx in idxs_users:
            self.fedpnslog[idx][0]+=1
        client_grads = [[a - b for a, b in zip(w, self.current_weights)] for w in client_weights]
        client_grads_flatten = [flatten_weights(w) for w in client_grads]
        grads_dict = {k:v for k,v in zip(idxs_users, client_grads_flatten)}
        avg_grad = np.mean(list(grads_dict.values()), axis=0)

        max_now = get_relation(grads_dict, avg_grad, idxs_users, conf=self.conf)
        expect_list = {}

        if isinstance(self.conf["server_opt"]["num_active_clients"], int):
            active_clients = self.conf["server_opt"]["num_active_clients"]
        else:
            active_clients = 0
        if active_clients == 0:
            active_clients = len(grads_dict)

        removed_ids = []
        while len(grads_dict)>0:
            expect_list = node_deleting(expect_list, max_now, idxs_users, grads_dict, conf=self.conf)
            print(expect_list)
            key = max(expect_list.items(), key=operator.itemgetter(1))[0]
            print("Key:", key)
            if expect_list[key]<=expect_list["all"]:
                break # Every client pulls towards the right direction
            else:
                self.fedpnslog[key][1] += 1
                expect_list.pop("all")
                loss_all, loss_pop = 1.0, 0.0 # Our server has no validation data
                if loss_all<loss_pop:
                    break
                else:
                    grads_dict.pop(key)
                    max_now=expect_list[key]
                    expect_list.pop(key)
                    idxs_users.remove(key)
                    removed_ids.append(key)
        node_prob = {k:v["fedpns_p"] for k,v in self.stored_client_data.items()}
        node_prob = probabilistic_selection(node_prob, self.fedpnslog, removed_ids,
                                            alpha=self.conf["server_opt"]["fedpns_alpha"],
                                            beta=self.conf["server_opt"]["fedpns_beta"])
        for k,v in node_prob.items():
            self.stored_client_data[k]["fedpns_p"] = v

        print("FedPNS_P:",node_prob)

    def track_group_dist(self, results):
        """Calculate client weights if we know the N matrix of all clients"""
        def getindex(k, conf):
            i = int(k.split('_')[-1])
            y = i//conf["dataset_options"]["num_groups"]
            g = i%conf["dataset_options"]["num_groups"]
            return f"interaction_matrix_y{y}g{g}"
        metric_list = [res.metrics for _, res in results]
        n_matrix = {}
        total = 0
        for m in metric_list:
            group_keys = [k for k in m.keys() if k.startswith("interaction_matrix_")]
            n_dict = {getindex(k, self.conf):v for k,v in m.items() if k in group_keys}
            for k in n_dict.keys():
                if k not in n_matrix.keys():
                    n_matrix[k] = 0
                n_matrix[k] += n_dict[k]
                total += n_dict[k]
        for k,v in n_matrix.items():
            n_matrix[k] = v/total
        self.round_interaction_matrix = n_matrix
        print("Round Interaction matrix:", n_matrix)
        