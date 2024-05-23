import flwr as fl
from logging import ERROR, INFO
from logging import WARNING
import os
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays

from src.utils import log
from src.models import model_utils


def fit_metrics_aggregation_fn(fit_metrics):
    losses = [a[1]["loss"] for a in fit_metrics]
    return {"train_loss": np.mean(losses), "train_lossStd": np.std(losses)}


def evaluate_metrics_aggregation_fn(eval_metrics):
    eval_res = {
        "loss": sum([e[1]["loss"] for e in eval_metrics]),
        "accuracy": sum([e[1]["accuracy"] for e in eval_metrics]),
    }
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
