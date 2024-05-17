import flwr as fl
from logging import ERROR, INFO
from logging import WARNING
import os
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from src.utils import log
from src.models import model_utils


def fit_metrics_aggregation_fn(fit_metrics):
    losses = [a[1]["loss"] for a in fit_metrics]
    return {"train_loss":np.mean(losses), "train_lossStd":np.std(losses)}

def evaluate_metrics_aggregation_fn(eval_metrics):
    eval_res = {
        "loss": sum([e[1]["loss"] for e in eval_metrics]),
        "accuracy": sum([e[1]["accuracy"] for e in eval_metrics]),
    }
    return eval_res


class MyStrategy(fl.server.strategy.FedAvg):
    """Custom strategy"""

    def __init__(self, conf, *args, **kwargs):
        self.conf = conf
        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
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

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
 
        weights_aggregated = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)


        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            self.train_metrics_aggregated = metrics_aggregated
            log(INFO, "aggregated fit results %s", str(metrics_aggregated))
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        self.aggregated_parameters = (
            parameters_aggregated  # Why can't I access this at eval?
        )

        return parameters_aggregated, metrics_aggregated


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
                "./dump/",
                "latest_model"
            )
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters
            )
            model = model_utils.init_model(
                conf=self.conf, weights=aggregated_weights
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

        return aggregated_result