import flwr as fl
from src.utils import log
from logging import ERROR, INFO
import numpy as np
import os
from src.models import model_utils
from src.datasets import data_preparation


class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""

    def __init__(self, cid, conf):
        self.test_data = None
        self.train_data = None
        self.test_len = None
        self.train_len = None
        self.model = None
        self.cid = cid
        self.conf = conf

    def init_model(self):
        model = model_utils.init_model(
            conf=self.conf,
        )
        self.model = model

    def load_data(self, train_ds, test_ds):
        self.train_len = len(train_ds)
        self.test_len = len(test_ds)
        self.train_data = train_ds
        self.test_data = test_ds

    def get_parameters(self, config):
        return model_utils.get_weights(self.model)

    def set_parameters(self, weights, config):
        model_utils.set_weights(self.model, weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        # return self.get_parameters(config), 1, {"client_id": self.cid, "loss":-1}
        try:
            self.set_parameters(weights, config)

            train_ds = self.train_data
            
            history = model_utils.fit(self.model, train_ds, self.conf)

            if np.isnan(
                    history.history["loss"][-1]
            ):  # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")

            shared_metrics = {"client_id": self.cid, "loss": history.history["loss"][-1]}

            client_weight = self.train_len if self.conf["weight_clients"] else 1

            trained_weights = model_utils.get_weights(self.model)

        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise RuntimeError("Client training terminated unexpectedly")
        return trained_weights, client_weight, shared_metrics

    def evaluate(self, weights, config):
        try:

            test_ds = self.test_data

            # Local model eval
            self.set_parameters(weights, config)
            loss, accuracy, group_acc, f1_score, group_f1_score = \
                model_utils.evaluate(self.model, test_ds, self.conf, verbose=0)
            metric_dict = {
                "cid": self.cid,
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_f1_score": f1_score
            }
            if self.conf["log_group_acc"]:
                metric_dict |= group_acc
            if self.conf["log_group_f1_score"]:
                metric_dict |= group_f1_score

            return (
                loss,
                self.test_len,
                metric_dict,
            )
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise RuntimeError("Client evaluate terminated unexpectedly")
