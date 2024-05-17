import flwr as fl
from src.utils import log
from logging import ERROR, INFO
import numpy as np
import os
from src.models import model_utils
from src.datasets import augmentation, data_preparation


class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""

    def __init__(self, cid, conf):
        self.cid = cid
        self.conf = conf

    def init_model(self):
        model = model_utils.init_model(
            conf=self.conf,
        )
        self.model = model

    def load_data(self, X, Y, X_test, Y_test):
        self.train_len = len(X)
        self.test_len = len(X_test)
        self.train_data = (X, Y)
        self.test_data = (X_test, Y_test)
        self.len_train_data = len(X)

    def get_parameters(self, config):
        return model_utils.get_weights(self.model)

    def set_parameters(self, weights, config):
            model_utils.set_weights(self.model, weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        # return self.get_parameters(config), 1, {"client_id": self.cid, "loss":-1}
        try:
            self.set_parameters(weights, config)
            

            train_ds = data_preparation.get_ds_from_np(self.train_data)
            if self.conf["aug"]:
                train_ds = augmentation.aug_data(train_ds, conf=self.conf)
            train_ds = data_preparation.preprocess_data(train_ds, conf=self.conf, shuffle=True)

            history = model_utils.fit(self.model, train_ds, self.conf)

            if np.isnan(
                history.history["loss"][-1]
            ):  # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")

            shared_metrics = {"client_id": self.cid, "loss": history.history["loss"]}

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
            
            test_ds = data_preparation.get_ds_from_np(self.test_data)
            test_ds = data_preparation.preprocess_data(test_ds, self.conf)
            # Local model eval
            self.set_parameters(weights, config)
            loss, accuracy = model_utils.evaluate(self.model, test_ds, self.conf, verbose=0)
            
            return (
                loss,
                self.test_len,
                {"cid": self.cid,
                 "loss":loss, 
                 "accuracy": accuracy},
            )
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise RuntimeError("Client evaluate terminated unexpectedly")
