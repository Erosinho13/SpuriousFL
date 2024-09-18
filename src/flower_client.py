import flwr as fl
from  src.optimizers import subpop_federated
from src.utils import log
from logging import ERROR, INFO
import numpy as np
import os
from src.models import model_utils
from src.datasets import data_preparation
from src.optimizers import subpopbench
from src.optimizers import optim_utils

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
        self.train_len = len(train_ds.dataset)
        self.test_len = len(test_ds.dataset)
        self.train_data = train_ds
        self.test_data = test_ds

    def get_parameters(self, config):
        return model_utils.get_weights(self.model)

    def set_parameters(self, weights, config):
        model_utils.set_weights(self.model, weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        # return self.get_parameters(config), 1, {"cid": self.cid, "loss":-1}
        try:
            #print(config)
            self.set_parameters(weights, config)

            train_ds = self.train_data
            
            opt = subpopbench.get_subpop_optimizer(self.model, train_ds, self.conf)
            self.align_client_opt(opt, config)
            history = optim_utils.fit(self.model, train_ds, self.conf, opt=opt)

            if np.isnan(
                    history.history["loss"][-1]
            ):  # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")

            shared_metrics = {"cid": self.cid, "loss": history.history["loss"][-1]}
            shared_metrics = self.share_client_opt_params(opt, shared_metrics)

            if "weight_clients" in self.conf["server_opt"].keys():
                weight_mode = self.conf["server_opt"]["weight_clients"]
                if weight_mode == "same":
                    client_weight = 1
                elif weight_mode == "datasize":
                    client_weight = self.train_len
                elif weight_mode == "reversesize":
                    client_weight = int(self.conf["len_total_data"]/self.train_len)
                elif weight_mode == "fromlist":
                    weight_list = self.conf["server_opt"]["weight_list"]
                    client_weight = weight_list[int(self.cid)]
                elif weight_mode.startswith("server_pre_"):
                    client_weight = config["client_weight"]     # Calculated by server
                elif weight_mode.startswith("server_post_"):
                    client_weight = 1                           # Overriden at server
                else:
                    raise NotImplementedError("Client weight method not recognized!")
            else: 
                raise NotImplementedError("client weights not set!")
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
            loss, accuracy, group_acc = optim_utils.evaluate(self.model, test_ds, self.conf, verbose=0)
            metric_dict = {
                "cid": self.cid,
                 "test_loss": loss,
                 "test_accuracy": accuracy,
            }
            metric_dict = metric_dict | group_acc

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

    def align_client_opt(self, opt, config):
        """Update client subpopbench optimizer with FL server config params"""
        subpop_federated.update_opt_with_shared_params(opt, config)

    def share_client_opt_params(self, opt, shared_metrics):
        """Pass client opt params to FL server within the shared metrics dict"""
        
        shared_metrics = subpop_federated.store_opt_params(opt, shared_metrics, self.train_data.dataset, self.conf)
        return shared_metrics