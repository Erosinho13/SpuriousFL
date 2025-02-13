from collections import Counter
import flwr as fl
from src import corr, utils
from src.datasets.dataset_utils import ModifiedDataset, SubsetDataset, count_groups
from src.optimizers.subpopbench import GeneralizedCrossEntropyLoss as GCELoss
from  src.optimizers import subpop_federated
from src.optimizers.dataloaders import InfiniteDataLoader, WeightedDataLoader
from src.optimizers.matrix_inference import biased_prediction, estimate_interaction_matrix, ground_truth_matrix, split_by_class, train_left_right, training
from src.utils import log
from logging import ERROR, INFO
import numpy as np
import os
from src.models import model_utils
from src.datasets import data_preparation
from src.optimizers import subpopbench
from src.optimizers import optim_utils
import copy
import torch

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
            shared_metrics = {"cid": self.cid, "datasize":self.train_len}
            train_ds = self.train_data
            opt = subpopbench.get_subpop_optimizer(self.model, train_ds.dataset, self.conf)
            # Update client info, eg N matrix with model params from server
            if config["update_info"]:
                shared_metrics = self.share_client_params_once(opt, shared_metrics, before_train=True)
            shared_metrics = self.share_client_params_always(opt, shared_metrics, before_train=True)
            # Update client optimizer data from server
            self.align_client_opt(opt, config)


            history = optim_utils.fit(self.model, train_ds, self.conf, opt=opt)

            if np.isnan(
                    history.history["loss"][-1]
            ):  # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")
            
            # Share client training metadata
            shared_metrics["loss"] = history.history["loss"][-1]
            if config["update_info"]:
                shared_metrics = self.share_client_params_once(opt, shared_metrics, before_train=False)
            shared_metrics = self.share_client_params_always(opt, shared_metrics, before_train=False)
            


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

    def share_client_params_once(self, opt, shared_metrics, before_train=False):
        """Pass client opt params to FL server within the shared metrics dict"""
        N = None
        if before_train:
            if "groupweights" in self.conf["server_opt"]["client_info"] or "triplets" in self.conf["server_opt"]["client_info"]:
                if "Npredicted" in self.conf["server_opt"]["client_info"]:
                    N = self.predict_n_matrix()
                else:
                    metadata = count_groups(self.train_data.dataset,
                                            num_attributes=self.conf["dataset_options"]["num_groups"],
                                            num_labels=self.conf["dataset_options"]["num_targets"])
                    group_sizes = metadata["group_sizes"]
                    N = np.resize(group_sizes, (self.conf["dataset_options"]["num_targets"],self.conf["dataset_options"]["num_groups"]))
                        
                if "groupweights" in self.conf["server_opt"]["client_info"]:
                    group_sizes = np.resize(N, (1, N.shape[0]*N.shape[1]))
                    group_sizes = list(group_sizes[0])
                    for i, v in enumerate(group_sizes):
                        shared_metrics["groupsize_"+str(i)] = int(v)
                if "triplets" in self.conf["server_opt"]["client_info"]:
                    shared_metrics["SC"] = corr.SC(N)
                    shared_metrics["AI"] = corr.AI(N)
                    shared_metrics["CI"] = corr.CI(N)
        return shared_metrics
    
    def share_client_params_always(self, opt, shared_metrics, before_train=False):
        shared_metrics = subpop_federated.store_opt_params(opt, shared_metrics, self.conf, before_train=before_train)

        if before_train:
            if "gloss" in self.conf["server_opt"]["client_info"]:
                loss, accuracy, group_acc = optim_utils.evaluate(self.model, self.train_data, self.conf, verbose=0)
                shared_metrics["gloss"] = loss
        else:
            if "oort" in self.conf["server_opt"]["client_info"]:
                oort_stats = optim_utils.oort_stat(self.model, self.train_data, self.conf)
                shared_metrics["oort"] = oort_stats["oort_util"]
            if "nova" in self.conf["server_opt"]["client_info"]:
                shared_metrics["weights"] = self.train_len
                if "num_steps" in self.conf["client_opt"] and isinstance(self.conf["client_opt"]["num_steps"], int):
                    local_steps = self.conf["client_opt"]["num_steps"]
                else:
                    local_steps = len(self.train_data) * self.conf["client_opt"]["epochs"]
                local_normalizing_vec = 0
                if self.conf["client_opt"]["momentum"]!=0:
                    local_counter = 0
                    for _ in range(local_steps):
                        local_counter = local_counter*self.conf["client_opt"]["momentum"] +1
                        local_normalizing_vec += local_counter
                        etamu = self.conf["client_opt"]["learning_rate"] * self.conf["client_opt"]["proximal_mu"]
                        if etamu != 0:
                            local_normalizing_vec *= 1 - etamu
                            local_normalizing_vec += 1
                else:
                    local_normalizing_vec = local_steps
                if self.conf["client_opt"]["proximal_mu"] != 0:
                    local_tau = local_steps * (self.train_len / self.conf["len_total_data"])
                else:
                    local_tau = local_normalizing_vec * (self.train_len / self.conf["len_total_data"])
                shared_metrics["tau"] = local_tau
                shared_metrics["local_norm"] = local_normalizing_vec
        return shared_metrics
    
    def predict_n_matrix(self):
        """Predict N matrix by training biased and spurious classifier"""
        train_ds = self.train_data.dataset
        gt_int_matrix, y_weights, _ = ground_truth_matrix(
            train_ds,
            self.conf["dataset_options"]["num_targets"],
            self.conf["dataset_options"]["num_groups"],
            self.conf["client_opt"]["batch_size"],
            verbose=0
        )

        device = utils.get_device(self.conf)
        model = copy.deepcopy(self.model).to(device)

        # ==============================================
        #log(INFO, "Training biased model")
        if self.conf["client_opt"]["biased_optimizer"] == 'ReSample':
            weights = y_weights
        else:
            weights = None
        train_loader = iter(
            InfiniteDataLoader(
                dataset=train_ds,
                weights=weights,
                batch_size=self.conf["client_opt"]["batch_size"]
            )
        )
        opt = subpopbench.get_base_optimizer(model.parameters(), self.conf)

        training(
            model,
            train_loader,
            opt,
            self.conf["client_opt"]["biased_trainer_steps"],
            GCELoss(self.conf["client_opt"]["generalized_cross_entropy_q"]),
            device,
            verbose=0
        )

        # ==============================================
        #log(INFO, "Get biased predictions")
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            shuffle=False,
            batch_size=self.conf["client_opt"]["batch_size"],
        )
        error_dataset = biased_prediction(model, train_loader, device, verbose=0)
        splits = split_by_class(error_dataset)

        # ==============================================
        #log(INFO, "Train Left-Right classifier")

        train_idx = min(splits, key=lambda k: splits[k][2])
        lr_split, weights, _ = splits[train_idx]
        lr_loader = iter(
            InfiniteDataLoader(
                lr_split,
                weights=weights,
                batch_size=self.conf["client_opt"]["batch_size"],
            )
        )
        lr_clf = torch.nn.Linear(model.classifier.in_features, 2).to(device)
        opt = subpopbench.get_base_optimizer(lr_clf.parameters(), self.conf)
        train_left_right(lr_loader, lr_clf, opt, self.conf["client_opt"]["left_right_trainer_steps"], device, verbose=0)

        # ==============================================
        #log(INFO, "Estimate interaction matrix")
        est_int_matrix = estimate_interaction_matrix(
            self.conf["dataset_options"]["num_targets"],
            self.conf["dataset_options"]["num_groups"],
            splits,
            train_idx,
            lr_clf,
            {
                "batch_size": self.conf["client_opt"]["batch_size"],
            },
            device,
            verbose=0
        )

        l = self.conf["dataset_options"]["num_targets"] * self.conf["dataset_options"]["num_groups"]
        m_true = np.resize(gt_int_matrix.cpu().numpy(),(1,l))
        m_pred = np.resize(est_int_matrix.cpu().numpy(),(1,l))
        log_msg = "True: " + str(m_true) + " Pred: " + str(m_pred)
        log(INFO, log_msg)
        return est_int_matrix.cpu().numpy()