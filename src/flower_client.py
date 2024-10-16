from collections import Counter
import flwr as fl
from src.datasets.dataset_utils import ModifiedDataset, SubsetDataset, count_groups
from  src.optimizers import subpop_federated
from src.optimizers.dataloaders import WeightedDataLoader
from src.utils import log
from logging import ERROR, INFO
import numpy as np
import os
from src.models import model_utils
from src.datasets import data_preparation
from src.optimizers import subpopbench
from src.optimizers import optim_utils
import copy

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
            shared_metrics = {"cid": self.cid}
            train_ds = self.train_data
            opt = subpopbench.get_subpop_optimizer(self.model, train_ds.dataset, self.conf)
            # Update client info, eg N matrix with model params from server
            if config["update_info"]:
                shared_metrics = self.share_client_opt_params(opt, shared_metrics, before_train=True)
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
                shared_metrics = self.share_client_opt_params(opt, shared_metrics, before_train=False)

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

    def share_client_opt_params(self, opt, shared_metrics, before_train=False):
        """Pass client opt params to FL server within the shared metrics dict"""
        N = None
        if before_train:
            if self.conf["server_opt"]["weight_clients"].startswith("server_post_groupweights") or self.conf["server_opt"]["weight_clients"].startswith("server_post_triplets"):
                if "Npredicted" in self.conf["server_opt"]["weight_clients"]:
                    N = self.predict_n_matrix()
                else:
                    metadata = count_groups(self.train_data.dataset,
                                            num_attributes=self.conf["dataset_options"]["num_groups"],
                                            num_labels=self.conf["dataset_options"]["num_targets"])
                    group_sizes = metadata["group_sizes"]
                    N = np.resize(group_sizes, (self.conf["dataset_options"]["num_targets"],self.conf["dataset_options"]["num_groups"]))
        shared_metrics = subpop_federated.store_opt_params(opt, shared_metrics, self.conf, n_matrix=N, before_train=before_train)
        return shared_metrics
    
    def predict_n_matrix(self):
        """Predict N matrix by training biased and spurious classifier"""
        num_targets = self.conf["dataset_options"]["num_targets"]
        num_groups = self.conf["dataset_options"]["num_groups"]
        N = np.ones((num_targets, num_groups))

        # Train the biased classifier
        print("Train biased classifier")
        biased_model = copy.deepcopy(self.model)

        biased_conf = copy.deepcopy(self.conf)
        biased_conf["client_opt"]["subpop_optimizer"] = "CRT"
        biased_conf["client_opt"]["loss_function"] = "generalized_cross_entropy"
        biased_conf["client_opt"]["generalized_cross_entropy_q"] = self.conf["client_opt"]["generalized_cross_entropy_q"]
        biased_conf["client_opt"]["epochs"] = self.conf["client_opt"]["biased_trainer_epochs"]
        train_ds = self.train_data.dataset
        history = optim_utils.fit(biased_model, self.train_data, biased_conf, verbose=1)

        # Get biased predictions
        print("Get biased predictions")
        train_loader_seq = WeightedDataLoader(dataset=train_ds, weights=None,
                                            batch_size=self.conf['client_opt']['batch_size'], shuffle=False)
        
        def predict_biased(indeces, images, labels, groups, predicted, outdict):
            if "biased_predictions" not in outdict:
                outdict["biased_predictions"] = {}
            for i, v in zip(indeces.to('cpu').numpy(), (predicted != labels).to('cpu').numpy()):
                    outdict["biased_predictions"][i] = int(v)

        loss, accuracy, group_accuracies, extra_dict = optim_utils.evaluate(biased_model,
                                                                            train_loader_seq,
                                                                            biased_conf,
                                                                            extra_eval_fn=predict_biased)
        print(f"Biased prediction accuracy: {accuracy:.4f}%")
        print(group_accuracies)
        biased_predictions = extra_dict["biased_predictions"]
        biased_model.to('cpu')

        # Classify majority-minority
        
        # print(predictions)

        # Train spurious classifier
        print("Train spurious classifier")
        metadata1 = count_groups(train_ds, update_ds=False, num_attributes=num_groups, num_labels=num_targets)
        print(metadata1["class_sizes"])
        order_by_size = sorted(range(len(metadata1["class_sizes"])), key=lambda i: metadata["class_sizes"][i], reverse=True)
        for selected_class in order_by_size:
            ids_selected = [i for i,_,(y,s) in train_ds if y==selected_class]
            filtered_predictions = {k: v for k, v in biased_predictions.items() if k in ids_selected}
            value_counts = Counter(filtered_predictions.values())
            print(selected_class, value_counts)
            if len(value_counts)>1:
                break
        else:
            print("Classifier predicted everything correctly, should return with low priority matrix")
            for i, class_size in enumerate(metadata1["class_sizes"]):
                # Divide the class size evenly among the groups
                group_share = class_size // num_groups
                remainder = class_size % num_groups
                
                # Set the base share for all groups
                N[i, :] = group_share
                
                # Distribute the remainder across the first few groups
                for j in range(remainder):
                    N[i, j] += 1
            return N


        print("Train on data for class: ", selected_class)
        # Display the result
        print("with class imbalance: ", value_counts)

        # print(ids_most_pop)
        spurious_ds = SubsetDataset(train_ds, ids_selected) # Filter for most populus class
        spurious_ds = ModifiedDataset(spurious_ds, predictions=biased_predictions, use_groups=True) # Swap label to pred
        spurious_loader = WeightedDataLoader(dataset=spurious_ds, weights=None,
                                        batch_size=self.conf['client_opt']['batch_size'], shuffle=True)
        print(len(ids_selected))

        #print(metadata.keys())


        spurious_conf = copy.deepcopy(self.conf)
        assert num_groups == 2
        spurious_conf["dataset_options"]["num_targets"] = num_groups         # We can predict between 2 groups
        spurious_conf["client_opt"]["subpop_optimizer"] = "ReWeightCRT"
        spurious_conf["client_opt"]["epochs"] = self.conf["client_opt"]["left_right_trainer_epochs"]
        spurious_conf["client_opt"]["loss_function"] = "cross_entropy"
        spurious_conf["dataset_options"]["num_groups"] = num_targets # Only for the most populus class
        spurious_model = copy.deepcopy(self.model)
        
        metadata2 = count_groups(spurious_ds, update_ds=True,
                                num_attributes=num_targets,
                                num_labels=num_groups,)
        print("Sanity check for new ds' group sizes:", metadata2['group_sizes'])

        optim_utils.fit(spurious_model, spurious_loader, spurious_conf, verbose=1)

        # Predict groups on orig train set

        def predict_all(indeces, images, labels, groups, predicted, outdict):
            if "predictions" not in outdict:
                outdict["predictions"] = {}
            for i, g, v in zip(indeces.to('cpu').numpy(), groups.to('cpu').numpy(), predicted.to('cpu').numpy()):
                    outdict["predictions"][i] = int(g) * num_targets + int(v)

        groups_ds = ModifiedDataset(train_ds, use_groups=True) # Swap label to pred
        group_loader_seq = WeightedDataLoader(dataset=groups_ds, weights=None,
                                            batch_size=self.conf['client_opt']['batch_size'], shuffle=False)
        loss, accuracy, group_accuracies, extra_dict = optim_utils.evaluate(spurious_model,
                                                                            group_loader_seq,
                                                                            spurious_conf,
                                                                            extra_eval_fn=predict_all)
        predicted_groups = extra_dict["predictions"]
        n_counter = Counter(predicted_groups.values())
        # Get the range of IDs from the counter
        id_range = range(min(n_counter), max(n_counter) + 1)

        # Create a list that fills in 0 for missing keys
        counts_list = [n_counter.get(i, 0) for i in id_range]
        N = np.resize(np.array(counts_list), (num_targets, num_groups))
        print("N matrix:", N)
        
        N_true = np.resize(np.array(metadata1["group_sizes"]), (num_targets, num_groups))
        print("Expected:", N_true)
        # Evaluate predicted N matrix
        print(f"Group prediction accuracy: {accuracy}%")
        print("Flipped (y,g):",group_accuracies)
        return N