import argparse
from datetime import datetime
import os
import copy
import numpy as np
from collections import Counter
from src.datasets.dataset_utils import ModifiedDataset, SubsetDataset, concat_subsets, count_groups
import torch
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from src.config_params import Config

from src import utils
from src.models import model_utils
from src.datasets import data_preparation
from src.datasets.data_preparation import subsample
from src.optimizers.dataloaders import WeightedDataLoader
from src.optimizers import optim_utils
from src.optimizers.subpopbench import ERM, get_subpop_optimizer, get_sample_weights, is_two_stage_optimizer


def train(conf, conf_name=None):
    conf["exp_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = utils.get_device(conf)

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)

    if 'local_training_id' in conf["dataset_options"].keys():
        if conf["dataset_options"]["local_training_id"] is not None:
            
            ds_split = data_preparation.split_data(
                    train_ds,
                    conf
                )
            if conf["dataset_options"]["local_training_id"]=="all":
                # Reconstruct one global dataset if data dropping happened
                total_length = sum([len(ds) for ds in ds_split[:conf["num_clients"]]])
                if len(train_ds)!=total_length:
                    train_ds = concat_subsets(ds_split, conf["num_clients"])
            else:
                # Feature to do local training for one client's data only
                train_ds = ds_split[conf["dataset_options"]["local_training_id"]]
    conf["len_total_data"] = len(train_ds)
    print("Dataset size: ", len(train_ds))

 
    # Loading model
    model = model_utils.init_model(conf).to(device)
    # model = mobilenet_v2(pretrained=False).to(device)
    model_utils.print_summary(model)


        

    # Train ERM for a few epoch
    print("Pre-train with ERM")
    train_loader = WeightedDataLoader(dataset=train_ds, weights=None,
                                      batch_size=conf['client_opt']['batch_size'], shuffle=True)
    

    erm_conf = copy.deepcopy(conf)
    erm_conf["client_opt"]["subpop_optimizer"] = "ERM"
    erm_conf["client_opt"]["epochs"] = conf["server_opt"]["pretrain_rounds"]
    erm_conf["client_opt"]["loss_function"] = "cross_entropy"

    history = optim_utils.fit(model, train_loader, erm_conf, verbose=1)

    model.to("cpu")

    # Train the biased classifier
    print("Train biased classifier")
    biased_model = copy.deepcopy(model).to(device)

    biased_conf = copy.deepcopy(conf)
    biased_conf["client_opt"]["subpop_optimizer"] = "CRT"
    biased_conf["client_opt"]["loss_function"] = "generalized_cross_entropy"
    biased_conf["client_opt"]["generalized_cross_entropy_q"] = conf["client_opt"]["generalized_cross_entropy_q"]
    biased_conf["client_opt"]["epochs"] = conf["client_opt"]["biased_trainer_epochs"]

    history = optim_utils.fit(biased_model, train_loader, biased_conf, verbose=1)

    # Get biased predictions
    print("Get biased predictions")
    train_loader_seq = WeightedDataLoader(dataset=train_ds, weights=None,
                                          batch_size=conf['client_opt']['batch_size'], shuffle=False)
    
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
    metadata = count_groups(train_ds)
    print(metadata["class_sizes"])
    order_by_size = sorted(range(len(metadata["class_sizes"])), key=lambda i: metadata["class_sizes"][i], reverse=True)
    for selected_class in order_by_size:
        ids_selected = [i for i,_,(y,s) in train_ds if y==selected_class]
        filtered_predictions = {k: v for k, v in biased_predictions.items() if k in ids_selected}
        value_counts = Counter(filtered_predictions.values())
        print(selected_class, value_counts)
        if len(value_counts)>1:
             break
    else:
         raise ValueError("Classifier predicted everything correctly")

    print("Train on data for class: ", selected_class)
    # Step 1: Filter the dictionary to include only keys that are in the ids_most_pop list
    # Step 2: Count the occurrences of each value in the filtered dictionary
    value_counts = Counter(filtered_predictions.values())
    # Display the result
    print("with class imbalance: ", value_counts)

    # print(ids_most_pop)
    spurious_ds = SubsetDataset(train_ds, ids_selected) # Filter for most populus class
    spurious_ds = ModifiedDataset(spurious_ds, predictions=biased_predictions, use_groups=True) # Swap label to pred
    spurious_loader = WeightedDataLoader(dataset=spurious_ds, weights=None,
                                      batch_size=conf['client_opt']['batch_size'], shuffle=True)
    print(len(ids_selected))

    #print(metadata.keys())

    metadata2 = count_groups(spurious_ds)
    print("Sanity check for new ds' group sizes:", metadata2['group_sizes'])

    spurious_conf = copy.deepcopy(conf)
    spurious_conf["dataset_options"]["num_targets"] = 2         # We can predict between 2 groups
    spurious_conf["client_opt"]["subpop_optimizer"] = "CRT"
    spurious_conf["client_opt"]["epochs"] = conf["client_opt"]["left_right_trainer_epochs"]
    spurious_conf["client_opt"]["loss_function"] = "cross_entropy"
    spurious_conf["dataset_options"]["num_groups"] = 1 # Only for the most populus class
    spurious_model = copy.deepcopy(model).to(device)
    

    optim_utils.fit(spurious_model, spurious_loader, spurious_conf, verbose=1)

    # Predict groups on orig train set

    def predict_all(indeces, images, labels, groups, predicted, outdict):
        if "predictions" not in outdict:
            outdict["predictions"] = {}
        for i, g, v in zip(indeces.to('cpu').numpy(), groups.to('cpu').numpy(), predicted.to('cpu').numpy()):
                outdict["predictions"][i] = int(g) * conf["dataset_options"]["num_targets"] + int(v)

    groups_ds = ModifiedDataset(train_ds, use_groups=True) # Swap label to pred
    group_loader_seq = WeightedDataLoader(dataset=groups_ds, weights=None,
                                          batch_size=conf['client_opt']['batch_size'], shuffle=False)
    loss, accuracy, group_accuracies, extra_dict = optim_utils.evaluate(spurious_model,
                                                                        group_loader_seq,
                                                                        spurious_conf,
                                                                        extra_eval_fn=predict_all)
    predicted_groups = extra_dict["predictions"]
    n_counter = Counter(predicted_groups.values())
    print("N matrix:", n_counter)
    # Get the range of IDs from the counter
    id_range = range(min(n_counter), max(n_counter) + 1)

    # Create a list that fills in 0 for missing keys
    counts_list = [n_counter.get(i, 0) for i in id_range]
    N = np.resize(np.array(counts_list), (conf["dataset_options"]["num_targets"], conf["dataset_options"]["num_groups"]))
    print("Predicted:")
    print(N)
    print("Expected:")
    N_true = np.resize(np.array(metadata["group_sizes"]), (conf["dataset_options"]["num_targets"], conf["dataset_options"]["num_groups"]))
    print(N_true)
    print("purity:")
    group_accuracies_matrix = np.array(utils.collect_values_to_2d_array(group_accuracies))
    for i in range(conf["dataset_options"]["num_targets"]):
         print(f"y{i}g0: correct: {int(N_true[i,0]*group_accuracies_matrix[i,0]*0.01)}, incorrect: {int(N_true[i,1]*(1-group_accuracies_matrix[i,1]*0.01))}")
         print(f"y{i}g1: correct: {int(N_true[i,1]*group_accuracies_matrix[i,1]*0.01)}, incorrect: {int(N_true[i,0]*(1-group_accuracies_matrix[i,0]*0.01))}")

    # Evaluate predicted N matrix
    print(f"Group prediction accuracy: {accuracy}%")
    print("Flipped (y,g):",group_accuracies)




cs = ConfigStore.instance()
cs.store(group="job", name="centralized_training", node=Config)

@hydra.main(config_path="conf", config_name="centralized_training", version_base=None)
def main(cfg: Config):
    hydra_cfg = HydraConfig.get()
    conf_name = hydra_cfg.job.config_name
    conf = OmegaConf.to_container(cfg, resolve=True)
    print(conf)
    train(conf, conf_name=conf_name)


if __name__ == "__main__":
    main()
