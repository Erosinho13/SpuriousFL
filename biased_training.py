import argparse
from datetime import datetime
import os
import copy
import numpy as np
from collections import Counter
from src.datasets.dataset_utils import ModifiedDataset, SubsetDataset, concat_subsets, count_groups
import torch
import wandb

from src import utils
from src.models import model_utils
from src.datasets import data_preparation
from src.datasets.data_preparation import subsample
from src.optimizers.dataloaders import WeightedDataLoader
from src.optimizers import optim_utils
from src.optimizers.subpopbench import ERM, get_subpop_optimizer, get_sample_weights, is_two_stage_optimizer


def train(conf, conf_path=None):
    conf["exp_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = utils.get_device(conf)

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)

    if 'local_training_id' in conf.keys():
        if conf["local_training_id"] is not None:
            
            ds_split = data_preparation.split_data(
                    train_ds,
                    conf
                )
            if conf["local_training_id"]=="all":
                # Reconstruct one global dataset if data dropping happened
                total_length = sum([len(ds) for ds in ds_split[:conf["num_clients"]]])
                if len(train_ds)!=total_length:
                    train_ds = concat_subsets(ds_split, conf["num_clients"])
            else:
                # Feature to do local training for one client's data only
                train_ds = ds_split[conf["local_training_id"]]
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
    erm_conf["epochs"] = 1
    erm_conf["client_opt"]["loss_function"] = "cross_entropy"

    history = optim_utils.fit(model, train_loader, erm_conf, verbose=1)

    model.to("cpu")

    # Train the biased classifier
    print("Train biased classifier")
    biased_model = copy.deepcopy(model).to(device)

    biased_conf = copy.deepcopy(conf)
    biased_conf["client_opt"]["subpop_optimizer"] = "CRT"
    biased_conf["client_opt"]["loss_function"] = "generalized_cross_entropy"
    biased_conf["client_opt"]["generalized_cross_entropy_q"] = 0.7
    biased_conf["epochs"] = 1

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
    most_populus_class = np.argmax(metadata["class_sizes"])

    ids_most_pop = [i for i,_,(y,s) in train_ds if y==most_populus_class]
    # print(ids_most_pop)
    spurious_ds = SubsetDataset(train_ds, ids_most_pop) # Filter for most populus class
    spurious_ds = ModifiedDataset(spurious_ds, predictions=biased_predictions, use_groups=True) # Swap label to pred
    spurious_loader = WeightedDataLoader(dataset=spurious_ds, weights=None,
                                      batch_size=conf['client_opt']['batch_size'], shuffle=True)
    print(len(ids_most_pop))

    #print(metadata.keys())

    print("Train on data for class: ", most_populus_class)
    # Step 1: Filter the dictionary to include only keys that are in the ids_most_pop list
    filtered_predictions = {k: v for k, v in biased_predictions.items() if k in ids_most_pop}
    # Step 2: Count the occurrences of each value in the filtered dictionary
    value_counts = Counter(filtered_predictions.values())
    # Display the result
    print("with class imbalance: ", value_counts)
    metadata = count_groups(spurious_ds)
    print("Sanity check for new ds' group sizes:", metadata['group_sizes'])

    spurious_conf = copy.deepcopy(conf)
    spurious_conf["dataset_options"]["num_targets"] = 2         # We can predict between 2 groups
    spurious_conf["client_opt"]["subpop_optimizer"] = "ReWeightCRT"
    spurious_conf["epochs"] = 1
    spurious_conf["client_opt"]["loss_function"] = "cross_entropy"
    spurious_conf["dataset_options"]["num_groups"] = 1 # Only for the most populus class
    spurious_model = copy.deepcopy(model).to(device)
    

    optim_utils.fit(spurious_model, spurious_loader, spurious_conf, verbose=1)

    # Predict groups on orig train set

    def predict_all(indeces, images, labels, groups, predicted, outdict):
        if "predictions" not in outdict:
            outdict["predictions"] = {}
        for i, v in zip(indeces.to('cpu').numpy(), predicted.to('cpu').numpy()):
                outdict["predictions"][i] = int(v)

    groups_ds = ModifiedDataset(train_ds, use_groups=True) # Swap label to pred
    group_loader_seq = WeightedDataLoader(dataset=groups_ds, weights=None,
                                          batch_size=conf['client_opt']['batch_size'], shuffle=False)
    loss, accuracy, group_accuracies, extra_dict = optim_utils.evaluate(spurious_model,
                                                                        group_loader_seq,
                                                                        spurious_conf,
                                                                        extra_eval_fn=predict_all)
    predicted_groups = extra_dict["predictions"]

    # Evaluate predicted N matrix
    print(f"Group prediction accuracy: {accuracy}%")
    print("Flipped (y,g):",group_accuracies)

def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Config path",
        default="config.yaml",
    )
    parser.add_argument(
        "--env_path",
        type=str,
        help="Environment path",
        default="env.yaml",
    )
    args = parser.parse_args()

    conf = utils.load_config(config_path=args.config_path, env_path=args.env_path)
    print(conf)
    train(conf, conf_path=args.config_path)


if __name__ == '__main__':
    main()
