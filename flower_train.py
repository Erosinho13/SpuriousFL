# Installed modules
import flwr as fl
from flwr.common import ndarrays_to_parameters
import os

# Own modules
from src import utils
from src.datasets import data_preparation
from src.models import model_utils
from src.flower_strategy import MyStrategy
from src.flower_client import FlowerClient

global conf
conf = utils.load_config()

global X_split
global Y_split
global X_val
global Y_val

#!TODO I don't know how to pass parameters to this function
def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client = FlowerClient(int(cid), conf)
    client.load_data(X_split[int(cid)], Y_split[int(cid)], X_val, Y_val)
    client.init_model()
    return client

def train():
    """Flower training simulation using global config"""
    global X_split
    global Y_split
    global X_val
    global Y_val
    train_ds, val_ds, test_ds = data_preparation.load_data(conf=conf)
    X_val, Y_val = data_preparation.get_np_from_ds(val_ds)
    X_train, Y_train = data_preparation.get_np_from_ds(train_ds)
    conf["len_total_data"] = len(X_train)
    X_split, Y_split = data_preparation.split_data(
        X_train,
        Y_train,
        conf["num_clients"],
        split_mode=conf["split_mode"],
        mode="clients",
        distribution_seed=conf["seed"],
        shuffle_seed=conf["data_shuffle_seed"],
        dirichlet_alpha=conf["dirichlet_alpha"],
    )

    initial_model = model_utils.init_model(
        conf=conf
    )

    model_utils.print_summary(initial_model)
    ws = model_utils.get_weights(initial_model)
    initial_parameters = ndarrays_to_parameters(
            ws
        )
    
    # Create FedAvg strategy
    strategy = MyStrategy(
        conf=conf,
        initial_parameters=initial_parameters,  # avoid smaller models as init
        fraction_fit=1.0,  # Sample 10% of available clients for training
        fraction_evaluate=0.1,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        # min_available_clients=1, # Wait until at least 75 clients are available
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=conf["num_clients"],
        config=fl.server.ServerConfig(num_rounds=conf["rounds"]),
        strategy=strategy,
        ray_init_args=conf["ray_init_args"],
        client_resources=conf["client_resources"],
    )
    #!TODO there is a new, better way of returning with latest model
    model_path = os.path.join(
                "./dump/",
                "latest_model"
            )
    model = model_utils.init_model(conf=conf, model_path=model_path)
    return model

if __name__ == "__main__":
    train()