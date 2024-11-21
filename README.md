# SpuriousFL


## Repository structure
```
project
│   README.md
│   .gitignore 
|   flower_train.py # Run flower training
|   centralized_training.py # Training centralized models
|   biased_training.py # Test N matrix prediction
│
└───src # Source code
│   │   flower_client.py # Client Flower class: client optimizers, client info
│   │   flower_strategy.py # Server Flower class: server optimizers, client weighting
|   |   flower_manager.py # Flower manager for client selection
|   |   corr.py # N matrix metrics SC,CI,AI
|   |   matrix_inference.py # N matrix inference
|   |   weighting_strategy.py # Helpers for imported selection and weighting strategies
│   └───datasets # Dataset loaders, splits
│   └───models # PyTorch models
|   └───optimizers # Handling subpopulation shift optimizers
└───conf # Config files following hydra
└───datasets # Dataset files or links
└───checkpoints # Experiment files
```

## Important config parameters:
 - FedProx is a client optimizer method: the `client_opt.subpop_optimizer` can be changed from `ERM` to `Prox` to enable
 - `FedAvgM` or `FedAvg` changes the server aggregation in `server_opt.optimizer`
 - Client participation can be controlled with weights or selection: `server_opt.participation` can be `weighting` or `selection`
   - If `weighting` is used, `server_opt.client_weights` sets the specific weighting method. Weighting takes into account even if `selection` is set (for FedPNS), should be `server_opt.client_weights=same` to disable.
   - If `selection` is used, `server_opt.selection_method` sets the specific selection method. It can be `random`, `groupweights` (original matrix with Oracle ReWeight), or `triplets_stochasticmatrix` for triplets.
 - Some methods use information from the client. The `server_opt.client_info` tells what data is shared.
   - If `groupweights`, `triplets` or `nova` is in the string, these infos will be passed to the server.
   - If `Npredicted` is in the string, numbers are generated with the N matrix prediction algorithm
 - The named data splits can be controlled with `dataset_options.split_mode`. Most options are in `src.datasets.data_splits.py`
   - `dataset_options.num_clients` must be set together with the `split_mode`.

## Methods
 - FedProx: `client_opt.subpop_optimizer=Prox`
 - FedAvgM: `server_opt.optimizer=FedAvgM`
 - FedDiverse: `server_opt.participation=selection`, `server_opt.participation=triplets_stochasticmatrix`,
 - FedNova: `nova` in `server_opt.client_info`, `server_opt.participation=weighting`, `server_opt.client_weights=server_post_nova`, `server_opt.optimizer=FedAvgM`
 - pow-d: `gloss` in `server_opt.client_info`, `server_opt.participation=weighting`, `server_opt.client_weights=server_post_powd`
 - round robin: `server_opt.participation=selection`, `server_opt.selection_method=roundrobin`
 - FedPNS: `server_opt.participation=selection`, `server_opt.selection_method=fedpns`, `server_opt.client_weights=fedpns`
 - FedPNS w/o weights: `server_opt.participation=selection`, `server_opt.selection_method=fedpns`, `server_opt.client_weights=same`

## Datasets distributions
 - Spawrious_GSC: `dataset_options.split_mode=spawrious2`, `dataset_options.num_clients=24`
 - Spawrious_GCI: `dataset_options.split_mode=spawrious_GCI`, `dataset_option.num_clients=24`
 - Spawrious_GAI: `dataset_options.split_mode=spawrious_GAI_2`, `dataset_option.num_clients=25`
 - Waterbirds_dist: `dataset_options.split_mode=waterbirds_dist`, `dataset_option.num_clients=30`
 - Spawrious_4: `dataset_options.split_mode=spawrious4`, `dataset_options.num_clients=25`, `dataset_options.num_targets=4`

## Run
Example run: `python flower_train.py`

To run the code, one has to add a wandb project to `flower_train.py` and `centralized_training.py` or set `wandb=false`. 