# SpuriousFL


# Repo structure
```
project
│   README.md
│   .gitignore 
|   config.yaml # Experiment params
|   env.yaml # Machine specific params
|   flower_train.py # Run flower training
|   centralized_training.py # Training centralized models
|   biased_training.py # Test N matrix prediction
│
└───src # Source code
│   │   flower_client.py # Client Flower class
│   │   flower_strategy.py # Server Flower class
|   |   corr.py # N matrix imbalance metrics
│   └───datasets # Dataset loaders, splits
│   └───models # PyTorch models
|   └───optimizers # Handling subpop optimizers
└───datasets # Dataset files or links
└───checkpoints # Experiment files
```

Some config:
 - FedProx is a client optimizer method: the `client_opt.subpop_optimizer` can be changed from `ERM` to `Prox` to enable
 - `FedAvgM` or `FedAvg` changes the server aggregation in `server_opt.optimizer`
 - Client participation can be controlled with weights or selection: `server_opt.participation` can be `weighting` or `selection`
   - If `weighting` is used, `server_opt.client_weights` sets the specific weighting method.
   - If `selection` is used, `server_opt.selection_method` sets the specific selection method. It can be `random`, `groupweights` (original matrix with Oracle ReWeight), or `triplets_stochasticmatrix` for triplets.
 - Some methods use information from the client. The `server_opt.client_info` tells what data is shared.
   - If `groupweights`, `triplets` or `nova` is in the string, these infos will be passed to the server.
   - If `Npredicted` is in the string, numbers are generated with the N matrix prediction algorithm
 - The named data splits can be controlled with `dataset_options.split_mode`. Most options are in `src.datasets.data_splits.py`
   - `dataset_options.num_clients` must be set together with the `split_mode`.