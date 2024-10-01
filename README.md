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
