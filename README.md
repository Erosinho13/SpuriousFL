# SpuriousFL


# Repo structure
```
project
│   README.md
│   .gitignore 
|   config.yaml # Experiment params
|   env.yaml # Machine specific params
|   flower_train.py # Run flower training
│
└───src # Source code
│   │   flower_client.py # Client Flower class
│   │   flower_strategy.py # Server Flower class
│   └───datasets # Dataset loaders, splits
│   └───models # PyTorch models
└───datasets # Dataset files or links
└───checkpoints # Experiment files
```
