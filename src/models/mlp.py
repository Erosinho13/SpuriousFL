import torch.nn as nn

from src.models.abstract_model import AbstractModel


class MLP(AbstractModel):
    def __init__(self,
                 input_shape=14,
                 num_classes=2,
                 hidden_sizes=[64],
                 use_bias=True):
        super(MLP, self).__init__()
        blocks = []
        blocks.append(nn.Linear(input_shape, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            blocks.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
        self.featurizer = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_features=hidden_sizes[-1], out_features=num_classes)
