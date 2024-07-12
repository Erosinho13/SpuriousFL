import torch
import torch.nn as nn

class AbstractModel(nn.Module):
    """Base class that set the structure of featurizer-classifier"""

    def __init__(self):
        super(AbstractModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.featurizer(x)
        output = self.classifier(representation)
        return output