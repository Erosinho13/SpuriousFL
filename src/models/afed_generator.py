# From AFed paper: https://arxiv.org/pdf/2501.02732

from src.utils import get_device
import torch.nn as nn
import torch
import numpy as np

class AFedGenerator(nn.Module):
    def __init__(self,
                 conf,
                 hidden_dim=32,
                 latent_dim=16,
                 input_channel=3,
                 n_class=2,
                 noise_dim=8,
                 embedding=False):
        super(AFedGenerator, self).__init__()

        self.hidden_dim = hidden_dim,
        self.latent_dim = latent_dim
        self.input_channel = input_channel,
        self.n_class = n_class
        self.noise_dim = noise_dim
        self.diversity_loss = DiversityLoss(metric='l1')
        self.embedding = embedding
        self.build_network()
        self.conf = conf

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(4, self.noise_dim)
            input_dim = self.noise_dim * 2
            self.fc_configs = [input_dim, self.hidden_dim[0]]
        else:
            input_dim = self.noise_dim + self.n_class
            self.fc_configs = [input_dim, self.hidden_dim[0]]
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(in_features=input_dim, out_features=out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]

        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, z):
        """
        G(Z|y,a)
        """
        for layer in self.fc_layers:
            z = layer(z.to(get_device(self.conf)))
        z = self.representation_layer(z)

        return z


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))