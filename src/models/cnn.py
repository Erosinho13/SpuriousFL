import numpy as np
import torch
import torch.nn as nn


class DiaoCNN(nn.Module):
    """Model following the diao et al paper.
    Emmiting LN, GN and IN as it is not straightforward to cast to TF,
    and the paper shows superiority of the BN

    https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/tree/master
    """

    def __init__(
        self,
        input_shape=(3, 32, 32),
        num_classes=10,
        default_hidden=[64, 128, 256, 512],
        use_bias=True,
    ):
        super(DiaoCNN, self).__init__()

        hidden_sizes = default_hidden

        conv = nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_sizes[0], kernel_size=3, stride=1, padding=1, bias=use_bias)


        norm = nn.BatchNorm2d(hidden_sizes[0], track_running_stats=False)

        blocks = [
            conv,
            norm,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]

        for i in range(len(hidden_sizes) - 1):
            conv = nn.Conv2d(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i + 1], kernel_size=3, stride=1, padding=1, bias=use_bias)
            norm = nn.BatchNorm2d(hidden_sizes[i + 1], track_running_stats=False)

            blocks.extend(
                [
                    conv,
                    norm,
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        linear = nn.Linear(in_features=hidden_sizes[-1], out_features=num_classes, bias=use_bias)

        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                linear,
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.blocks(x)
        return output


def get_diao_CNN(*args, **kwargs):
    return DiaoCNN(*args, **kwargs)
