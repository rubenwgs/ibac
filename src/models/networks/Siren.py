from src.models.networks import MLP


import torch
import numpy as np
from typing import override
import torch.nn as nn


class Siren(MLP):
    def __init__(
                self,
                in_features: int,
                out_features: int,
                hidden_features: int,
                num_hidden_layers: int
                 ):
        super(Siren, self).__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation='siren'
        )


    @override
    def create_network(self) -> None:
        assert (self.activation_str == 'siren')
        self.net = []
        for i in range(len(self.dims) - 1):
            layer = nn.Linear(self.dims[i], self.dims[i + 1])
            layer.apply(self.first_layer_sine_init if i == 0 else self.sine_init)

            print(self.first_layer_sine_init)
            self.net.append(layer)
        self.net = torch.nn.ModuleList(self.net)

    @staticmethod
    @torch.no_grad()
    def sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

    @staticmethod
    @torch.no_grad()
    def first_layer_sine_init(m):
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
