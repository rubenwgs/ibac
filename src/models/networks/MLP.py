import torch
from torch import Tensor
from typing import override
import torch.nn as nn
from src.models.components import get_activation
from src.utils import get_rank


class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 num_hidden_layers: int,
                 activation: str):
        super().__init__()
        self.net = None
        self.dims = ([in_features]
                     + [hidden_features for _ in range(num_hidden_layers)]
                     + [out_features])
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers
        self.activation = get_activation(activation)
        self.activation_str = activation
        self.rank = get_rank()

    def setup(self):
        self.create_network()

    def create_network(self) -> None:
        self.net = []
        for i in range(len(self.dims) - 1):
            layer = nn.Linear(self.dims[i], self.dims[i + 1])

            # apply takes as argument a function (Module)-> None
            layer.apply(self.init_weights_normal)
            self.net.append(layer)

        self.net = torch.nn.ModuleList(self.net)

    @staticmethod
    @torch.no_grad()
    def init_weights_normal(m: nn.Module) -> None:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, coords: Tensor) -> Tensor:
        x = coords
        for layer in self.net[:-1]:
            x = self.activation(layer(x))
        x = self.net[-1](x)
        return x
