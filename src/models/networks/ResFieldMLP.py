from src.models.networks import MLP
from src.models.components.layers import ResFieldLinear
import torch


class ResFieldMLP(MLP):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 num_hidden_layers: int,
                 activation: str,
                 resfield_layers: list[int],
                 composition_rank: int,
                 mode: str,
                 capacity: int,
                 coefficient_ratio: float,
                 fuse_mode: str,
                 compression: str
                 ):
        super(ResFieldMLP, self).__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation
        )

        self.resfield_layers = resfield_layers
        self.composition_rank = composition_rank
        self.mode = mode
        self.capacity = capacity
        self.coefficient_ratio = coefficient_ratio
        self.fuse_mode = fuse_mode
        self.compression = compression

    def create_network(self) -> None:
        self.net = []
        for i in range(len(self.dims) - 1):
            _rank = self.composition_rank if i in self.resfield_layers else 0
            _capacity = self.capacity if i in self.resfield_layers else 0

            layer = ResFieldLinear(
                self.dims[i],
                self.dims[i + 1],
                rank=_rank,
                capacity=_capacity,
                mode=self.mode,
                compression=self.compression,
                fuse_mode=self.fuse_mode,
                coeff_ratio=self.coefficient_ratio)

            if self.activation == 'relu':
                layer.apply(self.init_weights_normal)
            self.net.append(layer)
        self.net = torch.nn.ModuleList(self.net)

    def forward(self, coords: torch.Tensor, frame_id=None, input_time=None) -> torch.Tensor:
        x = coords
        for layer in self.net[:-1]:
            x = self.activation(layer(x, frame_id=frame_id, input_time=input_time))
            if layer.compression == 'resnet' and layer.capacity > 0:
                if frame_id.numel() == 1:
                    x = x + layer.resnet_vec[frame_id].view(1, 1, layer.resnet_vec.shape[-1])
                else:
                    x = x + layer.resnet_vec[:, None]  # T, S, F_out
        x = self.net[-1](x, frame_id=frame_id, input_time=input_time)
        return x
