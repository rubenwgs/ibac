from torch import Tensor
import torch
import torch.nn as nn


def get_activation(name: str, **kwargs) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    elif name == "siren":
        return Sine()
    elif name == "gaussian":
        return GaussianActivation(**kwargs)
    elif name == "quadratic":
        return QuadraticActivation(**kwargs)
    elif name == "multi_quadratic":
        return MultiQuadraticActivation(**kwargs)
    elif name == "laplacian":
        return LaplacianActivation(**kwargs)
    elif name == "super_gaussian":
        return SuperGaussianActivation(**kwargs)
    elif name == "exp_sin":
        return ExpSinActivation(**kwargs)
    else:
        return nn.Identity()


# different activation functions

class Sine(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * x)


class GaussianActivation(nn.Module):
    def __init__(self, a: float = 1., trainable: bool = True) -> None:
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-x ** 2 / (2 * self.a ** 2))


class QuadraticActivation(nn.Module):
    def __init__(self, a: float = 1., trainable: bool = True) -> None:
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (self.a * x) ** 2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a: float = 1., trainable: bool = True) -> None:
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + (self.a * x) ** 2) ** 0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a: float = 1., trainable: bool = True) -> None:
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-torch.abs(x) / self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a: float = 1., b: float = 1., trainable: bool = True) -> None:
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b * torch.ones(1), trainable))

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-x ** 2 / (2 * self.a ** 2)) ** self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a: float = 1., trainable: bool = True) -> None:
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-torch.sin(self.a * x))
