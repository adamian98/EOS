from flax import linen as nn
from .torch_layers import *
from typing import Callable


class MLP(nn.Module):
    activation: Callable
    n_classes: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = TorchLinear(200)(x)
        x = self.activation(x)
        x = TorchLinear(200)(x)
        x = self.activation(x)
        x = TorchLinear(self.n_classes)(x)
        return x


class CNN(nn.Module):
    activation: Callable
    n_classes: int

    @nn.compact
    def __call__(self, x):
        x = TorchConv(32, (3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, (2, 2), (2, 2), "SAME")
        x = TorchConv(32, (3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, (2, 2), (2, 2), "SAME")
        x = x.reshape(x.shape[0], -1)
        x = TorchLinear(self.n_classes)(x)
        return x
