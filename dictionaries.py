from flax import linen as nn
from jax import numpy as jnp
from models.simple import CNN, MLP
from models.resnet import ResNet18
from models.vgg import vgg11
from models.bert import bert_tiny
from models.transformer import Transformer
from transformers.modeling_flax_utils import ACT2FN
from data import (
    cifar10,
    cifar10_binary,
    sst2,
)

data_dict = {
    "cifar10": cifar10,
    "cifar10_binary": cifar10_binary,
    "sst2": sst2,
}

model_dict = {
    "mlp": MLP,
    "cnn": CNN,
    "resnet18": ResNet18,
    "vgg": vgg11,
    "bert_tiny": bert_tiny,
    "transformer": Transformer,
}

CrossEntropyLoss = lambda f, y: -nn.log_softmax(f)[y]
CategoricalMSELoss = lambda f, y: f @ f - 2 * f[y] + 1
BinaryMSELoss = lambda f, y: jnp.mean((f[0] - y) ** 2)
LogisticLoss = lambda f, y: -nn.log_sigmoid(f[0] * y)
criterion_dict = {
    "cat_mse": CategoricalMSELoss,
    "binary_mse": BinaryMSELoss,
    "ce": CrossEntropyLoss,
    "logistic": LogisticLoss,
}

activation_dict = ACT2FN
