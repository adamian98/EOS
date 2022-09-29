import jax
from jax_resnet import (
    ResNet,
    ResNetBlock,
    ConvBlock,
    ModuleDef,
)
from functools import partial
from flax import linen as nn


class CIFARStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    width: int = 64

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(
            self.width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
        )(x)


def small_resnet(stage_sizes, activation, base_width=16, **kwargs):
    config = {
        "stem_cls": partial(CIFARStem, width=base_width),
        "stage_sizes": stage_sizes,
        "block_cls": partial(ResNetBlock, activation=activation),
        "hidden_sizes": [base_width * (2**i) for i in range(len(stage_sizes))],
        "norm_cls": lambda *args, **kwargs: nn.GroupNorm(num_groups=16),
        "pool_fn": lambda x: x,
    }
    model = ResNet(**config, **kwargs)
    model.layers[-1].kernel_init = nn.initializers.zeros
    return model


ResNet18 = partial(small_resnet, stage_sizes=[2, 2, 2, 2])
ResNet20 = partial(small_resnet, stage_sizes=[3, 3, 3])
