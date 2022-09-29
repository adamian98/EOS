from flax import linen as nn
from functools import partial

torch_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")

# Note: The default PyTorch initialization uses the kernel's fan_in to initialize the bias, however this is a minor difference
TorchLinear = partial(
    nn.Dense,
    kernel_init=torch_init,
    bias_init=nn.initializers.zeros,
    dtype=None
)
TorchConv = partial(
    nn.Conv,
    kernel_init=torch_init,
    bias_init=nn.initializers.zeros,
    dtype=None
)
