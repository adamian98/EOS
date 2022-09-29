from jax import numpy as jnp
from flax import linen as nn
from flax.linen.attention import MultiHeadDotProductAttention
from typing import Callable


class Transformer(nn.Module):
    activation: Callable
    n_classes: int
    num_tokens: int = 33278
    hidden_size: int = 64
    max_position_embeddings: int = 64
    num_attention_heads: int = 2
    num_layers: int = 2
    norm_layer: Callable = nn.LayerNorm

    @nn.compact
    def __call__(self, x):
        pos_idx = jnp.broadcast_to(jnp.arange(x.shape[1]), x.shape)
        pos_enc = nn.Embed(
            self.max_position_embeddings,
            self.hidden_size,
            name="positional_embedding",
        )(pos_idx)
        x = nn.Embed(self.num_tokens, self.hidden_size, name="encoder")(x)
        x += pos_enc
        for _ in range(self.num_layers):
            attn = MultiHeadDotProductAttention(self.num_attention_heads)
            x = x + attn(x, x)
            x = self.norm_layer()(x)

            identity = x
            x = nn.Dense(self.hidden_size)(x)
            x = self.activation(x)
            x = nn.Dense(self.hidden_size)(x)
            x = x + identity
            x = self.norm_layer()(x)
        x = x.mean(1)
        x = nn.Dense(self.n_classes, kernel_init=nn.initializers.zeros, name="head")(x)
        return x
