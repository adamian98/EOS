from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from jax import numpy as jnp
from flax import linen as nn
import os


def cifar10(n):
    trainset = load_dataset("cifar10")["train"]
    x = np.stack(trainset[:n]["img"])
    y = np.array(trainset[:n]["label"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    x = (x - x.mean((0, 1, 2))) / x.std((0, 1, 2))
    return x, y


def cifar10_binary(n):
    trainset = load_dataset("cifar10")["train"]
    y = np.array(trainset["label"])
    cat_idx = np.argwhere(y == 3)[: n // 2, 0]
    dog_idx = np.argwhere(y == 5)[: n // 2, 0]
    idx = np.concatenate([cat_idx, dog_idx])
    y = 2 * (y[idx] == 5) - 1
    x = np.stack(trainset[idx]["img"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    x = (x - x.mean((0, 1, 2))) / x.std((0, 1, 2))
    return x, y


def sst2(n):
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "sst2")
    y = np.array(dataset["train"]["label"])
    neg_idx = np.argwhere(y == 0)[: n // 2, 0]
    pos_idx = np.argwhere(y == 1)[: n // 2, 0]
    idx = np.concatenate([neg_idx, pos_idx])
    y = 2 * y[idx] - 1
    x = dataset["train"][idx]["sentence"]
    x = tokenizer(x, padding=True)["input_ids"]
    x, y = jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)
    return x, y
