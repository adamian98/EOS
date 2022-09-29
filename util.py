import pickle
from pathlib import Path
from typing import Union
from jax import numpy as jnp, random
from jax.numpy import linalg as jla
from tqdm import trange
from jax.tree_util import tree_flatten, tree_unflatten
import hashlib
import json

suffix = ".pytree"


def UID(config):
    dhash = hashlib.md5()
    encoded = json.dumps(config, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def tree_save(data, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def tree_load(path: Union[str, Path]):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix != suffix:
        raise ValueError(f"Not a {suffix} file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def power_iter(A, v, num_iter=20, seed=0, show_progress=False):
    v = v / jla.norm(v)
    iterator = trange if show_progress else range
    for _ in iterator(num_iter):
        v = A(v)
        v = v / jla.norm(v)
    return v @ A(v), v


def tree_stack(trees):
    _, treedef = tree_flatten(trees[0])
    leaf_list = [tree_flatten(tree)[0] for tree in trees]
    leaf_stacked = [jnp.stack(leaves) for leaves in zip(*leaf_list)]
    return tree_unflatten(treedef, leaf_stacked)
