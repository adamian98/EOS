from functools import wraps
from pathlib import Path

import jax
import numpy as np
import optax
from absl import app, flags
from datasets import load_dataset
from jax import jit
from jax import numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from tqdm.auto import tqdm, trange
from functools import partial

from models.resnet import ResNet18
from util import tree_stack, tree_save
from loss import laxmean, D

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 0, "Learning Rate")
flags.DEFINE_integer("steps", 3000, "GD Steps")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_integer("width", 16, "resnet width")


def main(args):
    out_dir = Path("experiments") / "train_speed"
    out_dir.mkdir(exist_ok=True)

    model_key, eig_key = random.split(random.PRNGKey(FLAGS.seed))

    batch_size = 1000
    dataset = load_dataset("cifar10")
    trainset = dataset["train"]
    x = np.stack(trainset["img"])
    y = np.array(trainset["label"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    CIFAR_MEAN = x.mean((0, 1, 2))
    CIFAR_STD = x.std((0, 1, 2))
    x = (x - CIFAR_MEAN) / CIFAR_STD
    trainset = (x, y)
    n_batch = len(x) // batch_size
    trainset = tree_map(lambda x: x[: n_batch * batch_size], trainset)
    trainset = tree_map(
        lambda x: x.reshape(n_batch, batch_size, *x.shape[1:]), trainset
    )

    testset = dataset["test"]
    x = np.stack(testset["img"])
    y = np.array(testset["label"])
    x, y = jnp.array(x, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)
    x = (x - CIFAR_MEAN) / CIFAR_STD
    testset = (x, y)
    n_batch = len(x) // batch_size
    testset = tree_map(lambda x: x[: n_batch * batch_size], testset)
    testset = tree_map(lambda x: x.reshape(n_batch, batch_size, *x.shape[1:]), testset)

    model = ResNet18(activation=jax.nn.swish, n_classes=10, base_width=FLAGS.width)
    p = model.init(model_key, jnp.zeros((1, 32, 32, 3)))
    p, unravel = ravel_pytree(p)
    f = lambda p, x: model.apply(unravel(p), x)
    criterion = optax.softmax_cross_entropy_with_integer_labels
    acc_fn = lambda f, y: jnp.argmax(f) == y

    def batch_loss(p, batch):
        x, y = batch
        out = f(p, x)
        loss = criterion(out, y).mean()
        acc = vmap(acc_fn)(out, y).mean()
        return loss, acc

    def dataloop(f):
        @wraps(f)
        def _f(*args, batches, **kwargs):
            return laxmean(lambda batch: f(*args, **kwargs, batch=batch), batches)

        return jit(_f)

    loss_fn = dataloop(batch_loss)
    grad_fn = dataloop(jax.value_and_grad(batch_loss, has_aux=True))
    hvp_fn = dataloop(
        lambda p, v, batch: D(lambda p: batch_loss(p, batch=batch)[0], p, 2, v)
    )

    trainset1k = tree_map(lambda x: x[0][:1000], trainset)

    @jit
    def eig_fn(p, ref_U, data, tol=1e-9):
        _loss = lambda p: batch_loss(p, batch=data)[0]
        hvp_fn = lambda v: D(_loss, p, 2, v[:, 0])[:, None]
        eigs, U, _ = jax.experimental.sparse.linalg.lobpcg_standard(
            hvp_fn, ref_U, tol=tol
        )
        return eigs, U

    save_list = []
    U = random.normal(eig_key, (len(p), 1))
    pbar = trange(FLAGS.steps)
    for i in pbar:
        (train_loss, train_acc), dL = grad_fn(p, batches=trainset)
        test_loss, test_acc = loss_fn(p, batches=testset)
        if i % 40 == 0:
            (S,), U = eig_fn(p, U, data=trainset1k)
            save_list.append(
                dict(
                    step=i,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    S=S,
                )
            )
            pbar.set_description(
                f"t: {i}, S: {S:.2f}, L: {train_loss:.2f}, A: {train_acc:.2f}, testL: {test_loss:.2f}, testA: {test_acc:.2f}"
            )
        if FLAGS.lr > 0:
            p = p - FLAGS.lr * dL
        else:
            p = p - dL / S
    save_list = tree_stack(save_list)
    name = str(FLAGS.lr) if FLAGS.lr > 0 else "flow"
    name = f"{name}_{FLAGS.width}.pytree"
    tree_save(save_list, out_dir / name, overwrite=True)


if __name__ == "__main__":
    app.run(main)
