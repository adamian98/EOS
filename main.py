import jax

jax.config.update("jax_enable_x64", True)
from pathlib import Path
from absl import app, flags
from jax import numpy as jnp, random
from jax.numpy import linalg as jla
from tqdm.auto import tqdm, trange
from dictionaries import data_dict, model_dict, criterion_dict, activation_dict
from loss import build_loss
from util import UID, tree_save
from taylor import find_instability, track_dynamics
import json
from functools import partial
import shutil

FLAGS = flags.FLAGS
flags.DEFINE_enum("model", None, model_dict.keys(), "Model")
flags.DEFINE_enum("data", None, data_dict.keys(), "Data")
flags.DEFINE_integer("n_classes", None, "Number of classes")
flags.DEFINE_enum("criterion", None, criterion_dict.keys(), "Loss Criterion")
flags.DEFINE_enum("activation", "swish", activation_dict.keys(), "Activation Function")
flags.DEFINE_integer("n_samples", 5000, "Number of samples")
flags.DEFINE_float("lr", None, "Learning Rate")
flags.DEFINE_enum(
    "deriv_dtype", "f32", ["f32", "f64"], "Precision for higher order derivatives"
)
flags.DEFINE_enum(
    "hvp_dtype",
    "f32",
    ["f32", "f64"],
    "Precision for Hessian Vector Products for Eigenvalue Solver",
)
flags.DEFINE_enum(
    "solver_dtype", "f32", ["f32", "f64"], "Precision for Eigenvalue Solver"
)
flags.DEFINE_float("solver_tol", 1e-9, "Tolerance for Eigenvalue Solver")
flags.DEFINE_integer("ghost_batch_size", None, "Ghost batch size to save memory")
flags.DEFINE_integer("steps", 2000, "Number of steps")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_boolean(
    "generalized_pred", False, "Whether to use the generalized predicted dynamics"
)
flags.DEFINE_integer("num_proj_steps", 3, "number of linearized projections for PGD")


def main(args):
    config = {key: val.value for key, val in FLAGS._flags().items()}
    run_name = UID(config)
    save_dir = (
        Path("experiments") / FLAGS.model / FLAGS.data / FLAGS.criterion / run_name
    )
    if save_dir.exists():
        overwrite = input("File already exists. Overwrite? Y = yes, N = no\n")
        if overwrite.lower() == "y":
            shutil.rmtree(save_dir)
        else:
            quit()
    save_dir.mkdir(parents=True)

    print(f"Run Name: {run_name}")
    print(f"Save Directory: {save_dir}")
    with open(save_dir / "flags.json", "w") as file:
        json.dump(config, file, sort_keys=True, indent=4)

    activation = activation_dict[FLAGS.activation]
    criterion = criterion_dict[FLAGS.criterion]
    data = data_dict[FLAGS.data](FLAGS.n_samples)
    model = model_dict[FLAGS.model](activation=activation, n_classes=FLAGS.n_classes)
    model_key, eig_key = random.split(random.PRNGKey(FLAGS.seed))
    p, loss = build_loss(
        model=model,
        data=data,
        criterion=criterion,
        batch_size=FLAGS.ghost_batch_size,
        model_key=model_key,
    )
    dtype_dict = dict(f32=jnp.float32, f64=jnp.float64)
    loss = loss._replace(
        D=partial(loss.D, dtype=dtype_dict[FLAGS.deriv_dtype]),
        eig=partial(
            loss.eig,
            tol=FLAGS.solver_tol,
            hvp_dtype=dtype_dict[FLAGS.hvp_dtype],
            solver_dtype=dtype_dict[FLAGS.solver_dtype],
        ),
    )

    lr = FLAGS.lr
    p = p.astype(jnp.float32)
    U = random.normal(eig_key, (len(p), 2), dtype=p.dtype)

    print("Running Until Instability")
    p, U0, prelim = find_instability(p, U[:, :1], lr, loss)
    tree_save(prelim, save_dir / "prelim.pytree", overwrite=True)
    print("Switching to 64 bit")
    p = p.astype(jnp.float64)
    U = U.astype(jnp.float64).at[:, :1].set(U0)
    _, U = loss.eig(p, U)
    print("Tracking Dynamics")
    track_dynamics(
        p,
        U,
        lr,
        loss,
        steps=FLAGS.steps,
        num_proj_steps=FLAGS.num_proj_steps,
        generalized_pred=FLAGS.generalized_pred,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    app.run(main)
