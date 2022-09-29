import jax
from jax import numpy as jnp, random, vmap, jit, eval_shape, lax, jacobian, jvp
from jax.tree_util import tree_map, tree_leaves
from jax.flatten_util import ravel_pytree
from functools import partial
from jax.experimental.sparse.linalg import lobpcg_standard
from time import perf_counter
from collections import namedtuple
from absl import flags
from functools import wraps


def laxmean(f, data):
    n = len(tree_leaves(data)[0])
    x0 = tree_map(lambda x: x[0], data)
    out_tree = eval_shape(f, x0)
    avg_init = tree_map(lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype), out_tree)

    def step(avg, x):
        avg = tree_map(lambda a, b: a + b / n, avg, f(x))
        return avg, None

    return lax.scan(step, avg_init, data)[0]


# Computes the total derivative of f of order "order" at x evaluated in the tangent directions *vecs
def D(f, x, order=1, *vecs):
    if order == 0:
        return f(x)
    elif len(vecs) < order:
        _f = jacobian(f)
        vecs = vecs
    else:
        v, *vecs = vecs
        _f = lambda x: jvp(f, (x,), (v,))[1]
    return D(_f, x, order - 1, *vecs)


Loss = namedtuple("Loss", "value grad value_and_grad D eig")


def build_loss(model, data, criterion, batch_size=None, model_key=None, benchmark=True):
    data = tree_map(lambda x: jnp.array(x, dtype=x.dtype), data)
    p = model.init(model_key, data[0][:1])
    p = tree_map(lambda x: x.astype(jnp.float32), p)
    p, unravel = ravel_pytree(p)
    safe_unravel = lambda p: tree_map(lambda x: x.astype(p.dtype), unravel(p))
    f = lambda p, x: model.apply(safe_unravel(p), x)

    @jit
    def batch_loss(p, data):
        x, y = data
        if jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(p.dtype)
        out = f(p, x)
        out = out.reshape(-1, out.shape[-1])
        y = y.reshape(-1)
        return vmap(criterion)(out, y).mean(0)

    if batch_size is not None:
        n = len(data[0])
        n_batch = n // batch_size
        data = tree_map(lambda x: x[: n_batch * batch_size], data)
        data = tree_map(lambda x: x.reshape(n_batch, batch_size, *x.shape[1:]), data)

        def dataloop(f, **jit_kwargs):
            @partial(jit, **jit_kwargs)
            @wraps(f)
            def looped_f(*args, data, **kwargs):
                return laxmean(lambda batch: f(*args, **kwargs, data=batch), data)

            return looped_f

    else:
        dataloop = jit

    loss_fn = partial(dataloop(batch_loss), data=data)
    value_and_grad_fn = partial(dataloop(jax.value_and_grad(batch_loss)), data=data)
    grad_fn = lambda p: value_and_grad_fn(p)[1]

    @partial(dataloop, static_argnums=1)
    def _dL(p, order, *v, data):
        if len(v) == 0:
            v = D(partial(batch_loss, data=data), p, order)
        else:
            v_shape = v[0].shape
            v = tree_map(lambda x: x.reshape(x.shape[0], -1), v)
            v = vmap(lambda v: D(partial(batch_loss, data=data), p, order, *v), 1, -1)(
                v
            )
            v = v.reshape((*v.shape[:-1], *v_shape[1:]))
        return v

    def dL(p, order, *v, dtype=None):
        out_dtype = p.dtype
        if dtype is not None:
            p = p.astype(dtype)
            v = tree_map(lambda x: x.astype(dtype), v)
        v = _dL(p, order, *v, data=data)
        return v.astype(out_dtype)

    @jit
    def _eig_fn(p, ref_U, tol, data):
        hvp_dtype = p.dtype
        solver_dtype = ref_U.dtype

        def hvp_fn(v):
            v = v.astype(hvp_dtype)
            v = _dL(p, 2, v, data=data)
            return v.astype(solver_dtype)

        eigs, U, _ = lobpcg_standard(hvp_fn, ref_U, tol=tol)
        return eigs, U

    def eig_fn(p, ref_U, tol=1e-9, hvp_dtype=None, solver_dtype=None):
        out_dtype = p.dtype
        if hvp_dtype is not None:
            p = p.astype(hvp_dtype)
        if solver_dtype is not None:
            ref_U = ref_U.astype(solver_dtype)
        single_eval = False
        if ref_U.ndim == 1:
            ref_U = ref_U[:, None]
            single_eval = True
        eigs, U = _eig_fn(p, ref_U, tol, data=data)
        eigs, U = eigs.astype(out_dtype), U.astype(out_dtype)
        if single_eval:
            S, u = eigs[0], U[:, 0]
            return S, u
        else:
            return eigs, U

    def benchmark_fn(name, f, p, n_iter=3):
        f_ = lambda p: tree_map(lambda x: x.block_until_ready(), f(p))
        print(f"{name}")
        print("\tCompiling...", end=" ", flush=True)
        start_time = perf_counter()
        f_(p)
        end_time = perf_counter()
        print(f"Done ({end_time-start_time:.2f}s)")
        print("\tBenchmarking", end="", flush=True)
        start_time = perf_counter()
        for _ in range(n_iter):
            f_(p)
            print(".", end="", flush=True)
        end_time = perf_counter()
        print(f" {(end_time-start_time)/n_iter:.2f}s")

    if benchmark:
        benchmark_fn("Loss", loss_fn, p)
        benchmark_fn("Gradient", grad_fn, p)
        benchmark_fn("Gradient (64 bit)", grad_fn, p.astype(jnp.float64))
        benchmark_fn("Second Derivative", lambda p: dL(p, 2, p), p)
        benchmark_fn("Third Derivative", lambda p: dL(p, 3, p, p), p)
        benchmark_fn("Fourth Derivative", lambda p: dL(p, 4, p, p, p), p)
    return p, Loss(loss_fn, grad_fn, value_and_grad_fn, dL, eig_fn)
