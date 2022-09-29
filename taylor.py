from collections import namedtuple
from util import tree_stack
from tqdm.auto import tqdm
from jax import numpy as jnp
from jax.numpy import linalg as jla
from tqdm import trange
from math import ceil
from util import power_iter, tree_save
from jax.tree_util import tree_map
import time
from functools import wraps


def find_instability(p, U, lr, loss):
    pbar = tqdm()
    save_list = []
    t = 0
    prev = (t, p, jnp.inf)
    next_check = 0
    dt = 1
    while True:
        L, dL = loss.value_and_grad(p)
        if t >= next_check:
            p_next = p - lr * dL
            (S,), U = loss.eig((p + p_next) / 2, U)
            save_list.append(dict(t=t, L=L, S=S))
            pbar.set_description(
                f"t={t:.2f}, next={next_check:.2f}, S={S:.2f}/{2/lr:.2f}"
            )
            pbar.refresh()
            if S >= 2 / lr:
                if dt <= lr:
                    break
                else:
                    print(
                        f"\nBacktracking to t={prev[0]:.2f} where sharpness was {prev[2]:.2f}"
                    )
                    t, p, _ = prev
                    dt = 0
                    next_check = t
                    continue
            if S >= 1.5 / lr:
                dt = min(dt, 0.01)
            if S >= 1 / lr:
                dt = min(dt, 0.1)
            next_check = t + dt
            prev = (t, p, S)
        pbar.set_description(f"t={t:.2f}, next={next_check:.2f}, S={S:.2f}/{2/lr:.2f}")
        t += lr
        p = p - lr * dL
        pbar.update()
    pbar.set_description(f"t={t:.2f}, S={S:.2f}/{2/lr:.2f}")
    pbar.close()
    return p, U, tree_stack(save_list)


TaylorCenter = namedtuple("TaylorCenter", "p L dL S u dS")
EigenSystem = namedtuple("EigenSystem", "p S u")


def track_dynamics(
    p, ref_U, lr, loss, steps, num_proj_steps, generalized_pred=False, save_dir=None
):
    def check_eigengap(T: TaylorCenter, ref_U):
        p, L, dL, S, u, dS = T
        U = ref_U.at[:, 0].set(u)
        eigs, U = loss.eig(p, U)
        return eigs, U

    def taylor_center(p, ref_u):
        L, dL = loss.value_and_grad(p)
        S, u = loss.eig(p, ref_u)
        dS = loss.D(p, 3, u, u)
        return TaylorCenter(p, L, dL, S, u, dS)

    def linear_project(T: TaylorCenter):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        S_proj = -max(0, S - 2 / lr) * dS_perp / (dS_perp @ dS_perp)
        newton_proj = -u * ((u @ dL) / S)
        return taylor_center(p + S_proj + newton_proj, T.u)

    def project(T: TaylorCenter):
        for _ in range(num_proj_steps):
            T = linear_project(T)
        return T

    def constrained_step(T: TaylorCenter):
        return project(taylor_center(T.p - lr * T.dL, T.u))

    def gd_step(p):
        return p - lr * loss.grad(p)

    def gf_step(E: EigenSystem):
        p, S, u = E
        sub_steps = int(ceil(lr * S))
        for _ in range(sub_steps):
            p = p - (lr / sub_steps) * loss.grad(p.astype(jnp.float32)).astype(
                jnp.float64
            )
        S, u = loss.eig(p, u)
        return EigenSystem(p, S, u)

    def predicted_step(v, T: TaylorCenter):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        alpha = -dL @ dS_perp
        beta = dS_perp @ dS_perp
        delta = jnp.sqrt(2 * alpha / beta)

        x = u @ v
        y = dS_perp @ v
        v_perp = v - x * u
        if not generalized_pred:
            grad_terms = [
                (x**2 - delta**2) * dS_perp / 2,
                loss.D(p, 2, v),
                u * x * y,
            ]
        else:
            delta_dL = u * (u @ loss.grad(p + x * u))
            grad_terms = [
                (x**2 - delta**2) * dS_perp / 2,
                loss.D(p, 2, v_perp),
                delta_dL,
                u * x * y,
            ]
        v = v - lr * sum(grad_terms)
        return v

    def taylor_stats_fn(p, T: TaylorCenter):
        ref, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        v = p - ref

        def d2L_ratio(*vs):
            vs = tree_map(lambda v: v - u * (u @ v), vs)
            vs = tree_map(lambda v: v / jla.norm(v), vs)
            return jla.norm(loss.D(ref, 2, *vs))

        def d3L_ratio(*vs):
            vs = tree_map(lambda v: v - u * (u @ v), vs)
            vs = tree_map(lambda v: v / jla.norm(v), vs)
            return jla.norm(loss.D(ref, 3, *vs))

        output = dict(
            L=loss.value(p),
            S_taylor=S + dS @ v,
            S_avg=S + dS_perp @ v,
            S_exact=loss.eig(p, T.u)[0],
            x=u @ v,
            # d2L_v_ratio=d2L_ratio(v),
            # d2L_dL_ratio=d2L_ratio(dL),
            # d2L_dS_ratio=d2L_ratio(dS),
            d2L_v_v_ratio=d2L_ratio(v, v),
            # d2L_v_dL_ratio=d2L_ratio(v, dL),
            # d2L_v_dS_ratio=d2L_ratio(v, dS),
            # d2L_dL_dL_ratio=d2L_ratio(dL, dL),
            # d2L_dL_dS_ratio=d2L_ratio(dL, dS),
            # d2L_dS_dS_ratio=d2L_ratio(dS, dS),
            d3L_v_ratio=d3L_ratio(v, v),
            d3L_dL_ratio=d3L_ratio(dL, dL),
            dist=jla.norm(v),
        )
        return output

    def gf_stats_fn(E: EigenSystem, T: TaylorCenter):
        output = dict(
            L=loss.value(E.p),
            S_exact=E.S,
            dist=jla.norm(E.p - T.p),
        )
        return output

    def constants_fn(T: TaylorCenter):
        ref, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        alpha = -dL @ dS_perp
        beta = dS_perp @ dS_perp
        gamma = alpha / (jla.norm(dL) * jla.norm(dS_perp))
        delta = jnp.sqrt(2 * alpha / beta)
        dL_perp = dL - u * (u @ dL) - dS_perp * (dS_perp @ dL) / beta
        newton_x = u @ dL / S
        output = dict(
            L=L,
            S=S,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            newton_x=newton_x,
            dL_norm=jla.norm(dL),
            dL_perp_norm=jla.norm(dL_perp),
        )
        return output

    def update_betas(betas, T: TaylorCenter, step, save_step=False):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        for t, (rot, l) in betas.items():
            l.append(dS_perp @ rot)
            rot = rot - u * (u @ rot)
            rot = rot - lr * loss.D(T.p, 2, rot)
            betas[t] = (rot, l)
        if save_step:
            betas[step] = (dS_perp, [])
        return betas

    def slow_stats_fn(T: TaylorCenter):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        alpha = -dL @ dS
        beta = dS_perp @ dS_perp
        delta = jnp.sqrt(2 * alpha / beta)
        xs = jnp.linspace(-5 * delta, 5 * delta, 100)
        F = jnp.array([loss.value(p + x * u) for x in xs])
        rho3 = power_iter(lambda v: loss.D(p, 3, v, v), u)[0]
        rho4 = power_iter(lambda v: loss.D(p, 4, v, v, v), u)[0]
        return dict(F=(xs, F), rho3=rho3, rho4=rho4)

    ref_u = ref_U[:, 0]
    dagger = project(taylor_center(p, ref_u))
    gd = p
    gf = EigenSystem(p, 2 / lr, ref_u)
    v = gd - dagger.p
    save_list = []
    betas = {}
    slow_stats = {}
    pbar = trange(steps)
    for i in pbar:
        stats = dict(
            gd=taylor_stats_fn(gd, dagger),
            pred=taylor_stats_fn(dagger.p + v, dagger),
            gf=gf_stats_fn(gf, dagger),
            dagger=constants_fn(dagger),
        )
        save_list.append(stats)
        if i % 50 == 0:
            ref_U = ref_U.at[:, 0].set(dagger.u)
            eigs, ref_U = check_eigengap(dagger, ref_U)
            betas = update_betas(betas, dagger, step=i, save_step=True)
            slow_stats[i] = slow_stats_fn(dagger)
            if eigs[1] > 1.9 / lr:
                print(f"Step = {i}, eigenvalues at theta dagger: {eigs}")
                print("Second largest eigenvalue exceeded threshold")
                break
            dynamics = tree_stack(save_list)
            dynamics["beta"] = {key: jnp.array(val[1]) for key, val in betas.items()}
            dynamics["slow_stats"] = slow_stats
            tree_save(dynamics, save_dir / "dynamics.pytree", overwrite=True)
        else:
            betas = update_betas(betas, dagger, step=i, save_step=False)

        def timed(f):
            @wraps(f)
            def _f(*args, **kwargs):
                start_time = time.perf_counter()
                out = f(*args, **kwargs)
                # tree_map(lambda x: x.block_until_ready(), out)
                end_time = time.perf_counter()
                return out, end_time - start_time

            return _f

        gd, gd_time = timed(gd_step)(gd)
        gf, gf_time = timed(gf_step)(gf)
        v, pred_time = timed(predicted_step)(v, dagger)
        dagger, dagger_time = timed(constrained_step)(dagger)
        pbar.set_description(
            f"gd: {gd_time:.1f}, gf: {gf_time:.1f}, pred: {pred_time:.1f}, dagger: {dagger_time:.1f}"
        )
    output = tree_stack(save_list)
    output["beta"] = {key: jnp.array(val[1]) for key, val in betas.items()}
    output["slow_stats"] = slow_stats
    return output
