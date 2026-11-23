"""Thermodynamic integration of the pair charging free energy in JAX (jit + vmap over replicas).

Same protocol as reference.mc.run_ti: host equilibration, insertion of a neutral pair into a cavity,
then sequential coupling windows (the configuration is carried from window to window), Delta F = trapz.
Equilibration sweep counts must be multiples of 50 (the step-adaptation block).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..reference.mc import box_length
from .core import (State, dudlambda, equilibrate, make_consts, make_state, production, set_charges)


def init_positions(key, N, c):
    nside = int(np.ceil(N ** (1 / 3)))
    g = (jnp.arange(nside) + 0.5) * c.L / nside
    P = jnp.stack(jnp.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)[:N]
    amp = jnp.minimum(0.05, 0.15 * (c.L / nside - 1.0))
    return jnp.mod(P + amp * (2.0 * jax.random.uniform(key, P.shape) - 1.0), c.L)


def insert_pair(key, pos, c):
    """Random positions for a neutral pair that overlap with neither the host nor each other."""
    def min_dist(p):
        d = pos - p
        d = d - c.L * jnp.round(d / c.L)
        return jnp.min(jnp.sqrt(jnp.sum(d * d, axis=1)))

    def cond(v):
        return ~v[3]

    def body(v):
        key, _, _, _ = v
        key, k1, k2 = jax.random.split(key, 3)
        gp = c.L * jax.random.uniform(k1, (3,))
        gm = c.L * jax.random.uniform(k2, (3,))
        dpm = gp - gm
        dpm = dpm - c.L * jnp.round(dpm / c.L)
        ok = (min_dist(gp) >= 1.0) & (min_dist(gm) >= 1.0) & (jnp.sqrt(jnp.sum(dpm * dpm)) >= 1.0)
        return key, gp, gm, ok

    init = (key, jnp.zeros(3), jnp.zeros(3), jnp.array(False))
    _, gp, gm, _ = lax.while_loop(cond, body, init)
    return gp, gm


def make_ti_one(c_M, sigma_A=4.0, lB_A=7.13961, N=192, lambdas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                host_equil=2500, pair_equil=1500, win_equil=1500, win_prod=12000, sample_every=10):
    """Returns a function key -> dict(mean_dudl, sem_dudl, beta_DF_TI) for ONE replica (vmap it over keys)."""
    for n in (host_equil, pair_equil, win_equil):
        assert n % 50 == 0, "equilibration sweeps must be multiples of 50"
    assert win_prod % sample_every == 0
    L = box_length(c_M, sigma_A, N)
    c = make_consts(L, lB_A / sigma_A)
    lam_arr = jnp.asarray(lambdas, jnp.float64)
    nsamp = win_prod // sample_every

    def one(key):
        k_pos, k_q, k_host, k_ins, k_pair, k_win = jax.random.split(key, 6)
        qhost = jax.random.permutation(k_q, jnp.concatenate([jnp.ones(N // 2), -jnp.ones(N // 2)]))
        state = make_state(init_positions(k_pos, N, c), qhost, c)
        state, step = equilibrate(state, k_host, 0.15, host_equil // 50, c)
        gp, gm = insert_pair(k_ins, state.pos, c)
        state = make_state(jnp.vstack([state.pos, gp, gm]), jnp.concatenate([qhost, jnp.zeros(2)]), c)
        state, step = equilibrate(state, k_pair, step, pair_equil // 50, c)

        def window(carry, xs):
            st, stp = carry
            lam, k = xs
            ke, kp = jax.random.split(k)
            st = set_charges(st, jnp.concatenate([qhost, jnp.stack([lam, -lam])]))
            st, stp = equilibrate(st, ke, stp, win_equil // 50, c)
            st, vals = production(st, kp, stp, lam, nsamp, sample_every, c)
            nb = 10
            usable = (nsamp // nb) * nb
            blocks = vals[:usable].reshape(nb, -1).mean(axis=1)
            sem = jnp.std(blocks, ddof=1) / jnp.sqrt(nb)
            return (st, stp), (jnp.mean(vals), sem)

        keys = jax.random.split(k_win, len(lambdas))
        _, (means, sems) = lax.scan(window, (state, step), (lam_arr, keys))
        dF = jnp.sum(0.5 * (means[1:] + means[:-1]) * jnp.diff(lam_arr))
        w = jnp.zeros(len(lambdas)).at[:-1].add(jnp.diff(lam_arr) / 2).at[1:].add(jnp.diff(lam_arr) / 2)
        dF_sem = jnp.sqrt(jnp.sum((w * sems) ** 2))
        return dict(mean_dudl=means, sem_dudl=sems, beta_DF_TI=dF, beta_DF_TI_sem=dF_sem)

    return one, c


def ti_batch(c_M, seeds, **kw):
    """Run len(seeds) independent replicas in one compiled, vmapped call. Returns numpy results per replica."""
    one, c = make_ti_one(c_M, **kw)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, jnp.int64))
    out = jax.jit(jax.vmap(one))(keys)
    out = {k: np.asarray(v) for k, v in out.items()}
    out.update(c_M=c_M, L=float(c.L), Gamma=float(c.Gamma), ln_gamma_charge=out["beta_DF_TI"] / 2,
               ln_gamma_charge_sem=out["beta_DF_TI_sem"] / 2)
    return out
