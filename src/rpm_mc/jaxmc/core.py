"""JAX implementation of the RPM Ewald Monte Carlo core.

Mirrors rpm_mc.reference (NumPy) function by function, in double precision:
  energy            <-> reference.ewald.ewald_energy        (minimum-image real space, half-space k-vectors)
  propose           <-> reference.ewald.move_delta          (incremental energy change of one ion move)
  sweep             <-> reference.mc.Ensemble.sweep         (mixed local/global Metropolis sweep)
  dudlambda         <-> reference.mc.Ensemble.dudlambda     (exact dU/dlambda of the inserted pair)
  equilibrate       <-> reference.mc.adapt_equilibrate      (step adaptation every 50 sweeps)
All control flow uses lax primitives, so whole simulations can be jit-compiled and vmapped over replicas.
Reduced units: sigma = 1, kT = 1, Gamma = lambda_B / sigma.
"""
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import erfc

jax.config.update("jax_enable_x64", True)


class Consts(NamedTuple):
    L: jax.Array
    Gamma: jax.Array
    alpha: jax.Array
    rc: jax.Array
    k: jax.Array      # (nk, 3) half-space reciprocal vectors
    coef: jax.Array   # (nk,)  (4 pi / V) exp(-k^2 / 4 alpha^2) / k^2


class State(NamedTuple):
    pos: jax.Array    # (N, 3)
    q: jax.Array      # (N,)
    eik: jax.Array    # (N, nk) exp(i k . r_j), kept up to date
    S: jax.Array      # (nk,)   structure factor sum_j q_j exp(i k . r_j)


@lru_cache(maxsize=None)
def _half_space(kmax):
    rng = np.arange(-kmax, kmax + 1)
    n = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    keep = (n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
    return n[keep].astype(np.float64)


def make_consts(L, Gamma, alpha_factor=5.0, kmax=4):
    L = jnp.asarray(L, jnp.float64)
    Gamma = jnp.asarray(Gamma, jnp.float64)
    alpha = alpha_factor / L
    k = (2 * jnp.pi / L) * jnp.asarray(_half_space(kmax))
    k2 = jnp.sum(k * k, axis=1)
    coef = (4 * jnp.pi / L ** 3) * jnp.exp(-k2 / (4 * alpha ** 2)) / k2
    return Consts(L, Gamma, alpha, L / 2, k, coef)


def make_state(pos, q, c):
    pos = jnp.asarray(pos, jnp.float64)
    q = jnp.asarray(q, jnp.float64)
    eik = jnp.exp(1j * (pos @ c.k.T))
    return State(pos, q, eik, jnp.sum(q[:, None] * eik, axis=0))


def set_charges(state, q):
    q = jnp.asarray(q, jnp.float64)
    return State(state.pos, q, state.eik, jnp.sum(q[:, None] * state.eik, axis=0))


def _abs2(z):
    return z.real ** 2 + z.imag ** 2


def _dist(pos, p, i, c):
    """Minimum-image distances from point p to all ions; entry i (the moving ion itself) is set to infinity."""
    d = pos - p
    d = d - c.L * jnp.round(d / c.L)
    r = jnp.sqrt(jnp.sum(d * d, axis=1))
    return r.at[i].set(jnp.inf)


def _real(r, q, qi, c):
    m = r < c.rc
    rs = jnp.where(m, r, 1.0)
    return qi * jnp.sum(jnp.where(m, q * erfc(c.alpha * rs) / rs, 0.0))


def energy(pos, q, c):
    """Total Coulomb energy (kT), same conventions as reference.ewald.ewald_energy (default settings)."""
    n = q.shape[0]
    d = pos[None, :, :] - pos[:, None, :]
    d = d - c.L * jnp.round(d / c.L)
    eye = jnp.eye(n, dtype=bool)
    r = jnp.sqrt(jnp.where(eye, 1.0, jnp.sum(d * d, axis=-1)))
    m = (~eye) & (r < c.rc)
    u_real = 0.5 * jnp.sum(jnp.where(m, q[:, None] * q[None, :] * erfc(c.alpha * r) / r, 0.0))
    S = jnp.sum(q[:, None] * jnp.exp(1j * (pos @ c.k.T)), axis=0)
    u_rec = jnp.sum(c.coef * _abs2(S))
    u_self = -c.alpha / jnp.sqrt(jnp.pi) * jnp.sum(q ** 2)
    return c.Gamma * (u_real + u_rec + u_self)


def propose(state, i, new, c):
    """Energy change for moving ion i to `new`. Returns (dU, overlap, eik_new, S_new)."""
    r_new = _dist(state.pos, new, i, c)
    r_old = _dist(state.pos, state.pos[i], i, c)
    overlap = jnp.any(r_new < 1.0)
    qi = state.q[i]
    e_new = jnp.exp(1j * (c.k @ new))
    S_new = state.S + qi * (e_new - state.eik[i])
    d_rec = jnp.sum(c.coef * (_abs2(S_new) - _abs2(state.S)))
    dU = c.Gamma * (_real(r_new, state.q, qi, c) - _real(r_old, state.q, qi, c) + d_rec)
    return dU, overlap, e_new, S_new


def move(state, i, new, u_acc, c):
    dU, overlap, e_new, S_new = propose(state, i, new, c)
    accept = (~overlap) & ((dU <= 0.0) | (u_acc < jnp.exp(-dU)))
    pos = state.pos.at[i].set(jnp.where(accept, new, state.pos[i]))
    eik = state.eik.at[i].set(jnp.where(accept, e_new, state.eik[i]))
    S = jnp.where(accept, S_new, state.S)
    return State(pos, state.q, eik, S), accept


def sweep(state, key, step, c, global_fraction=0.2):
    """One sweep (every ion proposed once, random order). Returns (state, local accepted, local attempted)."""
    n = state.q.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    order = jax.random.permutation(k1, n)
    u_type = jax.random.uniform(k2, (n,))
    u_prop = jax.random.uniform(k3, (n, 3))
    u_acc = jax.random.uniform(k4, (n,))

    def body(carry, xs):
        st, na, nt = carry
        i, ut, up, ua = xs
        is_global = ut < global_fraction
        new = jnp.where(is_global, c.L * up, jnp.mod(st.pos[i] + step * (2.0 * up - 1.0), c.L))
        st, acc = move(st, i, new, ua, c)
        loc = ~is_global
        return (st, na + (acc & loc).astype(jnp.int32), nt + loc.astype(jnp.int32)), None

    zero = jnp.zeros((), jnp.int32)
    (state, na, nt), _ = lax.scan(body, (state, zero, zero), (order, u_type, u_prop, u_acc))
    return state, na, nt


def dudlambda(state, lam, c):
    """Exact dU/dlambda for charges [q_host, +lam, -lam] (the last two ions are the inserted pair)."""
    n = state.q.shape[0]
    p, m = n - 2, n - 1
    rp = _dist(state.pos, state.pos[p], p, c)
    rm = _dist(state.pos, state.pos[m], m, c)
    host = jnp.arange(n) < n - 2

    def phi(r):
        mk = host & (r < c.rc)
        rs = jnp.where(mk, r, 1.0)
        return jnp.sum(jnp.where(mk, state.q * erfc(c.alpha * rs) / rs, 0.0))

    r_pm = rp[m]
    mk = r_pm < c.rc
    rs = jnp.where(mk, r_pm, 1.0)
    real = phi(rp) - phi(rm) - 2.0 * lam * jnp.where(mk, erfc(c.alpha * rs) / rs, 0.0)
    dS = state.eik[p] - state.eik[m]
    rec = jnp.sum(c.coef * 2.0 * jnp.real(jnp.conj(state.S) * dS))
    self_d = -c.alpha / jnp.sqrt(jnp.pi) * 4.0 * lam
    return c.Gamma * (real + rec + self_d)


def equilibrate(state, key, step, nblocks, c, block=50, target=0.35):
    """nblocks * block sweeps; the local step is adapted after every block (x1.10 / /1.10 outside target +/- 0.05)."""
    max_step = 0.25 * c.L
    step = jnp.asarray(step, jnp.float64)

    def one_block(carry, k):
        st, stp = carry

        def one_sweep(cr, kk):
            s, na, nt = cr
            s, a, t = sweep(s, kk, stp, c)
            return (s, na + a, nt + t), None

        zero = jnp.zeros((), jnp.int32)
        (st, na, nt), _ = lax.scan(one_sweep, (st, zero, zero), jax.random.split(k, block))
        ac = na / jnp.maximum(nt, 1)
        stp = jnp.where(ac > target + 0.05, jnp.minimum(max_step, stp * 1.10),
                        jnp.where(ac < target - 0.05, jnp.maximum(0.01, stp / 1.10), stp))
        return (st, stp), None

    (state, step), _ = lax.scan(one_block, (state, step), jax.random.split(key, nblocks))
    return state, step


def production(state, key, step, lam, nsamples, every, c):
    """nsamples * every sweeps at fixed step; dU/dlambda is sampled after every `every` sweeps."""
    def one_sample(st, k):
        def one_sweep(s, kk):
            s, _, _ = sweep(s, kk, step, c)
            return s, None

        st, _ = lax.scan(one_sweep, st, jax.random.split(k, every))
        return st, dudlambda(st, lam, c)

    return lax.scan(one_sample, state, jax.random.split(key, nsamples))
