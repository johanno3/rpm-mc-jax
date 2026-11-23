"""JAX implementation against the NumPy reference. Skipped automatically when JAX is not installed."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from rpm_mc.jaxmc import core  # noqa: E402
from rpm_mc.jaxmc.ti import ti_batch  # noqa: E402
from rpm_mc.reference.ewald import ewald_energy, move_delta  # noqa: E402
from rpm_mc.reference.mc import run_ti  # noqa: E402
from test_ewald_reference import random_ions  # noqa: E402
from test_mc_reference import make  # noqa: E402


def _ions(N=40, L=6.0, seed=0):
    rng = np.random.default_rng(seed)
    pos, q = random_ions(N, L, rng)
    return pos, q, L, rng


def _state_from_numpy(ens):
    consts = core.make_consts(ens.L, ens.Gamma)
    return core.make_state(ens.pos, ens.q, consts), consts


def test_energy_matches_numpy():
    pos, q, L, _ = _ions()
    consts = core.make_consts(L, 1.8)
    e = float(core.energy(jnp.asarray(pos), jnp.asarray(q), consts))
    assert abs(e - ewald_energy(pos, q, L, 1.8)) < 1e-9


def test_move_energy_change_matches_numpy():
    pos, q, L, rng = _ions()
    consts = core.make_consts(L, 1.8)
    st = core.make_state(pos, q, consts)
    for _ in range(15):
        i = int(rng.integers(len(q)))
        new = (pos[i] + rng.normal(scale=0.4, size=3)) % L
        dU, overlap, _, _ = core.propose(st, i, jnp.asarray(new), consts)
        dU_ref, overlap_ref, _ = move_delta(i, new, pos, q, L, 1.8)
        assert bool(overlap) == overlap_ref
        if not overlap_ref:
            assert abs(float(dU) - dU_ref) < 1e-9


def test_dudlambda_matches_numpy_and_autodiff():
    ens, q, rng = make(c=0.3, N=48, seed=1, lam=0.7)
    st, consts = _state_from_numpy(ens)
    ref = ens.dudlambda(0.7)
    assert abs(float(core.dudlambda(st, 0.7, consts)) - ref) < 1e-9
    qh, pos = jnp.asarray(ens.q[:-2]), jnp.asarray(ens.pos)

    def U(lam):
        return core.energy(pos, jnp.concatenate([qh, jnp.stack([lam, -lam])]), consts)

    assert abs(float(jax.grad(U)(0.7)) - ref) < 1e-8  # autodiff reproduces the analytic derivative


def test_sweeps_keep_structure_factor_and_hard_core():
    ens, q, rng = make(c=2.5, N=48, seed=2, lam=0.0)
    st, consts = _state_from_numpy(ens)
    sweep = jax.jit(core.sweep)
    for k in jax.random.split(jax.random.PRNGKey(0), 20):
        st, na, nt = sweep(st, k, 0.5, consts)
    S_full = jnp.sum(st.q[:, None] * jnp.exp(1j * (st.pos @ consts.k.T)), axis=0)
    assert float(jnp.abs(st.S - S_full).max()) < 1e-9
    pos, L = np.asarray(st.pos), ens.L
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1)) + 10 * np.eye(len(pos))
    assert r.min() >= 1.0 - 1e-12


def test_vmap_over_random_keys():
    ens, q, rng = make(c=2.5, N=48, seed=2, lam=0.0)
    st, consts = _state_from_numpy(ens)
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    out = jax.vmap(lambda k: core.sweep(st, k, 0.5, consts)[0].pos)(keys)
    assert out.shape == (3, len(ens.q), 3)
    assert not np.allclose(out[0], out[1])


def test_lambda_zero_window_mean_is_zero():
    ens, q, rng = make(c=2.5, N=48, seed=3, lam=0.0)
    st, consts = _state_from_numpy(ens)
    st, step = core.equilibrate(st, jax.random.PRNGKey(0), 0.15, 8, consts)
    st, vals = core.production(st, jax.random.PRNGKey(1), step, 0.0, 400, 4, consts)
    vals = np.asarray(vals)
    blocks = vals[:400].reshape(10, -1).mean(axis=1)
    assert abs(vals.mean()) < 4 * blocks.std(ddof=1) / np.sqrt(10)


def test_free_energy_agrees_with_numpy_reference():
    kw = dict(N=48, host_equil=300, pair_equil=150, win_equil=150, win_prod=600, sample_every=5)
    jx = ti_batch(2.5, seeds=[1, 2, 3, 4], **kw)           # four replicas in one vmapped call
    assert np.std(jx["beta_DF_TI"]) > 0                      # replicas are independent
    ref = [run_ti(2.5, seed=s, **kw) for s in (1, 2)]
    m_j = jx["ln_gamma_charge"].mean()
    s_j = np.sqrt((jx["ln_gamma_charge_sem"] ** 2).sum()) / len(jx["ln_gamma_charge"])
    m_n = np.mean([r["ln_gamma_charge"] for r in ref])
    s_n = np.sqrt(sum(r["ln_gamma_charge_sem"] ** 2 for r in ref)) / len(ref)
    assert abs(m_j - m_n) < 4 * np.hypot(s_j, s_n) + 0.02
