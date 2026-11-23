import numpy as np
from rpm_mc.reference.mc import Ensemble, initialize_positions, box_length, adapt_equilibrate, block_sem
from rpm_mc.reference.ewald import ewald_energy


def make(c=0.3, N=48, seed=0, lam=0.7):
    rng = np.random.default_rng(seed)
    L, Gamma = box_length(c, 4.0, N), 7.13961 / 4.0
    q = rng.permutation(np.r_[np.ones(N // 2), -np.ones(N // 2)])
    pos = initialize_positions(N, L, rng)
    while True:  # place the pair without hard-core overlaps
        gp, gm = L * rng.random(3), L * rng.random(3)
        pts = np.vstack([pos, gp, gm])
        d = pts[:, None, :] - pts[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1)) + 10 * np.eye(len(pts))
        if r.min() >= 1.0:
            break
    ens = Ensemble(pts, np.r_[q, lam, -lam], L, Gamma)
    return ens, q, rng


def test_box_length_matches_data():
    # Run 20 file: c = 2.5 M, sigma = 4 A, N = 192 -> L = 9.98773 (units of sigma)
    assert abs(box_length(2.5, 4.0, 192) - 9.98773) < 1e-4
    # c = 0.3 M -> 20.2491
    assert abs(box_length(0.3, 4.0, 192) - 20.2491) < 1e-3


def test_energy_matches_ewald_reference_module():
    ens, q, rng = make()
    assert abs(ens.energy() - ewald_energy(ens.pos, ens.q, ens.L, ens.Gamma)) < 1e-9


def test_dudlambda_matches_finite_difference():
    ens, q, rng = make()
    lam, h, n = 0.7, 1e-5, len(ens.q)

    def U(l):
        e2 = Ensemble(ens.pos, np.r_[q, l, -l], ens.L, ens.Gamma)
        return e2.energy()
    fd = (U(lam + h) - U(lam - h)) / (2 * h)
    assert abs(ens.dudlambda(lam) - fd) < 1e-6


def test_tracked_energy_does_not_drift_and_hard_core_holds():
    ens, q, rng = make(c=2.5, N=48)
    step = adapt_equilibrate(ens, rng, 30, 0.15)
    for _ in range(20):
        ens.sweep(rng, step)
    # tracked structure factor equals a from-scratch one
    S_full = (ens.q[:, None] * np.exp(1j * (ens.pos @ ens.k.T))).sum(0)
    assert np.abs(ens.S - S_full).max() < 1e-9
    d = ens.pos[:, None, :] - ens.pos[None, :, :]
    d -= ens.L * np.round(d / ens.L)
    r = np.sqrt((d ** 2).sum(-1)) + np.eye(len(ens.q)) * 10
    assert r.min() >= 1.0 - 1e-12


def test_neutral_hard_sphere_acceptance_is_reasonable():
    # at lambda = 0 with neutral pair the local acceptance after adaptation should be near the 0.35 target
    ens, q, rng = make(c=2.5, N=48, lam=0.0)  # dense enough that the step is not capped at 0.25 L
    # adapt every 10 sweeps (the workers adapt every 50 over thousands of sweeps) so the test stays fast
    step = adapt_equilibrate(ens, rng, 500, 0.15, adapt=10)
    na = nt = 0
    for _ in range(30):
        a, t, _, _ = ens.sweep(rng, step); na += a; nt += t
    assert 0.25 < na / nt < 0.45


def test_lambda_zero_window_mean_is_zero():
    """For a neutral inserted pair the exact dU/dlambda averages to zero (the pair particles carry no charge,
    so the potential difference between them vanishes by symmetry)."""
    ens, q, rng = make(c=2.5, N=48, lam=0.0)
    step = adapt_equilibrate(ens, rng, 300, 0.15, adapt=10)
    samples = []
    for sw in range(1, 1601):
        ens.sweep(rng, step)
        if sw % 4 == 0:
            samples.append(ens.dudlambda(0.0))
    assert abs(np.mean(samples)) < 4 * block_sem(samples)
