import numpy as np
from rpm_mc.reference.ewald import ewald_energy, components, move_delta

MADELUNG_NACL = 1.7475645946  # relative to the nearest-neighbour distance


def rocksalt(n, L=1.0):
    m = 2 * n
    d = L / m
    idx = np.array([(i, j, k) for i in range(m) for j in range(m) for k in range(m)])
    return d * idx, np.where(idx.sum(1) % 2 == 0, 1.0, -1.0), d


def random_ions(N, L, rng, dmin=1.0):
    pos = []
    while len(pos) < N:
        p = rng.random(3) * L
        if all(np.linalg.norm((p - x) - L * np.round((p - x) / L)) > dmin for x in pos):
            pos.append(p)
    q = np.array([1.0] * (N // 2) + [-1.0] * (N // 2))
    return np.array(pos), q


def test_madelung_constant():
    for n in (1, 2):
        pos, q, d = rocksalt(n)
        U = ewald_energy(pos, q, 1.0, 1.0, alpha_factor=7.0, kmax=10, nimg=1)
        assert abs(U / len(q) - (-MADELUNG_NACL / (2 * d))) < 1e-6 * abs(U / len(q))


def test_independent_of_splitting_parameter():
    rng = np.random.default_rng(1)
    L = 6.0
    pos, q = random_ions(40, L, rng)
    u = [ewald_energy(pos, q, L, 1.8, alpha_factor=a, kmax=12, nimg=1) for a in (6.0, 7.0, 8.0)]
    assert max(u) - min(u) < 1e-6


def test_translation_and_permutation_invariance():
    rng = np.random.default_rng(2)
    L = 6.0
    pos, q = random_ions(40, L, rng)
    u0 = ewald_energy(pos, q, L, 1.8)
    assert abs(ewald_energy((pos + [0.3, -1.1, 2.2]) % L, q, L, 1.8) - u0) < 1e-9
    p = rng.permutation(len(q))
    assert abs(ewald_energy(pos[p], q[p], L, 1.8) - u0) < 1e-9


def test_incremental_move_matches_full_recompute():
    rng = np.random.default_rng(3)
    L = 6.0
    pos, q = random_ions(40, L, rng)
    for _ in range(20):
        i = rng.integers(len(q))
        new = (pos[i] + rng.normal(scale=0.4, size=3)) % L
        dU, overlap, _ = move_delta(i, new, pos, q, L, 1.8)
        if overlap:
            continue
        p2 = pos.copy()
        p2[i] = new
        assert abs(dU - (ewald_energy(p2, q, L, 1.8) - ewald_energy(pos, q, L, 1.8))) < 1e-9


def test_overlap_is_rejected():
    rng = np.random.default_rng(4)
    L = 6.0
    pos, q = random_ions(40, L, rng)
    _, overlap, _ = move_delta(0, pos[1] + [0.5, 0.0, 0.0], pos, q, L, 1.8)
    assert overlap
