"""NumPy reference implementation of the periodic Ewald energy of the restricted primitive model.

Conventions follow the original MATLAB workers (reduced units):
  * lengths in units of the ion diameter sigma (hard-core distance = 1), energies in kT;
  * Coulomb coupling Gamma = lambda_B / sigma, so the pair energy is Gamma q_i q_j / r;
  * cubic box of side L, neutral system (no background term needed);
  * splitting parameter alpha = alpha_factor / L (the workers use alpha_factor = 5);
  * reciprocal space: half-space integer vectors with |n_x|,|n_y|,|n_z| <= kmax (the workers use 4);
  * real space: minimum image with cutoff rc = L/2 (the workers), or a full image sum (accurate mode).

U = Gamma * (U_real + U_rec + U_self)
"""
import numpy as np
from scipy.special import erfc


def kvectors(L, alpha, kmax):
    """Half-space reciprocal vectors and their coefficients (4 pi / V) exp(-k^2 / 4 alpha^2) / k^2."""
    rng = np.arange(-kmax, kmax + 1)
    n = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    keep = (n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))
    k = (2 * np.pi / L) * n[keep]
    k2 = (k ** 2).sum(axis=1)
    coef = (4 * np.pi / L ** 3) * np.exp(-k2 / (4 * alpha ** 2)) / k2
    return k, coef


def structure_factor(pos, q, k):
    return (q[:, None] * np.exp(1j * (pos @ k.T))).sum(axis=0)


def real_space(pos, q, L, alpha, rc=None, nimg=0):
    """Real-space sum. nimg = 0: minimum image with cutoff rc (default L/2), as in the workers.
    nimg >= 1: sum over periodic images n in [-nimg, nimg]^3 with cutoff rc (default (nimg + 0.5) L)."""
    N = len(q)
    if nimg == 0:
        rc = L / 2 if rc is None else rc
        d = pos[None, :, :] - pos[:, None, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1))
        iu = np.triu_indices(N, 1)
        r, qq = r[iu], (q[:, None] * q[None, :])[iu]
        m = r < rc
        return float((qq[m] * erfc(alpha * r[m]) / r[m]).sum())
    rc = (nimg + 0.5) * L if rc is None else rc
    total = 0.0
    shifts = np.arange(-nimg, nimg + 1)
    base = pos[None, :, :] - pos[:, None, :]
    qq_full = q[:, None] * q[None, :]
    for nx in shifts:
        for ny in shifts:
            for nz in shifts:
                d = base + L * np.array([nx, ny, nz])
                r = np.sqrt((d ** 2).sum(-1))
                m = (r > 1e-12) & (r < rc)
                total += (qq_full[m] * erfc(alpha * r[m]) / r[m]).sum()
    return float(0.5 * total)


def components(pos, q, L, Gamma, alpha_factor=5.0, kmax=4, rc=None, nimg=0):
    alpha = alpha_factor / L
    k, coef = kvectors(L, alpha, kmax)
    S = structure_factor(pos, q, k)
    u_real = real_space(pos, q, L, alpha, rc=rc, nimg=nimg)
    u_rec = float((coef * np.abs(S) ** 2).sum())
    u_self = float(-alpha / np.sqrt(np.pi) * (q ** 2).sum())
    return Gamma * u_real, Gamma * u_rec, Gamma * u_self


def ewald_energy(pos, q, L, Gamma, alpha_factor=5.0, kmax=4, rc=None, nimg=0):
    """Total Coulomb energy in kT. Defaults reproduce the MATLAB workers."""
    return sum(components(pos, q, L, Gamma, alpha_factor, kmax, rc, nimg))


def move_delta(i, new_pos, pos, q, L, Gamma, alpha_factor=5.0, kmax=4, S=None):
    """Energy change for moving ion i to new_pos (worker conventions: minimum image, cutoff L/2).
    Returns (dU, overlap, S_new). overlap is True if the new position is closer than 1 to another ion."""
    alpha = alpha_factor / L
    rc = L / 2
    k, coef = kvectors(L, alpha, kmax)
    if S is None:
        S = structure_factor(pos, q, k)
    others = np.arange(len(q)) != i

    def f(p):
        d = pos[others] - p
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1))
        m = r < rc
        return r, float((q[i] * q[others][m] * erfc(alpha * r[m]) / r[m]).sum())

    r_old, e_old = f(pos[i])
    r_new, e_new = f(new_pos)
    if (r_new < 1.0).any():
        return 0.0, True, S
    S_new = S + q[i] * (np.exp(1j * (k @ new_pos)) - np.exp(1j * (k @ pos[i])))
    d_rec = float((coef * (np.abs(S_new) ** 2 - np.abs(S) ** 2)).sum())
    return Gamma * (e_new - e_old + d_rec), False, S_new
