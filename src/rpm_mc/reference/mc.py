"""NumPy reference Monte Carlo for the RPM charging free energy (thermodynamic integration).

Mirrors the MATLAB workers: N host ions (neutral overall) plus an inserted pair with charges
+lambda and -lambda; hard spheres of diameter 1; mixed local/global single-ion Metropolis moves with
step-size adaptation; exact Ewald dU/dlambda sampled every few sweeps; Delta F = trapz over lambda.
Reduced units (sigma = 1, kT = 1).  ln(gamma_charge) = Delta F / 2.
"""
import numpy as np
from scipy.special import erfc

from .ewald import kvectors

NA = 6.02214076e23


def box_length(c_M, sigma_A, N):
    """Box length in units of sigma for N host ions at formula-unit concentration c (mol/L)."""
    n_s = 1000.0 * NA * c_M * 1e-30          # formula units per A^3
    rho_star = 2.0 * n_s * sigma_A ** 3      # total ion number density * sigma^3
    return (N / rho_star) ** (1.0 / 3.0)


def initialize_positions(N, L, rng):
    nside = int(np.ceil(N ** (1 / 3)))
    g = (np.arange(nside) + 0.5) * L / nside
    P = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T[:N]
    assert L / nside > 1.02, "initial grid spacing too small"
    amp = min(0.05, 0.15 * (L / nside - 1))
    return (P + amp * (2 * rng.random(P.shape) - 1)) % L


class Ensemble:
    """Particles with charges q (host first, then the two pair particles), Ewald energy tracked incrementally."""

    def __init__(self, pos, q, L, Gamma, alpha_factor=5.0, kmax=4):
        self.L, self.Gamma = L, Gamma
        self.alpha, self.rc = alpha_factor / L, L / 2
        self.k, self.coef = kvectors(L, self.alpha, kmax)
        self.pos = np.array(pos, dtype=float)
        self.set_charges(q)

    def set_charges(self, q):
        self.q = np.array(q, dtype=float)
        self.eik = np.exp(1j * (self.pos @ self.k.T))
        self.S = (self.q[:, None] * self.eik).sum(axis=0)

    def _dist(self, p, i):
        d = self.pos - p
        d -= self.L * np.round(d / self.L)
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        r[i] = np.inf
        return r

    def _real(self, r, qi):
        m = r < self.rc
        return qi * float((self.q[m] * erfc(self.alpha * r[m]) / r[m]).sum())

    def try_move(self, i, new, rng):
        """Metropolis step for ion i. Returns True if accepted."""
        r_new = self._dist(new, i)
        if (r_new < 1.0).any():
            return False
        r_old = self._dist(self.pos[i], i)
        qi = self.q[i]
        e_eik = np.exp(1j * (self.k @ new))
        S_new = self.S + qi * (e_eik - self.eik[i])
        d_rec = float((self.coef * (np.abs(S_new) ** 2 - np.abs(self.S) ** 2)).sum())
        dU = self.Gamma * (self._real(r_new, qi) - self._real(r_old, qi) + d_rec)
        if dU <= 0 or rng.random() < np.exp(-dU):
            self.pos[i] = new
            self.eik[i] = e_eik
            self.S = S_new
            return True
        return False

    def sweep(self, rng, step, global_fraction=0.2):
        n = len(self.q)
        naL = ntL = naG = ntG = 0
        for i in rng.permutation(n):
            if rng.random() < global_fraction:
                new, is_global = self.L * rng.random(3), True
            else:
                new, is_global = (self.pos[i] + step * (2 * rng.random(3) - 1)) % self.L, False
            ok = self.try_move(i, new, rng)
            if is_global:
                ntG += 1; naG += ok
            else:
                ntL += 1; naL += ok
        return naL, ntL, naG, ntG

    def energy(self):
        """Total energy (kT) recomputed from scratch with the same Ewald settings (for drift checks)."""
        q, a = self.q, self.alpha
        d = self.pos[None, :, :] - self.pos[:, None, :]
        d -= self.L * np.round(d / self.L)
        r = np.sqrt((d ** 2).sum(-1))
        iu = np.triu_indices(len(q), 1)
        r, qq = r[iu], (q[:, None] * q[None, :])[iu]
        m = r < self.rc
        u_real = (qq[m] * erfc(a * r[m]) / r[m]).sum()
        u_rec = (self.coef * np.abs((q[:, None] * np.exp(1j * (self.pos @ self.k.T))).sum(0)) ** 2).sum()
        return float(self.Gamma * (u_real + u_rec - a / np.sqrt(np.pi) * (q ** 2).sum()))

    def dudlambda(self, lam):
        """Exact dU/dlambda for charges [q_host, +lam, -lam] (host charges fixed): same formula as the workers."""
        n = len(self.q)
        p, m = n - 2, n - 1
        a = self.alpha
        rp, rm = self._dist(self.pos[p], p), self._dist(self.pos[m], m)
        host = np.arange(n - 2)
        qh = self.q[host]
        mp, mm = rp[host] < self.rc, rm[host] < self.rc
        real = (qh[mp] * erfc(a * rp[host][mp]) / rp[host][mp]).sum() - (qh[mm] * erfc(a * rm[host][mm]) / rm[host][mm]).sum()
        r_pm = rp[m]
        if r_pm < self.rc:
            real += -2.0 * lam * erfc(a * r_pm) / r_pm
        dS = self.eik[p] - self.eik[m]
        rec = (self.coef * 2.0 * np.real(np.conj(self.S) * dS)).sum()
        self_d = -a / np.sqrt(np.pi) * 4.0 * lam
        return float(self.Gamma * (real + rec + self_d))


def adapt_equilibrate(ens, rng, nsw, step, target=0.35, adapt=50, gf=0.2, max_step=None):
    """Equilibration with the workers' step adaptation (x/÷1.10 outside target +/- 0.05)."""
    max_step = 0.25 * ens.L if max_step is None else max_step
    ab = tb = 0
    for sw in range(1, nsw + 1):
        naL, ntL, _, _ = ens.sweep(rng, step, gf)
        ab += naL; tb += ntL
        if sw % adapt == 0:
            ac = ab / max(tb, 1)
            if ac > target + 0.05:
                step = min(max_step, step * 1.10)
            elif ac < target - 0.05:
                step = max(0.01, step / 1.10)
            ab = tb = 0
    return step


def block_sem(x, nblocks=10):
    x = np.asarray(x, float)
    n = len(x) // nblocks * nblocks
    if n < nblocks:
        return float("nan")
    b = x[:n].reshape(nblocks, -1).mean(axis=1)
    return float(b.std(ddof=1) / np.sqrt(nblocks))


def run_ti(c_M, sigma_A=4.0, lB_A=7.13961, N=192, seed=0, lambdas=None,
           host_equil=2500, pair_equil=1500, win_equil=1500, win_prod=12000, sample_every=10, verbose=False):
    """Charging free energy of an inserted ion pair. Returns dict with ln_gamma_charge = beta*DeltaF/2."""
    rng = np.random.default_rng(seed)
    lambdas = np.linspace(0, 1, 6) if lambdas is None else np.asarray(lambdas, float)
    L, Gamma = box_length(c_M, sigma_A, N), lB_A / sigma_A
    qhost = rng.permutation(np.r_[np.ones(N // 2), -np.ones(N // 2)])
    ens = Ensemble(initialize_positions(N, L, rng), qhost, L, Gamma)
    step = adapt_equilibrate(ens, rng, host_equil, 0.15)
    # neutral hard-sphere pair in a cavity of the host
    for _ in range(200000):
        gp, gm = L * rng.random(3), L * rng.random(3)
        d = ens.pos - gp; d -= L * np.round(d / L)
        e = ens.pos - gm; e -= L * np.round(e / L)
        dpm = gp - gm; dpm -= L * np.round(dpm / L)
        if (np.linalg.norm(d, axis=1) >= 1).all() and (np.linalg.norm(e, axis=1) >= 1).all() and np.linalg.norm(dpm) >= 1:
            break
    else:
        raise RuntimeError("no cavity for the neutral pair")
    ens = Ensemble(np.vstack([ens.pos, gp, gm]), np.r_[qhost, 0.0, 0.0], L, Gamma)
    step = adapt_equilibrate(ens, rng, pair_equil, step)
    means, sems = [], []
    for lam in lambdas:
        ens.set_charges(np.r_[qhost, lam, -lam])
        step = adapt_equilibrate(ens, rng, win_equil, step)
        samples = []
        for sw in range(1, win_prod + 1):
            ens.sweep(rng, step)
            if sw % sample_every == 0:
                samples.append(ens.dudlambda(lam))
        means.append(float(np.mean(samples)))
        sems.append(block_sem(samples))
        if verbose:
            print(f"  lambda={lam:.2f}  <dU/dlambda>={means[-1]:+.4f} +/- {sems[-1]:.4f}  step={step:.3f}", flush=True)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    dF = float(trapz(means, lambdas))
    # propagate block SEMs through the trapezoid weights
    w = np.zeros(len(lambdas)); w[:-1] += np.diff(lambdas) / 2; w[1:] += np.diff(lambdas) / 2
    dF_sem = float(np.sqrt(((w * np.array(sems)) ** 2).sum()))
    return dict(c_M=c_M, L=L, Gamma=Gamma, lambdas=lambdas, mean_dudl=means, sem_dudl=sems,
                beta_DF_TI=dF, beta_DF_TI_sem=dF_sem, ln_gamma_charge=dF / 2, ln_gamma_charge_sem=dF_sem / 2)
