"""Run the NumPy reference TI at selected states and write results as JSON.
Usage: python validation/run_validation.py OUT.json [--production] [--c=2.5] [--seed=1]
Default: shortened sampling (fast check). --production: the workers' sweep counts."""
import json, sys, time
sys.path.insert(0, "src")
from rpm_mc.reference.mc import run_ti

out = sys.argv[1]
prod = "--production" in sys.argv
kw = dict(host_equil=2500, pair_equil=1500, win_equil=1500, win_prod=12000) if prod else \
     dict(host_equil=1000, pair_equil=300, win_equil=500, win_prod=4000)
results = []
states = [float(a.split("=")[1]) for a in sys.argv if a.startswith("--c=")] or [0.3, 0.03, 2.5]
for c in states:
    t = time.time()
    seed = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--seed=')), int(c * 1000) + 7))
    r = run_ti(c, sigma_A=4.0, lB_A=7.13961, N=192, seed=seed, verbose=True, **kw)
    r['seed'] = seed
    r["seconds"] = time.time() - t
    r = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
    results.append(r)
    print(f"c={c} M  ln_gamma_charge = {r['ln_gamma_charge']:.4f} +/- {r['ln_gamma_charge_sem']:.4f}  ({r['seconds']:.0f} s)", flush=True)
    json.dump(results, open(out, "w"), indent=1)
