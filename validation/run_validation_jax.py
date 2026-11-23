"""Run the JAX thermodynamic integration at selected states; replicas run in one vmapped call.
Usage: python validation/run_validation_jax.py OUT.json [--c 0.3 --c 2.5 ...] [--replicas 2] [--seed 1] [--production]
Output entries have the same fields as run_validation.py, so compare_with_dataset.py works on both."""
import argparse, json, sys, time
sys.path.insert(0, "src")
from rpm_mc.jaxmc.ti import ti_batch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--c", type=float, action="append")
ap.add_argument("--replicas", type=int, default=2)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--production", action="store_true")
a = ap.parse_args()
kw = dict(host_equil=2500, pair_equil=1500, win_equil=1500, win_prod=12000) if a.production else \
     dict(host_equil=1000, pair_equil=300, win_equil=500, win_prod=4000)
results = []
for c in (a.c or [0.3, 0.03, 2.5]):
    t = time.time()
    out = ti_batch(c, seeds=list(range(a.seed, a.seed + a.replicas)), N=192, **kw)
    secs = time.time() - t
    for j in range(a.replicas):
        results.append(dict(engine="jax", c_M=c, seed=a.seed + j, seconds=secs,
                            ln_gamma_charge=float(out["ln_gamma_charge"][j]),
                            ln_gamma_charge_sem=float(out["ln_gamma_charge_sem"][j]),
                            beta_DF_TI=float(out["beta_DF_TI"][j]), mean_dudl=out["mean_dudl"][j].tolist(),
                            sem_dudl=out["sem_dudl"][j].tolist()))
        print(f"c={c} M seed={a.seed + j}: ln_gamma_charge = {results[-1]['ln_gamma_charge']:+.4f} "
              f"+/- {results[-1]['ln_gamma_charge_sem']:.4f}   ({secs:.0f} s for {a.replicas} replicas)", flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
