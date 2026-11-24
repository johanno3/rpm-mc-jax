"""Make the validation figure: JAX replicates against the Run 20 reference data.
Usage: python validation/make_figure.py [OUT.png]        (run from the repository root; needs matplotlib)"""
import csv, glob, json, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "validation/validation_figure.png"
ref = defaultdict(list)
for r in csv.DictReader(open("validation/reference_run20_baseline.csv", newline="")):
    ref[float(r["c_M"])].append(float(r["ln_gamma_charge"]))
jax_res = defaultdict(list)
for f in sorted(glob.glob("validation/results/jax_ti_*.json")):
    for r in json.load(open(f)):
        jax_res[r["c_M"]].append(r["ln_gamma_charge"])

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
cs = np.array(sorted(ref))
a.plot(cs ** (1 / 3), [np.mean(ref[c]) for c in cs], "o", ms=3.5, color="0.55", label="Run 20 data (MATLAB, 2 replicates)")
for c, v in sorted(jax_res.items()):
    a.plot([c ** (1 / 3)] * len(v), v, ".", color="tab:red", ms=5, alpha=0.6)
a.plot([], [], ".", color="tab:red", label="JAX, 8 replicates")
a.set_xlabel(r"$c^{1/3}$ (M$^{1/3}$)")
a.set_ylabel(r"charging contribution, $\ln\gamma_{\mathrm{charge}}$")
a.legend(frameon=False, fontsize=8)
a.set_title(r"RPM, $\sigma$ = 4 Å, $\lambda_B$ = 7.14 Å, N = 192", fontsize=9)

xs, labels = [], []
for i, c in enumerate(sorted(jax_res)):
    v = np.array(jax_res[c]); r = np.array(ref[c])
    d = v - r.mean()
    b.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), d, ".", color="tab:red", alpha=0.7)
    sem = v.std(ddof=1) / np.sqrt(len(v))
    b.errorbar(i + 0.3, v.mean() - r.mean(), yerr=sem, fmt="s", color="k", ms=4, capsize=3)
    b.plot([i - 0.45] * 2, r - r.mean(), "_", color="0.4", ms=9)
    xs.append(i); labels.append(f"{c:g}")
b.axhline(0, color="0.7", lw=0.8)
b.set_xticks(xs); b.set_xticklabels(labels); b.set_xlabel("c (M)")
b.set_ylabel(r"difference from Run 20 mean in $\ln\gamma_{\mathrm{charge}}$")
b.set_title("red: JAX replicates; black: JAX mean ± SEM;\ngrey bars: the two Run 20 replicates", fontsize=8.5)
fig.tight_layout()
fig.savefig(out, dpi=200)
print("wrote", out)
