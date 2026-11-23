"""Compare reference-TI result files with the Run 20 baseline values in the research dataset.
Usage: python validation/compare_with_dataset.py DATASET_DIR validation/results/*.json
DATASET_DIR contains tables/ti_shell.csv (Mendeley record doi 10.17632/hkr6jbkypg.1, folder full_simulation_data)."""
import csv, json, sys
from collections import defaultdict
import numpy as np

ds, files = sys.argv[1], sys.argv[2:]
ref = defaultdict(list)
for r in csv.DictReader(open(f"{ds}/tables/ti_shell.csv", newline="")):
    if (r["campaign"] == "rpm2/run20_dense-parameter" and abs(float(r["lB_A"]) - 7.13961) < 1e-3
            and float(r["sigma_A"]) == 4.0 and r["is_first_for_run_key"] == "1"):
        ref[round(float(r["c_M"]), 5)].append(float(r["lngamma_charge"]))
print(f"{'file':44s} {'c (M)':>6s} {'sweeps':>9s} {'ln gamma_charge':>17s} {'Run 20 mean':>12s} {'diff/sigma':>10s}")
for f in files:
    for r in json.load(open(f)):
        c = round(r["c_M"], 5)
        m = float(np.mean(ref[c]))
        d = (r["ln_gamma_charge"] - m) / r["ln_gamma_charge_sem"]
        print(f"{f.split('/')[-1]:44s} {c:6.2f} {'':>9s} {r['ln_gamma_charge']:+8.4f} +/- {r['ln_gamma_charge_sem']:.4f} {m:+12.4f} {d:+10.1f}")
