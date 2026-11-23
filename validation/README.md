# Validation of the NumPy reference against the research data

The reference Monte Carlo (`rpm_mc.reference`) reproduces the thermodynamic-integration setup of the MATLAB
workers behind the research dataset (Nordstrand 2026, doi 10.17632/hkr6jbkypg.1, campaign
`rpm2/run20_dense-parameter`): N = 192 host ions plus an inserted +/- pair, sigma = 4 A, lambda_B = 7.14 A,
six coupling windows, ln(gamma_charge) = beta*DeltaF/2.

| c (M) | sampling | this code | Run 20 (mean of 2 replicates) | difference |
|---|---|---|---|---|
| 0.03 | short | -0.1732 +/- 0.0079 | -0.1699 | 0.4 sigma |
| 0.3 | short | -0.3802 +/- 0.0102 | -0.3995 | 1.9 sigma |
| 0.3 | production | -0.4085 +/- 0.0067 | -0.3995 | 1.3 sigma |
| 2.5 | short | -0.6892 +/- 0.0138 | -0.7207 | 2.3 sigma |
| 2.5 | production | -0.7133 +/- 0.0072 | -0.7207 | 1.0 sigma |

"Short" uses 1000/300/500/4000 sweeps (host equilibration / pair / per-window equilibration / per-window
production); "production" uses the workers' 2500/1500/1500/12000. Errors are block-averaging estimates for one run.
The two Run 20 replicates differ by about 0.003-0.009, smaller than the error bars here, so the comparison is limited
by the sampling in this code, not by the reference.

The shortened runs at 0.3 and 2.5 M sit about 2 sigma below the reference; the production-length runs do not.
The lambda = 0 window, whose exact expectation is zero, shows the same effect (short run at 2.5 M: +0.20 +/- 0.065;
production: -0.018 +/- 0.029). The test suite includes this check.

Not yet done: production-length 0.03 M, and more than one replicate per state. `slurm_numpy_ti.sh` runs 3 states x
2 replicas on a SLURM cluster (edit the account line first); `compare_with_dataset.py` compares the resulting JSON
files with the dataset.
