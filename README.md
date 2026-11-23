# rpm-mc-jax

Monte Carlo simulation of the restricted primitive model (RPM) electrolyte with periodic Ewald
summation, written first as a transparent NumPy reference and then in JAX. The reference follows
the conventions of the MATLAB workers used for the research data in Nordstrand (2026),
doi 10.17632/hkr6jbkypg.1 (reduced units, sigma = 1, energies in kT, Gamma = lambda_B / sigma).

Status: NumPy reference complete (Ewald, Metropolis moves, thermodynamic integration; 11 tests; validated against published data at 0.03, 0.3 and 2.5 M, see validation/README.md). JAX implementation (`rpm_mc.jaxmc`) added; tested against the NumPy reference in CI, production validation pending.

Planned: hard-sphere Metropolis moves, thermodynamic integration of the charging free energy,
validation against the published simulation data, then the JAX implementation (jit, vmap over
replicas and coupling windows, autodiff for dU/dlambda), CI and a container.

Run the tests: `pip install -e .[test] && pytest`  (or `python run_tests.py` without pytest).
