# Matched financial-duality comparison

The fixed input is `results/neural_certificate/seed_11_final.json`, SHA256
`47628e7e10fb3619a9ac44c01c35270e7074bc56e0b13e679765ebb21b9ad9d7`.
There is no training and no change to its mathematical greedy policy. Monte
Carlo uses time discretization and, optionally, a lookup approximation to the
stored network derivative. Neither approximation is an interval certificate.

## Source and adaptation

Haugh, Kogan and Wang (2006), *Evaluating Portfolio Policies: A Duality Approach*,
Operations Research 54(3), 405--418, equations (7)--(9), (15) and Section 5.
[Author-hosted article](https://web.mit.edu/wangj/www/pap/HaughKoganWang06.pdf).
This implements their price-density performance-bound construction for our
market, not a reproduction of their calibration or their reported experiments.
The variance-reduction and scalar-state implementation below are adaptations.

Write dZ=b dt+Gamma.dW+beta dW0, beta^2=Q-|Gamma|^2, with the last Brownian
unspanned by the 30 risky assets. For |p_i|<=cap_i, set nu=theta_risky-lambda
and delta=sum cap_i |nu_i|. The deflator obeys

    dD/D = -(r+delta) dt - theta_risky.dW - theta_0 dW0, D(0)=1.

For any admissible strategy p, XD has nonpositive drift
`-XD*(delta+p.nu)`. As a nonnegative local supermartingale it is a
supermartingale. Cauchy--Schwarz therefore gives, at unit initial wealth,

    g* = inf E[1/X_T] >= (E[sqrt(D_T)])^2,
    relative CE loss <= 1 - (E[sqrt(D_T)])^2 / E[1/X_T].

The directions are specific to gamma=2: utility is -1/X and CE is 1/g.
Inside the original domain, h=log(u_NN) and

    p = clamp((lambda+Gamma*h_z)/2),
    theta_risky = 2*p-Gamma*h_z, theta_0 = -beta*h_z.

This is the CRRA specialization of HKW equation (15). The box support is
finite for every nu, so no projection to a support-function domain is needed.
Inside, the coefficients are bounded; outside we use the myopic constrained
dual price, theta_risky=2*clamp(lambda/2), theta_0=0. Thus theta is globally
bounded and delta has at most linear growth in the Gaussian OU state.
The deployed primal strategy is zero outside and resumes the unchanged neural
greedy strategy on re-entry. The dual price candidate need not equal the
deployed primal strategy outside the computational domain.

## GPU simulation and control variate

All computations use float64, without TF32. The exact OU transition is coupled
to the same Brownian increment used by wealth and the deflator. The component
orthogonal to the state Brownian requires only one additional normal: inside,
theta_perp=2*p_perp; outside the primal exposure is zero. This retains all 30
asset terms and the unspanned variance Q-|Gamma|^2, rather than deleting risk.

We freeze predictable coefficients at each time step and update log wealth and
log density exponentially. This preserves positivity but remains a time
approximation to the continuous feedback strategy. At a frozen step, with
M=XD and a=delta+p.nu>=0, the discrete identity is exact:

    E[M_(j+1) | F_j] = M_j exp(-a_j dt).
    K = M_T + sum_j M_j (1-exp(-a_j dt)), E[K]=1.

Thus P_cv=1/X_T-g0*(K-1) and S_cv=sqrt(D_T)-sqrt(g0)*(K-1) have the correct
discrete means. The coefficient g0=u_NN(T,z_c) is fixed before sampling; it
is neither fitted on the reporting sample nor treated as a true value.
For an exact optimal policy/value the associated martingale tracks terminal
marginal utility, motivating the variance reduction. We retain raw estimates
and the martingale-mean check as diagnostics.

Reported gaps use 1-mean(S_cv)^2/mean(P_cv). The ratio/square plug-in has a
finite-sample bias; paired delta-method normal intervals are asymptotic
sampling intervals, not continuous-time upper certificates. There is no
clipping of negative estimates or selective exclusion of random seeds.
Multiple time resolutions, lookup checks, and direct-network runs assess
numerical sensitivity separately. Seeds repeat across resolutions but their
Brownian increments are not nested; convergence is not inferred from paired
differences. The five seeds are independent within each resolution.

## Scope of the comparison

The existing 9.662% whole-space upper bound includes continuous-domain
residuals, artificial-boundary error and outward rounding. The duality formula
is itself a continuous-time bound, but this Monte Carlo implementation has
sampling and time/lookup discretization error. Its numerical sharpness and cost
must be shown alongside, not equated with, the deterministic guarantee.
This comparison does not implement Picarelli--Reisinger's separate
Markov-chain discretization-error theorem.

## Reproduction

From the package root, with Python 3.12, NumPy, SciPy, mpmath, PyTorch and pytest:

```sh
python -B experiments/neural_duality_comparison.py --steps 512 1024 2048 --paths 1048576 --seeds 101 202 303 404 505 --space-points 8193 --output results/neural_duality/comparison_large.json
python -B experiments/neural_duality_pde.py --steps 512 1024 2048 4096 8192 --space 128 256 512 1024 1024
python -B experiments/neural_duality_pde.py --steps 16384 --space 1024 --output results/neural_duality/pde_time_refinement.json
python -B experiments/neural_duality_pde.py --steps 16384 --space 1024 --outer 2 --output results/neural_duality/pde_domain_check.json
python -B experiments/neural_duality_comparison.py --steps 256 1024 --paths 8192 --seeds 707 --direct --output results/neural_duality/direct_check.json
python -B experiments/neural_duality_comparison.py --steps 256 1024 --paths 8192 --seeds 707 --space-points 8193 --output results/neural_duality/direct_lookup_check.json
python -B experiments/neural_duality_comparison.py --steps 2048 --paths 65536 --seeds 707 --space-points 4097 --output results/neural_duality/lookup_refinement.json
python -B experiments/neural_duality_comparison.py --steps 2048 --paths 65536 --seeds 707 --space-points 8193 --output results/neural_duality/lookup_fine_check.json
python -B analysis/report_neural_duality.py
python -B -m pytest -p no:cacheprovider tests/test_neural_duality.py -q
```

The simulation defaults to CUDA. The direct/lookup sensitivity runs have
identical seeds and path counts, hence identical normal draws. The PDE solve
uses the direct neural derivative at every coefficient node and CPU banded
linear solves, with batched network/coefficient evaluation on GPU. Its
cell-centered spatial grid places the discontinuous outside-policy boundary
between nodes. The extra domain test holds the spatial step and time step
fixed while moving the far numerical boundary.

The supplied interval bounds can also be reproduced without GPU:

```sh
python -B experiments/neural_log_barrier.py
python -B experiments/neural_welfare_bound.py
```

Each simulation file contains source/checkpoint hashes, all five seed results,
raw and variance-reduced moment estimates, and timings. Regenerating outputs
changes their elapsed-time fields and thus their file hashes, even when all
numerical quantities agree. Run the report generator after regeneration.
