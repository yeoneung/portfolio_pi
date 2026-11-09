# Heterogeneous return predictors

This benchmark retains the cyclic factor drift, independent factor noise,
diagonal asset covariance, risk aversion 2, box constraints, one-year horizon,
and radius 3/2 of `results/highdim_local/`. The state and asset dimensions are
10, 20, and 50. The three return-predictor coefficients vary by +/-20%:

For zero-based coordinate i, set
`s_i = 2*((17*i+3) mod d)/(d-1)-1` and
`t_i = 2*((7*i+1) mod d)/(d-1)-1`. The coefficients are
`ell_i = (1+s_i/5)/5`, `a_i = (1+t_i/5)/2`, and
`b_i = (1-s_i/5)/4`. Excess returns are
`(ell_i+a_i*z_i+b_i*z_next)/sqrt(d)`. Every intercept is different.

The generating polynomial from the homogeneous example at the same dimension
is reused. The greedy trading rule uses the heterogeneous predictor. Lower and
upper evaluations have distinct coefficients at each coordinate. Each endpoint
uses one common telescoping potential from the corresponding homogeneous
candidate; independent coordinate potentials would not cancel around the cycle.

## Construction and verification

Three sequential linear programs fit each evaluation on an initial
17 x 49 x 49 time-state grid. If continuous verification fails, the proposal
grid is refined and that coordinate is refitted. The supplied 20-state candidate
uses one refinement at coordinate 16; the 50-state candidate uses refinements at
coordinates 30 and 39 (one-based indices). These refinements use 33 time nodes,
65 uniform spatial nodes, and 33 central spatial nodes before duplicate removal.
The 10-state candidate needs no refinement.

The candidate coefficients are exact multiples of 2^-36. Each local cylinder
is checked with tolerance 1/2000 by the original outward-rounded Bernstein
verifier. All compatible control-clamp branches and all trace quotients are
retained. Every cover has exact dyadic volume one. No symmetry reuse is used:
there are d separate verified cylinders, each with three variables.

The normalized whole-space residual is the average of the coordinate residuals.
Consequently the endpoint exponents use averages of the verified coordinate
corrections. The common localization bound uses the largest intercept and the
sum of the largest absolute self and neighboring predictor loadings. The state
drift, covariance, and admissible policy bounds are common across coordinates.
The finite-domain lower and upper endpoints are adjusted by this localization
bound before the CE ratio is computed with interval arithmetic.

## Results

| State dimension | Whole-space CE upper bound (bp, rounded upward) | Local cylinders | Accepted cells |
| --- | --- | --- | --- |
| 10 | 47.545 | 10 | 28693 |
| 20 | 47.854 | 20 | 45510 |
| 50 | 47.123 | 50 | 119311 |

The reported bounds concern the initial origin and the fixed deployed policy.
Candidate JSON files contain all coefficients and market parameters. Certificate
JSON files contain every local cover summary, the common localization envelope,
interval endpoints, input hashes, and timings. Verification times are measured
by serial verification of the supplied candidates; fitting is reported separately.

From the repository root:

```sh
python experiments/highdim_heterogeneous.py --dimensions 10 20 50
python -m pytest tests/test_highdim_heterogeneous.py tests/test_highdim_local.py -q
```

To regenerate evaluation proposals and verify them:

```sh
python experiments/highdim_heterogeneous.py --fit --refine-failed --dimensions 10 20 50
```

Fitting uses SciPy and the existing basis utilities; verification requires only
NumPy and mpmath. The generating policy is never retrained by these commands.
