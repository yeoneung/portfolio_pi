# High-dimensional certification experiment

The continuous verifier closes through **50 state dimensions** for the specified locally coupled, constrained portfolio family. The two-sided PDE welfare bounds stay below 39 bp in the main cube. These are upper bounds on loss, not measurements of the actual loss.

The structured policy is distinct from the supplied seed-11 neural checkpoint. Read [METHOD.md](METHOD.md) for the model, algebraic decomposition, bounded-domain comparison, localization proof, assumptions, and reproduction commands.

## Main continuous certificates

Both methods evaluate exactly the same policy. The baseline changes only the time-dependent part of the log evaluation; both methods can optimize telescoping potentials. The baseline has the denser proposal grid. All continuous checks use the same 1/2000 repair tolerance.

| State dimensions | Spatial evaluation: CE upper (bp) | Time-only evaluation: CE upper (bp) | Accepted local cells | Verification (s) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 38.344220 | 1477.926297 | 3,320 | 0.759 |
| 5 | 38.321375 | 1458.535033 | 2,950 | 0.701 |
| 10 | 37.605506 | 1492.041900 | 3,741 | 0.807 |
| 20 | 38.167352 | 1457.694240 | 3,033 | 0.720 |
| 50 | 37.831448 | 1479.216063 | 2,898 | 0.705 |

The time column is verification only, including interval localization and the analytic lower endpoint; it is not end-to-end training time. GPU proposal fitting took approximately 0.9–2.6 s; spatial LP fitting approximately 4.4–4.8 s, versus approximately 23–26 s for the denser time-only LP. Single-machine timings are descriptive, not an asymptotic-complexity experiment.

## Stronger elementary lower bound, applied fairly to both

The Gaussian-envelope/Jensen lower endpoint is stronger than the PDE lower endpoint in four main cases. It is therefore substituted into **both** methods where helpful, while keeping each method’s independently certified policy upper endpoint. This separates the contribution of the policy evaluation from that of the optimal-value lower bound.

| State dimensions | Spatial + best lower endpoint (bp) | Time-only + best lower endpoint (bp) | Internal upper-bound reduction |
| ---: | ---: | ---: | ---: |
| 3 | 38.344220 | 577.459240 | 15.06x |
| 5 | 37.412449 | 568.989371 | 15.21x |
| 10 | 35.687779 | 715.271558 | 20.04x |
| 20 | 35.507436 | 701.762448 | 19.76x |
| 50 | 34.399529 | 695.507321 | 20.22x |

These factors compare upper bounds for our own two evaluation constructions. They are not improvements in actual investment returns and not superiority factors over an external paper.

## Boundary sensitivity: same implemented policy

The implemented policy still uses the cube of radius 1.5. Only the certification cube is reduced to radius 1; the policy and all generating coefficients are unchanged. The localization term grows with the coordinate union bound.

| State dimensions | Smaller-cube two-sided PDE bound (bp) | Smaller-cube hybrid bound (bp) | Additive factor localization bound |
| ---: | ---: | ---: | ---: |
| 10 | 51.931927 | 42.836491 | 0.00069032 |
| 20 | 66.035105 | 49.429660 | 0.00134651 |
| 50 | 105.937006 | 68.439358 | 0.00329269 |

At 50 states the smaller-cube two-sided bound exceeds the 100 bp reporting threshold. Thus favorable main-cube results must not be read as dimension-independent localization.

## Genuine state coupling, and the structural restriction

There are d state coordinates, nonzero neighbor interactions in the drift and risky premia, full-rank return predictors, and positive unspanned state covariance. The independently sampled upper-constraint frequency is 41.2–42.1%; lower constraints also activate. These are stationary physical-state diagnostics, not certified controlled-path frequencies.

The speed comes from an exact identity: the full residual is the average of cyclic (remaining time, current state, neighboring state) terms, with cancelling potentials. The verifier covers these continuous three-variable domains and every compatible clipping branch. It does not replace a high-dimensional supremum with a Monte Carlo maximum.

However, the candidate is a symmetric additive log-quadratic with 15 coefficients, asset covariance is the identity, interactions are local, and opportunities/exposure caps are normalized by sqrt(d). This is not a test of a generic dense neural network, dense state dependence, or the manuscript’s full high-dimensional Newton/policy-iteration solver.

## Interpretation

The result supports **structure-exploiting high-dimensional certification** as a credible research direction. It does not establish a general solution to the dimensionality problem or a standalone literature novelty claim. The most defensible claim is the computable certificate for this locally coupled family, including its fixed-policy evaluation and boundary-to-whole-space error conversion.

Exact coefficient and source hashes are checked before this report is generated. The standalone validation record additionally documents independent re-verification.
