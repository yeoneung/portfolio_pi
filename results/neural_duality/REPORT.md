# Frozen neural-policy financial-duality comparison

## Main result

The HKW-style duality gap estimate at 2048 time steps and 5,242,880 paths is **7.5483 bp**, with a paired asymptotic 95% sampling interval **[7.0645, 8.0321] bp**. The independently discretized linear PDE gives **7.9262 bp**.

The unchanged outward-rounded whole-space upper bound is **966.1235 bp (9.662% rounded upward)**. The duality construction is sharper numerically in this example. This is not a new rigorous improvement to the 9.662% bound: Monte Carlo intervals account for sampling only, and the PDE cross-check is a mesh-refinement diagnostic.

This is a bound-gap estimate for the fixed deployed policy, not a measured distance to a known optimizer. Neither method changes the policy. The comparison supports a distinction in error coverage, not superiority over the literature in sharpness or matched-accuracy speed. All losses are relative terminal CE losses over the original three-year horizon, not annualized returns; one basis point is 0.0001 of wealth.

## Monte Carlo time refinement

| Steps | Paths | Duality gap (bp) | Sampling SE (bp) | 95% sampling interval (bp) | GPU simulation (s) |
|---:|---:|---:|---:|---:|---:|
| 512 | 5,242,880 | 7.7780 | 0.2471 | [7.2936, 8.2624] | 54.31 |
| 1024 | 5,242,880 | 7.3233 | 0.2082 | [6.9152, 7.7314] | 108.51 |
| 2048 | 5,242,880 | 7.5483 | 0.2468 | [7.0645, 8.0321] | 217.08 |

Five independent seeds (101, 202, 303, 404, 505) are retained at each resolution. Cross-resolution paths are not Brownian-nested. Timings are evaluation costs, excluding the common saved-network training. GPU table setup is separately recorded in the JSON files.

## Independent linear-PDE refinement

| Time steps | Spatial points | Primal factor | Dual factor | Duality gap (bp) | Seconds |
|---:|---:|---:|---:|---:|---:|
| 512 | 768 | 0.474380625 | 0.473877583 | 10.60419 | 0.29 |
| 1024 | 1536 | 0.474251825 | 0.473814314 | 9.22529 | 0.77 |
| 2048 | 3072 | 0.474187331 | 0.473782695 | 8.53325 | 3.07 |
| 4096 | 6144 | 0.474155061 | 0.473766890 | 8.18658 | 11.99 |
| 8192 | 6144 | 0.474138916 | 0.473758990 | 8.01297 | 23.95 |
| 16384 | 6144 | 0.474130843 | 0.473755040 | 7.92615 | 48.34 |

The PDE evaluates both moments of the same duality construction, using the direct network rather than derivative lookup. It does not compute or assume an exact optimal value. The artificial far boundary is tested separately by changing its distance; no rigorous tail-error enclosure is claimed for this numerical reference.

## Numerical sensitivity and variance checks

- 4097 versus 8193 lookup nodes, with the same random draws: signed gap change -0.00001717 bp.
- Far-domain half-width 2 versus 3 times the original half-width: signed gap change 0.00000004 bp.
- Finest-run paired-gap variance reduction relative to raw simulation: 242.21 times.
- Discrete martingale control mean 0.99996100, SE 0.00038606; theoretical mean is 1.
- Second-order plug-in bias estimate: -0.00000151 bp.
- Direct NN versus lookup, 256 steps and 8,192 paired paths: signed gap change 0.00000613 bp.
- Direct NN versus lookup, 1024 steps and 8,192 paired paths: signed gap change 0.00000634 bp.

These are numerical sensitivity checks, not uniform interpolation-error certificates or confidence intervals including discretization bias.

## Reproducibility and guarantee scope

Checkpoint SHA256: `47628e7e10fb3619a9ac44c01c35270e7074bc56e0b13e679765ebb21b9ad9d7`. All runs use float64 and TF32 is disabled. Device: NVIDIA GeForce RTX 5070. No training or policy update.

The supplied interval verifier recorded 404.21 CPU seconds. That is a different certification task and a recorded earlier run, not a matched-accuracy CPU/GPU timing experiment. The main submission PDF and its validated ZIPs are not replaced by this standalone comparison.

See [METHOD.md](METHOD.md) for equations, the unspanned-risk treatment, the discrete martingale proof, reproduction commands, and source attribution.
