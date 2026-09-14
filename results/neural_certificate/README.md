# Exact-weight neural-policy verification

This is a reproduced seed of the manuscript's constrained 30-asset, one-state
PINN-PI benchmark, not a polynomial replacement. The final binary network
weights and normalization buffers are frozen in `seed_11_final.json`.
The policy is the exact mathematical greedy map of that network, with the
stored model and constraints. GPU execution roundoff is outside this scope.

`neural_log_barrier.json` verifies a logarithmic comparison on the full closed
cylinder. `neural_welfare_bound.json` adds a global positive subsolution, a
time-dependent reaction enclosure, and an additive Lyapunov exit estimate to
certify whole-space terminal CE loss at the initial center. The outside-domain
extension is zero risky exposure; the original neural policy is used whenever
inside, including after re-entry. These bounds concern this one checkpoint.

Two-input quadratic Taylor forms and interval remainders enclose the network and
its derivatives; they do not replace its policy. Explicit binary64 dot-product
roundoff bounds and polynomial exponential remainders are included. Floating
probes only propose subdivision thresholds. Every reported residual bound uses
the separate outward full-cover enclosures, with exact dyadic slab area checks.

The separate `neural_audit.json` proves the failure of the scalar absolute-error
certificate's positive-endpoint condition. It is a necessary-condition audit,
not a lower bound on actual investment loss. The logarithmic comparison uses
a different error formula. The small bounds in `results/local_policy/` concern
a separate two-asset polynomial experiment.

From the package root, with NumPy and mpmath:

```sh
python experiments/neural_log_barrier.py
python experiments/neural_welfare_bound.py
```

With pytest additionally installed:

```sh
python -m pytest -p no:cacheprovider tests/test_neural_certificate_audit.py tests/test_neural_log_barrier.py tests/test_neural_welfare_bound.py -q
```

The scripts fail closed, check checkpoint/source hashes, and overwrite their own
result files. Verification needs neither training nor a reference PDE solution.
The recorded training statistics compare against a numerical FDM reference;
they are not used as rigorous error bounds.
