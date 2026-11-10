# Portfolio policy experiments

Reproducible constrained-portfolio experiments, fixed candidate policies,
outward-rounded residual verifiers, numerical comparisons, and regression tests.
This repository contains experimental material only: no manuscript, arXiv
upload, journal submission package, cover letter, or editorial archive.

## Experiments

| Experiment | Supplied records | Scope |
| --- | --- | --- |
| Locally coupled 3--50-state markets | `results/highdim_local/` | Continuous CE certificates, matched baselines, and boundary sensitivity |
| Heterogeneous 10--50-state markets | `results/highdim_heterogeneous/` | Distinct return predictors and evaluations, with every local cylinder verified |
| Correlated network markets with 10--50 states | `results/highdim_network/` | Risk aversion 3, an irregular factor tree, correlated assets, and all coordinate cylinders verified |
| Two-asset fixed-policy evaluation | `results/portfolio_welfare/`, `results/policy_loss/`, `results/local_policy/` | Direct, global, and local evaluation precision for the same policy |
| Frozen neural policy | `results/neural_certificate/` | Exact stored network weights and full-cylinder/whole-space verification |
| Separate neural-policy certification | `results/neural_separate_evaluation/` | A 13.345 bp whole-space bound from fixed-policy evaluation and an independent Bellman comparison |
| Financial-duality comparison | `results/neural_duality/` | Matched Monte Carlo and PDE-refinement estimates for the same network |
| Empirical portfolios | `code/empirical_src_bundle/`, `results/empirical_tables/` | Supplied monthly paths, costs, bootstrap/HAC, and subperiod statistics |

The markets and horizons differ; these are not cross-market performance rankings.
Read the experiment-specific `METHOD.md`/`REPORT.md` files and `DATA_SOURCES.md`.
The separate neural-policy proof and reproduction commands are in
`results/neural_separate_evaluation/METHOD.txt`; its original deployment rule
and network weights are unchanged.
Numerical duality estimates are distinct from continuous-time certificates.
The high-dimensional family uses local interactions and structured evaluations.

## Installation

Python 3.12 is the recorded environment. Minimal certificate dependencies:

```sh
python -m pip install -r requirements-core.txt
python run.py manifest
```

For tests, fitting, plotting, and empirical/duality calculations:

```sh
python -m pip install -r requirements.txt
```

Fitting and large Monte Carlo runs can use Torch/CUDA. Supplied candidates can
be verified without GPU training. No credentials or network data downloads are
needed to verify the supplied records.

## Verify supplied candidates

```sh
python run.py verify
python run.py verify-welfare
python run.py verify-policy-loss
python run.py verify-local-policy
python run.py verify-neural-welfare
python experiments/highdim_local_verify.py --dimensions 3 5 10 20 50
python experiments/highdim_local_verify.py --dimensions 3 5 10 20 50 --folder results/highdim_local/time_only
python experiments/highdim_local_verify.py --dimensions 10 20 50 --radius 1
python experiments/highdim_heterogeneous.py --dimensions 10 20 50
python experiments/highdim_network.py --dimensions 10 20 50
python experiments/verify_neural_separate_parallel.py results/neural_separate_evaluation/adapted_policy.json --workers 4
python experiments/verify_neural_separate_parallel.py results/neural_separate_evaluation/adapted_bellman.json --workers 4
python experiments/neural_separate_welfare.py results/neural_separate_evaluation/adapted_policy.json results/neural_separate_evaluation/adapted_bellman.json --output results/neural_separate_evaluation/whole_space_refined.json
```

Verification does not refit the generating policy. It recomputes and overwrites
its result files; global polynomial and neural checks can take several minutes.
`python run.py manifest` checks the exact supplied snapshot, so run it before
commands that regenerate tracked results. Work in a copy to preserve that
snapshot unchanged.

The certificate cores retain their exact source bytes and recorded hashes.
For scalar and two-asset benchmarks, `run.py` redirects their optional table
export to `results/tables/` without modifying numerical algorithms.

## Reports and tests

```sh
python run.py report-tables
python run.py test
python run.py empirical --update-tables
python run.py localization-budget
python run.py figures
python analysis/plot_neural_residual_envelopes.py
python analysis/report_highdim_network.py
```

Reports check candidate/source hashes where applicable. Optional numerical
LaTeX tables are generated under `results/tables/` and excluded from Git.
Selected experiment figures are supplied with their result records.
No paper build is needed.

## Fitting

```sh
python run.py fit-welfare
python run.py fit-policy-loss
python run.py fit-local-policy
python experiments/highdim_heterogeneous.py --fit --refine-failed --dimensions 10 20 50
python experiments/highdim_network.py --fit --dimensions 10 20 50
python run.py train-empirical --universe ff25 --prepare-only --device cpu
```

Fitting replaces candidates and invalidates their existing certificates. Remove
`--prepare-only` only when intentionally retraining an empirical model. Use the
individual scripts' `--help` for high-dimensional and neural fitting. Large
training and simulation runs are separate from snapshot verification.

## Contents

- `experiments/`: numerical solvers, fitting, and certificate verifiers.
- `analysis/`: reproduction entry points, financial statistics, and reports.
- `code/`: covariance-metric QP solvers and portable empirical inputs.
- `results/`: candidates, numerical outputs, certificates, and method notes.
- `tests/`: independent arithmetic, financial-unit, and reproduction checks.
- `figures/`: plotting scripts only.
- `MANIFEST.json`: exact supplied-file hashes and selected configurations.

Data provenance and interpretation are documented in `DATA_SOURCES.md` and
`COVARIANCE_INPUTS.md`. Upstream data sources retain their own terms; no
additional license is assigned by this export.
