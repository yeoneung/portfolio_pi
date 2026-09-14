# Experiments

Run commands from the parent directory of this folder.

| Experiment | Entry point | Output directory |
|---|---|---|
| Fixed-policy evaluation and improvement-gap certificate | `python analysis/reproduce_policy_loss.py` | `results/policy_loss/` |
| C1 local evaluation and finite-horizon localization | `python analysis/reproduce_local_policy.py` | `results/local_policy/` |
| Actual neural-policy scalar-certificate obstruction audit | `python experiments/neural_certificate_audit.py` | `results/neural_certificate/` |
| Exact-weight neural logarithmic and whole-space comparison | `python run.py verify-neural-welfare` | `results/neural_certificate/` |
| Two-asset whole-space welfare certificate | `python experiments/portfolio_welfare.py --verify-only` | `results/portfolio_welfare/` |
| Fit and verify the two-asset candidates | `python experiments/portfolio_welfare.py` | `results/portfolio_welfare/` |
| Analytic radius/horizon localization budgets | `python analysis/localization_budget.py` | `results/portfolio_welfare/` |
| Exact coefficient-ODE policy iteration | `python experiments/exact_pi_ode.py` | `results/exact_pi_ode/` |
| Verified continuous Bellman residual | `python experiments/direct_residual_certificate.py --verify-only` | `results/direct_certificate/` |
| Fit and verify the constrained polynomial | `python experiments/direct_residual_certificate.py` | `results/direct_certificate/` |
| Active-constraint FDM and five-seed PINN-PI comparison | `python experiments/constrained_benchmark.py` | `results/constrained_benchmark/` |
| Trace-exact neural residual and radius diagnostics | `python experiments/pipinn_hard_certificate.py` | `results/hard_certificate/` |
| Empirical performance and inference | `python analysis/reproduce_empirical.py` | `results/empirical_tables/` |

The fixed-policy certificate keeps the supplied greedy portfolio unchanged.
Two higher-degree polynomial evaluations solve its linear PDE with the same
trace. Exact rational restriction precedes outward Bernstein products,
derivatives, KKT branch checks, and subdivision. The residual and quadratic
Hamiltonian gap give a sharper whole-space loss bound, also uniform over a
specified inner cylinder. `--refit` refits the evaluations only. Verification
requires several minutes and its measured cost is reported with the bounds.

The local evaluations use shared nodal values and global y-derivatives, local
polynomial bubbles, and graded time. Exact rational checks verify terminal and
lateral traces and C1 spatial interfaces; interval Bernstein bounds cover every
cell in the cylinder. The finite-horizon localizer bounds the exit probability
uniformly over admissible policies by a partitioned exponential-martingale
inequality. Its comparison table applies both the quadratic and finite-horizon
localizers to the same verified evaluations. No policy update or boundary
change is made. `--refit` additionally requires SciPy; the supplied candidates
can be verified with the core dependencies alone. Different coefficient counts
and accuracy targets preclude a matched-cost solver-speed interpretation.

The two-asset experiment combines prescribed Dirichlet data, a non-diagonal
covariance-metric QP, all nine possible active-set assignments, whole-cylinder
outward residual enclosures, and explicit quadratic-weight localization. It
certifies relative terminal certainty-equivalent wealth loss for the stated
whole-space strategy. The myopic comparison is separately evaluated with the
same QP and boundary. The overlapping CE intervals are reported explicitly.

The one-asset direct certificate uses a supplied dyadic polynomial,
exact rational algebra, and outward-rounded Bernstein subdivision. The
verification-only command requires NumPy and mpmath, with no training, GPU,
reference PDE solution, or random validation cloud. It checks the whole
bounded cylinder, including active-set switching cells. Its scalar barrier
bounds value error and greedy-policy performance loss for the prescribed
Dirichlet problem. Appendix G gives the enclosure formulas. The certificate does
not bound pointwise policy distance or certify the whole-space/neural runs.
Verification replaces its result JSON and LaTeX table; fitting also replaces
the candidate. A failed run returns no successful bound.

The neural diagnostic in Appendix F preserves each evaluation's frozen policy and
reports its linear-PDE residual separately from the final greedy Bellman
residual. Its sampling mixture has exact importance weights, but the scanned
residual envelope is empirical, not a verified certificate. The diagnostic
filename `pipinn_hard_certificate.py` does not imply a rigorous neural bound.

The constrained benchmark uses matching finite-domain boundary data for FDM
and PINN-PI. Grid self-convergence checks FDM accuracy, while multiple neural
seeds assess the evaluation error floor. Neural scripts require PyTorch and
select CUDA if available; `PIPINN_DEVICE=cpu` selects CPU. CPU/GPU arithmetic
and retraining need not reproduce fixed supplied outputs bit for bit.

`analysis/reproduce_empirical.py` recomputes the published statistics from
supplied policy paths. `analysis/train_empirical.py` separately prepares or
trains the selected specifications using a common covariance-metric QP.
These entry points do not rerun predictor selection. All empirical inference
is conditional on the selected specifications and fixed strategy paths.
