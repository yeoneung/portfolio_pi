"""Build a hash-checked report of the independent high-dimensional experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/highdim_local'
DIMS = [3, 5, 10, 20, 50]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_certificate(d, baseline=False, radius=False):
    folder = OUT/'time_only' if baseline else OUT
    suffix = '_radius_1_1' if radius else ''
    path = folder/f'candidate_d{d}.json'
    proposal = json.loads(path.read_text())
    record = json.loads((folder/f'certificate_d{d}{suffix}.json').read_text())
    if record['candidate_sha256'] != digest(path):
        raise ValueError('Stale candidate certificate: '+str(path))
    if record['verifier_sha256'] != digest(ROOT/'experiments/highdim_local_verify.py'):
        raise ValueError('Stale verifier certificate')
    if proposal['fit_source_sha256'] != digest(ROOT/'experiments/highdim_local_fit.py'):
        raise ValueError('Stale proposal source')
    if not record['whole_space_certified'] or not record['cover']['cover_complete'] or record['cover']['exact_covered_volume'] != '1':
        raise ValueError('Incomplete proof')
    if record['tolerance_exact'] != '1/2000' or record['implementation_radius_exact'] != '3/2':
        raise ValueError('Unmatched certification settings')
    if record['certification_radius_exact'] != ('1' if radius else '3/2'):
        raise ValueError('Unexpected certification radius')
    return proposal, record


def main():
    rows, stress = [], []
    for d in DIMS:
        proposal, result = read_certificate(d)
        baseline, bound = read_certificate(d, baseline=True)
        k = proposal['time_terms']
        if baseline['base'] != proposal['base'] or baseline['model'] != proposal['model']:
            raise ValueError('Baseline policy/model mismatch')
        for side in ('lower', 'upper'):
            if baseline[side]['coeff'][k:] != proposal['base'][k:]:
                raise ValueError('Baseline changed spatial evaluation')
        diagnostic = json.loads((OUT/f'diagnostic_d{d}.json').read_text())
        if diagnostic['candidate_sha256'] != digest(OUT/f'candidate_d{d}.json') or diagnostic['certified']:
            raise ValueError('Stale or mislabeled sampling diagnostic')
        source = digest(ROOT/'experiments/highdim_local_diagnostics.py')
        if diagnostic['diagnostic_source_sha256'] != source or baseline['diagnostic_source_sha256'] != source:
            raise ValueError('Stale diagnostic source')
        rows.append({'state_dimension': d, 'asset_dimension': d,
                     'two_sided_pde_ce_upper_bp': result['relative_ce_loss_upper_bp'],
                     'time_only_two_sided_pde_ce_upper_bp': bound['relative_ce_loss_upper_bp'],
                     'hybrid_ce_upper_bp': result['hybrid_relative_ce_loss_upper_bp'],
                     'time_only_hybrid_ce_upper_bp': bound['hybrid_relative_ce_loss_upper_bp'],
                     'hybrid_bound_reduction_factor': bound['hybrid_relative_ce_loss_upper_bp']/result['hybrid_relative_ce_loss_upper_bp'],
                     'accepted_local_cells': result['cover']['accepted_cells'],
                     'verification_seconds': result['seconds'],
                     'gpu_proposal_seconds': proposal['training']['seconds'],
                     'spatial_barrier_fit_seconds': proposal['barrier_fit_seconds'],
                     'time_only_barrier_fit_seconds': baseline['baseline_fit_seconds'],
                     'exit_probability_upper': result['exit_probability_upper'],
                     'localization_upper': result['localization_upper'],
                     'upper_active_fraction_sampled': diagnostic['upper_active_fraction_at_tau_one'],
                     'lower_active_fraction_sampled': diagnostic['lower_active_fraction_at_tau_one'],
                     'joint_residual_rms_sampled': diagnostic['joint_residual_rms_sampled'],
                     'hybrid_lower_endpoint_source': result['hybrid_lower_endpoint_source']})
        if d >= 10:
            _, smaller = read_certificate(d, radius=True)
            stress.append({'state_dimension': d, 'implementation_radius': '3/2',
                           'certification_radius': '1',
                           'two_sided_pde_ce_upper_bp': smaller['relative_ce_loss_upper_bp'],
                           'hybrid_ce_upper_bp': smaller['hybrid_relative_ce_loss_upper_bp'],
                           'localization_upper': smaller['localization_upper'],
                           'exit_probability_upper': smaller['exit_probability_upper']})
    summary = {'complete': True, 'main_cases': rows, 'smaller_cube_same_policy': stress,
               'scope': 'cyclic local coupling, identity asset covariance, additive log-quadratic evaluation, normalized opportunities; not generic dense-neural certification',
               'financial_metric': 'relative certainty-equivalent loss upper bound at initial origin, gamma=2, T=1',
               'comparison_scope': 'internal time-only versus spatial evaluation; no literature-superiority claim',
               'candidate_state_dimension_is_not_asset_count_only': True,
               'policy_iteration_or_newton_scaling_tested': False,
               'source_sha256': {name: digest(ROOT/'experiments'/name) for name in
                                  ['highdim_local_fit.py', 'highdim_local_verify.py', 'highdim_local_diagnostics.py']}}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    lines = [
        '# High-dimensional certification experiment', '',
        'The continuous verifier closes through **50 state dimensions** for the specified locally coupled, '
        'constrained portfolio family. The two-sided PDE welfare bounds stay below 39 bp in the main cube. '
        'These are upper bounds on loss, not measurements of the actual loss.', '',
        'The structured policy is distinct from the supplied seed-11 neural checkpoint. '
        'Read [METHOD.md](METHOD.md) for the model, algebraic decomposition, '
        'bounded-domain comparison, localization proof, assumptions, and reproduction commands.', '',
        '## Main continuous certificates', '',
        'Both methods evaluate exactly the same policy. The baseline changes only the time-dependent '
        'part of the log evaluation; both methods can optimize telescoping potentials. The baseline '
        'has the denser proposal grid. All continuous checks use the same 1/2000 repair tolerance.', '',
        '| State dimensions | Spatial evaluation: CE upper (bp) | Time-only evaluation: CE upper (bp) | Accepted local cells | Verification (s) |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for x in rows:
        lines.append(f'| {x["state_dimension"]} | {x["two_sided_pde_ce_upper_bp"]:.6f} | '
                     f'{x["time_only_two_sided_pde_ce_upper_bp"]:.6f} | {x["accepted_local_cells"]:,} | {x["verification_seconds"]:.3f} |')
    lines += ['', 'The time column is verification only, including interval localization and the analytic lower '
              'endpoint; it is not end-to-end training time. GPU proposal fitting took approximately 0.9–2.6 s; '
              'spatial LP fitting approximately 4.4–4.8 s, versus approximately 23–26 s for the denser time-only LP. '
              'Single-machine timings are descriptive, not an asymptotic-complexity experiment.', '',
              '## Stronger elementary lower bound, applied fairly to both', '',
              'The Gaussian-envelope/Jensen lower endpoint is stronger than the PDE lower endpoint in four '
              'main cases. It is therefore substituted into **both** methods where helpful, while keeping '
              'each method’s independently certified policy upper endpoint. This separates the contribution '
              'of the policy evaluation from that of the optimal-value lower bound.', '',
              '| State dimensions | Spatial + best lower endpoint (bp) | Time-only + best lower endpoint (bp) | Internal upper-bound reduction |',
              '| ---: | ---: | ---: | ---: |']
    for x in rows:
        lines.append(f'| {x["state_dimension"]} | {x["hybrid_ce_upper_bp"]:.6f} | {x["time_only_hybrid_ce_upper_bp"]:.6f} | {x["hybrid_bound_reduction_factor"]:.2f}x |')
    lines += ['', 'These factors compare upper bounds for our own two evaluation constructions. '
              'They are not improvements in actual investment returns and not superiority factors over an external paper.', '',
              '## Boundary sensitivity: same implemented policy', '',
              'The implemented policy still uses the cube of radius 1.5. Only the certification cube is '
              'reduced to radius 1; the policy and all generating coefficients are unchanged. The localization '
              'term grows with the coordinate union bound.', '',
              '| State dimensions | Smaller-cube two-sided PDE bound (bp) | Smaller-cube hybrid bound (bp) | Additive factor localization bound |',
              '| ---: | ---: | ---: | ---: |']
    for x in stress:
        lines.append(f'| {x["state_dimension"]} | {x["two_sided_pde_ce_upper_bp"]:.6f} | {x["hybrid_ce_upper_bp"]:.6f} | {x["localization_upper"]:.8f} |')
    lines += ['', 'At 50 states the smaller-cube two-sided bound exceeds the 100 bp reporting threshold. '
              'Thus favorable main-cube results must not be read as dimension-independent localization.', '',
              '## Genuine state coupling, and the structural restriction', '',
              'There are d state coordinates, nonzero neighbor interactions in the drift and risky premia, '
              'full-rank return predictors, and positive unspanned state covariance. The independently sampled '
              'upper-constraint frequency is 41.2–42.1%; lower constraints also activate. These are stationary '
              'physical-state diagnostics, not certified controlled-path frequencies.', '',
              'The speed comes from an exact identity: the full residual is the average of cyclic '
              '(remaining time, current state, neighboring state) terms, with cancelling potentials. '
              'The verifier covers these continuous three-variable domains and every compatible clipping '
              'branch. It does not replace a high-dimensional supremum with a Monte Carlo maximum.', '',
              'However, the candidate is a symmetric additive log-quadratic with 15 coefficients, '
              'asset covariance is the identity, interactions are local, and opportunities/exposure caps '
              'are normalized by sqrt(d). This is not a test of a generic dense neural network, dense '
              'state dependence, or the manuscript’s full high-dimensional Newton/policy-iteration solver.', '',
              '## Interpretation', '',
              'The result supports **structure-exploiting high-dimensional certification** as a credible '
              'research direction. It does not establish a general solution to the dimensionality problem '
              'or a standalone literature novelty claim. The most defensible claim is the computable '
              'certificate for this locally coupled family, including its fixed-policy evaluation and '
              'boundary-to-whole-space error conversion.', '',
              'Exact coefficient and source hashes are checked before this report is generated. '
              'The standalone validation record additionally documents independent re-verification.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('High-dimensional report regenerated: 5 main, 5 matched baseline, 3 same-policy boundary sensitivity certificates.')


if __name__ == '__main__':
    main()
