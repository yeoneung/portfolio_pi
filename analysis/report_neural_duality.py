"""Build the matched-comparison report from completed, hash-checked results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/neural_duality'


def read(name):
    d = json.loads((OUT/name).read_text())
    if not d.get('complete') or d.get('continuous_time_certified'):
        raise ValueError('Expected a completed non-certified diagnostic: '+name)
    if hashlib.sha256((ROOT/'results/neural_certificate/seed_11_final.json').read_bytes()).hexdigest() != d['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch: '+name)
    hashes = d['source_sha256']
    if isinstance(hashes, str):
        hashes = {'neural_duality_comparison.py': hashes}
    for source, digest in hashes.items():
        if hashlib.sha256((ROOT/'experiments'/source).read_bytes()).hexdigest() != digest:
            raise ValueError('Stale experiment source: '+name)
    return d


def main():
    mc = read('comparison_large.json')
    pde = read('pde_refinement.json')
    pde['levels'] += read('pde_time_refinement.json')['levels']
    domain = read('pde_domain_check.json')
    direct = read('direct_check.json')
    lookup = read('direct_lookup_check.json')
    coarse = read('lookup_refinement.json')
    fine = read('lookup_fine_check.json')
    cert = json.loads((ROOT/'results/neural_certificate/neural_welfare_bound.json').read_text())
    log = json.loads((ROOT/'results/neural_certificate/neural_log_barrier.json').read_text())
    if not cert['whole_space_certified']:
        raise ValueError('Expected the unchanged successful interval certificate')
    for source, digest in cert['input_sha256'].items():
        folder = 'experiments' if source.endswith('.py') else 'results/neural_certificate'
        if hashlib.sha256((ROOT/folder/source).read_bytes()).hexdigest() != digest:
            raise ValueError('Stale certificate input: '+source)
    last = mc['levels'][-1]
    cv, raw = last['pooled']['control_variate'], last['pooled']['raw']
    lastp = pde['levels'][-1]
    sensitivity = []
    for a, b in zip(direct['levels'], lookup['levels']):
        if a['steps'] != b['steps'] or a['pooled']['paths'] != b['pooled']['paths']:
            raise ValueError('Unmatched direct/lookup sensitivity run')
        ca, cb = a['pooled']['control_variate'], b['pooled']['control_variate']
        sensitivity.append({'steps': a['steps'], 'paths': a['pooled']['paths'],
                            'gap_difference_bp': 1e4*(cb['relative_ce_gap_plugin']-ca['relative_ce_gap_plugin']),
                            'policy_factor_difference': cb['policy_factor']-ca['policy_factor']})
    grid_difference = 1e4*(fine['levels'][-1]['pooled']['control_variate']['relative_ce_gap_plugin']-
                           coarse['levels'][-1]['pooled']['control_variate']['relative_ce_gap_plugin'])
    domain_difference = 1e4*(lastp['relative_ce_duality_gap']-domain['levels'][-1]['relative_ce_duality_gap'])
    ci = cv['gap_95_interval']
    summary = {'checkpoint_sha256': mc['checkpoint_sha256'], 'policy_updates': 0,
               'rigorous_relative_ce_upper_bp': 1e4*cert['relative_ce_loss_upper'],
               'mc_finest_gap_bp': 1e4*cv['relative_ce_gap_plugin'],
               'mc_finest_sampling_95_interval_bp': [1e4*x for x in ci],
               'mc_finest_sampling_se_bp': 1e4*cv['gap_standard_error'],
               'pde_finest_gap_bp': 1e4*lastp['relative_ce_duality_gap'],
               'direct_lookup_sensitivity': sensitivity,
               'lookup_grid_gap_difference_bp': grid_difference,
               'far_domain_gap_difference_bp': domain_difference,
               'mc_variance_reduction_factor': (raw['gap_standard_error']/cv['gap_standard_error'])**2,
               'mc_simulated_paths_all_resolutions': sum(x['pooled']['paths'] for x in mc['levels']),
               'mc_table_and_simulation_seconds': sum(x['table_seconds']+x['simulation_seconds'] for x in mc['levels']),
               'input_sha256': {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in
                               ('comparison_large.json', 'pde_refinement.json', 'pde_time_refinement.json', 'pde_domain_check.json',
                                'direct_check.json', 'direct_lookup_check.json',
                                'lookup_refinement.json', 'lookup_fine_check.json')},
               'comparison_establishes_certified_speed_or_accuracy_superiority': False}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    lines = ['# Frozen neural-policy financial-duality comparison', '',
             '## Main result', '',
             f"The HKW-style duality gap estimate at {last['steps']} time steps and "
             f"{last['pooled']['paths']:,} paths is **{1e4*cv['relative_ce_gap_plugin']:.4f} bp**, "
             f"with a paired asymptotic 95% sampling interval "
             f"**[{1e4*ci[0]:.4f}, {1e4*ci[1]:.4f}] bp**. "
             f"The independently discretized linear PDE gives **{1e4*lastp['relative_ce_duality_gap']:.4f} bp**.", '',
             f"The unchanged outward-rounded whole-space upper bound is "
             f"**{summary['rigorous_relative_ce_upper_bp']:.4f} bp (9.662% rounded upward)**. "
             'The duality construction is sharper numerically in this example. This is not a '
             'new rigorous improvement to the 9.662% bound: Monte Carlo intervals account '
             'for sampling only, and the PDE cross-check is a mesh-refinement diagnostic.', '',
             'This is a bound-gap estimate for the fixed deployed policy, not a measured '
             'distance to a known optimizer. Neither method changes the policy. '
             'The comparison supports a distinction in error coverage, not superiority '
             'over the literature in sharpness or matched-accuracy speed. All losses '
             'are relative terminal CE losses over the original three-year horizon, '
             'not annualized returns; one basis point is 0.0001 of wealth.', '',
             '## Monte Carlo time refinement', '',
             '| Steps | Paths | Duality gap (bp) | Sampling SE (bp) | 95% sampling interval (bp) | GPU simulation (s) |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in mc['levels']:
        est = row['pooled']['control_variate']
        l, u = est['gap_95_interval']
        lines.append(f"| {row['steps']} | {row['pooled']['paths']:,} | {1e4*est['relative_ce_gap_plugin']:.4f} "
                     f"| {1e4*est['gap_standard_error']:.4f} | [{1e4*l:.4f}, {1e4*u:.4f}] "
                     f"| {row['simulation_seconds']:.2f} |")
    lines += ['', 'Five independent seeds (101, 202, 303, 404, 505) are retained at each '
              'resolution. Cross-resolution paths are not Brownian-nested. Timings are '
              'evaluation costs, excluding the common saved-network training. '
              'GPU table setup is separately recorded in the JSON files.', '',
              '## Independent linear-PDE refinement', '',
              '| Time steps | Spatial points | Primal factor | Dual factor | Duality gap (bp) | Seconds |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in pde['levels']:
        lines.append(f"| {row['steps']} | {row['space_points']} | {row['policy_factor']:.9f} "
                     f"| {row['dual_factor']:.9f} | {1e4*row['relative_ce_duality_gap']:.5f} | {row['seconds']:.2f} |")
    lines += ['', 'The PDE evaluates both moments of the same duality construction, using '
              'the direct network rather than derivative lookup. It does not compute '
              'or assume an exact optimal value. The artificial far boundary is tested '
              'separately by changing its distance; no rigorous tail-error enclosure '
              'is claimed for this numerical reference.', '',
              '## Numerical sensitivity and variance checks', '',
              f"- 4097 versus 8193 lookup nodes, with the same random draws: signed gap change {grid_difference:.8f} bp.",
              f"- Far-domain half-width 2 versus 3 times the original half-width: signed gap change {domain_difference:.8f} bp.",
              f"- Finest-run paired-gap variance reduction relative to raw simulation: {summary['mc_variance_reduction_factor']:.2f} times.",
              f"- Discrete martingale control mean {last['pooled']['martingale_control_mean']:.8f}, "
              f"SE {last['pooled']['martingale_control_standard_error']:.8f}; theoretical mean is 1.",
              f"- Second-order plug-in bias estimate: {1e4*cv['estimated_second_order_plugin_bias']:.8f} bp."]
    for row in sensitivity:
        lines.append(f"- Direct NN versus lookup, {row['steps']} steps and {row['paths']:,} paired paths: "
                     f"signed gap change {row['gap_difference_bp']:.8f} bp.")
    lines += ['', 'These are numerical sensitivity checks, not uniform interpolation-error '
              'certificates or confidence intervals including discretization bias.', '',
              '## Reproducibility and guarantee scope', '',
              f"Checkpoint SHA256: `{mc['checkpoint_sha256']}`. All runs use float64 and "
              f"TF32 is disabled. Device: {mc['environment']['gpu']}. No training or policy update.", '',
              f"The supplied interval verifier recorded {log['seconds']+cert['seconds']:.2f} CPU seconds. "
              'That is a different certification task and a recorded earlier run, not '
              'a matched-accuracy CPU/GPU timing experiment. The main submission PDF '
              'and its validated ZIPs are not replaced by this standalone comparison.', '',
              'See [METHOD.md](METHOD.md) for equations, the unspanned-risk treatment, '
              'the discrete martingale proof, reproduction commands, and source attribution.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
