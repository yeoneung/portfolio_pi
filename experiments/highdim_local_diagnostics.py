"""Matched time-only baseline and NON-certifying joint-state diagnostics.

All methods use the exact stored generating coefficients and hence the same
policy. This script never trains or replaces that policy.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import linprog

from highdim_local_fit import basis, design, local_residual, rounded

ROOT = Path(__file__).resolve().parents[1]


def time_only(candidate):
    start = time.perf_counter()
    base = np.array([float(F(v)) for v in candidate['base']])
    m = {k: float(F(v)) for k, v in candidate['model'].items()}
    k, d = candidate['time_terms'], candidate['dimension']
    # The time-only envelope has sharper extrema: give this baseline a denser
    # proposal grid. Verification accuracy is identical for both methods.
    ts = (1-np.cos(np.linspace(0, np.pi, 129)))/2
    zs = np.unique(np.r_[np.linspace(-m['radius'], m['radius'], 65), np.linspace(-.4, .4, 17)])
    tt, xx, yy = np.meshgrid(ts, zs, zs, indexing='ij')
    t, x, y = tt.ravel(), xx.ravel(), yy.ravel()
    cb, _, bt = basis(t, k)
    pot = np.concatenate((cb*(y-x)[:, None], cb*(y*y-x*x)[:, None]), axis=1)
    matrix = np.concatenate((bt, pot), axis=1)
    trace = np.concatenate((basis(ts, k)[0], np.zeros((len(ts), 2*k))), axis=1)
    residual = local_residual(base, t, x, y, d, fixed=base)[0]
    endpoint = np.r_[np.ones(k), np.zeros(2*k)]
    output = copy.deepcopy(candidate)
    for side in ('lower', 'upper'):
        sign = 1 if side == 'lower' else -1
        aa = sign*np.concatenate((matrix, trace))
        bb = np.r_[-sign*residual-1e-8, np.zeros(len(ts))]
        sol = linprog(-sign*endpoint, A_ub=aa, b_ub=bb, bounds=[(-3, 3)]*(3*k), method='highs')
        if not sol.success:
            raise RuntimeError(f'Time-only baseline LP failed: {sol.message}')
        # Keep all spatial coefficients EXACTLY, not round-tripping float64.
        a = [str(F(candidate['base'][i])+F(rounded([sol.x[i]])[0])) for i in range(k)]
        output[side] = {'coeff': a+candidate['base'][k:], 'potential': rounded(sol.x[k:])}
    output['baseline'] = 'same policy and spatial log-evaluation coefficients; time-only evaluation corrections plus optimized telescoping potentials'
    output['baseline_fit_seconds'] = time.perf_counter()-start
    output['baseline_collocation_shape'] = [len(ts), len(zs), len(zs)]
    output['parent_candidate_sha256'] = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
    return output


def diagnostic(candidate, count=65536):
    """One independent stationary joint-state sample, not a global bound."""
    d, k = candidate['dimension'], candidate['time_terms']
    m = {k: float(F(v)) for k, v in candidate['model'].items()}
    shift = np.roll(np.eye(d), -1, axis=0)
    drift = m['kappa']*np.eye(d)-m['drift_neighbor']*shift
    stationary = solve_continuous_lyapunov(drift, m['variance']*np.eye(d))
    rng = np.random.default_rng(9152026)
    z = rng.normal(size=(count, d))@np.linalg.cholesky(stationary).T
    base = np.array([float(F(v)) for v in candidate['base']])
    # Time one is a specified state-space diagnostic, not a path occupancy rate.
    b, c = base[k:2*k].sum(), base[2*k:].sum()
    signal = m['lambda_constant']+m['lambda_self']*z+m['lambda_neighbor']*np.roll(z, -1, axis=1)
    signal += m['cross']/math.sqrt(d)*(b+2*c*z)
    upper, lower = signal >= 2*m['cap'], signal <= -2*m['cap']
    # Random times and all coordinates enter the held-out JOINT residual.
    tau = rng.uniform(0, 1, count)
    chunks = []
    for first in range(0, count, 1024):
        zz, t = z[first:first+1024], tau[first:first+1024]
        residual = local_residual(base, np.repeat(t, d), zz.ravel(), np.roll(zz, -1, axis=1).ravel(), d, fixed=base)[0]
        chunks.append(residual.reshape(-1, d).mean(1))
    rr = np.concatenate(chunks)
    return {'certified': False, 'sample_kind': 'independent stationary physical-state Gaussian; not a controlled-state or path-occupancy law',
            'sample_count': count, 'seed': 9152026, 'state_dimension': d,
            'upper_active_fraction_at_tau_one': float(upper.mean()),
            'lower_active_fraction_at_tau_one': float(lower.mean()),
            'interior_fraction_at_tau_one': float((~upper & ~lower).mean()),
            'joint_residual_rms_sampled': float(np.sqrt(np.mean(rr**2))),
            'joint_residual_max_abs_sampled': float(np.abs(rr).max()),
            'stationary_state_std': float(np.sqrt(stationary[0, 0])),
            'neighbor_state_correlation': float(stationary[0, 1]/stationary[0, 0]),
            'drift_neighbor_to_self_ratio': m['drift_neighbor']/m['kappa'],
            'premium_neighbor_to_self_ratio': m['lambda_neighbor']/m['lambda_self'],
            'state_covariance_smallest_eigenvalue': float(np.linalg.eigvalsh(stationary).min()),
            'premium_loading_smallest_singular_value_before_normalization': float(np.linalg.svd(m['lambda_self']*np.eye(d)+m['lambda_neighbor']*shift, compute_uv=False).min())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions', type=int, nargs='+', default=[3, 5, 10, 20, 50])
    args = parser.parse_args()
    folder = ROOT/'results/highdim_local'
    out = folder/'time_only'
    out.mkdir(exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for d in args.dimensions:
        path = folder/f'candidate_d{d}.json'
        input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        candidate = json.loads(path.read_text())
        baseline = time_only(candidate)
        baseline['diagnostic_source_sha256'] = digest
        diag = diagnostic(candidate)
        diag.update(candidate_sha256=input_hash, diagnostic_source_sha256=digest)
        if input_hash != hashlib.sha256(path.read_bytes()).hexdigest() or digest != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError('Input or source changed')
        (out/f'candidate_d{d}.json').write_text(json.dumps(baseline, indent=2)+'\n')
        (folder/f'diagnostic_d{d}.json').write_text(json.dumps(diag, indent=2)+'\n')
        print(f'd={d} matched time-only proposal; active upper={diag["upper_active_fraction_at_tau_one"]:.3%}, '
              f'lower={diag["lower_active_fraction_at_tau_one"]:.3%}; sampled diagnostics NOT CERTIFIED', flush=True)


if __name__ == '__main__':
    main()
