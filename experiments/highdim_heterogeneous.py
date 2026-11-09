"""Fit and certify coordinate-dependent return predictors on a factor cycle.

Every coordinate has its own predictor and evaluation polynomials. A common
potential telescopes around the cycle, and every local cylinder is verified.
The original generating polynomial is fixed; no policy training occurs here.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from mpmath import iv

import highdim_local_verify as verifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/highdim_heterogeneous'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictor_models(reference):
    """Exact, nonperiodic +/-20% variation in three return coefficients."""
    d = reference['dimension']
    if d not in (10, 20, 50):
        raise ValueError('This benchmark specifies dimensions 10, 20, and 50')
    models = []
    for i in range(d):
        s = F(2*((17*i+3) % d), d-1)-1
        t = F(2*((7*i+1) % d), d-1)-1
        model = dict(reference['model'])
        for key, multiplier in (('lambda_constant', 1+s/5),
                                ('lambda_self', 1+t/5),
                                ('lambda_neighbor', 1-s/5)):
            model[key] = str(F(model[key])*multiplier)
        models.append(model)
    assert len({m['lambda_constant'] for m in models}) == d
    return models


def fit_evaluations(reference, model, mode, refinement=1):
    from scipy.optimize import linprog
    from highdim_local_fit import basis, design, rounded

    start = time.perf_counter()
    m = {key: float(F(value)) for key, value in model.items()}
    d, k = reference['dimension'], reference['time_terms']
    base = np.array([float(F(v)) for v in reference['base']])
    ts = (1-np.cos(np.linspace(0, np.pi, 16*refinement+1)))/2
    zs = np.unique(np.r_[np.linspace(-m['radius'], m['radius'], 32*refinement+1),
                         np.linspace(-.4, .4, 16*refinement+1)])
    tt, xx, yy = np.meshgrid(ts, zs, zs, indexing='ij')
    t, x, y = tt.ravel(), xx.ravel(), yy.ravel()
    _, vt, vx, vxx, _ = design(t, x, k)
    cb = basis(t, k)[0]
    potential_design = np.concatenate((cb*(y-x)[:, None], cb*(y*y-x*x)[:, None]), axis=1)
    tu, xu = np.meshgrid(ts, zs, indexing='ij')
    _, _, dx, _, trace = design(tu.ravel(), xu.ravel(), k)
    lam = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y
    g = m['cross']/np.sqrt(d)
    fixed_policy = np.clip((lam+g*(vx@base))/2, -m['cap'], m['cap'])
    active = np.arange(3*k if mode == 'spatial' else k)
    objective = np.r_[np.ones(k), np.zeros(2*k)][active]
    outputs, diagnostics = {}, []
    for side in ('lower', 'upper'):
        prev = base.copy()
        potential = np.array([float(F(v)) for v in reference[side]['potential']])
        potential_values = potential_design@potential
        trust = .08 if mode == 'spatial' else 0.
        for iteration in range(3):
            grad = vx@prev
            policy = fixed_policy if side == 'upper' else np.clip((lam+g*grad)/2, -m['cap'], m['cap'])
            residual = vt@prev+(m['kappa']*x-m['drift_neighbor']*y)*grad
            residual -= m['variance']/2*(vxx@prev+grad*grad/d)
            residual += policy*(lam+g*grad)-policy*policy
            drift = m['kappa']*x-m['drift_neighbor']*y-m['variance']/d*grad+policy*g
            operator = vt+drift[:, None]*vx-m['variance']/2*vxx
            matrix = operator[:, active]
            rhs = matrix@prev[active]-residual-potential_values
            tr = trace[:, active]
            trace_rhs = trace@(base-prev)+tr@prev[active]
            if side == 'lower':
                aa, bb = np.concatenate((matrix, tr)), np.r_[rhs-1e-8, trace_rhs]
                cost = -objective
            else:
                aa, bb = np.concatenate((-matrix, -tr)), np.r_[
                    -rhs-1e-8-m['variance']/(2*d)*trust**2, -trace_rhs]
                if mode == 'spatial':
                    dd = dx[:, active]
                    aa = np.concatenate((aa, dd, -dd))
                    bb = np.r_[bb, dd@prev[active]+trust, -dd@prev[active]+trust]
                cost = objective
            sol = linprog(cost, A_ub=aa, b_ub=bb, bounds=[(-3., 3.)]*len(active), method='highs')
            if not sol.success:
                raise RuntimeError(f'{side} {mode}: {sol.message}')
            prev[active] = sol.x
            diagnostics.append({'side': side, 'iteration': iteration, 'objective': float(sol.fun)})
        outputs[side] = {'coeff': rounded(prev), 'potential': reference[side]['potential']}
    return outputs, {'seconds': time.perf_counter()-start, 'lp': diagnostics,
                     'grid_refinement': refinement}


def fit(reference, mode):
    start = time.perf_counter()
    blocks = []
    for i, model in enumerate(predictor_models(reference)):
        evaluations, log = fit_evaluations(reference, model, mode)
        block = {key: copy.deepcopy(reference[key]) for key in ('dimension', 'assets', 'time_terms', 'base')}
        block.update(model=model, **evaluations)
        blocks.append(block)
        print(f"d={reference['dimension']} {mode} fitted {i+1}/{reference['dimension']}", flush=True)
    return {'dimension': reference['dimension'], 'mode': mode, 'blocks': blocks,
            'fit_seconds': time.perf_counter()-start, 'variation_exact': '1/5',
            'potential_choice': 'common lower/upper potentials from the homogeneous reference',
            'policy_training_steps': 0, 'last_fit_diagnostics': log}


def validate_blocks(candidate):
    blocks, d = candidate['blocks'], candidate['dimension']
    if len(blocks) != d:
        raise ValueError('One block is required for every coordinate')
    reference = blocks[0]
    variable = {'lambda_constant', 'lambda_self', 'lambda_neighbor'}
    for block in blocks:
        if block['dimension'] != d or block['assets'] != d:
            raise ValueError('Block dimension mismatch')
        for key, value in reference['model'].items():
            if key not in variable and block['model'][key] != value:
                raise ValueError('This localization requires common state dynamics and caps')
        for side in ('lower', 'upper'):
            if block[side]['potential'] != reference[side]['potential']:
                raise ValueError('Coordinate-dependent potentials do not telescope in this verifier')
    return blocks


def combine(candidate, covers):
    blocks = validate_blocks(candidate)
    d = candidate['dimension']
    if len(covers) != d or any(not c['cover_complete'] or c['exact_covered_volume'] != '1' for c in covers):
        raise ValueError('Every coordinate cylinder must have a complete verified cover')
    reference = copy.deepcopy(blocks[0])
    for key in ('lambda_constant', 'lambda_self', 'lambda_neighbor'):
        reference['model'][key] = str(max(abs(F(b['model'][key])) for b in blocks))
    r, horizon = F(reference['model']['risk_free_rate']), F(reference['model']['horizon'])
    lower_exp, upper_exp, trace_exp = -r*horizon, -r*horizon, F(0)
    for block, cover in zip(blocks, covers):
        k = block['time_terms']
        eta_l = F(max(0., cover['lower_residual_upper'], cover['lower_trace_rate_upper']))
        eta_u = F(max(0., -cover['upper_residual_lower'], cover['upper_trace_rate_upper']))
        lower_exp += (sum(map(F, block['lower']['coeff'][:k]))-eta_l*horizon)/d
        upper_exp += (sum(map(F, block['upper']['coeff'][:k]))+eta_u*horizon)/d
        trace_exp += F(cover['base_factor_exponent_upper'])/d
    loc = verifier.localization(reference, trace_exp)
    lower = iv.exp(verifier.scalar(lower_exp))-verifier.scalar(F(loc['localization_upper']))
    upper = iv.exp(verifier.scalar(upper_exp))+verifier.scalar(F(loc['localization_upper']))
    if float(lower.a) <= 0:
        raise ArithmeticError('The whole-space lower endpoint is not positive')
    loss = (1-lower/upper)*10000
    return {'whole_space_certified': True, 'dimension': d, 'mode': candidate['mode'],
            'relative_ce_loss_upper_bp': verifier.ends(loss)[1],
            'whole_space_optimal_factor_lower': verifier.ends(lower)[0],
            'whole_space_policy_factor_upper': verifier.ends(upper)[1],
            'localization_envelope_model': reference['model'], **loc,
            'local_cylinders_verified': d, 'total_accepted_cells': sum(c['accepted_cells'] for c in covers),
            'maximum_local_dimension': 3, 'covers': covers,
            'scope': 'initial origin, T=1; heterogeneous return predictors, common cyclic state dynamics; fixed deployed policy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', action='store_true', help='Propose evaluation polynomials before verification')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[10, 20, 50])
    parser.add_argument('--modes', nargs='+', choices=['spatial', 'time_only'], default=['spatial'])
    parser.add_argument('--tolerance', default='1/2000')
    parser.add_argument('--refine-failed', action='store_true',
                        help='Refit a block on a finer proposal grid if continuous verification fails')
    args = parser.parse_args()
    iv.dps = 60
    tolerance = F(args.tolerance)
    if tolerance <= 0:
        parser.error('The verification tolerance must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    for d in args.dimensions:
        reference_path = ROOT/f'results/highdim_local/candidate_d{d}.json'
        reference = json.loads(reference_path.read_text())
        for mode in args.modes:
            path = OUT/f'candidate_d{d}_{mode}.json'
            if args.fit:
                proposal = fit(reference, mode)
                proposal['generating_candidate_sha256'] = digest(reference_path)
                proposal['fit_source_sha256'] = digest(Path(__file__))
                path.write_text(json.dumps(proposal, indent=2)+'\n', encoding='utf-8')
            candidate = json.loads(path.read_text())
            if candidate['dimension'] != d or candidate['mode'] != mode:
                raise ValueError('Candidate metadata mismatch')
            if candidate['generating_candidate_sha256'] != digest(reference_path):
                raise ValueError('Generating candidate changed')
            input_hash = digest(path)
            blocks = validate_blocks(candidate)
            start = time.perf_counter()
            covers, refinements, refit_seconds = [], [], 0.
            for i, block in enumerate(blocks):
                for refinement in (None, 2, 4):
                    if refinement is not None:
                        evaluations, fit_log = fit_evaluations(reference, block['model'], mode, refinement)
                        block.update(evaluations)
                        candidate['fit_seconds'] += fit_log['seconds']
                        refit_seconds += fit_log['seconds']
                        refinements.append({'coordinate': i+1, **fit_log})
                    try:
                        cover = verifier.verify_cover(block, tolerance=tolerance, max_depth=36)
                        covers.append(cover)
                        break
                    except ArithmeticError as error:
                        if not args.refine_failed or refinement == 4 or not str(error).startswith('Cover incomplete'):
                            raise
                        print(f'd={d} {mode}: refining proposal grid for coordinate {i+1}', flush=True)
            if refinements:
                candidate.setdefault('refinements', []).extend(refinements)
                candidate['fit_source_sha256'] = digest(Path(__file__))
                path.write_text(json.dumps(candidate, indent=2)+'\n', encoding='utf-8')
                input_hash = digest(path)
            result = combine(candidate, covers)
            result.update(verification_seconds=time.perf_counter()-start-refit_seconds,
                          fit_seconds=candidate['fit_seconds'], tolerance_exact=str(tolerance),
                          candidate_sha256=input_hash, source_sha256=digest(Path(__file__)),
                          local_verifier_sha256=digest(Path(verifier.__file__)))
            if input_hash != digest(path):
                raise RuntimeError('Candidate changed during verification')
            (OUT/f'certificate_d{d}_{mode}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(f'd={d} {mode}: CERTIFIED {result["relative_ce_loss_upper_bp"]:.6f} bp, '
                  f'{result["verification_seconds"]:.2f} s', flush=True)


if __name__ == '__main__':
    main()
