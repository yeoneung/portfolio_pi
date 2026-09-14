"""One-sided whole-space extension of the neural logarithmic certificate.

Inside the unchanged domain use the exact greedy policy of the checkpoint.
Outside use zero risky exposure, including re-entry into the domain. Only
the interior strategy was present in the training experiment; this explicit
Markov extension defines the whole-space strategy certified here.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from mpmath import iv

from neural_log_barrier import Affine, pack, down, up, times, plus, sumup
from neural_certificate_audit import scalar, endpoints, evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/neural_certificate'


def reaction_upper(c, cells):
    """Enclose the actual policy reaction, requiring only one NN derivative."""
    cells = np.asarray(cells, dtype=float)
    a = Affine.constant
    tau = Affine.input(cells[:, 0:1], cells[:, 1:2], 1)*c['horizon']
    m, s = c['market'], c['state_dict']
    z = m['xbar'][0]+Affine.input(cells[:, 2:3], cells[:, 3:4], 2)*(a(c['domain_scale'])*m['eta'][0])
    nh = s['half_ax'][0][0]
    inverse = Affine.interval_constant((down(1/nh), up(1/nh)))
    inverse_t = Affine.interval_constant((down(1/c['horizon']), up(1/c['horizon'])))
    y = (z-s['center'][0][0])*inverse
    d, dz = 1-y*y, -2*y*inverse
    zero = a(np.zeros(tau.c[0].shape))
    value = pack((tau*inverse_t, y, d))
    derivative = pack((zero, zero+inverse, dz))
    for layer in range(c['config']['depth']+1):
        w, bias = s[f'net.{2*layer}.weight'], s[f'net.{2*layer}.bias']
        value = value.linear(w, bias)
        derivative = derivative.linear(w, np.zeros(len(bias)))
        if layer < c['config']['depth']:
            value = value.tanh()
            derivative = (1-value*value)*derivative
    hz = tau*(dz*value+d*derivative)
    lam = np.asarray(m['lam0'])[None, :]+z*np.asarray(m['Lam'])[:, 0][None, :]
    signal = lam+hz*np.asarray(m['Gamma'])[:, 0][None, :]
    slo, shi = signal.bounds()
    cap = np.asarray(c['caps'])[None, :]
    plo, phi = np.maximum(-cap, np.minimum(cap, down(slo/2))), np.maximum(-cap, np.minimum(cap, up(shi/2)))
    llo, lhi = lam.bounds()
    # The convex quadratic p^2-p*lambda attains its box maximum at a vertex.
    vertices = []
    for p in (plo, phi):
        for l in (llo, lhi):
            product = times((p, p), plus((p, p), (-l, -l)))
            vertices.append(product[1])
    upper = np.maximum.reduce(vertices)
    ans = up(sumup(upper.T)-c['risk_free_rate'])
    if not np.isfinite(ans).all():
        raise ArithmeticError('Nonfinite reaction enclosure')
    return ans


def verify_reaction(c, target=None, level=6, max_depth=20):
    """Verify slab targets over a full cover; point probes only propose targets."""
    if (target is not None and target < 0) or not 1 <= level <= 12:
        raise ValueError('Invalid reaction target or partition')
    n = 2**level
    targets = []
    if target is None:
        for ti in range(n):
            probes = [[t, 0., z, 0.] for t in (ti/n, (ti+.5)/n, (ti+1)/n)
                      for z in np.linspace(-1, 1, 17)]
            proposed = max(0., float(np.max(reaction_upper(c, probes))))
            targets.append(F(int(np.ceil(proposed*100))+1, 100))
    else:
        targets = [target]*n
    pending = [((ti+.5)/n, .5/n, -1+(zi+.5)*2/n, 1/n, 0, ti)
               for ti in range(n) for zi in range(n)]
    accepted, maximum = 0, -np.inf
    area = [F(0) for _ in range(n)]
    while pending:
        batch = pending[-128:]
        del pending[-128:]
        bounds = reaction_upper(c, [item[:4] for item in batch])
        for cell, bound in zip(batch, bounds):
            t, dt, z, dz, depth, ti = cell
            if bound <= down(float(targets[ti])):
                accepted += 1
                area[ti] += F(2, n*n*2**depth)
                maximum = max(maximum, bound)
            elif depth >= max_depth:
                raise ArithmeticError(f'Reaction target not proved at {cell}')
            elif dt >= dz/2:
                pending.extend([(t-dt/2, dt/2, z, dz, depth+1, ti), (t+dt/2, dt/2, z, dz, depth+1, ti)])
            else:
                pending.extend([(t, dt, z-dz/2, dz/2, depth+1, ti), (t, dt, z+dz/2, dz/2, depth+1, ti)])
    if any(a != F(2, n) for a in area):
        raise ArithmeticError('Incomplete reaction cover')
    return {'reaction_upper_exact': str(max(targets)), 'reaction_enclosure_upper': maximum,
            'reaction_upper_by_slab_exact': [str(x) for x in targets],
            'reaction_integral_upper_exact': str(F(c['horizon'])*sum(targets)/n),
            'reaction_accepted_cells': accepted, 'reaction_full_cylinder_covered': True}


def jensen_lower(c):
    """Explicit positive global subsolution for every feasible strategy.

    J solves J_tau = (Q/2) J_zz + b_tilde J_z + r+|lambda|^2/4.
    For w=exp(-J), the unconstrained forward Bellman residual is
    -(Q-|Gamma|^2/2) J_z^2 w/2 <= 0; all constrained frozen residuals
    are smaller. Thus w <= g* without a numerical Riccati solver.
    """
    m = c['market']
    q, k, k0, z, t = map(F, (m['Q'][0][0], m['K'][0][0], m['k0'][0], m['xbar'][0], c['horizon']))
    gam = [F(x[0]) for x in m['Gamma']]
    l0, l1 = list(map(F, m['lam0'])), [F(x[0]) for x in m['Lam']]
    alpha = q-sum((g*g for g in gam), F(0))/2
    kt = k+sum((g*l for g, l in zip(gam, l1)), F(0))/2
    if alpha <= 0 or kt <= 0 or c['risk_free_rate'] < 0:
        raise ValueError('Unsupported global subsolution parameters')
    mean = (k0-sum((g*l for g, l in zip(gam, l0)), F(0))/2)/kt
    aa = [a+b*mean for a, b in zip(l0, l1)]
    bb = [b*(z-mean) for b in l1]
    e1 = (1-iv.exp(-scalar(kt*t)))/scalar(kt)
    e2 = (1-iv.exp(-scalar(2*kt*t)))/scalar(2*kt)
    cost = scalar(sum((a*a for a in aa), F(0))*t)
    cost += scalar(2*sum((a*b for a, b in zip(aa, bb)), F(0)))*e1
    cost += scalar(sum((b*b for b in bb), F(0)))*e2
    cost += scalar(sum((b*b for b in l1), F(0))*q/(2*kt))*(scalar(t)-e2)
    j = scalar(F(c['risk_free_rate'])*t)+cost/4
    return {'global_optimal_factor_lower': endpoints(iv.exp(-j))[0],
            'jensen_exponent_interval': endpoints(j), 'distorted_variance_exact': str(alpha),
            'distorted_mean_reversion_exact': str(kt)}


def exit_probability(c, powers=range(16, 31), bins=1024):
    """Additive Lyapunov/Dynkin exit bound; the tilted process is not Gaussian.

    For V=exp(a*x^2), LV <= C and V>=1 before exit imply
    P(exit<=T) <= (V(x0)-1+C*T)/(V(R)-1). Here x0=0.
    C encloses the generator on every spatial subinterval, for all controls.
    """
    m = c['market']
    k, q, t, center = map(F, (m['K'][0][0], m['Q'][0][0], c['horizon'], m['xbar'][0]))
    radius = F(c['domain_scale'])*F(m['eta'][0])
    drift = abs(F(m['k0'][0])-k*center)+sum((F(cap)*abs(F(g[0])) for cap, g in zip(c['caps'], m['Gamma'])), F(0))
    if k <= 0 or q <= 0 or radius <= 0 or bins < 1:
        raise ValueError('Invalid exit-bound model')
    best = None
    for j in powers:
        if not 0 < j < 32:
            raise ValueError('Lyapunov exponents must lie strictly below kappa/Q')
        a = k/q*F(j, 32)
        d = 2*a*k-2*q*a*a
        maximum = 0.
        for i in range(bins):
            left, right = radius*F(i, bins), radius*F(i+1, bins)
            # A separate positive envelope avoids interval sign ambiguity.
            polynomial_upper = a*q-d*left*left+2*a*drift*right
            if polynomial_upper > 0:
                value = endpoints(scalar(polynomial_upper)*iv.exp(scalar(a*right*right)))[1]
                maximum = max(maximum, value)
        probability = min(1., endpoints(scalar(F(maximum)*t)/(iv.exp(scalar(a*radius*radius))-1))[1])
        item = {'exit_probability_upper': probability, 'lyapunov_exponent_exact': str(a),
                'generator_upper': maximum, 'drift_shift_upper_exact': str(drift),
                'spatial_enclosure_bins': bins, 'method': 'additive stopped Lyapunov bound'}
        if best is None or probability < best['exit_probability_upper']:
            best = item
    if best is None:
        raise ValueError('No Lyapunov candidate')
    return best


def combine(c, logarithmic, reaction):
    iv.dps = 50
    if not logarithmic.get('bounded_domain_certified') or not reaction.get('reaction_full_cylinder_covered'):
        raise ValueError('Completed full-domain enclosures required')
    initial = evaluate(c, F(c['horizon']), F(c['market']['xbar'][0]))
    cmax = F(reaction['reaction_upper_exact'])
    if cmax < 0:
        raise ValueError('Nonnegative global reaction upper bound required')
    t, r = F(c['horizon']), F(c['risk_free_rate'])
    loc = exit_probability(c)
    lower = jensen_lower(c)
    integrated = F(reaction['reaction_integral_upper_exact'])
    if integrated < 0:
        raise ValueError('Invalid nonnegative reaction envelope')
    discrepancy = (iv.exp(scalar(integrated))-iv.exp(-scalar(r*t)))*scalar(F(loc['exit_probability_upper']))
    dirichlet_upper = scalar(F(initial['neural_value_interval'][1]))*iv.exp(scalar(F(logarithmic['upper_log_correction_exact'])))
    whole_upper = dirichlet_upper+discrepancy
    ce = 1-scalar(F(lower['global_optimal_factor_lower']))/whole_upper
    return {'whole_space_certified': True, 'scope': 'continuous-time whole-space terminal CE loss at the initial center',
            'outside_policy': 'zero exposure outside the original domain; unchanged neural greedy policy after re-entry',
            'whole_space_policy_factor_upper': endpoints(whole_upper)[1],
            'bounded_domain_policy_factor_upper': endpoints(dirichlet_upper)[1],
            'one_sided_policy_localization_upper': endpoints(discrepancy)[1],
            'relative_ce_loss_upper': endpoints(ce)[1],
            'relative_ce_loss_upper_bp': endpoints(ce*10000)[1],
            **reaction, **loc, **lower}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=OUT/'seed_11_final.json')
    p.add_argument('--log-certificate', type=Path, default=OUT/'neural_log_barrier.json')
    p.add_argument('--output', type=Path, default=OUT/'neural_welfare_bound.json')
    p.add_argument('--reaction-only', action='store_true')
    args = p.parse_args()
    if args.output.resolve() in (args.checkpoint.resolve(), args.log_certificate.resolve(), Path(__file__).resolve()):
        p.error('Output must not overwrite an input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('{"whole_space_certified": false}\n')
    inputs = [args.checkpoint, Path(__file__), Path(__file__).with_name('neural_log_barrier.py'), Path(__file__).with_name('neural_certificate_audit.py')]
    if not args.reaction_only:
        inputs.append(args.log_certificate)
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    c = json.loads(args.checkpoint.read_text())
    if c['gamma'] != 2 or c['format'] != 'hard_value_net_exact_binary_weights_v1':
        raise ValueError('Unsupported mathematical policy')
    start = time.perf_counter()
    print('Enclosing the unchanged neural policy reaction...', flush=True)
    result = verify_reaction(c)
    print(json.dumps(result), flush=True)
    if not args.reaction_only:
        log = json.loads(args.log_certificate.read_text())
        if (log['checkpoint_sha256'] != hashes[str(args.checkpoint.resolve())]
                or log['verifier_source_sha256'] != hashes[str(Path(__file__).with_name('neural_log_barrier.py').resolve())]
                or log['trace_helper_sha256'] != hashes[str(Path(__file__).with_name('neural_certificate_audit.py').resolve())]):
            raise ValueError('Stale logarithmic certificate')
        result = combine(c, log, result)
    else:
        result['whole_space_certified'] = False
    for path in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.resolve())]:
            raise ArithmeticError('Inputs changed during verification')
    result.update({'input_sha256': {path.name: hashes[str(path.resolve())] for path in inputs},
                   'checkpoint_sha256': hashes[str(args.checkpoint.resolve())],
                   'policy_updates_after_checkpoint': 0, 'seconds': time.perf_counter()-start})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
