"""Continuous, outward-rounded clique certificates for a coupled factor cycle.

Only (remaining time, z_i, z_(i+1)) is subdivided. Exact algebra proves that
the full d-state normalized PDE residual is the average of these edge terms.
All compatible clamp branches are retained. See METHOD.md for scope.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
import hashlib
from math import comb
from pathlib import Path
import json
import time

import numpy as np
from mpmath import iv

ROOT = Path(__file__).resolve().parents[1]


def scalar(x):
    x = F(x)
    return iv.mpf(x.numerator)/x.denominator


def ends(x):
    return float(np.nextafter(float(x.a), -np.inf)), float(np.nextafter(float(x.b), np.inf))


class Poly:
    """Three-variable power polynomial with high-precision interval coefficients."""
    def __init__(self, coefficients=None):
        self.c = coefficients or {}

    @classmethod
    def constant(cls, x):
        if not hasattr(x, '_mpi_'):
            x = scalar(x)
        return cls({(0, 0, 0): x})

    @classmethod
    def variable(cls, axis):
        key = [0, 0, 0]
        key[axis] = 1
        return cls({tuple(key): scalar(1)})

    def __add__(self, other):
        if not isinstance(other, Poly):
            other = Poly.constant(other)
        result = dict(self.c)
        for key, value in other.c.items():
            result[key] = result.get(key, scalar(0))+value
        return Poly(result)

    __radd__ = __add__

    def __neg__(self):
        return Poly({k: -v for k, v in self.c.items()})

    def __sub__(self, other):
        return self+(-other if isinstance(other, Poly) else Poly.constant(-other))

    def __rsub__(self, other):
        return -self+other

    def __mul__(self, other):
        if not isinstance(other, Poly):
            other = Poly.constant(other)
        result = {}
        for ka, va in self.c.items():
            for kb, vb in other.c.items():
                key = tuple(a+b for a, b in zip(ka, kb))
                result[key] = result.get(key, scalar(0))+va*vb
        return Poly(result)

    __rmul__ = __mul__

    def derivative(self, axis):
        result = {}
        for key, value in self.c.items():
            if key[axis]:
                new = list(key)
                new[axis] -= 1
                result[tuple(new)] = key[axis]*value
        return Poly(result)

    def divide_time(self):
        ans = {}
        for key, value in self.c.items():
            if key[0] == 0:
                # Every value polynomial is built with an explicit time factor.
                if float(value.a) != 0 or float(value.b) != 0:
                    raise ArithmeticError('Nonzero terminal correction')
            else:
                ans[(key[0]-1, key[1], key[2])] = value
        return Poly(ans)

    def at(self, t, x, y):
        inputs = list(map(scalar, (t, x, y)))
        result = scalar(0)
        for key, value in self.c.items():
            term = value
            for z, k in zip(inputs, key):
                term *= z**k
            result += term
        return result


def chebyshev(k):
    t = Poly.variable(0)
    out = [Poly.constant(1)]
    if k > 1:
        out.append(2*t-1)
    for _ in range(2, k):
        out.append(2*(2*t-1)*out[-1]-out[-2])
    return out


def factor(coeff):
    k = len(coeff)//3
    t, x = Poly.variable(0), Poly.variable(1)
    cb = chebyshev(k)
    ans = Poly.constant(0)
    for j, spatial in enumerate((Poly.constant(1), x, x*x)):
        for i in range(k):
            ans += F(coeff[j*k+i])*t*cb[i]*spatial
    return ans


def potential(coeff):
    k = len(coeff)//2
    x, y = Poly.variable(1), Poly.variable(2)
    cb = chebyshev(k)
    ans = Poly.constant(0)
    for j, spatial in enumerate((y-x, y*y-x*x)):
        for i in range(k):
            ans += F(coeff[j*k+i])*cb[i]*spatial
    return ans


def residual_polys(candidate):
    d, k = candidate['dimension'], candidate['time_terms']
    m = {key: F(value) for key, value in candidate['model'].items()}
    if d < 3 or candidate['assets'] != d or m['gamma'] != 2 or m['horizon'] != 1:
        raise ValueError('Unsupported model')
    if k < 1 or m['cap'] <= 0 or m['radius'] <= 0 or m['cross'] < 0 or m['lambda_constant'] < 0 or m['risk_free_rate'] < 0:
        raise ValueError('Unsupported sign or degree in the localization model')
    if F(candidate.get('implementation_radius_exact', m['radius'])) < m['radius']:
        raise ValueError('Certification cube must stay inside the implemented greedy-policy cube')
    if m['variance'] <= m['cross']**2 or not 0 <= m['drift_neighbor'] < m['kappa']:
        raise ValueError('Invalid covariance or dissipativity')
    if not m['lambda_self'] > abs(m['lambda_neighbor']) > 0:
        raise ValueError('Need full-rank, nonzero coupled premia')
    for name in ('lower', 'upper'):
        if len(candidate[name]['coeff']) != 3*k or len(candidate[name]['potential']) != 2*k:
            raise ValueError('Coefficient dimensions disagree')
    if len(candidate['base']) != 3*k:
        raise ValueError('Base coefficient dimensions disagree')
    base = factor(candidate['base'])
    lower, upper = factor(candidate['lower']['coeff']), factor(candidate['upper']['coeff'])
    x, y = Poly.variable(1), Poly.variable(2)
    lam = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y
    g = scalar(m['cross'])/iv.sqrt(scalar(d))
    base_signal = lam+base.derivative(1)*g
    arrays = []
    signals = []
    for name, val, fixed in (('lower', lower, None), ('upper', upper, base_signal), ('base', base, None)):
        dz, dzz = val.derivative(1), val.derivative(1).derivative(1)
        core = val.derivative(0)+(m['kappa']*x-m['drift_neighbor']*y)*dz
        core -= m['variance']/2*dzz+m['variance']/F(2*d)*dz*dz
        if name != 'base':
            core += potential(candidate[name]['potential'])
        target = lam+dz*g
        sig = target if fixed is None else fixed
        cap = m['cap']
        # negative, free, positive; the free upper branch uses the BASE policy.
        terms = (-cap*target-cap*cap,
                 sig*target*F(1, 2)-sig*sig*F(1, 4), cap*target-cap*cap)
        arrays.extend(core+h for h in terms)
        signals.append(sig)
    arrays += signals
    arrays += [(lower-base).divide_time(), (base-upper).divide_time(), base]
    return arrays, m


def bernstein(poly, radius, degrees):
    """Affine restriction to [0,1]^3, then exact-rational Bernstein weights."""
    power = {}
    r = F(radius)
    for (it, ix, iy), value in poly.c.items():
        for j in range(ix+1):
            for k in range(iy+1):
                key = (it, j, k)
                weight = F(comb(ix, j)*comb(iy, k))*(-r)**(ix+iy-j-k)*(2*r)**(j+k)
                power[key] = power.get(key, scalar(0))+value*scalar(weight)
    lo, hi = np.empty(tuple(n+1 for n in degrees)), np.empty(tuple(n+1 for n in degrees))
    for index in np.ndindex(lo.shape):
        total = scalar(0)
        for key, value in power.items():
            if all(a <= b for a, b in zip(key, index)):
                weight = F(1)
                for n, p, j in zip(degrees, key, index):
                    weight *= F(comb(j, p), comb(n, p))
                total += value*scalar(weight)
        lo[index], hi[index] = ends(total)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ArithmeticError('Nonfinite Bernstein coefficients')
    return lo, hi


def split(lo, hi, axis):
    """Outward de Casteljau bisection; input includes a leading polynomial axis."""
    lo, hi = np.moveaxis(lo, axis+1, 0), np.moveaxis(hi, axis+1, 0)
    ll, lh, rl, rh = [np.empty_like(lo) for _ in range(4)]
    ll[0], lh[0], rl[-1], rh[-1] = lo[0], hi[0], lo[-1], hi[-1]
    for k in range(1, len(lo)):
        lo = np.nextafter((lo[:-1]+lo[1:])/2, -np.inf)
        hi = np.nextafter((hi[:-1]+hi[1:])/2, np.inf)
        ll[k], lh[k], rl[-1-k], rh[-1-k] = lo[0], hi[0], lo[-1], hi[-1]
    return tuple(np.moveaxis(v, 0, axis+1) for v in (ll, lh, rl, rh))


def cell_bounds(lo, hi, cap):
    l, h = lo.min(axis=(1, 2, 3)), hi.max(axis=(1, 2, 3))
    # The threshold is enclosed; overlapping masks retain uncertain branches.
    thlo, thhi = ends(2*scalar(cap))
    lows, highs = [], []
    for i in range(3):
        a, b = l[9+i], h[9+i]
        possible = (a <= -thlo, a <= thhi and b >= -thhi, b >= thlo)
        chosen = [3*i+j for j in range(3) if possible[j]]
        if not chosen:
            raise ArithmeticError('No compatible clamp branch')
        lows.append(float(min(l[j] for j in chosen)))
        highs.append(float(max(h[j] for j in chosen)))
    return {'lower_residual_upper': highs[0], 'upper_residual_lower': lows[1],
            'base_residual_lower': lows[2], 'base_residual_upper': highs[2],
            'lower_trace_rate_upper': float(h[12]), 'upper_trace_rate_upper': float(h[13]),
            'base_factor_exponent_upper': float(h[14])}


def verify_cover(candidate, tolerance=F(1, 10000), max_depth=27, max_cells=2000000):
    start = time.perf_counter()
    polys, m = residual_polys(candidate)
    degrees = tuple(max(key[i] for poly in polys for key in poly.c) for i in range(3))
    raw = [bernstein(poly, m['radius'], degrees) for poly in polys]
    rootlo, roothi = np.stack([v[0] for v in raw]), np.stack([v[1] for v in raw])
    pending = [(rootlo, roothi, 0)]
    accepted, visited, depthmax, volume = 0, 0, 0, F(0)
    worst = {'lower_residual_upper': -np.inf, 'upper_residual_lower': np.inf,
             'base_residual_lower': np.inf, 'base_residual_upper': -np.inf,
             'lower_trace_rate_upper': -np.inf, 'upper_trace_rate_upper': -np.inf,
             'base_factor_exponent_upper': -np.inf}
    target = float(tolerance)
    while pending:
        lo, hi, depth = pending.pop()
        visited += 1
        bounds = cell_bounds(lo, hi, m['cap'])
        excess = max(bounds['lower_residual_upper'], -bounds['upper_residual_lower'],
                     bounds['lower_trace_rate_upper'], bounds['upper_trace_rate_upper'])
        if excess <= target:
            accepted += 1
            volume += F(1, 2**depth)
            depthmax = max(depthmax, depth)
            for key, val in bounds.items():
                worst[key] = min(worst[key], val) if key.endswith('_lower') else max(worst[key], val)
        else:
            if depth >= max_depth or visited >= max_cells:
                raise ArithmeticError(f'Cover incomplete: visited={visited}, depth={depth}, excess={excess:g}, target={target:g}')
            # Cyclic subdivision also refines time near terminal compatibility.
            axis = depth % 3
            ll, lh, rl, rh = split(lo, hi, axis)
            pending.append((rl, rh, depth+1))
            pending.append((ll, lh, depth+1))
        if visited % 50000 == 0:
            print(f"d={candidate['dimension']} visited={visited} accepted={accepted} depth={depthmax}", flush=True)
    if volume != 1:
        raise ArithmeticError('Exact dyadic cover volume is not one')
    return {'cover_complete': True, 'covered_local_coordinates': ['tau', 'z_i', 'z_next'],
            'accepted_cells': accepted, 'visited_cells': visited, 'maximum_depth': depthmax,
            'exact_covered_volume': str(volume), 'polynomial_degrees': list(degrees),
            'seconds': time.perf_counter()-start, **worst}


def localization(candidate, trace_exponent_upper, bins=128):
    """Policy-uniform stopped-coordinate union bound; controlled state need not be Gaussian."""
    d = candidate['dimension']
    m = {k: scalar(v) for k, v in candidate['model'].items()}
    k, e, Q, G, C, R, T = [m[key] for key in
                          ('kappa', 'drift_neighbor', 'variance', 'cross', 'cap', 'radius', 'horizon')]
    shift = e*R+G*C/iv.sqrt(scalar(d))
    if float((R-shift/k).a) <= 0:
        raise ArithmeticError('Exit threshold not positive')
    probability = scalar(0)
    for j in range(1, bins+1):
        a = iv.exp(k*T*scalar(F(j-1, bins)))*(R-shift/k)+shift/k
        n = Q/(2*k)*(iv.exp(2*k*T*scalar(F(j, bins)))-1)
        probability += 2*iv.exp(-a*a/(2*n))
    probability *= d
    plo, phi = ends(probability)
    if phi >= 1:
        probability = scalar(1)
        phi = 1.
    # omega=exp(a*|z|^2/d), x=|z|/sqrt(d).
    weight = scalar(F(1, 20))
    denom = 2*weight*(k-e)-2*Q*weight*weight/d
    if float(denom.a) <= 0:
        raise ArithmeticError('Quadratic Lyapunov domination is not positive')
    c0 = C*C+C*m['lambda_constant']  # dropping -r increases the bound
    c1 = C*(abs(m['lambda_self'])+abs(m['lambda_neighbor']))
    linear = 2*weight*G*C/iv.sqrt(scalar(d))+c1
    exponent = weight*Q+c0+linear*linear/(4*denom)
    continuation = iv.exp(exponent*T+weight*R*R)
    trace_upper = iv.exp(scalar(F(trace_exponent_upper)))
    cp = -m['risk_free_rate']+C*C+C*(abs(m['lambda_constant'])+
                                      (abs(m['lambda_self'])+abs(m['lambda_neighbor']))*R)
    cp = scalar(max(F(0), F(ends(cp)[1])))
    loc = iv.exp(cp*T)*(continuation+trace_upper)*probability
    return {'exit_probability_upper': phi, 'localization_upper': ends(loc)[1],
            'quadratic_weight_exact': '1/20', 'weight_generator_exponent_upper': ends(exponent)[1],
            'continuation_upper_at_cube_boundary': ends(continuation)[1],
            'coordinate_drift_shift_upper': ends(shift)[1],
            'exit_partition_bins': bins, 'union_over_coordinates': d}


def jensen_lower(candidate, bins=1024):
    """Elementary policy-uniform Gaussian-envelope/Jensen alternative.

    This is NOT a standalone bound on the candidate's loss: the separately
    certified policy upper endpoint is still needed. The right Riemann sum
    encloses a nondecreasing moment majorant on the full continuous horizon.
    """
    m = {k: scalar(v) for k, v in candidate['model'].items()}
    d = candidate['dimension']
    a = m['kappa']-m['drift_neighbor']
    loading = abs(m['lambda_self'])+abs(m['lambda_neighbor'])
    integral = scalar(0)
    for j in range(1, bins+1):
        t = m['horizon']*scalar(F(j, bins))
        gaussian = m['lambda_constant']**2+loading**2*m['variance']/(2*a)*(1-iv.exp(-2*a*t))
        controlled_shift = loading*m['cross']*m['cap']/(a*iv.sqrt(scalar(d)))*(1-iv.exp(-a*t))
        integral += (iv.sqrt(gaussian)+controlled_shift)**2*m['horizon']/bins
    # An upper endpoint on the integral gives a lower endpoint on exp(-I/4).
    upper_integral = scalar(F(ends(integral)[1]))
    lower = iv.exp(-m['risk_free_rate']*m['horizon']-upper_integral/4)
    return {'analytic_optimal_factor_lower': ends(lower)[0],
            'analytic_moment_integral_upper': ends(integral)[1],
            'analytic_right_riemann_bins': bins}


def combine(candidate, cover):
    k = candidate['time_terms']
    r, T = F(candidate['model']['risk_free_rate']), F(candidate['model']['horizon'])
    repair_l = max(0., cover['lower_residual_upper'], cover['lower_trace_rate_upper'])
    repair_u = max(0., -cover['upper_residual_lower'], cover['upper_trace_rate_upper'])
    a_l = sum(map(F, candidate['lower']['coeff'][:k]))-F(repair_l)*T-r*T
    a_u = sum(map(F, candidate['upper']['coeff'][:k]))+F(repair_u)*T-r*T
    lo, hi = iv.exp(scalar(a_l)), iv.exp(scalar(a_u))
    loc = localization(candidate, cover['base_factor_exponent_upper'])
    whole_lo, whole_hi = lo-scalar(F(loc['localization_upper'])), hi+scalar(F(loc['localization_upper']))
    if float(whole_lo.a) <= 0:
        raise ArithmeticError('Nonpositive global lower endpoint')
    loss = 1-whole_lo/whole_hi
    analytic = jensen_lower(candidate)
    hybrid_lower = max(ends(whole_lo)[0], analytic['analytic_optimal_factor_lower'])
    hybrid_loss = 1-scalar(F(hybrid_lower))/whole_hi
    basewidth = T*(F(max(0., cover['base_residual_upper']))+F(max(0., -cover['base_residual_lower'])))
    return {'whole_space_certified': True, 'dimension': candidate['dimension'],
            'scope': 'initial origin, unit wealth, T=1; coupled constrained cycle; fixed log-quadratic generating policy',
            'certification_radius_exact': candidate['model']['radius'],
            'implementation_radius_exact': candidate.get('implementation_radius_exact', candidate['model']['radius']),
            'lower_time_repair_rate': repair_l, 'upper_time_repair_rate': repair_u,
            'dirichlet_optimal_factor_lower': ends(lo)[0], 'dirichlet_policy_factor_upper': ends(hi)[1],
            'whole_space_optimal_factor_lower': ends(whole_lo)[0], 'whole_space_policy_factor_upper': ends(whole_hi)[1],
            'relative_ce_loss_upper': ends(loss)[1], 'relative_ce_loss_upper_bp': ends(loss*10000)[1],
            'hybrid_relative_ce_loss_upper_bp': ends(hybrid_loss*10000)[1],
            'hybrid_lower_endpoint_source': 'local PDE' if ends(whole_lo)[0] >= analytic['analytic_optimal_factor_lower'] else 'elementary Jensen moment bound',
            'direct_base_dirichlet_ce_gap_upper_bp': ends((1-iv.exp(-scalar(F(basewidth))))*10000)[1],
            'policy_updates_during_certification': 0,
            'verification_arithmetic': 'mpmath.iv 60-digit coefficient algebra, rational transforms, outward binary64 Bernstein bisection',
            **loc, **analytic}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimensions', nargs='+', type=int, default=[3, 5, 10, 20])
    p.add_argument('--folder', default='results/highdim_local')
    p.add_argument('--tolerance', default='1/2000')
    p.add_argument('--max-depth', type=int, default=30)
    p.add_argument('--radius', help='Smaller certification cube only; original whole-space policy stays fixed')
    args = p.parse_args()
    iv.dps = 60
    tolerance = F(args.tolerance)
    if tolerance <= 0 or args.max_depth < 1:
        p.error('Positive tolerance and depth required')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = ROOT/args.folder
    for d in args.dimensions:
        start = time.perf_counter()
        path = out/f'candidate_d{d}.json'
        candidate_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        candidate = json.loads(path.read_text())
        if candidate['dimension'] != d:
            raise ValueError('Dimension mismatch')
        suffix = ''
        if args.radius is not None:
            radius = F(args.radius)
            if not 0 < radius <= F(candidate['model']['radius']):
                raise ValueError('Radius override must be positive and no larger than the implemented cube')
            candidate = copy.deepcopy(candidate)
            candidate['implementation_radius_exact'] = candidate['model']['radius']
            candidate['model']['radius'] = str(radius)
            suffix = f'_radius_{radius.numerator}_{radius.denominator}'
        cover = verify_cover(candidate, tolerance, args.max_depth)
        result = {**combine(candidate, cover), 'cover': cover,
                  'candidate_sha256': candidate_hash, 'verifier_sha256': digest,
                  'tolerance_exact': str(tolerance), 'seconds': time.perf_counter()-start}
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Input changed during verification')
        (out/f'certificate_d{d}{suffix}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(f'd={d} CERTIFIED CE={result["relative_ce_loss_upper_bp"]:.6f} bp, '
              f'cells={cover["accepted_cells"]}, seconds={result["seconds"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
