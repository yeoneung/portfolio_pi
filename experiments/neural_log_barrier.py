"""Time-resolved logarithmic comparison for the unchanged neural policy.

The verifier encloses every point in a partition of the closed cylinder.
Two-input quadratic Taylor forms retain cancellations through the stored network;
their remainder radii and all elementary interval operations round outward.
Matrix products use the standard binary64 dot-product error bound (including
FMA/reassociation), with an absolute allowance for gradual underflow. No
sampled residual maximum is accepted as a bound.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import time

# Small dense products are faster and more reproducible with one BLAS thread.
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
from mpmath import iv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/neural_certificate"
TINY = 2.0**-1000


def up(x):
    return np.nextafter(x, np.inf)


def down(x):
    return np.nextafter(x, -np.inf)


def plus(a, b):
    return down(a[0]+b[0]), up(a[1]+b[1])


def times(a, b):
    products = np.stack(np.broadcast_arrays(a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return down(products.min(axis=0)), up(products.max(axis=0))


def divide_positive(a, b):
    if np.any(np.asarray(b[0]) <= 0):
        raise ArithmeticError("Nonpositive interval denominator")
    return times(a, (down(1/b[1]), up(1/b[0])))


def point(x):
    x = np.asarray(x, dtype=np.float64)
    return x, x


def tanh_interval(x):
    """Enclose tanh at binary64 points without a library transcendental bound.

    For |x| <= 20, exp(2|x|) is evaluated by a degree-24 Taylor polynomial
    at |x|/32 followed by six squarings. Its positive remainder is at most
    2*(5/8)^25/25!, because exp(5/8)<2. Larger arguments use
    1-2^-52 < tanh(|x|) < 1 (2 exp(-40)<2^-52).
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.minimum(abs(x), 20.)
    reduced = (np.maximum(0., down(ax/32)), up(ax/32))
    coeff = []
    for k in range(25):
        value = float(F(1, math.factorial(k)))
        coeff.append((down(value), up(value)))
    value = coeff[-1]
    for a in reversed(coeff[:-1]):
        value = down(a[0]+down(reduced[0]*value[0])), up(a[1]+up(reduced[1]*value[1]))
    tail = up(float(2*F(5, 8)**25/math.factorial(25)))
    value = value[0], up(value[1]+tail)
    for _ in range(6):
        value = down(value[0]*value[0]), up(value[1]*value[1])
    ans = (down(down(value[0]-1)/up(value[1]+1)),
           up(up(value[1]-1)/down(value[0]+1)))
    lo = np.where(abs(x) > 20, down(1.-2.**-52), np.maximum(0., ans[0]))
    hi = np.where(abs(x) > 20, 1., np.minimum(1., ans[1]))
    return np.where(x < 0, -hi, lo), np.where(x < 0, -lo, hi)


def gamma(n):
    return up(float(F(n, 2**53-n)))


def matup(a, b):
    """Nonnegative matrix product upper bound, allowing 2n operations per dot."""
    n = a.shape[-1]
    return up(up(a @ b)/down(1.-gamma(2*n+4))+TINY)


def sumup(items):
    result = np.zeros_like(items[0])
    for item in items:
        result = up(result+item)
    return result


class Affine:
    """Quadratic Taylor form in (1,x,y,x^2,xy,y^2), plus [-r,r]."""

    __array_priority__ = 1000

    def __init__(self, c, r=0.):
        self.c = np.asarray(c, dtype=np.float64)
        self.r = np.broadcast_to(np.asarray(r, dtype=np.float64), self.c.shape[1:])

    @classmethod
    def constant(cls, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return cls(np.stack((x, *(np.zeros_like(x) for _ in range(5)))))

    @classmethod
    def input(cls, center, radius, axis):
        ans = cls.constant(center)
        ans.c[axis] = radius
        return ans

    @classmethod
    def interval_constant(cls, bounds):
        lo, hi = np.broadcast_arrays(*bounds)
        mid = lo/2+hi/2
        return cls(cls.constant(mid).c, up(np.maximum(up(mid-lo), up(hi-mid))))

    def width(self):
        return up(sumup(abs(self.c[1:]))+self.r)

    def bounds(self):
        w = self.width()
        return down(self.c[0]-w), up(self.c[0]+w)

    def __neg__(self):
        return Affine(-self.c, self.r)

    def __add__(self, other):
        if not isinstance(other, Affine):
            other = Affine.constant(other)
        c = self.c+other.c
        roundoff = np.maximum(up(c)-c, c-down(c))
        err = sumup(roundoff)
        return Affine(c, up(up(self.r+other.r)+err))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other if isinstance(other, Affine) else -np.asarray(other))

    def __rsub__(self, other):
        return -self+other

    def __mul__(self, other):
        if not isinstance(other, Affine):
            other = Affine.constant(other)
        a, b = self.c, other.c
        c = np.stack(np.broadcast_arrays(a[0]*b[0], a[1]*b[0]+a[0]*b[1], a[2]*b[0]+a[0]*b[2],
                        a[0]*b[3]+a[3]*b[0]+a[1]*b[1],
                        a[0]*b[4]+a[4]*b[0]+a[1]*b[2]+a[2]*b[1],
                        a[0]*b[5]+a[5]*b[0]+a[2]*b[2]))
        al, bl = sumup(abs(a[1:3])), sumup(abs(b[1:3]))
        aq, bq = sumup(abs(a[3:])), sumup(abs(b[3:]))
        at, bt = up(up(abs(a[0])+al)+aq), up(up(abs(b[0])+bl)+bq)
        rounding = up(up(gamma(8)*up(at*bt))+TINY)
        radius = sumup([up(al*bq), up(aq*bl), up(aq*bq), up(at*other.r), up(bt*self.r), up(self.r*other.r)])
        return Affine(c, up(radius+rounding))

    __rmul__ = __mul__

    def linear(self, weights, bias):
        w = np.asarray(weights, dtype=np.float64).T
        bias = np.asarray(bias, dtype=np.float64)
        c = self.c @ w
        c[0] += bias
        mag = sumup(abs(self.c))
        total = up(matup(mag, abs(w))+abs(bias))
        err = up(up(gamma(2*w.shape[0]+8)*total)+TINY)
        return Affine(c, up(matup(self.r, abs(w))+err))

    def tanh(self):
        value = Affine.interval_constant(tanh_interval(self.c[0]))
        deriv = 1-value*value
        delta = Affine(np.concatenate((np.zeros_like(self.c[0:1]), self.c[1:])), self.r)
        ans = value+deriv*delta-value*deriv*delta*delta
        # Taylor at the exact center; saturation sharply reduces both terms.
        vlo, vhi = value.bounds()
        width = self.width()
        flo, fhi = np.maximum(-1., down(vlo-width)), np.minimum(1., up(vhi+width))
        minabs = np.where((flo <= 0) & (fhi >= 0), 0., np.minimum(abs(flo), abs(fhi)))
        maxabs = np.maximum(abs(flo), abs(fhi))
        maxdf = np.minimum(1., up(1.-down(minabs*minabs)))
        # f'''=-2(1-f^2)(1-3f^2), |f'''|<=2, and <=4 max(1-f^2).
        maxthird = np.minimum(2., up(4.*maxdf))
        remainder = up(up(maxthird/6)*up(up(width*width)*width))
        return Affine(ans.c, up(ans.r+remainder))

    def hamiltonian(self, cap):
        """H(s)=max_|a|<=cap(a*s-a^2), H'=clip(s/2), Lip(H')=1/2."""
        cap = np.asarray(cap, dtype=np.float64)
        p = np.clip(self.c[0]/2, -cap, cap)
        # Division by two is exact away from underflow (covered below).
        value = times(point(p), plus(point(self.c[0]), point(-p)))
        ans = Affine.interval_constant(value)
        linear = Affine(np.concatenate((np.zeros_like(self.c[0:1]), self.c[1:])))
        ans = ans+p*linear
        fallback = Affine(ans.c, up(up(up(ans.r+up(abs(p)*self.r))+up(.25*up(self.width()**2)))+TINY))
        lo, hi = self.bounds()
        free = (lo >= 2*(-cap)) & (hi <= 2*cap)
        pos, neg = lo >= 2*cap, hi <= -2*cap
        square = .25*self*self
        positive = cap*self-Affine.constant(cap)*cap
        negative = -cap*self-Affine.constant(cap)*cap
        result = fallback
        for mask, item in ((free, square), (pos, positive), (neg, negative)):
            result = Affine(np.where(mask[None, ...], item.c, result.c), np.where(mask, item.r, result.r))
        return result


def pack(features):
    return Affine(np.concatenate([f.c for f in features], axis=-1),
                  np.concatenate([f.r for f in features], axis=-1))


def jadd(a, b):
    return tuple(x+y for x, y in zip(a, b))


def jmul(a, b):
    v, t, z, zz = a
    w, s, y, yy = b
    return v*w, t*w+v*s, z*w+v*y, zz*w+2*z*y+v*yy


def network_residual(c, cells):
    """cells rows are (tau-unit midpoint, half-width, state-unit midpoint, half-width).

    State units use the canonical center +/- half-width, not the NN buffers.
    Both model geometry and unchanged normalization are propagated outward.
    """
    cells = np.asarray(cells, dtype=np.float64)
    a = lambda x: Affine.constant(x)
    tau = Affine.input(cells[:, 0:1], cells[:, 1:2], 1)*c['horizon']
    market, state = c['market'], c['state_dict']
    zz = Affine.input(cells[:, 2:3], cells[:, 3:4], 2)
    z = market['xbar'][0] + zz*(a(c['domain_scale'])*market['eta'][0])
    nc, nh = state['center'][0][0], state['half_ax'][0][0]
    inv_half = Affine.interval_constant((down(1/nh), up(1/nh)))
    inv_t = Affine.interval_constant((down(1/c['horizon']), up(1/c['horizon'])))
    y = (z-nc)*inv_half
    d = 1-y*y
    shape = tau.c[0].shape
    zero, one = a(np.zeros(shape)), a(np.ones(shape))
    tj = tau, one, zero, zero
    dj = d, zero, -2*y*inv_half, -2*inv_half*inv_half+zero
    features = (pack((tau*inv_t, y, d)),
                pack((one*inv_t, zero, zero)),
                pack((zero, one*inv_half, dj[2])),
                pack((zero, zero, dj[3])))
    for layer in range(c['config']['depth']+1):
        w, bias = state[f'net.{2*layer}.weight'], state[f'net.{2*layer}.bias']
        f = tuple(x.linear(w, bias if i == 0 else np.zeros(len(bias))) for i, x in enumerate(features))
        if layer < c['config']['depth']:
            v, t, x, xx = f
            value = v.tanh()
            df = 1-value*value
            features = value, df*t, df*x, df*xx-2*value*df*x*x
        else:
            features = f
    h, ht, hz, hzz = jadd(jmul(jmul(tj, dj), features),
                            (-c['risk_free_rate']*tau, -c['risk_free_rate']*one, zero, zero))
    b = market['k0'][0]-market['K'][0][0]*z
    rho = ht-b*hz-.5*market['Q'][0][0]*(hzz+hz*hz)+c['risk_free_rate']
    lam0 = np.asarray(market['lam0'])[None, :]
    lam1 = np.asarray(market['Lam'])[:, 0][None, :]
    cross = np.asarray(market['Gamma'])[:, 0][None, :]
    signal = lam0+z*lam1+hz*cross
    ham = signal.hamiltonian(np.asarray(c['caps'])[None, :])
    rho = rho+ham.linear(np.ones((1, len(c['caps']))), np.zeros(1))
    lo, hi = rho.bounds()
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ArithmeticError('Nonfinite enclosure: use a finer starting partition')
    return lo[:, 0], hi[:, 0], rho.c[0, :, 0]


def propose_envelope(c, left, right, padding):
    """Fast floating-point probes propose thresholds, never certified bounds.

    Every subsequently accepted cell has a separate outward enclosure.
    Even an inaccurate proposal cannot produce a false certificate.
    """
    m, s = c['market'], c['state_dict']
    ts, zs = np.meshgrid(np.linspace(left, right, 9), np.linspace(-1, 1, 257), indexing='ij')
    t = ts.reshape(-1, 1)*c['horizon']
    z = m['xbar'][0]+zs.reshape(-1, 1)*c['domain_scale']*m['eta'][0]
    nh = s['half_ax'][0][0]
    y = (z-s['center'][0][0])/nh
    d, dz, dzz = 1-y*y, -2*y/nh, np.full_like(y, -2/nh**2)
    zero = np.zeros_like(t)
    value = np.concatenate((t/c['horizon'], y, d), axis=1)
    dt = np.concatenate((np.full_like(t, 1/c['horizon']), zero, zero), axis=1)
    dx = np.concatenate((zero, np.full_like(t, 1/nh), dz), axis=1)
    dxx = np.concatenate((zero, zero, dzz), axis=1)
    for layer in range(c['config']['depth']+1):
        w, bias = np.asarray(s[f'net.{2*layer}.weight']).T, np.asarray(s[f'net.{2*layer}.bias'])
        value, dt, dx, dxx = value@w+bias, dt@w, dx@w, dxx@w
        if layer < c['config']['depth']:
            value = np.tanh(value)
            df = 1-value*value
            dt, dx, dxx = df*dt, df*dx, df*dxx-2*value*df*dx*dx
    ht = -c['risk_free_rate']+d*value+t*d*dt
    hz = t*(dz*value+d*dx)
    hzz = t*(dzz*value+2*dz*dx+d*dxx)
    lam = np.asarray(m['lam0'])[None, :]+z*np.asarray(m['Lam'])[:, 0][None, :]
    signal = lam+hz*np.asarray(m['Gamma'])[:, 0][None, :]
    pi = np.clip(signal/2, -np.asarray(c['caps']), np.asarray(c['caps']))
    rho = ht-(m['k0'][0]-m['K'][0][0]*z)*hz-.5*m['Q'][0][0]*(hzz+hz*hz)+c['risk_free_rate']
    rho += np.sum(pi*(signal-pi), axis=1, keepdims=True)
    return float(rho.min()-padding), float(rho.max()+padding)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=OUT/'seed_11_final.json')
    p.add_argument('--output', type=Path, default=OUT/'neural_log_barrier.json')
    p.add_argument('--time-level', type=int, default=7)
    p.add_argument('--space-level', type=int, default=7)
    p.add_argument('--tolerance', type=float, default=.005)
    p.add_argument('--max-depth', type=int, default=22)
    p.add_argument('--batch', type=int, default=128)
    args = p.parse_args()
    if args.output.resolve() in (args.checkpoint.resolve(), Path(__file__).resolve()):
        p.error('Output must not overwrite an input')
    if not (1 <= args.time_level <= 14 and 1 <= args.space_level <= 14
            and args.tolerance > 0 and math.isfinite(args.tolerance)
            and 0 <= args.max_depth <= 30 and args.batch >= 1):
        p.error('Invalid verifier configuration')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('{"bounded_domain_certified": false, "whole_space_certified": false}\n')
    payload = args.checkpoint.read_bytes()
    c = json.loads(payload)
    if (c['format'] != 'hard_value_net_exact_binary_weights_v1' or c['gamma'] != 2
            or len(c['market']['xbar']) != 1 or c['horizon'] <= 0
            or c['domain_scale'] <= 0 or c['state_dict']['half_ax'][0][0] <= 0
            or c['risk_free_rate'] < 0 or any(x < 0 for x in c['caps'])):
        raise ValueError('Unsupported checkpoint')
    digest = hashlib.sha256(payload).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper = Path(__file__).with_name('neural_certificate_audit.py')
    helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
    start = time.perf_counter()
    nt, nz = 2**args.time_level, 2**args.space_level
    # All cells are dyadic: splitting preserves an exact cover, including edges.
    low = np.full(nt, np.inf)
    high = np.full(nt, -np.inf)
    count, deepest = 0, 0
    widened = 0
    for ti in range(nt):
        area = F(0)
        proposed_lo, proposed_hi = propose_envelope(c, ti/nt, (ti+1)/nt, args.tolerance)
        pending = [((ti+.5)/nt, .5/nt, -1+(zi+.5)*2/nz, 1/nz, 0) for zi in range(nz)]
        while pending:
            batch = pending[-args.batch:]
            del pending[-args.batch:]
            lo, hi, mid = network_residual(c, [x[:4] for x in batch])
            for cell, l, u, v in zip(batch, lo, hi, mid):
                t, dt, z, dz, depth = cell
                # Narrow rigorous boxes may correct an underestimated proposal.
                # Only the actual enclosure endpoints enter the final sums.
                if depth >= 16 and max(u-v, v-l) <= args.tolerance:
                    if l < proposed_lo or u > proposed_hi:
                        widened += 1
                    proposed_lo, proposed_hi = min(proposed_lo, l), max(proposed_hi, u)
                if l >= proposed_lo and u <= proposed_hi:
                    low[ti], high[ti] = min(low[ti], l), max(high[ti], u)
                    count += 1
                    area += F(2, nt*nz*2**depth)
                    deepest = max(deepest, depth)
                else:
                    if depth >= args.max_depth:
                        raise ArithmeticError(f'Maximum depth exceeded at {cell}; width {u-l}')
                    # Alternate bisection after respecting the normalized aspect ratio.
                    if dt >= dz/2:
                        pending.extend([(t-dt/2, dt/2, z, dz, depth+1), (t+dt/2, dt/2, z, dz, depth+1)])
                    else:
                        pending.extend([(t, dt, z-dz/2, dz/2, depth+1), (t, dt, z+dz/2, dz/2, depth+1)])
        if area != F(2, nt):
            raise ArithmeticError('Incomplete or overlapping subdivision area')
        if ti % 8 == 0 or ti == nt-1:
            print(f'Time slabs {ti+1}/{nt}; accepted {count}; rho [{low[:ti+1].min():.6g},{high[:ti+1].max():.6g}]; {time.perf_counter()-start:.1f}s', flush=True)
    from neural_certificate_audit import trace_mismatch, scalar, endpoints
    iv.dps = 50
    # Keep exact rational sums of the outward binary64 envelopes.
    upper = F(c['horizon']) * sum((max(F(0), F(x)) for x in high), F(0))/nt
    lower = F(c['horizon']) * sum((max(F(0), -F(x)) for x in low), F(0))/nt
    zeta = F(trace_mismatch(c)['lateral_log_trace_error_upper'])
    log_width = upper+lower+2*zeta
    ce = endpoints(1-iv.exp(-scalar(log_width)))[1]
    result = {'bounded_domain_certified': True, 'whole_space_certified': False,
              'scope': 'relative CE loss for the same prescribed-trace Dirichlet problem',
              'checkpoint_sha256': digest, 'verifier_source_sha256': source_hash,
              'trace_helper_sha256': helper_hash,
              'policy_updates_after_checkpoint': 0, 'full_closed_cylinder_covered': True,
              'normalized_covered_area_exact': '2',
              'time_level': args.time_level, 'space_level': args.space_level,
              'proposal_padding': args.tolerance, 'accepted_cells': count,
              'proposal_role': 'subdivision thresholds only; all reported bounds use full-cover outward enclosures',
              'rigorous_proposal_widenings': widened,
              'deepest_subdivision': deepest, 'lower_residual_by_slab': low.tolist(),
              'upper_residual_by_slab': high.tolist(),
              'lower_log_correction_exact': str(upper+zeta),
              'upper_log_correction_exact': str(lower+zeta),
              'total_log_width_interval': endpoints(scalar(log_width)),
              'relative_ce_loss_upper': ce, 'relative_ce_loss_upper_bp': endpoints(scalar(F(ce))*10000)[1],
              'arithmetic_model': 'IEEE-754 binary64 basic arithmetic and dot-product forward-error bounds; polynomial transcendental enclosure',
              'seconds': time.perf_counter()-start}
    if (hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != digest
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash
            or hashlib.sha256(helper.read_bytes()).hexdigest() != helper_hash):
        raise ArithmeticError('Inputs changed during verification')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_by_slab')}, indent=2))


if __name__ == '__main__':
    main()
