"""GPU proposals for coupled multi-factor portfolio certificates.

The candidate is explicitly log-quadratic in space, NOT a dense neural net.
Sampled optimization never certifies a result. highdim_local_verify.py checks
the rounded polynomial proposals independently over continuous domains.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = {'kappa': '1', 'drift_neighbor': '1/5', 'variance': '1/16',
         'cross': '1/8', 'lambda_constant': '1/5', 'lambda_self': '1/2',
         'lambda_neighbor': '1/4', 'cap': '11/100', 'risk_free_rate': '3/100',
         'horizon': '1', 'radius': '3/2', 'gamma': '2'}


def parameters():
    return {k: float(F(v)) for k, v in MODEL.items()}


def basis(t, degree, xp=np):
    """Return Chebyshev, tau*Chebyshev, and its time derivative (T=1)."""
    one = t*0+1
    cheb, deriv = [one], [t*0]
    if degree > 1:
        cheb.append(2*t-1)
        deriv.append(2*one)
    for k in range(2, degree):
        cheb.append(2*(2*t-1)*cheb[-1]-cheb[-2])
        deriv.append(4*cheb[-2]+2*(2*t-1)*deriv[-1]-deriv[-2])
    axis = -1
    if xp is torch:
        cc, dd = torch.stack(cheb, dim=axis), torch.stack(deriv, dim=axis)
    else:
        cc, dd = np.stack(cheb, axis=axis), np.stack(deriv, axis=axis)
    return cc, t[..., None]*cc, cc+t[..., None]*dd


def design(t, x, degree):
    cheb, b, bt = basis(t, degree)
    value = np.concatenate((b, b*x[:, None], b*x[:, None]**2), axis=1)
    derivative = np.concatenate((b*0, b, 2*b*x[:, None]), axis=1)
    second = np.concatenate((b*0, b*0, 2*b), axis=1)
    dt = np.concatenate((bt, bt*x[:, None], bt*x[:, None]**2), axis=1)
    trace = np.concatenate((cheb, cheb*x[:, None], cheb*x[:, None]**2), axis=1)
    return value, dt, derivative, second, trace


def local_residual(coeff, t, x, y, dimension, fixed=None):
    m = parameters()
    degree = len(coeff)//3
    v, vt, vx, vxx, _ = design(t, x, degree)
    fz = vx@coeff
    g = m['cross']/math.sqrt(dimension)
    lam = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y
    policy_grad = fz if fixed is None else vx@fixed
    pi = np.clip((lam+g*policy_grad)/2, -m['cap'], m['cap'])
    ans = vt@coeff+(m['kappa']*x-m['drift_neighbor']*y)*fz
    ans -= m['variance']/2*(vxx@coeff+fz*fz/dimension)
    ans += pi*(lam+g*fz)-pi*pi
    return ans, pi, fz


def train(dimension, degree, epochs, seed=1103, device='cuda'):
    m = parameters()
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = torch.Generator(device=device).manual_seed(seed)
    coeff = torch.zeros((3, degree), device=device, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        coeff[0, 0] = -.01
        coeff[1, 0] = -.02
        coeff[2, 0] = -.01
    optimizer = torch.optim.Adam([coeff], lr=.001)
    shift = np.roll(np.eye(dimension), -1, axis=1)
    drift = m['kappa']*np.eye(dimension)-m['drift_neighbor']*shift
    stationary = solve_continuous_lyapunov(drift, m['variance']*np.eye(dimension))
    chol = torch.tensor(np.linalg.cholesky(stationary), device=device, dtype=torch.float64)
    n = 2048
    for epoch in range(epochs):
        t = torch.rand(n, generator=rng, device=device, dtype=torch.float64)
        z = torch.randn((n, dimension), generator=rng, device=device, dtype=torch.float64)@chol.T
        tail = torch.rand((n, dimension), generator=rng, device=device, dtype=torch.float64)*3-1.5
        mask = torch.rand(n, generator=rng, device=device, dtype=torch.float64) < .02
        z = torch.where(mask[:, None], tail, z)
        zn = z.roll(-1, dims=1)
        _, bt, dt = basis(t, degree, torch)
        f, ft = bt@coeff.T, dt@coeff.T
        fz = f[:, 1:2]+2*f[:, 2:3]*z
        ftime = ft[:, 0:1]+ft[:, 1:2]*z+ft[:, 2:3]*z*z
        lam = m['lambda_constant']+m['lambda_self']*z+m['lambda_neighbor']*zn
        signal = lam+m['cross']/math.sqrt(dimension)*fz
        pi = torch.clamp(signal/2, -m['cap'], m['cap'])
        residual = ftime+(m['kappa']*z-m['drift_neighbor']*zn)*fz
        residual -= m['variance']/2*(2*f[:, 2:3]+fz*fz/dimension)
        residual += pi*signal-pi*pi
        loss = residual.mean(1).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return coeff.detach().cpu().numpy().ravel(), {'seconds': time.perf_counter()-start,
                                                 'final_sampled_loss': float(loss.detach()),
                                                 'epochs': epochs, 'seed': seed, 'device': device,
                                                 'gpu': torch.cuda.get_device_name() if device.startswith('cuda') else None}


def fit_barriers(base, dimension, iterations=3):
    """Sampled sequential LP proposals, with telescoping edge potentials."""
    start = time.perf_counter()
    m, k = parameters(), len(base)//3
    ts = (1-np.cos(np.linspace(0, np.pi, 17)))/2
    zs = np.unique(np.r_[np.linspace(-m['radius'], m['radius'], 33), np.linspace(-.4, .4, 17)])
    tt, xx, yy = np.meshgrid(ts, zs, zs, indexing='ij')
    t, x, y = tt.ravel(), xx.ravel(), yy.ravel()
    _, vt, vx, vxx, trace = design(t, x, k)
    cb = basis(t, k)[0]
    pot = np.concatenate((cb*(y-x)[:, None], cb*(y*y-x*x)[:, None]), axis=1)
    tu, xu = np.meshgrid(ts, zs, indexing='ij')
    _, _, dx_unary, _, trace_unary = design(tu.ravel(), xu.ravel(), k)
    end = np.r_[np.ones(k), np.zeros(4*k)]
    bounds = [(-3., 3.)]*(5*k)
    outputs, logs = {}, []
    for side in ('lower', 'upper'):
        prev = base.copy()
        # The trust restriction makes the concave upper-PDE tangent conservative.
        trust = .08
        for iteration in range(iterations):
            residual, pi, grad = local_residual(prev, t, x, y, dimension,
                                                fixed=base if side == 'upper' else None)
            b = m['kappa']*x-m['drift_neighbor']*y-m['variance']/dimension*grad
            b += pi*m['cross']/math.sqrt(dimension)
            op = vt+b[:, None]*vx-m['variance']/2*vxx
            matrix = np.concatenate((op, pot), axis=1)
            rhs = op@prev-residual
            tr = np.concatenate((trace_unary, np.zeros((len(trace_unary), 2*k))), axis=1)
            if side == 'lower':
                aa = np.concatenate((matrix, tr))
                bb = np.r_[rhs-1e-8, trace_unary@base]
                objective = -end
            else:
                trust_mat = np.concatenate((dx_unary, np.zeros((len(dx_unary), 2*k))), axis=1)
                aa = np.concatenate((-matrix, -tr, trust_mat, -trust_mat))
                bb = np.r_[-rhs-1e-8-m['variance']/(2*dimension)*trust**2,
                            -trace_unary@base, dx_unary@prev+trust, -dx_unary@prev+trust]
                objective = end
            sol = linprog(objective, A_ub=aa, b_ub=bb, bounds=bounds, method='highs')
            if not sol.success:
                raise RuntimeError(f'{side} LP failed: {sol.message}')
            prev = sol.x[:3*k]
            edge = sol.x[3*k:]
            actual = local_residual(prev, t, x, y, dimension,
                                    fixed=base if side == 'upper' else None)[0]+pot@edge
            logs.append({'side': side, 'iteration': iteration,
                         'endpoint': float(end@sol.x), 'sampled_residual_min': float(actual.min()),
                         'sampled_residual_max': float(actual.max()), 'lp_objective': float(sol.fun)})
        outputs[side] = {'coeff': prev, 'potential': edge}
    outputs['seconds'] = time.perf_counter()-start
    outputs['log'] = logs
    return outputs


def rounded(values):
    return [str(F(round(float(v)*2**36), 2**36)) for v in values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dimensions', nargs='+', type=int, default=[3, 5, 10, 20])
    p.add_argument('--degree', type=int, default=5)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', default='results/highdim_local')
    args = p.parse_args()
    if min(args.dimensions) < 3 or not 2 <= args.degree <= 12 or args.epochs < 1:
        p.error('At least 3 states, 2..12 time terms, and positive epochs required')
    out = ROOT/args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for d in args.dimensions:
        base, training = train(d, args.degree, args.epochs, device=args.device)
        print(f'd={d} trained: {training}', flush=True)
        barriers = fit_barriers(base, d)
        data = {'model': MODEL, 'dimension': d, 'assets': d, 'time_terms': args.degree,
                'candidate_type': 'additive log-quadratic, polynomial in time; not a dense neural network',
                'normalization': 'lambda and box exposure caps divided by exact sqrt(d)',
                'state_interaction': 'directed nearest-neighbor cycle in drift and risky premia',
                'base': rounded(base),
                'lower': {k: rounded(v) for k, v in barriers['lower'].items()},
                'upper': {k: rounded(v) for k, v in barriers['upper'].items()},
                'prescribed_trace': 'trace of the frozen positive generating candidate on the cube',
                'outside_policy': 'zero outside the cube; same coupled greedy policy upon re-entry',
                'training': training, 'barrier_fit_seconds': barriers['seconds'],
                'barrier_fit_diagnostics': barriers['log'], 'fit_source_sha256': digest,
                'certified': False}
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Fit source changed during run')
        target = out/f'candidate_d{d}.json'
        target.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
        lo = sum(map(F, data['lower']['coeff'][:args.degree]))
        hi = sum(map(F, data['upper']['coeff'][:args.degree]))
        print(f'd={d} sampled-proposal endpoint width={float(hi-lo):.8g}, '
              f'gap_bp={1e4*(-math.expm1(float(lo-hi))):.6f}; NOT CERTIFIED', flush=True)


if __name__ == '__main__':
    main()
