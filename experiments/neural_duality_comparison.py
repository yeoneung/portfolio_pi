"""GPU primal/dual diagnostics for the unchanged seed-11 neural policy.

Adaptation of Haugh--Kogan--Wang (2006), equations (7)--(9), (15).
See results/neural_duality/METHOD.md for signs, the unspanned Brownian risk,
and the exact discrete martingale control variate. These Monte Carlo outputs
are NOT outward-rounded continuous-time welfare certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'results/neural_certificate/seed_11_final.json'
EXPECTED = '47628e7e10fb3619a9ac44c01c35270e7074bc56e0b13e679765ebb21b9ad9d7'


class FrozenNetwork:
    """Analytic forward derivative, with the stored normalization untouched."""

    def __init__(self, c, device='cpu', dtype=torch.float64):
        self.c, self.device, self.dtype = c, device, dtype
        s = c['state_dict']
        self.center, self.half = s['center'][0][0], s['half_ax'][0][0]
        self.weights = [self.tensor(s[f'net.{2*i}.weight']).T.contiguous()
                        for i in range(c['config']['depth']+1)]
        self.biases = [self.tensor(s[f'net.{2*i}.bias'])
                       for i in range(c['config']['depth']+1)]

    def tensor(self, a):
        return torch.as_tensor(a, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def jet(self, tau, z):
        tau, z = torch.broadcast_tensors(self.tensor(tau), self.tensor(z))
        y = (z-self.center)/self.half
        d, dz = 1-y*y, -2*y/self.half
        v = torch.stack((tau/self.c['horizon'], y, d), -1)
        vz = torch.stack((torch.zeros_like(y), torch.ones_like(y)/self.half, dz), -1)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            v, vz = v@w+b, vz@w
            if i < len(self.weights)-1:
                v = torch.tanh(v)
                vz = (1-v*v)*vz
        v, vz = v[..., 0], vz[..., 0]
        h = -self.c['risk_free_rate']*tau+tau*d*v
        return h, tau*(dz*v+d*vz)


class Market:
    def __init__(self, c, net):
        self.c, self.net = c, net
        m = c['market']
        if c['gamma'] != 2 or len(m['xbar']) != 1:
            raise ValueError('Only the supplied gamma=2, one-state model is implemented')
        self.l0 = net.tensor(m['lam0'])
        self.l1 = net.tensor(np.asarray(m['Lam'])[:, 0])
        self.cross = net.tensor(np.asarray(m['Gamma'])[:, 0])
        self.cap = net.tensor(c['caps'])
        self.q = m['Q'][0][0]
        if self.q <= float((self.cross*self.cross).sum()):
            raise ValueError('Joint covariance must be positive definite')
        self.sq = math.sqrt(self.q)
        self.k, self.k0 = m['K'][0][0], m['k0'][0]
        self.center = m['xbar'][0]
        self.half = c['domain_scale']*m['eta'][0]
        self.r, self.T = c['risk_free_rate'], c['horizon']

    def coefficients(self, z, hz):
        """Return projected diffusions and drifts; no unspanned risk is dropped.

        theta=(2*pdual-Gamma*hz, -sqrt(Q-|Gamma|^2)*hz).
        Orthogonal to the state Brownian, theta_perp=2*pdual_perp.
        Outside p=0, while the dual uses a feasible myopic-price candidate.
        """
        inside = (z-self.center).abs() <= self.half
        hz = torch.where(inside, hz, 0.)
        lam = self.l0+z[..., None]*self.l1
        pd = torch.clamp((lam+hz[..., None]*self.cross)/2, -self.cap, self.cap)
        pd2 = (pd*pd).sum(-1)
        pg = (pd*self.cross).sum(-1)
        psd = pg/self.sq
        pod = torch.sqrt(torch.clamp(pd2-psd*psd, min=0.))
        ps, po = torch.where(inside, psd, 0.), torch.where(inside, pod, 0.)
        p2 = torch.where(inside, pd2, 0.)
        plam = torch.where(inside, (pd*lam).sum(-1), 0.)
        ts, to = 2*psd-self.sq*hz, 2*pod
        theta2 = ts*ts+to*to
        nu = 2*pd-hz[..., None]*self.cross-lam
        delta = (nu.abs()*self.cap).sum(-1)
        # In the interior the exact clamp gives support complementarity.
        # Retain the computed nonnegative slack, rather than assume equality.
        a = delta+torch.where(inside, (pd*nu).sum(-1), 0.)
        a = torch.clamp(a, min=0.)
        return self.r+plam-p2/2, -(self.r+delta)-theta2/2, ps, po, ts, to, a


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


@torch.no_grad()
def tabulate(m, steps, space_points, batch=32768):
    """Only h_z is interpolated. Time nodes match each simulation step exactly."""
    if steps <= 0 or space_points < 3:
        raise ValueError('Invalid lookup grid')
    start = time.perf_counter()
    z = torch.linspace(m.center-m.half, m.center+m.half, space_points,
                       device=m.net.device, dtype=m.net.dtype)
    total = steps*space_points
    table = torch.empty(total, device=m.net.device, dtype=m.net.dtype)
    for first in range(0, total, batch):
        ix = torch.arange(first, min(total, first+batch), device=m.net.device)
        tau = m.T*(1-(ix//space_points).to(m.net.dtype)/steps)
        table[first:first+len(ix)] = m.net.jet(tau, z[ix % space_points])[1]
    synchronize(m.net.device)
    return table.reshape(steps, space_points), time.perf_counter()-start


def lookup(m, row, z):
    x = torch.clamp((z-m.center+m.half)/(2*m.half)*(len(row)-1), 0., len(row)-1.)
    ix = x.long().clamp(max=len(row)-2)
    f = x-ix
    return row[ix]*(1-f)+row[ix+1]*f


@torch.no_grad()
def lookup_check(m, table, seed=987123, size=16384):
    rng = np.random.default_rng(seed)
    ti = rng.integers(0, len(table), size)
    # Include concentrated probes next to both prescribed boundaries.
    y = rng.uniform(-1, 1, size)
    y[:1000] = rng.choice([-1, 1], 1000)*(1-rng.uniform(0, 1e-3, 1000))
    z = m.net.tensor(m.center+m.half*y)
    tau = m.net.tensor(m.T*(1-ti/len(table)))
    _, exact = m.net.jet(tau, z)
    rows = table[torch.as_tensor(ti, device=m.net.device)]
    x = (z-m.center+m.half)/(2*m.half)*(table.shape[1]-1)
    ix = x.long().clamp(0, table.shape[1]-2)
    f = x-ix
    ii = torch.arange(size, device=m.net.device)
    approx = rows[ii, ix]*(1-f)+rows[ii, ix+1]*f
    error = float((exact-approx).abs().max())
    # Clamp is 1-Lipschitz; this is a SAMPLED bound only.
    return {'sample_count': size, 'sampled_max_log_gradient_error': error,
            'sampled_policy_coordinate_error_upper': error*float(m.cross.abs().max())/2,
            'certified': False}


def summarize(samples, reference_factor):
    """Paired delta-method normal intervals for discrete MC expectations."""
    x = np.asarray(samples, dtype=float)
    n = len(x)
    p, s, k = x.T
    cv = np.column_stack((p-reference_factor*(k-1),
                          s-math.sqrt(reference_factor)*(k-1)))
    def estimate(y):
        means = y.mean(0)
        covariance = np.cov(y.T, ddof=1)/n
        gp, ds = means
        gap = 1-ds*ds/gp
        derivative = np.array([ds*ds/gp**2, -2*ds/gp])
        se = math.sqrt(max(0., derivative@covariance@derivative))
        bias = -ds*ds*covariance[0, 0]/gp**3+2*ds*covariance[0, 1]/gp**2-covariance[1, 1]/gp
        return {'policy_factor': float(gp), 'dual_sqrt_density_mean': float(ds),
                'dual_factor_plugin': float(ds*ds), 'relative_ce_gap_plugin': float(gap),
                'gap_standard_error': se, 'gap_95_interval': [float(gap-1.96*se), float(gap+1.96*se)],
                'gap_one_sided_95_upper': float(gap+1.6448536269514722*se),
                'estimated_second_order_plugin_bias': float(bias),
                'covariance_of_means': covariance.tolist()}
    return {'paths': n, 'raw': estimate(x[:, :2]), 'control_variate': estimate(cv),
            'martingale_control_mean': float(k.mean()),
            'martingale_control_standard_error': float(k.std(ddof=1)/math.sqrt(n)),
            'cv_coefficients_fixed_before_sampling': [reference_factor, math.sqrt(reference_factor)]}


@torch.no_grad()
def simulate(m, steps, paths, seed, table=None):
    """Exact OU/state--Brownian coupling; left-frozen log-wealth/deflator.

    There are only three independent normals per step: the state increment,
    its exact-OU bridge, and the one needed orthogonal financial projection.
    All 30 assets and the unspanned factor remain in the coefficients.
    """
    start = time.perf_counter()
    gen = torch.Generator(device=m.net.device).manual_seed(seed)
    dt, sqdt = m.T/steps, math.sqrt(m.T/steps)
    decay = math.exp(-m.k*dt)
    bridge_mean = -math.expm1(-m.k*dt)/(m.k*dt)
    bridge_sd = math.sqrt(max(0., -math.expm1(-2*m.k*dt)/(2*m.k)-bridge_mean**2*dt))
    z = m.net.tensor(np.full(paths, m.center))
    lx, ld, comp = torch.zeros_like(z), torch.zeros_like(z), torch.zeros_like(z)
    outside_steps = torch.zeros_like(z)
    for i in range(steps):
        hz = lookup(m, table[i], z) if table is not None else m.net.jet(m.T-i*dt, z)[1]
        bx, bd, ps, po, ts, to, slack = m.coefficients(z, hz)
        noise = torch.randn((3, paths), generator=gen, device=m.net.device, dtype=m.net.dtype)
        db, do = sqdt*noise[0], sqdt*noise[2]
        comp += torch.exp(lx+ld)*(-torch.expm1(-slack*dt))
        lx += bx*dt+ps*db+po*do
        ld += bd*dt-ts*db-to*do
        outside_steps += ((z-m.center).abs() > m.half).to(m.net.dtype)
        z = m.k0/m.k+(z-m.k0/m.k)*decay+m.sq*(bridge_mean*db+bridge_sd*noise[1])
    samples = torch.stack((torch.exp(-lx), torch.exp(ld/2), torch.exp(lx+ld)+comp), -1)
    if not torch.isfinite(samples).all():
        raise ArithmeticError('Nonfinite Monte Carlo path payoff')
    data = samples.cpu().numpy()
    synchronize(m.net.device)
    return data, {'seed': seed, 'steps': steps, 'paths': paths,
                  'seconds': time.perf_counter()-start,
                  'outside_time_fraction': float(outside_steps.mean()/steps),
                  'paths_with_outside_grid_point_fraction': float((outside_steps > 0).double().mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, nargs='+', default=[256, 512, 1024])
    p.add_argument('--paths', type=int, default=65536)
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303, 404, 505])
    p.add_argument('--space-points', type=int, default=4097)
    p.add_argument('--device', default='cuda')
    p.add_argument('--direct', action='store_true', help='Evaluate the unchanged NN at every path point')
    p.add_argument('--output', default='results/neural_duality/comparison.json')
    args = p.parse_args()
    if args.paths < 3 or min(args.steps) < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error('Positive steps, at least three paths, and distinct seeds required')
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError('Frozen checkpoint changed; refusing this matched comparison')
    c = json.loads(CHECKPOINT.read_text())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    net = FrozenNetwork(c, args.device)
    m = Market(c, net)
    reference = math.exp(float(net.jet(m.T, m.center)[0]))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {'method': 'HKW-2006 CRRA/box/incomplete-market adaptation with discrete martingale CV',
              'checkpoint_sha256': digest, 'source_sha256': source_hash,
              'continuous_time_certified': False, 'policy_retrained': False,
              'model': 'unchanged seed-11 30-asset, one-state, gamma=2, T=3',
              'outside_policy': 'zero exposure, NN greedy upon re-entry',
              'dual_outside': 'myopic constrained price candidate; zero log gradient',
              'precision': 'torch.float64; TF32 disabled',
              'policy_evaluation': 'direct NN' if args.direct else 'spatially interpolated NN log gradient, exact clamp',
              'ci_scope': 'asymptotic Monte Carlo sampling only; excludes discretization and interpolation bias',
              'refinement_coupling': 'same RNG seeds, but not nested Brownian increments; marginal SEs reported',
              'reference_factor_cv': reference, 'args': vars(args),
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'numpy': np.__version__, 'device': args.device,
                              'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None},
              'levels': []}
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for steps in args.steps:
        table, elapsed = (None, 0.) if args.direct else tabulate(m, steps, args.space_points)
        check = None if table is None else lookup_check(m, table)
        print(f'table steps={steps} seconds={elapsed:.2f} check={check}', flush=True)
        level = {'steps': steps, 'table_seconds': elapsed, 'lookup_check': check, 'runs': []}
        all_samples = []
        for seed in args.seeds:
            samples, timing = simulate(m, steps, args.paths, seed, table)
            all_samples.append(samples)
            row = {**timing, **summarize(samples, reference)}
            level['runs'].append(row)
            cv = row['control_variate']
            print(f'steps={steps} seed={seed} sec={row["seconds"]:.2f} '
                  f'gap_bp={1e4*cv["relative_ce_gap_plugin"]:.5f} '
                  f'se_bp={1e4*cv["gap_standard_error"]:.5f}', flush=True)
        level['pooled'] = summarize(np.concatenate(all_samples), reference)
        level['simulation_seconds'] = sum(row['seconds'] for row in level['runs'])
        result['levels'].append(level)
        result['complete'] = False
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        del table
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during simulation')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source changed during simulation')
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
