"""Independent one-state linear-PDE check of the financial dual MC moments.

Float64 central-space/implicit-time refinement is a diagnostic, not a certified
continuum enclosure. No neural weights or policies are optimized here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_banded
import torch

from neural_duality_comparison import CHECKPOINT, EXPECTED, FrozenNetwork, Market

ROOT = Path(__file__).resolve().parents[1]


def implicit_step(previous, drift, reaction, Q, dz, dt, boundary):
    drift, reaction = np.asarray(drift)[1:-1], np.asarray(reaction)[1:-1]
    lower = Q/(2*dz*dz)-drift/(2*dz)
    upper = Q/(2*dz*dz)+drift/(2*dz)
    if min(lower.min(), upper.min()) < 0:
        raise ArithmeticError('Nonmonotone spatial operator; refine the mesh')
    diagonal = 1+dt*Q/(dz*dz)-dt*reaction
    if np.min(1-dt*reaction) <= 0:
        raise ArithmeticError('Implicit comparison condition failed')
    ab = np.zeros((3, len(diagonal)))
    ab[0, 1:], ab[1], ab[2, :-1] = -dt*upper[:-1], diagonal, -dt*lower[1:]
    rhs = previous[1:-1].copy()
    rhs[0] += dt*lower[0]*boundary[0]
    rhs[-1] += dt*upper[-1]*boundary[1]
    out = np.empty_like(previous)
    out[0], out[-1] = boundary
    out[1:-1] = solve_banded((1, 1), ab, rhs, check_finite=False)
    if not np.isfinite(out).all() or out.min() <= 0:
        raise ArithmeticError('Nonpositive or nonfinite PDE moment')
    return out


@torch.no_grad()
def solve_moments(m, steps, points_per_half, outer_multiple=3):
    start = time.perf_counter()
    # Cell centers put the discontinuous deployment boundary between nodes.
    dz = m.half/points_per_half
    z = m.center+(np.arange(-outer_multiple*points_per_half,
                           outer_multiple*points_per_half)+.5)*dz
    dt = m.T/steps
    pvalue, dvalue = np.ones(len(z)), np.ones(len(z))
    coefficients_seconds, solve_seconds = 0., 0.
    # Direct NN evaluation: no lookup interpolation in this independent check.
    for first in range(1, steps+1, 16):
        ti = np.arange(first, min(first+16, steps+1))
        tt, zz = np.broadcast_arrays((ti*dt)[:, None], z[None, :])
        tick = time.perf_counter()
        tg, zg = m.net.tensor(tt.copy().ravel()), m.net.tensor(zz.copy().ravel())
        hz = m.net.jet(tg, zg)[1]
        bx, bd, ps, po, ts, to, _ = m.coefficients(zg, hz)
        b = m.k0-m.k*zg
        rows = torch.stack((b-m.sq*ps, -bx+(ps*ps+po*po)/2,
                            b-m.sq*ts/2, bd/2+(ts*ts+to*to)/8), -1)
        rows = rows.cpu().numpy().reshape(len(ti), len(z), 4)
        coefficients_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        for i, row in zip(ti, rows):
            pboundary = np.exp(-m.r*i*dt)
            dboundary = np.exp(-m.r*i*dt/2)
            pvalue = implicit_step(pvalue, row[:, 0], row[:, 1], m.q, dz, dt, (pboundary, pboundary))
            dvalue = implicit_step(dvalue, row[:, 2], row[:, 3], m.q, dz, dt, (dboundary, dboundary))
        solve_seconds += time.perf_counter()-tick
    gp, md = np.interp(m.center, z, pvalue), np.interp(m.center, z, dvalue)
    return {'steps': steps, 'space_points': len(z), 'points_per_original_half_width': points_per_half,
            'outer_multiple': outer_multiple, 'dz': dz, 'dt': dt,
            'policy_factor': float(gp), 'dual_sqrt_density_mean': float(md),
            'dual_factor': float(md*md), 'relative_ce_duality_gap': float(1-md*md/gp),
            'seconds': time.perf_counter()-start, 'gpu_coefficients_seconds': coefficients_seconds,
            'cpu_banded_solve_seconds': solve_seconds,
            'continuous_time_certified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, nargs='+', default=[512, 1024, 2048, 4096])
    p.add_argument('--space', type=int, nargs='+', default=[128, 256, 512, 1024])
    p.add_argument('--outer', type=int, default=3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default='results/neural_duality/pde_refinement.json')
    args = p.parse_args()
    if len(args.steps) != len(args.space) or min(args.steps+args.space) <= 0 or args.outer < 2:
        p.error('Pair positive time/space resolutions, with outer >= 2')
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Changed fixed checkpoint')
    c = json.loads(CHECKPOINT.read_text())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = Market(c, FrozenNetwork(c, args.device))
    result = {'checkpoint_sha256': EXPECTED, 'continuous_time_certified': False,
              'method': 'direct-NN coefficients, monotone central-space / implicit-Euler linear PDEs',
              'far_boundary': 'risk-free primal and square-root-risk-free dual; assessed by domain refinement',
              'source_sha256': {name: hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest()
                                for name in ('neural_duality_pde.py', 'neural_duality_comparison.py')},
              'args': vars(args), 'levels': [], 'complete': False}
    out = ROOT/args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    for steps, space in zip(args.steps, args.space):
        row = solve_moments(m, steps, space, args.outer)
        result['levels'].append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for name, digest in result['source_sha256'].items():
        if hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed during PDE evaluation')
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('Checkpoint changed during PDE evaluation')
    result['complete'] = True
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
