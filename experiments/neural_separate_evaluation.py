"""Separate linear-PDE evaluation of the unchanged seed-11 neural policy.

Numerical fitting is not a certificate. The deployed policy is zero outside
its original domain, including when the evaluation domain is enlarged.
The positive factor is represented as exp(-r*tau) P(tau,z), with P=1 on
the terminal/lateral trace. Shared Hermite coefficients impose exact C1
spatial matching. This module never updates the deployed policy.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
import torch

from local_policy_evaluation import spatial_basis, spatial_values, time_basis
from neural_duality_comparison import CHECKPOINT, EXPECTED, FrozenNetwork, Market

ROOT = Path(__file__).resolve().parents[1]


def knots_for(level, outer, boundary_levels):
    # y is in [0,1]; original deployment boundaries are included exactly.
    n = 2**level
    points = {F(i, 2*outer*n) for i in range(2*outer*n+1)}
    for location in (F(0), F(1), F(outer-1, 2*outer), F(outer+1, 2*outer)):
        for j in range(1, boundary_levels+1):
            for sign in (-1, 1):
                y = location+sign*F(1, 2*outer*n*2**j)
                if 0 < y < 1:
                    points.add(y)
    return sorted(points)


@torch.no_grad()
def coefficients(m, tau, z, inside):
    """Take the interior/exterior one-sided limit on interface endpoints."""
    zt, tt = m.net.tensor(z), m.net.tensor(tau)
    if inside:
        hz = m.net.jet(tt, zt)[1]
        lam = m.l0+zt[:, None]*m.l1
        pi = torch.clamp((lam+hz[:, None]*m.cross)/2, -m.cap, m.cap)
        shift = pi@m.cross
        reaction = (pi*(pi-lam)).sum(-1)
    else:
        shift, reaction = torch.zeros_like(zt), torch.zeros_like(zt)
    return ((m.k0-m.k*zt-shift).cpu().numpy(), reaction.cpu().numpy())


def cell_data(m, knots, cell, dt, dz, s, x, outer, power):
    h = float(knots[cell+1]-knots[cell])
    y = float(knots[cell])+h*x
    z = m.center+outer*m.half*(2*y-1)
    midpoint = (knots[cell]+knots[cell+1])/2
    inside = F(outer-1, 2*outer) < midpoint < F(outer+1, 2*outer)
    drift, reaction = coefficients(m, m.T*s**power, z, inside)
    basis = spatial_basis(knots, cell, dz)
    bx, by, byy = spatial_values(basis, x)
    length = 2*outer*m.half*h
    by, byy = by/length, byy/length**2
    bs, bt = time_basis(dt, s, power)
    # L P = P_tau - (Q/2 P_zz + beta P_z + c0 P), c0=c+r.
    a = bt[:, :, None]*bx[:, None, :]/m.T
    a -= bs[:, :, None]*(m.q/2*byy+drift[:, None]*by+reaction[:, None]*bx)[:, None, :]
    return a.reshape(len(s), -1), -reaction, [i for i, _ in basis], (bx, by, byy, bs, bt), inside


def fit(m, degree_time=20, degree_space=6, level=3, outer=1,
        power=2, boundary_levels=3, reweights=0):
    start = time.perf_counter()
    knots = knots_for(level, outer, boundary_levels)
    n, nb = len(knots)-1, (degree_space-1)*(len(knots)-1)
    sa = (1-np.cos(np.linspace(0, np.pi, 2*degree_time+9)))/2
    xa = (1-np.cos(np.linspace(0, np.pi, 2*degree_space+9)))/2
    ss, xx = np.meshgrid(sa, xa, indexing='ij')
    s, x = ss.ravel(), xx.ravel()
    rows, columns, entries, offsets = [], [], [], []
    for cell in range(n):
        a, b, indices, _, _ = cell_data(m, knots, cell, degree_time, degree_space, s, x, outer, power)
        cols = (np.arange(degree_time+1)[:, None]*nb+np.array(indices)[None, :]).ravel()
        rows.append(np.repeat(np.arange(len(s))+cell*len(s), len(cols)))
        columns.append(np.tile(cols, len(s)))
        entries.append(a.ravel())
        offsets.append(b)
    a = coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
                   shape=(n*len(s), (degree_time+1)*nb)).tocsr()
    b = np.concatenate(offsets)
    del rows, columns, entries, offsets
    # Integrate in physical time. Equal weights at s=0 would force the
    # incompatible terminal/corner PDE at a set of time measure zero and
    # spread that discrepancy over the whole three-year horizon.
    time_weights = np.sqrt(np.maximum(0., power*s**(power-1)*np.sqrt(s*(1-s))))
    weights = np.tile(time_weights, n)
    history = []
    for iteration in range(reweights+1):
        weighted = a.multiply(weights[:, None]).tocsr()
        norms = np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).ravel())
        scaled = weighted@diags(1/norms)
        factor = splu((scaled.T@scaled).tocsc())
        rhs = -b*weights
        solution = factor.solve(np.asarray(scaled.T@rhs).ravel())
        for _ in range(3):
            solution += factor.solve(np.asarray(scaled.T@(rhs-scaled@solution)).ravel())
        coeff = solution/norms
        residual = a@coeff+b
        maximum = float(abs(residual).max())
        history.append(maximum)
        print(f'Fit {iteration}: sampled absolute residual {maximum:.6g}; {time.perf_counter()-start:.1f}s', flush=True)
        if iteration < reweights:
            target = np.maximum(abs(residual), .01*maximum)
            weights = np.sqrt(weights*target/np.mean(target))
            weights /= max(weights)
    den = 2**44
    if np.max(abs(coeff))*den >= 2**62:
        raise ArithmeticError('Coefficient quantization overflow')
    nums = np.rint(coeff*den).astype(np.int64).reshape(degree_time+1, nb)
    return {'format': 'neural_fixed_policy_separate_evaluation_v1',
            'checkpoint_sha256': EXPECTED, 'policy_updates_after_checkpoint': 0,
            'outside_policy': 'zero outside original domain, unchanged neural rule after re-entry',
            'knots_exact': list(map(str, knots)), 'degree_time': degree_time,
            'degree_space': degree_space, 'outer_multiple': outer, 'time_power': power,
            'coefficient_denominator': str(den),
            'coefficient_numerators': [[str(int(v)) for v in row] for row in nums],
            'sampled_fitting_residual_NOT_certificate': history,
            'fitting_seconds': time.perf_counter()-start, 'certified': False}


def fit_marching(m, degree_space=6, level=3, outer=1, boundary_levels=4,
                 degree_time=5, time_level=5, comparison=False, knots_override=None):
    """Independent polynomial time slabs with exact continuous joining."""
    start = time.perf_counter()
    knots = knots_for(level,outer,boundary_levels) if knots_override is None else list(map(F,knots_override))
    if knots[0]!=0 or knots[-1]!=1 or any(a>=b for a,b in zip(knots,knots[1:])):
        raise ValueError('Invalid fitting mesh')
    n, nb = len(knots)-1, (degree_space-1)*(len(knots)-1)
    # Time knots are dyadic; geometric refinement resolves the terminal layer.
    ts = sorted({F(i,2**time_level) for i in range(2**time_level+1)} |
                {F(1,2**j) for j in range(time_level+1,time_level+12)})
    sa = (1-np.cos(np.linspace(0,np.pi,2*degree_time+7)))/2
    xa = (1-np.cos(np.linspace(0,np.pi,2*degree_space+7)))/2
    ss, xx = np.meshgrid(sa,xa,indexing='ij')
    s,x = ss.ravel(),xx.ravel()
    bs,bt = time_basis(degree_time,s,1)
    blocks=[]
    for cell in range(n):
        h=float(knots[cell+1]-knots[cell])
        z=m.center+outer*m.half*(2*(float(knots[cell])+h*x)-1)
        midpoint=(knots[cell]+knots[cell+1])/2
        inside=F(outer-1,2*outer)<midpoint<F(outer+1,2*outer)
        basis=spatial_basis(knots,cell,degree_space)
        bx,by,byy=spatial_values(basis,x)
        length=2*outer*m.half*h
        indices=np.array([i for i,_ in basis])
        cols=(np.arange(degree_time+1)[:,None]*nb+indices[None,:]).ravel()
        blocks.append((z,inside,indices,bx,by/length,byy/length**2,cols))
    rows=np.concatenate([np.repeat(np.arange(len(s))+j*len(s),len(b[-1])) for j,b in enumerate(blocks)])
    cols=np.concatenate([np.tile(b[-1],len(s)) for b in blocks])
    den=2**44
    current=np.zeros(nb,dtype=np.int64)
    slabs=[]
    for ti,(left,right) in enumerate(zip(ts[:-1],ts[1:])):
        delta=float(right-left)*m.T
        tau=m.T*(float(left)+float(right-left)*s)
        prior=np.zeros((degree_time+1,nb))
        for improvement in range(8 if comparison else 1):
            entries,offsets=[],[]
            for z,inside,indices,bx,by,byy,_ in blocks:
                if comparison:
                    pv=1+bx@(current[indices]/den)+np.einsum('ni,ij,nj->n',bs,prior[:,indices],bx)
                    pz=by@(current[indices]/den)+np.einsum('ni,ij,nj->n',bs,prior[:,indices],by)
                    if pv.min()<=0:
                        raise ArithmeticError('Nonpositive comparison iterate')
                    lam=m.l0.cpu().numpy()+z[:,None]*m.l1.cpu().numpy()
                    cross=m.cross.cpu().numpy();cap=m.cap.cpu().numpy()
                    pi=np.clip((lam+(pz/pv)[:,None]*cross)/2,-cap,cap)
                    drift=m.k0-m.k*z-pi@cross
                    reaction=np.sum(pi*(pi-lam),axis=1)
                else:
                    drift,reaction=coefficients(m,tau,z,inside)
                spatial=m.q/2*byy+drift[:,None]*by+reaction[:,None]*bx
                a=bt[:,:,None]*bx[:,None,:]/delta-bs[:,:,None]*spatial[:,None,:]
                base=-reaction-spatial@(current[indices]/den)
                entries.append(a.ravel());offsets.append(base)
            a=coo_matrix((np.concatenate(entries),(rows,cols)),shape=(n*len(s),(degree_time+1)*nb)).tocsr()
            b=np.concatenate(offsets)
            norms=np.sqrt(np.asarray(a.power(2).sum(axis=0)).ravel())
            scaled=a@diags(1/norms)
            factor=splu((scaled.T@scaled).tocsc())
            sol=factor.solve(np.asarray(scaled.T@(-b)).ravel())
            for _ in range(3):
                sol+=factor.solve(np.asarray(scaled.T@(-b-scaled@sol)).ravel())
            cf=sol/norms
            updated=cf.reshape(degree_time+1,nb)
            if np.max(abs(updated-prior))<1e-10:
                break
            prior=updated
        nums=np.rint(cf*den).astype(np.int64).reshape(degree_time+1,nb)
        residual=a@(nums/den).ravel()+b
        slabs.append({'left_exact':str(left),'right_exact':str(right),
                      'initial_numerators':list(map(str,current)),
                      'coefficient_numerators':[[str(int(v)) for v in row] for row in nums],
                      'sampled_residual_max_NOT_certificate':float(abs(residual).max())})
        current=current+nums.sum(axis=0)
        if ti%8==0 or ti==len(ts)-2:
            print(f'Slab {ti+1}/{len(ts)-1}: tau={float(right)*m.T:.6g}, sampled residual {abs(residual).max():.5g}; {time.perf_counter()-start:.1f}s',flush=True)
    return {'format':'neural_fixed_policy_marching_evaluation_v1',
            'checkpoint_sha256':EXPECTED,'policy_updates_after_checkpoint':0,
            'comparison_kind':'bellman' if comparison else 'frozen_policy',
            'outside_policy':'zero outside original domain, unchanged neural rule after re-entry',
            'knots_exact':list(map(str,knots)), 'degree_time':degree_time,'degree_space':degree_space,
            'outer_multiple':outer,'coefficient_denominator':str(den),'slabs':slabs,
            'fitting_seconds':time.perf_counter()-start,'certified':False}


def marching_diagnostics(m,candidate,nt=17,nx=21):
    knots=list(map(F,candidate['knots_exact']))
    outer,ds,dz=candidate['outer_multiple'],candidate['degree_time'],candidate['degree_space']
    den=int(candidate['coefficient_denominator'])
    sa=(1-np.cos(np.linspace(0,np.pi,nt)))/2
    ss,xx=np.meshgrid(sa,np.linspace(0,1,nx),indexing='ij')
    s,x=ss.ravel(),xx.ravel()
    bs,bt=time_basis(ds,s,1)
    reports=[];initial=None
    for slab in candidate['slabs']:
        left,right=F(slab['left_exact']),F(slab['right_exact'])
        delta=m.T*float(right-left)
        current=np.asarray(slab['initial_numerators'],float)/den
        cf=np.asarray(slab['coefficient_numerators'],float)/den
        lows,highs,gaps=np.inf,-np.inf,0.
        minimum=np.inf;peak_size=-1.;peak=None
        for cell in range(len(knots)-1):
            basis=spatial_basis(knots,cell,dz)
            ix=np.array([i for i,_ in basis]);weights=cf[:,ix]
            bx,by,byy=spatial_values(basis,x)
            length=2*outer*m.half*float(knots[cell+1]-knots[cell])
            by,byy=by/length,byy/length**2
            value=1+bx@current[ix]+np.einsum('ni,ij,nj->n',bs,weights,bx)
            vt=np.einsum('ni,ij,nj->n',bt,weights,bx)/delta
            vz=by@current[ix]+np.einsum('ni,ij,nj->n',bs,weights,by)
            vzz=byy@current[ix]+np.einsum('ni,ij,nj->n',bs,weights,byy)
            z=m.center+outer*m.half*(2*(float(knots[cell])+float(knots[cell+1]-knots[cell])*x)-1)
            tau=m.T*(float(left)+float(right-left)*s)
            mid=(knots[cell]+knots[cell+1])/2
            inside=F(outer-1,2*outer)<mid<F(outer+1,2*outer)
            if candidate.get('comparison_kind')=='bellman':
                lam=m.l0.cpu().numpy()+z[:,None]*m.l1.cpu().numpy()
                cross=m.cross.cpu().numpy();cap=m.cap.cpu().numpy()
                pi=np.clip((lam+(vz/value)[:,None]*cross)/2,-cap,cap)
                drift=m.k0-m.k*z-pi@cross
                reaction=np.sum(pi*(pi-lam),axis=1)
            else:
                drift,reaction=coefficients(m,tau,z,inside)
            residual=(vt-m.q/2*vzz-drift*vz-reaction*value)/value
            imax=int(np.argmax(abs(residual)))
            if abs(residual[imax])>peak_size:
                peak_size=float(abs(residual[imax]))
                peak={'cell':cell,'tau':float(tau[imax]),'z':float(z[imax]),'normalized_residual':float(residual[imax])}
            minimum=min(minimum,float(value.min()))
            lows,highs=min(lows,float(residual.min())),max(highs,float(residual.max()))
            if inside:
                hz=m.net.jet(m.net.tensor(tau),m.net.tensor(z))[1].cpu().numpy()
                gaps=max(gaps,float((m.cross*m.cross).sum())/4*float(((vz/value-hz)**2).max()))
            if right==1 and knots[cell]<=F(1,2)<=knots[cell+1]:
                xc=float((F(1,2)-knots[cell])/(knots[cell+1]-knots[cell]))
                bsp=spatial_values(basis,np.array([xc]))[0]
                initial=float(np.exp(-m.r*m.T)*(1+(bsp@(current[ix]+weights.sum(axis=0)))[0]))
        reports.append({'tau_width':delta,'sampled_normalized_residual_lower':lows,
                        'sampled_normalized_residual_upper':highs,'sampled_gap_upper':gaps,
                        'sampled_factor_min':minimum,'peak':peak})
    return {'certified':False,'policy_factor_sampled':initial,
            'sampled_integral_negative_residual':sum(r['tau_width']*max(0,-r['sampled_normalized_residual_lower']) for r in reports),
            'sampled_integral_positive_residual':sum(r['tau_width']*max(0,r['sampled_normalized_residual_upper']) for r in reports),
            'sampled_integral_interior_improvement_gap':sum(r['tau_width']*r['sampled_gap_upper'] for r in reports),
            'slabs':reports}


def diagnostics(m, candidate, nt=257, nx=33):
    knots = list(map(F, candidate['knots_exact']))
    cf = np.asarray(candidate['coefficient_numerators'], float)/int(candidate['coefficient_denominator'])
    dt, dz, power, outer = [candidate[k] for k in ('degree_time','degree_space','time_power','outer_multiple')]
    sa = (1-np.cos(np.linspace(0, np.pi, nt)))/2
    ss, xx = np.meshgrid(sa, np.linspace(0, 1, nx), indexing='ij')
    s, x = ss.ravel(), xx.ravel()
    lows, highs, gapmax = np.full(nt, np.inf), np.full(nt, -np.inf), np.zeros(nt)
    minimum, initial = np.inf, None
    for cell in range(len(knots)-1):
        a, b, indices, basis, inside = cell_data(m, knots, cell, dt, dz, s, x, outer, power)
        bx, by, byy, bs, bt = basis
        weights = cf[:, indices]
        value = 1+np.einsum('ni,ij,nj->n', bs, weights, bx)
        vz = np.einsum('ni,ij,nj->n', bs, weights, by)
        residual = (a@weights.ravel()+b)/value
        if not np.isfinite(residual).all() or value.min() <= 0:
            raise ArithmeticError('Nonpositive candidate on diagnostic grid')
        lows = np.minimum(lows, residual.reshape(nt,nx).min(1))
        highs = np.maximum(highs, residual.reshape(nt,nx).max(1))
        minimum = min(minimum, float(value.min()))
        z = m.center+outer*m.half*(2*(float(knots[cell])+float(knots[cell+1]-knots[cell])*x)-1)
        zz, tau = m.net.tensor(z), m.net.tensor(m.T*s**power)
        lam = m.l0+zz[:, None]*m.l1
        hz = m.net.jet(tau, zz)[1]
        frozen = torch.clamp((lam+hz[:, None]*m.cross)/2, -m.cap, m.cap) if inside else torch.zeros_like(lam)
        signal = lam+m.net.tensor(vz/value)[:,None]*m.cross
        greedy = torch.clamp(signal/2, -m.cap, m.cap)
        gap = ((greedy-frozen)*(signal-greedy-frozen)).sum(-1).cpu().numpy()
        gapmax = np.maximum(gapmax, gap.reshape(nt,nx).max(1))
        if knots[cell] <= F(1,2) <= knots[cell+1]:
            xc = float((F(1,2)-knots[cell])/(knots[cell+1]-knots[cell]))
            bsp = spatial_values(spatial_basis(knots,cell,dz), np.array([xc]))[0]
            bst = time_basis(dt,np.array([1.]),power)[0]
            initial = float(np.exp(-m.r*m.T)*(1+(bst@weights@bsp.T)[0,0]))
    tau = m.T*sa**power
    integral = np.trapezoid if hasattr(np,'trapezoid') else np.trapz
    return {'certified': False, 'policy_factor_sampled': initial,
            'positive_factor_sampled_min': minimum,
            'frozen_normalized_residual_sampled_range': [float(min(lows)),float(max(highs))],
            'sampled_integral_negative_residual': float(integral(np.maximum(-lows,0),tau)),
            'sampled_integral_positive_residual': float(integral(np.maximum(highs,0),tau)),
            'sampled_integral_improvement_gap': float(integral(gapmax,tau)),
            'slab_probe_tau': tau.tolist(), 'sampled_residual_lower': lows.tolist(),
            'sampled_residual_upper': highs.tolist(), 'sampled_gap_upper': gapmax.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--degree-time', type=int, default=20)
    p.add_argument('--degree-space', type=int, default=6)
    p.add_argument('--level', type=int, default=3)
    p.add_argument('--outer', type=int, default=1)
    p.add_argument('--time-power', type=int, default=2)
    p.add_argument('--boundary-levels', type=int, default=3)
    p.add_argument('--reweights', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--march', action='store_true')
    p.add_argument('--comparison', action='store_true', help='Fit an independent Bellman comparison function; deployed policy unchanged')
    p.add_argument('--time-level',type=int,default=5)
    p.add_argument('--knots',type=Path,help='JSON list of exact spatial knots')
    p.add_argument('--output', type=Path, default=ROOT/'results/neural_separate_evaluation/pilot.json')
    args = p.parse_args()
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Changed neural policy checkpoint')
    if not (0 <= args.level <= 6 and 1 <= args.outer <= 6 and 0 <= args.boundary_levels <= 8
            and 0 <= args.reweights <= 10 and 1 <= args.degree_time <= 64
            and 4 <= args.degree_space <= 12 and args.time_power in (1,2,3,4)):
        p.error('Invalid approximation parameters')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    c = json.loads(CHECKPOINT.read_bytes())
    m = Market(c,FrozenNetwork(c,args.device))
    candidate = (fit_marching(m,args.degree_space,args.level,args.outer,args.boundary_levels,
                             args.degree_time,args.time_level,args.comparison,
                             json.loads(args.knots.read_bytes()) if args.knots else None) if args.march else
                 fit(m,args.degree_time,args.degree_space,args.level,args.outer,args.time_power,
                     args.boundary_levels,args.reweights))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(candidate,indent=2)+'\n',encoding='utf-8')
    report = marching_diagnostics(m,candidate) if args.march else diagnostics(m,candidate)
    args.output.with_name(args.output.stem+'_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list) or len(v)<3},indent=2),flush=True)


if __name__ == '__main__':
    main()
