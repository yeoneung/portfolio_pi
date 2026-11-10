"""Outward-rounded continuous residual verification for separate evaluation.

The candidate is continuous in time and C1 in space, piecewise polynomial.
Spatial second derivatives may jump across interfaces. Every rectangle is
covered, including the two one-sided limits at the deployment boundary.
No sampled residual maximum is used as a certified bound.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from mpmath import iv

from local_policy_evaluation import spatial_basis
import portfolio_welfare as w
from neural_log_barrier import Affine, pack, up, down, times, plus, divide_positive, sumup
from neural_certificate_audit import scalar, endpoints
from neural_duality_comparison import CHECKPOINT, EXPECTED

ROOT=Path(__file__).resolve().parents[1]


def fraction_bounds(x):
    x=F(x); y=float(x)
    return (y if F(y)<=x else float(down(y)), y if F(y)>=x else float(up(y)))


def exact_constant(x):
    return Affine.interval_constant(fraction_bounds(x))


def reciprocal(a):
    """Quadratic Taylor reciprocal, with its exact algebraic remainder."""
    lo,hi=a.bounds()
    if np.any(lo<=0):
        raise ArithmeticError('Positive denominator required')
    c=a.c[0]
    ci=Affine.interval_constant((down(1/c),up(1/c)))
    delta=a-Affine.constant(c)
    ans=ci-ci*ci*delta+ci*ci*ci*delta*delta
    width=a.width()
    remainder=divide_positive(times(times((width,width),(width,width)),(width,width)),
                              times(times((c,c),(c,c)),times((c,c),(lo,lo))))[1]
    return Affine(ans.c,up(ans.r+remainder))


def clipped_half(signal,caps):
    caps=np.asarray(caps)
    lo,hi=signal.bounds()
    result=Affine.interval_constant((np.clip(down(lo/2),-caps,caps),np.clip(up(hi/2),-caps,caps)))
    free=(lo>=-2*caps)&(hi<=2*caps)
    free_form=.5*signal
    return Affine(np.where(free[None,...],free_form.c,result.c),np.where(free,free_form.r,result.r))


def neural_hz(c,tau,z):
    state=c['state_dict']
    inverse=exact_constant(1/F(state['half_ax'][0][0]))
    inverse_t=exact_constant(1/F(c['horizon']))
    y=(z-state['center'][0][0])*inverse
    d=1-y*y; dz=-2*y*inverse
    zero=Affine.constant(np.zeros_like(tau.c[0]))
    value=pack((tau*inverse_t,y,d))
    derivative=pack((zero,zero+inverse,dz))
    for layer in range(c['config']['depth']+1):
        weights,bias=state[f'net.{2*layer}.weight'],state[f'net.{2*layer}.bias']
        value=value.linear(weights,bias)
        derivative=derivative.linear(weights,np.zeros(len(bias)))
        if layer<c['config']['depth']:
            value=value.tanh()
            derivative=(1-value*value)*derivative
    return tau*(dz*value+d*derivative)


def validate_candidate(candidate):
    if candidate['format']!='neural_fixed_policy_marching_evaluation_v1' or candidate['checkpoint_sha256']!=EXPECTED:
        raise ValueError('Unsupported candidate or changed policy')
    knots=list(map(F,candidate['knots_exact']))
    if knots[0]!=0 or knots[-1]!=1 or any(a>=b for a,b in zip(knots,knots[1:])):
        raise ValueError('Invalid spatial partition')
    outer=candidate['outer_multiple']
    if not isinstance(outer,int) or outer<1:
        raise ValueError('Invalid outer domain')
    if F(outer-1,2*outer) not in knots or F(outer+1,2*outer) not in knots:
        raise ValueError('Partition must split at both deployment interfaces')
    ds,dz=candidate['degree_time'],candidate['degree_space']
    if not (0<=ds<=10 and 4<=dz<=12):
        raise ValueError('Unsupported polynomial degree')
    den=int(candidate['coefficient_denominator'])
    if den<=0:
        raise ValueError('Invalid denominator')
    nb=(dz-1)*(len(knots)-1)
    expected=[0]*nb; right=F(0)
    for slab in candidate['slabs']:
        left,next_right=F(slab['left_exact']),F(slab['right_exact'])
        if left!=right or next_right<=left:
            raise ValueError('Time cover is not contiguous')
        current=list(map(int,slab['initial_numerators']))
        values=[list(map(int,row)) for row in slab['coefficient_numerators']]
        if current!=expected or len(values)!=ds+1 or any(len(row)!=nb for row in values):
            raise ValueError('Time trace does not join exactly')
        expected=[a+sum(row[j] for row in values) for j,a in enumerate(current)]
        right=next_right
    if right!=1 or not candidate['slabs']:
        raise ValueError('Incomplete horizon')
    return knots


def slab_polynomials(candidate,slab,knots):
    """Exact change of basis, followed by directed binary64 enclosure."""
    ds,dz=candidate['degree_time'],candidate['degree_space']
    den=int(candidate['coefficient_denominator'])
    initial=[F(int(a),den) for a in slab['initial_numerators']]
    cf=[[F(int(a),den) for a in row] for row in slab['coefficient_numerators']]
    chebs=[w.exact.mul({(1,0):F(1)},p) for p in w.exact.shifted_chebyshev(ds,0)]
    shape=(len(knots)-1,ds+2,dz+1)
    low,high=np.zeros(shape),np.zeros(shape)
    for cell in range(len(knots)-1):
        poly={(0,0):F(1)}
        for index,basis in spatial_basis(knots,cell,dz):
            poly=w.exact.add(poly,w.exact.scale(basis,initial[index]))
            for i,tc in enumerate(chebs):
                poly=w.exact.add(poly,w.exact.scale(w.exact.mul(tc,basis),cf[i][index]))
        for (i,j),v in poly.items():
            low[cell,i,j],high[cell,i,j]=fraction_bounds(v)
    return low,high


def polynomial_jet(bounds,cells,q,x):
    lo,hi=bounds
    cellid=np.asarray([int(row[0]) for row in cells])
    def constant(i,j):
        return Affine.interval_constant((lo[cellid,i,j,None],hi[cellid,i,j,None]))
    zero=Affine.constant(np.zeros_like(q.c[0]))
    value,qt,z,zz=zero,zero,zero,zero
    for i in range(lo.shape[1]-1,-1,-1):
        vx,dx,dxx=zero,zero,zero
        for j in range(lo.shape[2]-1,-1,-1):
            dxx=x*dxx+2*dx
            dx=x*dx+vx
            vx=x*vx+constant(i,j)
        qt=q*qt+value
        value=q*value+vx
        z=q*z+dx
        zz=q*zz+dxx
    return value,qt,z,zz


def residual(c,candidate,slab,knots,bounds,cells):
    # rows: space-cell, q midpoint/radius, x midpoint/radius, depth.
    a=np.asarray(cells,float)
    q=Affine.input(a[:,1:2],a[:,2:3],1)
    x=Affine.input(a[:,3:4],a[:,4:5],2)
    value,pt,px,pxx=polynomial_jet(bounds,cells,q,x)
    ids=[int(row[0]) for row in cells]
    widths=[knots[i+1]-knots[i] for i in ids]
    lower=[knots[i] for i in ids]
    def rational_vector(seq):
        ll,uu=zip(*(fraction_bounds(v) for v in seq))
        return Affine.interval_constant((np.array(ll)[:,None],np.array(uu)[:,None]))
    market=c['market']; outer=candidate['outer_multiple']
    half=F(c['domain_scale'])*F(market['eta'][0])
    radius=outer*half
    y=rational_vector(lower)+x*rational_vector(widths)
    z=market['xbar'][0]+exact_constant(radius)*(2*y-1)
    left,right=F(slab['left_exact']),F(slab['right_exact'])
    tau=exact_constant(F(c['horizon']))*(exact_constant(left)+q*exact_constant(right-left))
    length_inverse=rational_vector([1/(2*radius*h) for h in widths])
    pt=pt*exact_constant(1/(F(c['horizon'])*(right-left)))
    pz=px*length_inverse; pzz=pxx*length_inverse*length_inverse
    drift=market['k0'][0]-market['K'][0][0]*z
    base=pt-.5*market['Q'][0][0]*pzz-drift*pz
    lam=np.asarray(market['lam0'])[None,:]+z*np.asarray(market['Lam'])[:,0][None,:]
    cross=np.asarray(market['Gamma'])[:,0][None,:]
    caps=np.asarray(c['caps'])[None,:]
    vl,vu=value.bounds()
    good=vl[:,0]>0
    safe=Affine(np.where(good[None,:,None],value.c,Affine.constant(np.ones_like(vl)).c),
                np.where(good[:,None],value.r,0.))
    if candidate.get('comparison_kind','frozen_policy')=='bellman':
        signal=lam+(pz*reciprocal(safe))*cross
        ham=signal.hamiltonian(caps).linear(np.ones((1,len(c['caps']))),np.zeros(1))
        out=base+value*ham
    else:
        inside=np.array([F(outer-1,2*outer)<(knots[i]+knots[i+1])/2<F(outer+1,2*outer) for i in ids])
        # Avoid evaluating the neural net on remote exterior cells.
        if inside.any():
            def select(v): return Affine(v.c[:,inside],v.r[inside])
            hz=neural_hz(c,select(tau),select(z))
            signal=select(lam)+hz*cross
            ham=signal.hamiltonian(caps).linear(np.ones((1,len(c['caps']))),np.zeros(1))
            shift=clipped_half(signal,caps).linear(cross,np.zeros(1))
            interior=select(base)+select(value)*ham+(select(pz)-select(value)*hz)*shift
            oc,orr=base.c.copy(),base.r.copy()
            oc[:,inside],orr[inside]=interior.c,interior.r
            out=Affine(oc,orr)
        else:
            out=base
    lows,highs=divide_positive(out.bounds(),(np.maximum(vl,np.finfo(float).tiny),np.maximum(vu,np.finfo(float).tiny)))
    lows,highs=lows[:,0],highs[:,0]
    lows[~good],highs[~good]=-np.inf,np.inf
    return lows,highs,vl[:,0],vu[:,0]


def initial_factor(c,candidate,knots):
    slab=candidate['slabs'][-1]
    den=int(candidate['coefficient_denominator'])
    coeff=[F(int(a)+sum(int(row[j]) for row in slab['coefficient_numerators']),den)
           for j,a in enumerate(slab['initial_numerators'])]
    cell=next(i for i in range(len(knots)-1) if knots[i]<=F(1,2)<=knots[i+1])
    x=(F(1,2)-knots[cell])/(knots[cell+1]-knots[cell])
    value=F(1)
    for index,basis in spatial_basis(knots,cell,candidate['degree_space']):
        value+=coeff[index]*sum((v*x**j for (i,j),v in basis.items()),F(0))
    iv.dps=60
    return {'initial_polynomial_exact':str(value),
            'initial_value_interval':endpoints(scalar(value)*iv.exp(-scalar(F(c['risk_free_rate'])*F(c['horizon']))))}


def verify(c,candidate,diagnostics,padding=.0003,batch_size=128,max_depth=28,output=None,slab_limit=None):
    start=time.perf_counter();knots=validate_candidate(candidate)
    reports=[];lower_integral=F(0);upper_integral=F(0);accepted=0;tested=0
    for si,slab in enumerate(candidate['slabs']):
        if slab_limit is not None and si>=slab_limit: break
        bounds=slab_polynomials(candidate,slab,knots)
        proposal=diagnostics['slabs'][si]
        goal_lo=proposal['sampled_normalized_residual_lower']-padding
        goal_hi=proposal['sampled_normalized_residual_upper']+padding
        pending=[(i,.5,.5,.5,.5,0) for i in range(len(knots)-1)]
        area=F(0);lo=np.inf;hi=-np.inf;minimum=np.inf;count=0;depthmax=0
        while pending:
            batch=pending[-batch_size:];del pending[-batch_size:]
            ll,uu,vl,vu=residual(c,candidate,slab,knots,bounds,batch)
            tested+=len(batch)
            for row,l,u,pmin,pmax in zip(batch,ll,uu,vl,vu):
                cell,t,dt,x,dx,depth=row
                # A missed extremum in the numerical proposal cannot invalidate
                # verification. A narrow rigorous enclosure replaces the proposal.
                if pmin>0 and u-l<=2*padding:
                    goal_lo=min(goal_lo,l);goal_hi=max(goal_hi,u)
                if pmin>0 and l>=goal_lo and u<=goal_hi:
                    lo=min(lo,float(l));hi=max(hi,float(u));minimum=min(minimum,float(pmin))
                    area+=F(1,2**depth);count+=1;depthmax=max(depthmax,depth)
                else:
                    if depth>=max_depth:
                        raise ArithmeticError(f'Max depth in slab {si}, cell {row}: residual [{l},{u}], Pmin={pmin}')
                    if dt>=dx:
                        pending.extend([(cell,t-dt/2,dt/2,x,dx,depth+1),(cell,t+dt/2,dt/2,x,dx,depth+1)])
                    else:
                        pending.extend([(cell,t,dt,x-dx/2,dx/2,depth+1),(cell,t,dt,x+dx/2,dx/2,depth+1)])
        if area!=len(knots)-1:
            raise ArithmeticError('Incomplete rectangular cover')
        delta=F(c['horizon'])*(F(slab['right_exact'])-F(slab['left_exact']))
        lower_integral+=delta*max(F(0),-F(lo))
        upper_integral+=delta*max(F(0),F(hi))
        accepted+=count
        reports.append({'left_exact':slab['left_exact'],'right_exact':slab['right_exact'],
                        'normalized_residual_lower':lo,'normalized_residual_upper':hi,
                        'positive_factor_lower':minimum,'accepted_cells':count,'max_depth':depthmax,
                        'full_slab_covered':True})
        record={'bounded_domain_certified':False,'whole_space_certified':False,
                'comparison_kind':candidate.get('comparison_kind','frozen_policy'),
                'lower_log_correction_exact':str(upper_integral),
                'upper_log_correction_exact':str(lower_integral),
                'lower_log_correction_upper':float(up(float(upper_integral))),
                'upper_log_correction_upper':float(up(float(lower_integral))),
                'accepted_cells':accepted,'tested_cells':tested,'seconds':time.perf_counter()-start,'slabs':reports}
        if output:
            output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(f'Certified slabs {si+1}/{len(candidate["slabs"])}; cells {accepted}; log(-,+)=({float(upper_integral):.6g},{float(lower_integral):.6g}); {time.perf_counter()-start:.1f}s',flush=True)
    complete=len(reports)==len(candidate['slabs'])
    record.update(initial_factor(c,candidate,knots))
    record['bounded_domain_certified']=complete
    record['policy_updates_after_checkpoint']=0
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path)
    p.add_argument('--padding',type=float,default=.0003)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--max-depth',type=int,default=28)
    p.add_argument('--slab-limit',type=int)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not np.isfinite(args.padding) or args.padding<=0 or args.batch<1 or not 0<=args.max_depth<=40:
        p.error('Invalid verifier parameters')
    candidate=json.loads(args.candidate.read_bytes())
    diagnostic=args.candidate.with_name(args.candidate.stem+'_diagnostics.json')
    output=args.output or args.candidate.with_name(args.candidate.stem+'_certificate.json')
    inputs=[CHECKPOINT,args.candidate,Path(__file__),Path(__file__).with_name('neural_log_barrier.py'),
            Path(__file__).with_name('local_policy_evaluation.py')]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    if hashes[str(CHECKPOINT)]!=EXPECTED:
        raise ValueError('Changed checkpoint')
    if output.resolve() in [path.resolve() for path in inputs]+[diagnostic.resolve()]:
        raise ValueError('Output must not overwrite an input')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('{"bounded_domain_certified":false,"whole_space_certified":false}\n')
    result=verify(json.loads(CHECKPOINT.read_bytes()),candidate,json.loads(diagnostic.read_bytes()),
                  args.padding,args.batch,args.max_depth,output,args.slab_limit)
    for path in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[str(path)]:
            raise RuntimeError('Input changed during verification')
    result.update({'input_sha256':{Path(k).name:v for k,v in hashes.items()},'checkpoint_sha256':EXPECTED,
                   'candidate_sha256':hashes[str(args.candidate)],'padding_proposal':args.padding})
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
