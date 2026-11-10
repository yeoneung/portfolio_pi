"""Verify disjoint time slabs concurrently, then check their exact full cover.

Uses the same outward arithmetic as verify_neural_separate. Workers return
partial certificates; only the parent may mark the full cylinder complete.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_name,'1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import verify_neural_separate as v

_DATA=None


def initialize(checkpoint,candidate,diagnostics,padding,batch,depth):
    global _DATA
    _DATA=(json.loads(Path(checkpoint).read_bytes()),json.loads(Path(candidate).read_bytes()),
           json.loads(Path(diagnostics).read_bytes()),padding,batch,depth)


def verify_slab(si):
    c,candidate,diagnostics,padding,batch_size,max_depth=_DATA
    start=time.perf_counter()
    knots=list(map(F,candidate['knots_exact']));slab=candidate['slabs'][si]
    bounds=v.slab_polynomials(candidate,slab,knots)
    proposal=diagnostics['slabs'][si]
    goal_lo=proposal['sampled_normalized_residual_lower']-padding
    goal_hi=proposal['sampled_normalized_residual_upper']+padding
    pending=[(i,.5,.5,.5,.5,0) for i in range(len(knots)-1)]
    area=F(0);lo=np.inf;hi=-np.inf;minimum=np.inf;count=0;tested=0;depthmax=0
    while pending:
        batch=pending[-batch_size:];del pending[-batch_size:]
        ll,uu,vl,vu=v.residual(c,candidate,slab,knots,bounds,batch)
        tested+=len(batch)
        for row,l,u,pmin,pmax in zip(batch,ll,uu,vl,vu):
            cell,t,dt,x,dx,depth=row
            if pmin>0 and u-l<=2*padding:
                goal_lo=min(goal_lo,l);goal_hi=max(goal_hi,u)
            if pmin>0 and l>=goal_lo and u<=goal_hi:
                lo=min(lo,float(l));hi=max(hi,float(u));minimum=min(minimum,float(pmin))
                area+=F(1,2**depth);count+=1;depthmax=max(depthmax,depth)
            else:
                if depth>=max_depth:
                    raise ArithmeticError(f'Maximum depth: slab {si}, cell {row}, residual [{l},{u}]')
                if dt>=dx:
                    pending.extend([(cell,t-dt/2,dt/2,x,dx,depth+1),(cell,t+dt/2,dt/2,x,dx,depth+1)])
                else:
                    pending.extend([(cell,t,dt,x-dx/2,dx/2,depth+1),(cell,t,dt,x+dx/2,dx/2,depth+1)])
    if area!=len(knots)-1:
        raise ArithmeticError('Incomplete spatial/time rectangle cover')
    return {'slab_index':si,'left_exact':slab['left_exact'],'right_exact':slab['right_exact'],
            'normalized_residual_lower':lo,'normalized_residual_upper':hi,'positive_factor_lower':minimum,
            'accepted_cells':count,'tested_cells':tested,'max_depth':depthmax,
            'full_slab_covered':True,'worker_seconds':time.perf_counter()-start}


def aggregate(c,candidate,reports):
    knots=v.validate_candidate(candidate)
    if len(reports)!=len(candidate['slabs']):
        raise ValueError('Missing time slab')
    a,b=F(0),F(0)
    for i,(slab,report) in enumerate(zip(candidate['slabs'],reports)):
        if (report['slab_index']!=i or not report['full_slab_covered'] or report['positive_factor_lower']<=0
                or any(report[k]!=slab[k] for k in ('left_exact','right_exact'))):
            raise ValueError('Missing, duplicate, invalid, or mismatched time slab')
        dt=F(c['horizon'])*(F(slab['right_exact'])-F(slab['left_exact']))
        a+=dt*max(F(0),F(report['normalized_residual_upper']))
        b+=dt*max(F(0),-F(report['normalized_residual_lower']))
    return {'bounded_domain_certified':True,'whole_space_certified':False,
            'comparison_kind':candidate.get('comparison_kind','frozen_policy'),
            'policy_updates_after_checkpoint':0,
            'lower_log_correction_exact':str(a),'upper_log_correction_exact':str(b),
            'lower_log_correction_upper':float(v.up(float(a))),
            'upper_log_correction_upper':float(v.up(float(b))),
            'accepted_cells':sum(r['accepted_cells'] for r in reports),
            'tested_cells':sum(r['tested_cells'] for r in reports),
            **v.initial_factor(c,candidate,knots),'slabs':reports}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--padding',type=float,default=3e-5)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--max-depth',type=int,default=36)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not (1<=args.workers<=16 and args.padding>0 and np.isfinite(args.padding) and args.batch>0 and 0<=args.max_depth<=40):
        p.error('Invalid verification parameters')
    candidate=json.loads(args.candidate.read_bytes());v.validate_candidate(candidate)
    c=json.loads(v.CHECKPOINT.read_bytes())
    diag=args.candidate.with_name(args.candidate.stem+'_diagnostics.json')
    output=args.output or args.candidate.with_name(args.candidate.stem+'_certificate.json')
    helpers=['verify_neural_separate.py','neural_log_barrier.py','neural_certificate_audit.py',
             'local_policy_evaluation.py','portfolio_welfare.py','direct_residual_certificate.py']
    inputs=[args.candidate,v.CHECKPOINT,Path(__file__)]+[Path(__file__).with_name(n) for n in helpers]
    if output.resolve() in [path.resolve() for path in inputs]+[diag.resolve()]:
        raise ValueError('Output must not overwrite an input')
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    if hashes[v.CHECKPOINT.name]!=v.EXPECTED:
        raise ValueError('Changed checkpoint')
    output.parent.mkdir(parents=True,exist_ok=True)
    reports=[None]*len(candidate['slabs']);start=time.perf_counter()
    output.write_text('{"bounded_domain_certified":false,"whole_space_certified":false}\n')
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(str(v.CHECKPOINT),str(args.candidate),str(diag),args.padding,args.batch,args.max_depth)) as pool:
        futures=[pool.submit(verify_slab,i) for i in range(len(reports))]
        for done,future in enumerate(as_completed(futures),1):
            result=future.result();reports[result['slab_index']]=result
            progress={'bounded_domain_certified':False,'whole_space_certified':False,
                      'completed_slabs':done,'total_slabs':len(reports),
                      'elapsed_seconds':time.perf_counter()-start,
                      'slabs':[r for r in reports if r is not None]}
            output.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
            print(f'Covered {done}/{len(reports)} time slabs; {sum(r["accepted_cells"] for r in reports if r)} cells; {time.perf_counter()-start:.1f}s',flush=True)
    for path in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[path.name]:
            raise RuntimeError('Input changed during verification')
    result=aggregate(c,candidate,reports)
    result.update({'input_sha256':hashes,'candidate_sha256':hashes[args.candidate.name],
                   'checkpoint_sha256':v.EXPECTED,'padding_proposal':args.padding,'workers':args.workers,
                   'seconds':time.perf_counter()-start})
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
