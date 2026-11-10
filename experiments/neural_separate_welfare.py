"""Combine two independently verified residual enclosures and exit control.

For gamma=2, relative CE loss = 1 - g_star/g_pi. A fixed-policy
supersolution supplies g_pi's upper endpoint; a separately verified Bellman
subsolution supplies g_star's lower endpoint. The deployed policy is unchanged.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from mpmath import iv

from neural_certificate_audit import scalar, endpoints
from neural_duality_comparison import CHECKPOINT, EXPECTED
from neural_welfare_bound import exit_probability
from verify_neural_separate import validate_candidate


def max_reaction(c,radius):
    """Uniform maximum on the centered cylinder over the full exposure box."""
    m=c['market'];center=F(m['xbar'][0])
    total=-F(c['risk_free_rate'])
    for cap,l0,l1 in zip(c['caps'],m['lam0'],m['Lam']):
        p=F(cap);a=F(l0);b=F(l1[0])
        lm=max(abs(a+b*(center-radius)),abs(a+b*(center+radius)))
        total+=p*p+p*lm
    return max(F(0),total)


def combine(c,policy_candidate,policy,bellman_candidate,bellman):
    iv.dps=60
    for candidate,certificate,kind in ((policy_candidate,policy,'frozen_policy'),(bellman_candidate,bellman,'bellman')):
        validate_candidate(candidate)
        if not certificate.get('bounded_domain_certified') or certificate.get('comparison_kind')!=kind:
            raise ValueError('Two completed continuous-domain certificates are required')
        if candidate.get('comparison_kind','frozen_policy')!=kind:
            raise ValueError('Certificate kind does not match candidate')
        if certificate.get('checkpoint_sha256')!=EXPECTED:
            raise ValueError('Changed deployed policy')
    outer=policy_candidate['outer_multiple']
    if outer!=bellman_candidate['outer_multiple']:
        raise ValueError('Comparison domains must agree')
    # Only the geometric exit-bound input is enlarged. Neural normalization,
    # deployment boundaries, parameters, and weights are never modified.
    geometry=copy.deepcopy(c)
    scale=F(c['domain_scale'])*outer
    if F(float(scale))!=scale:
        raise ValueError('Exit-bound radius conversion must be exact')
    geometry['domain_scale']=float(scale)
    loc=exit_probability(geometry)
    radius=scale*F(c['market']['eta'][0])
    original_radius=F(c['domain_scale'])*F(c['market']['eta'][0])
    horizon=F(c['horizon'])
    c_opt=max_reaction(c,radius)
    c_pi=max_reaction(c,original_radius)  # exterior pi=0 gives c=-r<=0
    probability=scalar(F(loc['exit_probability_upper']))
    # Optimal continuation factor is >=0 and artificial trace is <=1.
    optimal_exit=iv.exp(scalar(c_opt*horizon))*probability
    # Policy continuation <=exp(c_pi*T), and pre-exit weight <=exp(c_pi*T).
    policy_exit=iv.exp(scalar(2*c_pi*horizon))*probability
    a=F(bellman['lower_log_correction_exact'])
    b=F(policy['upper_log_correction_exact'])
    lv=scalar(F(bellman['initial_value_interval'][0]))
    uv=scalar(F(policy['initial_value_interval'][1]))
    bounded_lower=lv*iv.exp(-scalar(a))
    bounded_upper=uv*iv.exp(scalar(b))
    lower=bounded_lower-optimal_exit
    upper=bounded_upper+policy_exit
    if endpoints(lower)[0]<=0:
        raise ArithmeticError('No positive optimal factor lower endpoint')
    ce=1-lower/upper
    central_gap=1-lv/uv
    # Additive algebraic accounting in this order: candidate gap, Bellman
    # residual, frozen residual, localization. Each term has outward endpoints.
    after_bellman=1-bounded_lower/uv
    after_frozen=1-bounded_lower/bounded_upper
    budget={
        'candidate_factor_gap_bp_upper':endpoints(10000*central_gap)[1],
        'bellman_residual_bp_upper':endpoints(10000*(after_bellman-central_gap))[1],
        'frozen_policy_residual_bp_upper':endpoints(10000*(after_frozen-after_bellman))[1],
        'localization_bp_upper':endpoints(10000*(ce-after_frozen))[1]}
    return {'whole_space_certified':True,'scope':'initial center; T=3; gamma=2; unchanged 30-asset neural policy',
            'checkpoint_sha256':EXPECTED,'policy_updates_after_checkpoint':0,
            'outside_policy':'zero outside original domain, unchanged neural rule after re-entry',
            'evaluation_domain_outer_multiple':outer,
            'whole_space_optimal_factor_lower':endpoints(lower)[0],
            'whole_space_policy_factor_upper':endpoints(upper)[1],
            'relative_ce_loss_upper_bp':endpoints(10000*ce)[1],
            'bounded_domain_optimal_factor_lower':endpoints(bounded_lower)[0],
            'bounded_domain_policy_factor_upper':endpoints(bounded_upper)[1],
            'optimal_localization_upper':endpoints(optimal_exit)[1],
            'policy_localization_upper':endpoints(policy_exit)[1],
            'uniform_bellman_reaction_upper_exact':str(c_opt),
            'global_policy_reaction_upper_exact':str(c_pi),
            'budget_bp':budget,'exit_bound':loc,
            'bellman_log_correction_upper':endpoints(scalar(a))[1],
            'frozen_policy_log_correction_upper':endpoints(scalar(b))[1]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('policy_candidate',type=Path)
    p.add_argument('bellman_candidate',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    inputs=[CHECKPOINT,args.policy_candidate,args.bellman_candidate,
            args.policy_candidate.with_name(args.policy_candidate.stem+'_certificate.json'),
            args.bellman_candidate.with_name(args.bellman_candidate.stem+'_certificate.json')]
    if args.output.resolve() in [path.resolve() for path in inputs]:
        raise ValueError('Output must not overwrite an input')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('{"whole_space_certified":false}\n')
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    if hashes[str(CHECKPOINT)]!=EXPECTED:
        raise ValueError('Changed checkpoint')
    c,pc,bc,pv,bv=[json.loads(path.read_bytes()) for path in inputs]
    for candidate,certificate in ((args.policy_candidate,pv),(args.bellman_candidate,bv)):
        if certificate['candidate_sha256']!=hashes[str(candidate)]:
            raise ValueError('Stale certificate: candidate changed')
        for name,digest in certificate['input_sha256'].items():
            helper=Path(__file__).with_name(name)
            if helper.is_file() and hashlib.sha256(helper.read_bytes()).hexdigest()!=digest:
                raise ValueError('Stale certificate: verification source changed')
    result=combine(c,pc,pv,bc,bv)
    for path in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[str(path)]:
            raise RuntimeError('Input changed during combination')
    result['input_sha256']={Path(k).name:v for k,v in hashes.items()}
    result['combination_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
