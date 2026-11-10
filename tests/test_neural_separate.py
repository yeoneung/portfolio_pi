"""Independent scalar differentiation, interface tests, and fail-closed checks."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np
from mpmath import mp
import pytest

from test_neural_certificate_audit import small_checkpoint, real, reference_value
from local_policy_evaluation import spatial_basis
import portfolio_welfare as w
import verify_neural_separate as v
from neural_duality_comparison import EXPECTED
from neural_separate_welfare import combine, max_reaction


def candidate(kind='frozen_policy'):
    return {'format':'neural_fixed_policy_marching_evaluation_v1',
            'checkpoint_sha256':EXPECTED,'policy_updates_after_checkpoint':0,
            'comparison_kind':kind,'knots_exact':['0','1/3','2/3','1'],
            'outer_multiple':3,'degree_time':1,'degree_space':4,
            'coefficient_denominator':str(2**16),
            'slabs':[{'left_exact':'0','right_exact':'1','initial_numerators':['0']*9,
                      'coefficient_numerators':[['-11','-17','3','-5','7','-2','9','-13','5'],
                                                ['3','-2','-1','1','-2','3','-2','2','1']]}]}


def exact_poly(cand,cell):
    slab=cand['slabs'][0];knots=list(map(F,cand['knots_exact']))
    den=int(cand['coefficient_denominator'])
    chebs=w.exact.shifted_chebyshev(cand['degree_time'],0)
    poly={(0,0):F(1)}
    for index,basis in spatial_basis(knots,cell,cand['degree_space']):
        poly=w.exact.add(poly,w.exact.scale(basis,F(int(slab['initial_numerators'][index]),den)))
        for k,cheb in enumerate(chebs):
            poly=w.exact.add(poly,w.exact.scale(w.exact.mul(w.exact.mul({(1,0):F(1)},cheb),basis),
                                               F(int(slab['coefficient_numerators'][k][index]),den)))
    return poly


def reference(c,cand,cell,q,x):
    """mp.diff of the original factor, then the defining linear/HJB operator."""
    knots=list(map(F,cand['knots_exact']));poly=exact_poly(cand,cell)
    radius=real(cand['outer_multiple'])*real(c['domain_scale'])*real(c['market']['eta'][0])
    center=real(c['market']['xbar'][0]);T=real(c['horizon']);r=real(c['risk_free_rate'])
    left=real(F(cand['slabs'][0]['left_exact']))
    delta=real(F(cand['slabs'][0]['right_exact'])-F(cand['slabs'][0]['left_exact']))
    length=2*radius*real(knots[cell+1]-knots[cell])
    zleft=center+radius*(2*real(knots[cell])-1)
    def f(t,z):
        qq=(t/T-left)/delta;xx=(z-zleft)/length
        p=sum((real(a)*qq**i*xx**j for (i,j),a in poly.items()),mp.mpf(0))
        return mp.exp(-r*t)*p
    tau=T*(left+delta*real(q));z=zleft+length*real(x)
    value=f(tau,z)
    ft,fz,fzz=[mp.diff(f,(tau,z),o) for o in ((1,0),(0,1),(0,2))]
    market=c['market']
    result=ft-real(market['Q'][0][0])/2*fzz-(real(market['k0'][0])-real(market['K'][0][0])*z)*fz+r*value
    inside=F(cand['outer_multiple']-1,2*cand['outer_multiple'])<(knots[cell]+knots[cell+1])/2<F(cand['outer_multiple']+1,2*cand['outer_multiple'])
    if cand['comparison_kind']=='bellman':
        hz=fz/value
    elif inside:
        hz=mp.diff(lambda zz:mp.log(reference_value(c,tau,zz)),z)
    else:
        hz=mp.mpf(0)
    for cap,l0,l1,cross in zip(c['caps'],market['lam0'],market['Lam'],market['Gamma']):
        lam=real(l0)+real(l1[0])*z
        pi=min(real(cap),max(-real(cap),(lam+real(cross[0])*hz)/2)) if inside or cand['comparison_kind']=='bellman' else mp.mpf(0)
        result+=(pi*lam-pi*pi)*value+pi*real(cross[0])*fz
    return result/value


@pytest.mark.parametrize('kind',['frozen_policy','bellman'])
def test_full_cells_enclose_independent_mpdiff(kind):
    mp.dps=80;c=small_checkpoint();cand=candidate(kind)
    knots=v.validate_candidate(cand);slab=cand['slabs'][0]
    bounds=v.slab_polynomials(cand,slab,knots)
    cells=[(i,.5,.0625,.5,.0625,0) for i in range(3)]
    lo,hi,_,_=v.residual(c,cand,slab,knots,bounds,cells)
    for cell in range(3):
        for q in (F(7,16),F(1,2),F(9,16)):
            for x in (F(7,16),F(1,2),F(9,16)):
                assert real(lo[cell])<=reference(c,cand,cell,q,x)<=real(hi[cell])


@pytest.mark.parametrize('kind',['frozen_policy','bellman'])
def test_point_enclosures_tight_and_correct_at_interfaces(kind):
    mp.dps=80;c=small_checkpoint();cand=candidate(kind)
    knots=v.validate_candidate(cand);slab=cand['slabs'][0]
    bounds=v.slab_polynomials(cand,slab,knots)
    cells=[(i,.625,0.,x,0.,0) for i in range(3) for x in (0.,1.)]
    lo,hi,_,_=v.residual(c,cand,slab,knots,bounds,cells)
    for row,l,u in zip(cells,lo,hi):
        exact=reference(c,cand,row[0],F(5,8),F(row[3]))
        assert real(l)<=exact<=real(u)
        assert u-l<1e-8


def test_exact_terminal_lateral_and_c1_traces():
    cand=candidate();knots=list(map(F,cand['knots_exact']))
    polys=[exact_poly(cand,i) for i in range(3)]
    for p in polys:
        assert {k:a for k,a in p.items() if k[0]==0}=={(0,0):F(1)}
    for q in (F(0),F(3,7),F(1)):
        ev=lambda p,x:sum((a*q**i*x**j for (i,j),a in p.items()),F(0))
        assert ev(polys[0],F(0))==ev(polys[-1],F(1))==1
        for i in (0,1):
            assert ev(polys[i],F(1))==ev(polys[i+1],F(0))
            dl=ev(w.exact.derivative(polys[i],1),F(1))/(knots[i+1]-knots[i])
            dr=ev(w.exact.derivative(polys[i+1],1),F(0))/(knots[i+2]-knots[i+1])
            assert dl==dr


def test_reject_time_discontinuity_and_missing_interface():
    cand=candidate();cand['slabs'][0]['initial_numerators'][0]='1'
    with pytest.raises(ValueError,match='join'):v.validate_candidate(cand)
    cand=candidate();cand['knots_exact']=['0','1/4','2/3','1']
    with pytest.raises(ValueError,match='interfaces'):v.validate_candidate(cand)


def test_reciprocal_contains_high_precision_values():
    mp.dps=80
    a=v.Affine(np.array([.75,.02,-.03,.001,.002,-.001])[:,None,None],1e-7)
    out=v.reciprocal(a)
    lo,hi=out.bounds()
    for x in np.linspace(-1,1,13):
        for y in np.linspace(-1,1,13):
            for sign in (-1,1):
                exact=sum(real(c)*real(x)**i*real(y)**j for c,(i,j) in zip(a.c[:,0,0],((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))))+sign*real(a.r[0,0])
                assert real(lo[0,0])<=1/exact<=real(hi[0,0])


def test_uniform_reaction_bound_includes_adverse_controls():
    c=small_checkpoint();radius=F(3)
    upper=max_reaction(c,radius)
    for x in (-radius,F(0),radius):
        z=F(c['market']['xbar'][0])+x
        for signs in ((-1,-1),(-1,1),(1,-1),(1,1)):
            value=-F(c['risk_free_rate'])
            for s,cap,l0,l1 in zip(signs,c['caps'],c['market']['lam0'],c['market']['Lam']):
                pi=s*F(cap);lam=F(l0)+F(l1[0])*z
                value+=pi*pi-pi*lam
            assert value<=upper


def test_combination_rejects_incomplete_certificate():
    with pytest.raises(ValueError,match='completed'):
        combine(small_checkpoint(),candidate(),{'bounded_domain_certified':False},candidate('bellman'),{})


def test_parallel_aggregation_rejects_missing_duplicate_or_wrong_slab():
    from verify_neural_separate_parallel import aggregate
    c=small_checkpoint();cand=candidate()
    report={'slab_index':0,'left_exact':'0','right_exact':'1','full_slab_covered':True,
            'positive_factor_lower':.9,'normalized_residual_lower':-.125,
            'normalized_residual_upper':.25,'accepted_cells':3,'tested_cells':3}
    result=aggregate(c,cand,[report])
    assert F(result['lower_log_correction_exact'])==F(3,4)
    assert F(result['upper_log_correction_exact'])==F(3,8)
    for reports in ([],[report,report],[{**report,'slab_index':1}],[{**report,'right_exact':'1/2'}]):
        with pytest.raises(ValueError):aggregate(c,cand,reports)


def test_actual_network_and_final_polynomial_against_mpdiff():
    mp.dps=80
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'results/neural_certificate/seed_11_final.json').read_bytes())
    cand=json.loads((root/'results/neural_separate_evaluation/adapted_policy.json').read_bytes())
    cand['slabs']=[cand['slabs'][80]]
    knots=list(map(F,cand['knots_exact']));slab=cand['slabs'][0]
    bounds=v.slab_polynomials(cand,slab,knots)
    # Interior center, near the original deployment interface, and exterior.
    selected=[next(i for i in range(len(knots)-1) if knots[i]<=y<knots[i+1])
              for y in (F(1,2),F(33,50),F(4,5))]
    cells=[(i,.625,0.,.375,0.,0) for i in selected]
    lo,hi,_,_=v.residual(c,cand,slab,knots,bounds,cells)
    for row,l,u in zip(cells,lo,hi):
        exact=reference(c,cand,row[0],F(5,8),F(3,8))
        assert real(l)<=exact<=real(u)
        assert u-l<1e-7


def test_resume_rejects_changed_input_before_starting_workers(tmp_path,monkeypatch):
    import sys
    import resume_neural_separate as resume
    source=tmp_path/'candidate.json';source.write_text(json.dumps(candidate()))
    saved=tmp_path/'checkpoint.json';saved.write_text(json.dumps({'verified_input_sha256':{'candidate.json':'wrong'}}))
    monkeypatch.setattr(sys,'argv',['resume',str(source),str(saved)])
    with pytest.raises(ValueError,match='Changed verification input'):
        resume.main()
