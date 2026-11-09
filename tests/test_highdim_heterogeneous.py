"""Independent full-generator and certificate-assembly checks for heterogeneity."""
import copy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys

from mpmath import iv, mp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import highdim_heterogeneous as heterogeneous
import highdim_local_verify as local

iv.dps = 60
mp.dps = 80


def candidate(d=10):
    return json.loads((ROOT/f'results/highdim_heterogeneous/candidate_d{d}_spatial.json').read_text())


@pytest.mark.parametrize('d', [10, 20, 50])
def test_predictors_have_no_rotational_symmetry_and_remain_full_rank(d):
    ref = json.loads((ROOT/f'results/highdim_local/candidate_d{d}.json').read_text())
    models = heterogeneous.predictor_models(ref)
    for key in ('lambda_constant', 'lambda_self', 'lambda_neighbor'):
        ratios = [F(m[key])/F(ref['model'][key]) for m in models]
        assert min(ratios) == F(4, 5) and max(ratios) == F(6, 5)
    assert len({m['lambda_constant'] for m in models}) == d
    a = np.array([float(F(m['lambda_self'])) for m in models])
    b = np.array([float(F(m['lambda_neighbor'])) for m in models])
    matrix = np.diag(a)
    matrix[np.arange(d), (np.arange(d)+1) % d] += b
    assert np.linalg.svd(matrix, compute_uv=False).min() >= a.min()-abs(b).max()-1e-14


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_full_generator_with_distinct_evaluations_matches_local_enclosures(side):
    c = candidate()
    blocks, d = c['blocks'], c['dimension']
    assert len({tuple(b[side]['coeff']) for b in blocks}) == d
    rng = np.random.default_rng(517)
    polynomials = [local.residual_polys(b)[0] for b in blocks]
    common = {k: float(F(v)) for k, v in blocks[0]['model'].items()}
    def jet(data, tau, z):
        k = len(data)//3
        p = [np.polynomial.Chebyshev([float(F(v)) for v in data[j*k:(j+1)*k]],
                                    domain=[0, 1]).convert(kind=np.polynomial.Polynomial)
             * np.polynomial.Polynomial([0, 1]) for j in range(3)]
        return (sum(p[j].deriv()(tau)*z**j for j in range(3)),
                p[1](tau)+2*p[2](tau)*z, 2*p[2](tau))
    for tau in (.17, .63, 1.):
        z = rng.uniform(-1.5, 1.5, d)
        jets = np.array([jet(b[side]['coeff'], tau, z[i]) for i, b in enumerate(blocks)])
        ft, grad, diagonal = jets[:, 0], jets[:, 1]/d, jets[:, 2]/d
        policy_gradient = np.array([jet(b['base'], tau, z[i])[1]/d for i, b in enumerate(blocks)]) if side == 'upper' else grad
        mu = np.array([(float(F(b['model']['lambda_constant']))
                        +float(F(b['model']['lambda_self']))*z[i]
                        +float(F(b['model']['lambda_neighbor']))*z[(i+1) % d])/math.sqrt(d)
                       for i, b in enumerate(blocks)])
        signal = mu+common['cross']*policy_gradient
        cap = common['cap']/math.sqrt(d)
        pi = np.clip(signal/2, -cap, cap)
        drift = -common['kappa']*z+common['drift_neighbor']*np.roll(z, -1)-common['cross']*pi
        cost = -common['risk_free_rate']-pi@mu+pi@pi
        generator = (-common['risk_free_rate']+ft.mean()-drift@grad
                     -common['variance']/2*(diagonal.sum()+grad@grad)-cost)
        offset = 0 if side == 'lower' else 3
        values = []
        for i, polys in enumerate(polynomials):
            branch = 0 if signal[i] < -2*cap else 2 if signal[i] > 2*cap else 1
            interval = polys[offset+branch].at(F(tau), F(float(z[i])), F(float(z[(i+1) % d])))
            values.append(sum(local.ends(interval))/2)
        assert np.mean(values) == pytest.approx(generator, abs=4e-14)


def test_non_telescoping_potentials_and_missing_blocks_are_rejected():
    c = candidate()
    altered = copy.deepcopy(c)
    altered['blocks'][2]['lower']['potential'][0] = '1/7'
    with pytest.raises(ValueError, match='do not telescope'):
        heterogeneous.validate_blocks(altered)
    altered = copy.deepcopy(c)
    altered['blocks'].pop()
    with pytest.raises(ValueError, match='every coordinate'):
        heterogeneous.validate_blocks(altered)


@pytest.mark.parametrize('d', [10, 20, 50])
def test_endpoint_assembly_and_all_coordinate_covers(d):
    c = candidate(d)
    result = json.loads((ROOT/f'results/highdim_heterogeneous/certificate_d{d}_spatial.json').read_text())
    assert result['local_cylinders_verified'] == d
    assert all(v['cover_complete'] and v['exact_covered_volume'] == '1' for v in result['covers'])
    def number(value):
        q = F(value)
        return mp.mpf(q.numerator)/q.denominator
    exponents = {side: -number(c['blocks'][0]['model']['risk_free_rate']) for side in ('lower', 'upper')}
    for block, cover in zip(c['blocks'], result['covers']):
        k = block['time_terms']
        lower_repair = max(0., cover['lower_residual_upper'], cover['lower_trace_rate_upper'])
        upper_repair = max(0., -cover['upper_residual_lower'], cover['upper_trace_rate_upper'])
        exponents['lower'] += (sum(number(x) for x in block['lower']['coeff'][:k])-number(lower_repair))/d
        exponents['upper'] += (sum(number(x) for x in block['upper']['coeff'][:k])+number(upper_repair))/d
    loc = number(result['localization_upper'])
    lower, upper = mp.exp(exponents['lower'])-loc, mp.exp(exponents['upper'])+loc
    assert number(result['whole_space_optimal_factor_lower']) <= lower
    assert number(result['whole_space_policy_factor_upper']) >= upper
    assert number(result['relative_ce_loss_upper_bp']) >= (1-lower/upper)*10000
    assert abs(float((1-lower/upper)*10000)-result['relative_ce_loss_upper_bp']) < 1e-10
    with pytest.raises(ValueError, match='Every coordinate'):
        heterogeneous.combine(c, result['covers'][:-1])


def test_reproduce_one_continuous_local_cover():
    c = candidate()
    cover = local.verify_cover(c['blocks'][3], tolerance=F(1, 2000), max_depth=36)
    assert cover['exact_covered_volume'] == '1'
    assert max(cover['lower_residual_upper'], -cover['upper_residual_lower'],
               cover['lower_trace_rate_upper'], cover['upper_trace_rate_upper']) <= .0005
