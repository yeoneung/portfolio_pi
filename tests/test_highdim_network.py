"""Independent matrix, full-generator, and continuous-certificate checks."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

from mpmath import iv, mp
import numpy as np
import pytest
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import highdim_network as network
import highdim_local_verify as arithmetic

iv.dps = 60
mp.dps = 80


def candidate(d=10):
    return json.loads((ROOT/f'results/highdim_network/candidate_d{d}.json').read_text())


def matrices(c):
    return [np.array([[float(F(x)) for x in row] for row in c[key]])
            for key in ('network_weights', 'asset_covariance')]


@pytest.mark.parametrize('d', [10, 20, 50])
def test_graph_predictors_and_covariances_have_full_dimension(d):
    c = network.model(d)
    network.validate(c)
    P, S = matrices(c)
    adj = (S != np.eye(d)).astype(int)
    assert adj.sum()//2 == d-1
    assert adj.sum(0).max() == 3
    assert sorted(set(adj.sum(0))) == [1, 2, 3]
    # Every vertex is reachable, while the graph is a tree, not a cycle.
    seen = {0}
    for _ in range(d):
        seen |= {j for i in seen for j in range(d) if adj[i, j]}
    assert len(seen) == d
    assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)
    lo = np.diag([float(F(b['model']['risk_lower'])) for b in c['blocks']])
    hi = np.diag([float(F(b['model']['risk_upper'])) for b in c['blocks']])
    assert np.linalg.eigvalsh(S-lo).min() > -1e-14
    assert np.linalg.eigvalsh(hi-S).min() > -1e-14
    joint = np.block([[S, np.eye(d)/8], [np.eye(d)/8, np.eye(d)/16]])
    assert np.linalg.eigvalsh(joint).min() > .03
    a, b = [np.array([float(F(v['model'][key])) for v in c['blocks']])
            for key in ('lambda_self', 'lambda_neighbor')]
    assert np.linalg.matrix_rank(np.diag(a)+np.diag(b)@P) == d


def jet(coeff, tau, z):
    k = len(coeff)//3
    p = [np.polynomial.Chebyshev([float(F(x)) for x in coeff[j*k:(j+1)*k]], domain=[0, 1])
         .convert(kind=np.polynomial.Polynomial)*np.polynomial.Polynomial([0, 1]) for j in range(3)]
    return (sum(p[j].deriv()(tau)*z**j for j in range(3)),
            p[1](tau)+2*p[2](tau)*z, 2*p[2](tau))


@pytest.mark.parametrize('side', ['lower', 'upper'])
@pytest.mark.parametrize('d', [10, 50])
def test_local_enclosures_bound_the_full_correlated_generator(side, d):
    c = candidate(d)
    network.validate(c)
    d, blocks = c['dimension'], c['blocks']
    P, S = matrices(c)
    ms = [{k: float(F(v)) for k, v in b['model'].items()} for b in blocks]
    m, gamma = ms[0], ms[0]['gamma']
    h, cap = gamma-1, m['cap']/np.sqrt(d)
    polys = [network.residual_polys(b)[0] for b in blocks]
    rng = np.random.default_rng(1110)
    for tau in (.03, .23, .71, 1.):
        for scale in (.15, 1.5):
            z = rng.uniform(-scale, scale, d)
            y = P@z
            jets = np.array([jet(b[side]['coeff'], tau, x) for b, x in zip(blocks, z)])
            grad, diag = jets[:, 1]/d, jets[:, 2]/d
            mu = np.array([v['lambda_constant']+v['lambda_self']*x+v['lambda_neighbor']*yy
                           for v, x, yy in zip(ms, z, y)])/np.sqrt(d)
            signal = mu+m['cross']*grad
            if side == 'lower':
                sol = minimize(lambda p: gamma/2*p@S@p-signal@p, np.zeros(d),
                               jac=lambda p: gamma*S@p-signal, bounds=[(-cap, cap)]*d,
                               method='L-BFGS-B', options={'gtol':1e-12, 'ftol':1e-15, 'maxiter':1000})
                assert sol.success
                policy = sol.x
            else:
                base_grad = np.array([jet(b['base'], tau, x)[1]/d for b, x in zip(blocks, z)])
                policy = np.clip((mu+m['cross']*base_grad)/gamma, -cap, cap)
            drift = -m['kappa']*z+m['drift_neighbor']*y-h*m['cross']*policy
            reaction = -h*m['risk_free_rate']-h*policy@mu+h*gamma/2*policy@S@policy
            full = -h*m['risk_free_rate']+jets[:, 0].mean()-drift@grad-m['variance']/2*(diag.sum()+grad@grad)-reaction
            values, potential_sum = [], 0.
            for i, (b, pp) in enumerate(zip(blocks, polys)):
                sig = arithmetic.ends(pp[6 if side == 'lower' else 7].at(F(tau), F(z[i]), F(y[i])))
                divisor = gamma*(ms[i]['risk_lower'] if side == 'lower' else 1)
                point = sum(sig)/2
                branch = 0 if point < -divisor*m['cap'] else 2 if point > divisor*m['cap'] else 1
                value = pp[branch+(0 if side == 'lower' else 3)].at(F(tau), F(z[i]), F(y[i]))
                values.append(sum(arithmetic.ends(value))/2)
                potential_sum += sum(arithmetic.ends(arithmetic.potential(b[side]['potential'])
                                                    .at(F(tau), F(z[i]), F(y[i]))))/2
            if side == 'lower':
                assert potential_sum >= -1e-12
                assert full <= np.mean(values)+1e-12
            else:
                assert potential_sum <= 1e-12
                assert full >= np.mean(values)-1e-12


def test_invalid_matrix_and_jensen_certificates_are_rejected():
    c = candidate()
    bad = copy.deepcopy(c)
    bad['blocks'][0]['model']['risk_lower'] = '1'
    with pytest.raises(ValueError, match='quadratic-form'):
        network.validate(bad)
    bad = copy.deepcopy(c)
    k = bad['blocks'][0]['time_terms']
    for b in bad['blocks']:
        b['lower']['potential'][k] = '1/10'
    with pytest.raises(ValueError, match='curvature'):
        network.validate(bad)
    bad = copy.deepcopy(c)
    bad['blocks'][1]['upper']['potential'][0] = '1/4'
    with pytest.raises(ValueError, match='common'):
        network.validate(bad)


@pytest.mark.parametrize('dimension', [10, 20, 50])
def test_complete_covers_and_independent_crra_transformation(dimension):
    c = candidate(dimension)
    record = json.loads((ROOT/f'results/highdim_network/certificate_d{dimension}.json').read_text())
    covers = record['covers']
    d = c['dimension']
    def number(x):
        f = F(x)
        return mp.mpf(f.numerator)/f.denominator
    m = c['blocks'][0]['model']
    q = 1-number(m['gamma'])
    low = high = q*number(m['risk_free_rate'])
    for b, v in zip(c['blocks'], covers):
        assert v['cover_complete'] and v['exact_covered_volume'] == '1'
        k = b['time_terms']
        eta_l = max(0., v['lower_residual_upper'], v['lower_trace_rate_upper'])
        eta_u = max(0., -v['upper_residual_lower'], v['upper_trace_rate_upper'])
        low += (sum(number(x) for x in b['lower']['coeff'][:k])-number(eta_l))/d
        high += (sum(number(x) for x in b['upper']['coeff'][:k])+number(eta_u))/d
    error = number(record['localization_upper'])
    low, high = mp.exp(low)-error, mp.exp(high)+error
    loss = (1-(low/high)**(-1/q))*10000
    assert number(record['whole_space_optimal_factor_lower']) <= low
    assert number(record['whole_space_policy_factor_upper']) >= high
    assert number(record['relative_ce_loss_upper_bp']) >= loss
    assert abs(float(loss)-record['relative_ce_loss_upper_bp']) < 1e-9
    with pytest.raises(ValueError, match='Every coordinate'):
        network.combine(c, covers[:-1])


def test_reproduce_one_full_continuous_cover():
    b = candidate()['blocks'][3]
    report = network.verify_block(b)
    assert report['cover_complete'] and report['exact_covered_volume'] == '1'
    assert max(report['lower_residual_upper'], -report['upper_residual_lower'],
               report['lower_trace_rate_upper'], report['upper_trace_rate_upper']) <= .001


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_each_local_branch_matches_independent_polynomial_differentiation(side):
    c = candidate()
    branches = set()
    for b in (c['blocks'][0], c['blocks'][5]):
        polys, thresholds, _ = network.residual_polys(b)
        m = {k: float(F(v)) for k, v in b['model'].items()}
        risk = m['risk_lower' if side == 'lower' else 'risk_upper']
        for tau in (.12, .74, 1.):
            for x in (-1.4, -.3, 0., 1.4):
                for y in (-1.1, .2, 1.1):
                    ft, fx, fxx = jet(b[side]['coeff'], tau, x)
                    signal = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y+m['cross']*fx/np.sqrt(10)
                    used = signal if side == 'lower' else (m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y
                            +m['cross']*jet(b['base'], tau, x)[1]/np.sqrt(10))
                    divisor = 3*(risk if side == 'lower' else 1)
                    p = np.clip(used/divisor, -.11, .11)
                    branch = 0 if used < -.11*divisor else 2 if used > .11*divisor else 1
                    branches.add(branch)
                    phi = np.array([float(F(v)) for v in b[side]['potential']])
                    linear = np.polynomial.chebyshev.chebval(2*tau-1, phi[:5])
                    quadratic = np.polynomial.chebyshev.chebval(2*tau-1, phi[5:])
                    direct = ft+(x-.2*y)*fx-.03125*(fxx+fx*fx/10)+2*(p*signal-1.5*risk*p*p)
                    direct += linear*(y-x)+quadratic*(y*y-x*x)
                    value = polys[branch+(0 if side == 'lower' else 3)].at(F(tau), F(x), F(y))
                    assert sum(arithmetic.ends(value))/2 == pytest.approx(direct, abs=2e-12)
    assert branches == {0, 1, 2}


def test_report_checks_sources_and_rounds_every_loss_upward():
    from decimal import Decimal
    import report_highdim_network as report
    rows = report.records()
    latex = report.table(rows)
    for row in rows:
        line = next(s for s in latex.splitlines() if s.startswith(str(row['dimension'])+' & '))
        displayed = Decimal(line.split('&')[1].strip())
        assert displayed >= Decimal.from_float(row['relative_ce_loss_upper_bp'])
        assert displayed-Decimal.from_float(row['relative_ce_loss_upper_bp']) < Decimal('.001')
