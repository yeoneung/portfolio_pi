"""Independent generator identities and continuous-cover certificate checks."""
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
import highdim_local_verify as cert

iv.dps = 60
mp.dps = 80


def candidate(d=3):
    return json.loads((ROOT/f'results/highdim_local/candidate_d{d}.json').read_text())


def number(x):
    q = F(x)
    return mp.mpf(q.numerator)/q.denominator


def interval_contains(interval, value):
    # The independent evaluation is at 80 digits, versus 60 for the verifier.
    lo, hi = [mp.mpf(x) for x in interval._mpi_]
    assert lo-mp.mpf('1e-70') <= value <= hi+mp.mpf('1e-70')


def independent_jet(coeff, t, x):
    """mpmath differentiation, without the verifier's polynomial recurrence."""
    k = len(coeff)//3
    a = list(map(number, coeff))
    def f(tau, z):
        return sum(a[j*k+i]*tau*mp.chebyt(i, 2*tau-1)*z**j
                   for j in range(3) for i in range(k))
    return f(t, x), mp.diff(lambda s: f(s, x), t), mp.diff(lambda z: f(t, z), x), mp.diff(lambda z: f(t, z), x, 2)


@pytest.mark.parametrize('d', [3, 5, 10, 20, 50])
def test_covariance_and_full_rank_coupling(d):
    c = candidate(d)
    m = {k: float(F(v)) for k, v in c['model'].items()}
    shift = np.roll(np.eye(d), -1, axis=0)  # (Sz)_i = z_(i+1)
    drift = -m['kappa']*np.eye(d)+m['drift_neighbor']*shift
    loading = m['lambda_self']*np.eye(d)+m['lambda_neighbor']*shift
    cov = np.block([[np.eye(d), m['cross']*np.eye(d)],
                    [m['cross']*np.eye(d), m['variance']*np.eye(d)]])
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.linalg.eigvalsh(-(drift+drift.T)/2).min() >= .8-1e-14
    assert np.linalg.svd(loading, compute_uv=False).min() >= .25-1e-14
    assert np.count_nonzero(shift-np.diag(np.diag(shift))) == d


@pytest.mark.parametrize('side', ['lower', 'upper', 'base'])
def test_interval_branch_polynomials_against_independent_high_precision(side):
    c = candidate(10)
    polys, _ = cert.residual_polys(c)
    offset = {'lower': 0, 'upper': 3, 'base': 6}[side]
    m = {k: number(v) for k, v in c['model'].items()}
    coeff = c['base'] if side == 'base' else c[side]['coeff']
    for ts, xs, ys in [('1/3', '-4/3', '-1'), ('7/8', '0', '0'), ('1', '5/4', '3/2')]:
        t, x, y = map(number, (ts, xs, ys))
        _, ft, fx, fxx = independent_jet(coeff, t, x)
        target = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y+m['cross']/mp.sqrt(10)*fx
        fixedfx = independent_jet(c['base'], t, x)[2] if side == 'upper' else fx
        signal = m['lambda_constant']+m['lambda_self']*x+m['lambda_neighbor']*y+m['cross']/mp.sqrt(10)*fixedfx
        core = ft+(m['kappa']*x-m['drift_neighbor']*y)*fx-m['variance']/2*(fxx+fx*fx/10)
        if side != 'base':
            p = list(map(number, c[side]['potential']))
            k = len(p)//2
            core += sum(mp.chebyt(i, 2*t-1)*(p[i]*(y-x)+p[k+i]*(y*y-x*x)) for i in range(k))
        for j, p in enumerate((-m['cap'], signal/2, m['cap'])):
            interval_contains(polys[offset+j].at(ts, xs, ys), core+p*target-p*p)


@pytest.mark.parametrize('d', [3, 10, 50])
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_full_dimensional_generator_equals_telescoping_local_sum(d, side):
    c = candidate(d)
    m = {k: float(F(v)) for k, v in c['model'].items()}
    rng = np.random.default_rng(662)
    coeff = c[side]['coeff']
    k = c['time_terms']
    polys, _ = cert.residual_polys(c)
    for tau in (.13, .64, 1.):
        z = rng.uniform(-1.5, 1.5, d)
        # Independent Chebyshev library and derivatives in physical time.
        def jet(data):
            time_polys = [np.polynomial.Chebyshev(list(map(lambda v: float(F(v)), data[j*k:(j+1)*k])), domain=[0, 1]).convert(kind=np.polynomial.Polynomial)*np.polynomial.Polynomial([0, 1]) for j in range(3)]
            a, b, cc = [p(tau) for p in time_polys]
            at, bt, ct = [p.deriv()(tau) for p in time_polys]
            return at+bt*z+ct*z*z, b+2*cc*z, np.full(d, 2*cc)
        ft, fx, fxx = jet(coeff)
        grad, hessian = fx/d, np.diag(fxx/d)
        basegrad = jet(c['base'])[1]/d if side == 'upper' else grad
        lam = (m['lambda_constant']+m['lambda_self']*z+m['lambda_neighbor']*np.roll(z, -1))/math.sqrt(d)
        pi = np.clip((lam+m['cross']*basegrad)/2, -m['cap']/math.sqrt(d), m['cap']/math.sqrt(d))
        drift = -m['kappa']*z+m['drift_neighbor']*np.roll(z, -1)-m['cross']*pi
        running = -m['risk_free_rate']-pi@lam+pi@pi
        full = -m['risk_free_rate']+ft.mean()-drift@grad-m['variance']/2*np.trace(hessian+np.outer(grad, grad))-running
        offset = 0 if side == 'lower' else 3
        signal = lam+m['cross']*basegrad
        local = []
        for i in range(d):
            branch = 0 if signal[i] < -2*m['cap']/math.sqrt(d) else 2 if signal[i] > 2*m['cap']/math.sqrt(d) else 1
            val = polys[offset+branch].at(F(tau), F(float(z[i])), F(float(z[(i+1)%d])))
            local.append(sum(cert.ends(val))/2)
        assert np.mean(local) == pytest.approx(full, abs=3e-14)


def test_fixed_policy_upper_is_not_regreedified():
    c = candidate(3)
    polys, _ = cert.residual_polys(c)
    # Locate an interior control where the two evaluation gradients differ.
    t, x, y = number('3/4'), number('0'), number('0')
    gx = independent_jet(c['base'], t, x)[2]
    ux = independent_jet(c['upper']['coeff'], t, x)[2]
    assert abs(gx-ux) > number('1/100000')
    expected = number('1/5')+number('1/8')/mp.sqrt(3)*gx
    interval_contains(polys[10].at('3/4', '0', '0'), expected)


def test_all_uncertain_clamp_branches_are_kept():
    lo = np.zeros((15, 1, 1, 1))
    hi = lo.copy()
    lo[:9, 0, 0, 0] = -np.arange(1, 10)
    hi[:9, 0, 0, 0] = np.arange(1, 10)
    lo[9:12] = -1
    hi[9:12] = 1
    b = cert.cell_bounds(lo, hi, F(11, 100))
    assert b['lower_residual_upper'] == 3
    assert b['upper_residual_lower'] == -6
    assert b['base_residual_upper'] == 9


def test_bernstein_and_bisection_enclose_exact_polynomial():
    t, x, y = [cert.Poly.variable(i) for i in range(3)]
    p = F(2, 7)+t*x*x-F(3, 11)*y*t*t+x*y*cert.iv.sqrt(cert.scalar(2))
    lo, hi = cert.bernstein(p, F(3, 2), (2, 2, 1))
    cells = [(lo[None], hi[None], [(F(0), F(1)), (F(-3, 2), F(3, 2)), (F(-3, 2), F(3, 2))])]
    for axis in (0, 1, 2, 0, 1, 2):
        new = []
        for a, b, box in cells:
            ll, lh, rl, rh = cert.split(a, b, axis)
            mid = sum(box[axis])/2
            lb, rb = list(box), list(box)
            lb[axis], rb[axis] = (box[axis][0], mid), (mid, box[axis][1])
            new += [(ll, lh, lb), (rl, rh, rb)]
        cells = new
    for a, b, box in cells:
        for choose in [(0, 0, 0), (1, 1, 1), (F(1, 3), F(2, 5), F(3, 7))]:
            point = [u+(v-u)*s for (u, v), s in zip(box, choose)]
            v = p.at(*point)
            assert number(F(float(a.min()))) <= mp.mpf(v._mpi_[0])
            assert number(F(float(b.max()))) >= mp.mpf(v._mpi_[1])


def test_trace_terminal_and_time_only_repair_identity():
    c = candidate()
    base, lower = cert.factor(c['base']), cert.factor(c['lower']['coeff'])
    correction = F(1, 2000)*cert.Poly.variable(0)
    repaired = lower-correction
    for x in ('-3/2', '0', '7/8'):
        interval_contains(base.at(0, x, 0), mp.mpf(0))
        interval_contains(lower.at(0, x, 0), mp.mpf(0))
        interval_contains((repaired.derivative(1)-lower.derivative(1)).at('1/2', x, 0), mp.mpf(0))
        interval_contains((repaired.derivative(0)-lower.derivative(0)).at('1/2', x, 0), -number('1/2000'))
        q = (lower-base).divide_time()
        value = (lower-base).at('2/3', x, 0)
        check = q.at('2/3', x, 0)*cert.scalar('2/3')
        assert abs(float(value.a)-float(check.a)) < 1e-50


def test_exit_threshold_and_lyapunov_generator():
    c = candidate(10)
    m = {k: number(v) for k, v in c['model'].items()}
    d = c['dimension']
    output = cert.localization(c, .1)
    shift = m['drift_neighbor']*m['radius']+m['cross']*m['cap']/mp.sqrt(d)
    probability = mp.mpf(0)
    for j in range(1, 129):
        threshold = mp.exp(m['kappa']*(j-1)/128)*(m['radius']-shift/m['kappa'])+shift/m['kappa']
        variance = m['variance']/(2*m['kappa'])*(mp.exp(2*m['kappa']*j/128)-1)
        probability += 2*d*mp.exp(-threshold**2/(2*variance))
    assert number(F(output['exit_probability_upper'])) >= probability
    rng = np.random.default_rng(503)
    for _ in range(200):
        z = rng.normal(size=d)*rng.uniform(0, 5)
        pi = rng.uniform(-1, 1, d)*float(m['cap'])/math.sqrt(d)
        lam = (float(m['lambda_constant'])+float(m['lambda_self'])*z+float(m['lambda_neighbor'])*np.roll(z, -1))/math.sqrt(d)
        grad = 2*.05*z/d
        drift = -float(m['kappa'])*z+float(m['drift_neighbor'])*np.roll(z, -1)-float(m['cross'])*pi
        generator_ratio = drift@grad+float(m['variance'])/2*(.1+grad@grad)-float(m['risk_free_rate'])-pi@lam+pi@pi
        assert generator_ratio <= output['weight_generator_exponent_upper']


def test_jensen_right_sum_encloses_independent_integral():
    c = candidate(10)
    m = {k: number(v) for k, v in c['model'].items()}
    a = m['kappa']-m['drift_neighbor']
    loading = m['lambda_self']+abs(m['lambda_neighbor'])
    def majorant(t):
        variance = m['lambda_constant']**2+loading**2*m['variance']/(2*a)*(1-mp.exp(-2*a*t))
        shift = loading*m['cross']*m['cap']/(a*mp.sqrt(10))*(1-mp.exp(-a*t))
        return (mp.sqrt(variance)+shift)**2
    integral = mp.quad(majorant, [0, 1])
    result = cert.jensen_lower(c)
    assert number(F(result['analytic_moment_integral_upper'])) > integral
    assert number(F(result['analytic_optimal_factor_lower'])) <= mp.exp(-m['risk_free_rate']-integral/4)


def test_constant_market_jensen_limit():
    c = candidate()
    c['model'].update(lambda_self='0', lambda_neighbor='0', cross='0')
    result = cert.jensen_lower(c)
    exact = mp.exp(-number('3/100')-number('1/5')**2/4)
    got = number(F(result['analytic_optimal_factor_lower']))
    assert got <= exact and exact-got < number('1/1000000000000000')


def test_coupled_pde_does_not_collapse_to_mean_and_variance():
    c = candidate(10)
    coeff = c['base']
    poly, _ = cert.residual_polys(c)
    z1, z2 = [F(1), F(-1)]+[F(0)]*8, [F(1), F(0), F(-1)]+[F(0)]*7
    assert sum(z1) == sum(z2) and sum(x*x for x in z1) == sum(x*x for x in z2)
    def residual(z):
        total = 0.
        for i, x in enumerate(z):
            y = z[(i+1)%10]
            signal = float(poly[11].at(1, x, y).a)
            branch = 0 if signal < -.22 else 2 if signal > .22 else 1
            total += float(poly[6+branch].at(1, x, y).a)/10
        return total
    assert abs(residual(z1)-residual(z2)) > 1e-4
    # The structured candidate agrees at these states, but its PDE does not.
    f = cert.factor(coeff)
    v1, v2 = sum(float(f.at(1, x, 0).a) for x in z1), sum(float(f.at(1, x, 0).a) for x in z2)
    assert v1 == pytest.approx(v2, abs=1e-15)


def test_radius_sensitivity_keeps_the_implemented_policy_fixed():
    c = candidate(50)
    base = list(c['base'])
    c['implementation_radius_exact'] = '3/2'
    c['model']['radius'] = '1'
    cover = cert.verify_cover(c, F(1, 2000), max_depth=30)
    result = cert.combine(c, cover)
    assert c['base'] == base
    assert result['implementation_radius_exact'] == '3/2'
    assert result['certification_radius_exact'] == '1'
    assert result['relative_ce_loss_upper_bp'] > 100
    assert result['localization_upper'] > .003
    c['implementation_radius_exact'] = '3/4'
    with pytest.raises(ValueError, match='inside'):
        cert.residual_polys(c)


def test_incomplete_cover_fails_closed():
    with pytest.raises(ArithmeticError, match='Cover incomplete'):
        cert.verify_cover(candidate(), F(1, 10**12), max_depth=1)


def test_unsupported_localization_inputs_fail_closed():
    for key, value in [('cross', '-1/8'), ('lambda_constant', '-1/5'), ('cap', '0'), ('radius', '-1')]:
        c = candidate()
        c['model'][key] = value
        with pytest.raises(ValueError):
            cert.residual_polys(c)


@pytest.mark.parametrize('d', [3, 5, 10, 20, 50])
def test_complete_cover_and_ce_orientation(d):
    c = candidate(d)
    cover = cert.verify_cover(c, F(1, 2000), max_depth=30)
    result = cert.combine(c, cover)
    assert cover['cover_complete'] and cover['exact_covered_volume'] == '1'
    assert 0 < result['relative_ce_loss_upper_bp'] < 100
    # Gamma=2: CE = 1/g, so relative loss is 1-g*/g_policy.
    lower = number(F(result['whole_space_optimal_factor_lower']))
    upper = number(F(result['whole_space_policy_factor_upper']))
    displayed_bound = 1-lower/upper
    assert abs(float(displayed_bound)-result['relative_ce_loss_upper']) < 2e-15
    assert result['policy_updates_during_certification'] == 0
