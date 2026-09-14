"""Independent checks of the one-sided financial comparison and exit inputs."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import numpy as np
from mpmath import mp, iv
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import neural_welfare_bound as welfare
from test_neural_certificate_audit import small_checkpoint, real, reference_value


def test_global_subsolution_against_independent_moment_quadrature():
    c = json.loads((ROOT/'results/neural_certificate/seed_11_final.json').read_text())
    m = c['market']
    mp.dps, iv.dps = 80, 50
    l0, l1, cross = list(map(real, m['lam0'])), [real(a[0]) for a in m['Lam']], [real(a[0]) for a in m['Gamma']]
    k = real(m['K'][0][0])+sum(g*l for g, l in zip(cross, l1))/2
    mean = (real(m['k0'][0])-sum(g*l for g, l in zip(cross, l0))/2)/k
    def cost(s):
        mz = mean+(real(m['xbar'][0])-mean)*mp.exp(-k*s)
        variance = real(m['Q'][0][0])*(-mp.expm1(-2*k*s))/(2*k)
        return real(c['risk_free_rate'])+sum((a+b*mz)**2+b*b*variance for a, b in zip(l0, l1))/4
    j = mp.quad(cost, [0, real(c['horizon'])])
    result = welfare.jensen_lower(c)
    lo, hi = result['jensen_exponent_interval']
    assert real(lo) <= j <= real(hi)
    assert real(result['global_optimal_factor_lower']) <= mp.exp(-j)


@pytest.mark.parametrize('tau,z', [(.05, -.95), (.75, .95), (1., -1.), (1., 0.)])
def test_reaction_encloses_actual_network_policy(tau, z):
    c = json.loads((ROOT/'results/neural_certificate/seed_11_final.json').read_text())
    m = c['market']
    mp.dps = 80
    t = real(tau)*real(c['horizon'])
    state = real(m['xbar'][0])+real(z)*real(c['domain_scale'])*real(m['eta'][0])
    hz = mp.diff(lambda x: mp.log(reference_value(c, t, x)), state)
    reaction = -real(c['risk_free_rate'])
    for i, cap in enumerate(c['caps']):
        lam = real(m['lam0'][i])+real(m['Lam'][i][0])*state
        pi = max(-real(cap), min(real(cap), (lam+real(m['Gamma'][i][0])*hz)/2))
        reaction += pi*pi-pi*lam
    upper = welfare.reaction_upper(c, [[tau, 0., z, 0.]])[0]
    assert reaction <= real(upper)
    assert real(upper)-reaction < mp.mpf('1e-6')


def test_additive_exit_generator_enclosure():
    c = small_checkpoint()
    mp.dps, iv.dps = 80, 50
    out = welfare.exit_probability(c, powers=(20,), bins=256)
    a, drift = real(F(out['lyapunov_exponent_exact'])), real(F(out['drift_shift_upper_exact']))
    k, q = real(c['market']['K'][0][0]), real(c['market']['Q'][0][0])
    radius = real(c['domain_scale'])*real(c['market']['eta'][0])
    d = 2*a*k-2*q*a*a
    for i in range(1001):
        x = radius*i/1000
        generator = (a*q-d*x*x+2*a*drift*x)*mp.exp(a*x*x)
        assert generator <= real(out['generator_upper'])
    probability = min(mp.mpf(1), real(out['generator_upper'])*real(c['horizon'])/mp.expm1(a*radius**2))
    assert probability <= real(out['exit_probability_upper'])


def test_time_resolved_continuation_difference():
    # Nonnegative, nonconstant reaction envelope in remaining-time coordinates.
    mp.dps = 80
    total, rate = mp.mpf(3), mp.mpf('.03')
    integral = lambda t: t*t/40
    bound = mp.exp(integral(total))-mp.exp(-rate*total)
    for elapsed in np.linspace(0, 3, 101):
        s = real(elapsed)
        weighted = mp.exp(integral(total))-mp.exp(integral(total)-integral(total-s)-rate*(total-s))
        assert weighted <= bound+mp.mpf('1e-75')


def test_missing_full_cover_is_rejected():
    with pytest.raises(ValueError):
        welfare.combine(small_checkpoint(), {'bounded_domain_certified': False}, {})


def test_exact_subsolution_square_identity():
    # The identity uses h=-J; check signs without a fitted or numerical PDE.
    q, cross = F(1, 4), F(1, 10)
    alpha = q-cross**2/2
    for jz, lam, pi in ((F(1, 3), F(1, 5), F(-1, 10)), (F(-1, 2), F(1, 8), F(1, 5))):
        residual = -q*jz*jz/2-lam*lam/4+cross*lam*jz/2+pi*(lam-cross*jz)-pi*pi
        square = -alpha*jz*jz/2-(pi-(lam-cross*jz)/2)**2
        assert residual == square <= 0
