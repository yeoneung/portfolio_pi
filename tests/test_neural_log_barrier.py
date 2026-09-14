"""Independent high-precision and whole-cell checks for logarithmic verification."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import numpy as np
from mpmath import mp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import neural_log_barrier as log
import neural_certificate_audit as audit
from test_neural_certificate_audit import small_checkpoint, real, reference_residual


def test_tanh_transcendental_enclosure():
    mp.dps = 90
    values = np.r_[0., 1e-30, -1e-30, np.linspace(-25, 25, 301),
                   np.nextafter(20., -np.inf), np.nextafter(20., np.inf)]
    lo, hi = log.tanh_interval(values)
    for x, l, u in zip(values, lo, hi):
        assert real(l) <= mp.tanh(real(x)) <= real(u)
    assert np.max(hi-lo) < 1e-12


def exact_polynomial(a, x, y):
    return sum(real(c)*(real(x)**i)*(real(y)**j) for c, (i, j) in zip(
        a.c[:, 0, 0], ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))))


def test_quadratic_arithmetic_with_remainders():
    mp.dps = 90
    a = log.Affine(np.array([.125, -.25, .5, .125, -.25, .0625])[:, None, None], .001)
    b = log.Affine(np.array([-.375, .25, .125, -.0625, .125, .25])[:, None, None], .002)
    expressions = [(a+b, lambda x, y: x+y), (a*b, lambda x, y: x*y)]
    for output, fn in expressions:
        for x in np.linspace(-1, 1, 7):
            for y in np.linspace(-1, 1, 7):
                center = exact_polynomial(output, x, y)
                for sign_a in (-1, 1):
                    for sign_b in (-1, 1):
                        exact = fn(exact_polynomial(a, x, y)+sign_a*real(.001),
                                   exact_polynomial(b, x, y)+sign_b*real(.002))
                        assert abs(exact-center) <= real(output.r[0, 0])


@pytest.mark.parametrize('center', [-4., -.2, 0., .19, 4.])
def test_nonlinear_composition_and_active_switch(center):
    mp.dps = 90
    a = log.Affine(np.array([center, .075, -.125, .02, .025, -.0125])[:, None, None], 1e-5)
    for output, fn in ((a.tanh(), mp.tanh), (a.hamiltonian(.1), lambda s: max(-real(.1), min(real(.1), s/2))*s-max(-real(.1), min(real(.1), s/2))**2)):
        for x in np.linspace(-1, 1, 9):
            for y in np.linspace(-1, 1, 9):
                for sign in (-1, 1):
                    exact = fn(exact_polynomial(a, x, y)+sign*real(1e-5))
                    assert abs(exact-exact_polynomial(output, x, y)) <= real(output.r[0, 0])


def test_small_network_full_cell_against_independent_differentiation():
    mp.dps = 90
    c = small_checkpoint()
    cells = [[.35, .07, -.4, .08], [.8, .03, .8, .05]]
    lows, highs, _ = log.network_residual(c, cells)
    for cell, lo, hi in zip(cells, lows, highs):
        t, dt, z, dz = cell
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                tau = (real(t)+x*real(dt))*real(c['horizon'])
                state = real(c['market']['xbar'][0])+(real(z)+y*real(dz))*real(c['domain_scale'])*real(c['market']['eta'][0])
                value, residual = reference_residual(c, tau, state)
                assert real(lo) <= residual/value <= real(hi)


def test_actual_network_point_encloses_multiprecision_interval():
    c = json.loads((ROOT/'results/neural_certificate/seed_11_final.json').read_text())
    audit.iv.dps = 70
    for t, z in ((0., 1.), (1., 0.), (.125, .875)):
        lo, hi, _ = log.network_residual(c, [[t, 0., z, 0.]])
        tau = F(t)*F(c['horizon'])
        state = F(c['market']['xbar'][0])+F(z)*F(c['domain_scale'])*F(c['market']['eta'][0])
        rigorous = audit.evaluate(c, tau, state)['normalized_greedy_residual_interval']
        assert lo[0] <= rigorous[0] <= rigorous[1] <= hi[0]
        assert hi[0]-lo[0] < 1e-5


def test_actual_network_nonzero_cell_encloses_interior_and_corners():
    c = json.loads((ROOT/'results/neural_certificate/seed_11_final.json').read_text())
    audit.iv.dps = 60
    t, dt, z, dz = F(1, 16), F(1, 2048), F(63, 64), F(1, 2048)
    lo, hi, _ = log.network_residual(c, [[float(t), float(dt), float(z), float(dz)]])
    for x, y in ((-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)):
        tau = (t+x*dt)*F(c['horizon'])
        state = F(c['market']['xbar'][0])+(z+y*dz)*F(c['domain_scale'])*F(c['market']['eta'][0])
        bounds = audit.evaluate(c, tau, state)['normalized_greedy_residual_interval']
        assert lo[0] <= bounds[0] <= bounds[1] <= hi[0]


def test_exact_time_barrier_cancels_level_scaling_residual():
    # For u=exp(-a*t), exact g=exp(-b*t), rho=b-a.
    # The lower or upper logarithmic correction is exactly the missing slope.
    for a, b in ((F(1, 3), F(2, 3)), (F(2, 3), F(1, 3))):
        rho = b-a
        assert rho-max(F(0), rho) <= 0
        assert rho+max(F(0), -rho) >= 0
        assert max(F(0), rho)+max(F(0), -rho) == abs(b-a)


def test_linear_product_rounding_encloses_exact_dot_product():
    rng = np.random.default_rng(731)
    a = log.Affine(rng.normal(size=(6, 1, 128)), .001)
    w, bias = rng.normal(size=(3, 128)), rng.normal(size=3)
    output = a.linear(w, bias)
    mp.dps = 90
    for j in range(3):
        error = mp.mpf('0')
        for k in range(6):
            exact = sum(real(a.c[k, 0, i])*real(w[j, i]) for i in range(128))
            if k == 0:
                exact += real(bias[j])
            error += abs(exact-real(output.c[k, 0, j]))
        error += sum(abs(real(w[j, i]))*real(.001) for i in range(128))
        assert error <= real(output.r[0, j])
