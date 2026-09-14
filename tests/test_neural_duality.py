"""Financial identities and independent neural-derivative checks for MC."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import neural_duality_comparison as dual


@pytest.fixture
def market():
    c = json.loads(dual.CHECKPOINT.read_text())
    return dual.Market(c, dual.FrozenNetwork(c))


def test_analytic_network_gradient_against_autograd(market):
    m, net = market, market.net
    z = torch.tensor([m.center-m.half, m.center, m.center+.91*m.half], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor([.03, 3., 1.1], dtype=torch.float64)
    y = (z-net.center)/net.half
    d = 1-y*y
    v = torch.stack((tau/m.T, y, d), -1)
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        v = v@w+b
        if i < len(net.weights)-1:
            v = torch.tanh(v)
    h = -m.r*tau+tau*d*v[:, 0]
    expected = torch.autograd.grad(h.sum(), z)[0]
    got_h, got_hz = net.jet(tau, z)
    torch.testing.assert_close(got_h, h, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(got_hz, expected, atol=2e-12, rtol=2e-12)


def test_full_augmented_covariance_and_support(market):
    m = market
    z = m.net.tensor([m.center-m.half*1.2, m.center-.4*m.half,
                      m.center+.99*m.half, m.center+1.3*m.half])
    hz = m.net.jet(2., z)[1]
    got = m.coefficients(z, hz)
    beta = math.sqrt(m.q-float((m.cross*m.cross).sum()))
    for j in range(len(z)):
        inside = abs(float(z[j])-m.center) <= m.half
        h = float(hz[j]) if inside else 0.
        lam = m.l0+z[j]*m.l1
        pd = torch.clamp((lam+m.cross*h)/2, -m.cap, m.cap)
        pi = pd if inside else pd*0
        theta = torch.cat((2*pd-m.cross*h, m.net.tensor([-beta*h])))
        pp = torch.cat((pi, m.net.tensor([0.])))
        ps, po, ts, to = (v[j] for v in got[2:6])
        torch.testing.assert_close(ps*ps+po*po, (pp*pp).sum())
        torch.testing.assert_close(ts*ts+to*to, (theta*theta).sum())
        torch.testing.assert_close(ps*ts+po*to, (pp*theta).sum())
        nu = theta[:-1]-lam
        delta = (m.cap*nu.abs()).sum()
        assert float(delta+(pi*nu).sum()) >= -1e-14
        if inside:
            assert abs(float(got[-1][j])) < 1e-13
        else:
            assert ps == 0 and po == 0
        rng = np.random.default_rng(123)
        for _ in range(20):
            other = m.net.tensor(rng.uniform(-1, 1, len(m.cap)))*m.cap
            assert float(delta+(other*nu).sum()) >= -1e-14


def test_discrete_compensator_identity():
    # Gaussian exponential moment of M_(j+1)/M_j exactly exp(-a dt).
    dt, a, mp, theta = .01, .07, np.array([.2, -.3]), np.array([.5, .1])
    r, plam = .03, .08
    delta = a+plam-mp@theta
    bx, bd = r+plam-.5*(mp@mp), -(r+delta)-.5*(theta@theta)
    expected_ratio = math.exp((bx+bd+.5*np.sum((mp-theta)**2))*dt)
    assert expected_ratio+(-math.expm1(-a*dt)) == pytest.approx(1., abs=1e-15)


def test_exact_constant_market_has_zero_cv_gap():
    # Unconstrained constant Merton, gamma=2: pi=lambda/2, theta=lambda.
    # The same identity holds pathwise at any discretization or random sample.
    rng = np.random.default_rng(301)
    lam, r, horizon = .3, .03, 3.
    w = rng.normal(size=1000)*math.sqrt(horizon)
    lx = (r+3*lam**2/8)*horizon+lam*w/2
    ld = -(r+lam**2/2)*horizon-lam*w
    g = math.exp(-(r+lam**2/4)*horizon)
    samples = np.column_stack((np.exp(-lx), np.exp(ld/2), np.exp(lx+ld)))
    result = dual.summarize(samples, g)['control_variate']
    assert result['policy_factor'] == pytest.approx(g, abs=1e-13)
    assert result['dual_sqrt_density_mean'] == pytest.approx(math.sqrt(g), abs=1e-13)
    assert abs(result['relative_ce_gap_plugin']) < 1e-13
    assert result['gap_standard_error'] < 1e-14


def test_lookup_refinement(market):
    coarse, _ = dual.tabulate(market, 8, 65)
    fine, _ = dual.tabulate(market, 8, 257)
    a = dual.lookup_check(market, coarse, size=1024)
    b = dual.lookup_check(market, fine, size=1024)
    assert b['sampled_max_log_gradient_error'] < a['sampled_max_log_gradient_error']/4


def test_ou_bridge_covariance():
    k, dt = 1.7464749783336673, 3/1024
    a = -math.expm1(-k*dt)/(k*dt)
    v = -math.expm1(-2*k*dt)/(2*k)
    b2 = v-a*a*dt
    assert b2 > 0
    assert a*dt == pytest.approx((1-math.exp(-k*dt))/k)
    assert a*a*dt+b2 == pytest.approx(v)


def test_dual_ce_sign_and_paired_error():
    rng = np.random.default_rng(23)
    x = np.column_stack((.5+.01*rng.normal(size=500),
                         .69+.001*rng.normal(size=500), np.ones(500)))
    out = dual.summarize(x, .5)['raw']
    assert out['relative_ce_gap_plugin'] == pytest.approx(1-x[:, 1].mean()**2/x[:, 0].mean())
    assert out['gap_95_interval'][0] < out['relative_ce_gap_plugin'] < out['gap_95_interval'][1]


def test_banded_implicit_operator():
    pytest.importorskip('scipy')
    from neural_duality_pde import implicit_step
    z = np.linspace(-1, 1, 41)
    rate, dt = -.1, .01
    v = implicit_step(np.ones_like(z), -z, np.full_like(z, rate), .25,
                      z[1]-z[0], dt, (1/(1-dt*rate),)*2)
    np.testing.assert_allclose(v, 1/(1-dt*rate), atol=2e-15)


def test_nonmonotone_pde_rejected():
    pytest.importorskip('scipy')
    from neural_duality_pde import implicit_step
    with pytest.raises(ArithmeticError, match='Nonmonotone'):
        implicit_step(np.ones(5), np.full(5, 100.), np.zeros(5), .1, .5, .01, (1., 1.))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is optional for identity tests')
def test_gpu_float64_matches_cpu_network(market):
    gpu = dual.FrozenNetwork(market.c, 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(179)
    t = rng.uniform(0, market.T, 256)
    z = rng.uniform(market.center-market.half, market.center+market.half, 256)
    a, b = market.net.jet(t, z), gpu.jet(t, z)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left.numpy(), right.cpu().numpy(), atol=1e-11, rtol=1e-11)
