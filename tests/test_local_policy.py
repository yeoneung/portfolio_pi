"""Independent checks of local gluing, transformed PDEs, and exit estimates."""
from fractions import Fraction as F
import hashlib
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
from mpmath import mp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
import local_policy_evaluation as local
import verify_local_policy as verify
import finite_horizon_localization as finite
import portfolio_welfare as w
import bernstein_arithmetic as ba


def value(poly, s, x):
    return sum((a*s**i*x**j for (i, j), a in poly.items()), F(0))


def enclosed_at(enclosure, s, x):
    n, m = (int(a)-1 for a in enclosure[0].shape)
    return [sum((F(float(c[i, j]))*comb(n, i)*s**i*(1-s)**(n-i)
                 *comb(m, j)*x**j*(1-x)**(m-j)
                 for i in range(n+1) for j in range(m+1)), F(0)) for c in enclosure]


def candidate(power=2):
    knots = local.mesh(1, 1)
    nb = 4*(len(knots)-1)
    return {"market": w.Market().json(), "mode": "C1_fixed_greedy_evaluation",
            "knots_exact": list(map(str, knots)), "degree_time": 2, "degree_space": 5,
            "time_power": power, "coefficient_denominator": str(2**20),
            "coefficient_numerators": [[str((i*nb+j)%17-8) for j in range(nb)] for i in range(3)],
            "trace_slopes_exact": list(map(str, w.trace_slopes(w.Market())))}


@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_exact_C1_and_bounded_graded_derivative(power):
    c = candidate(power)
    polys = local.local_polynomials(c)
    assert verify.check_traces(c, polys)["spatial_C1_exact"]
    for poly in polys:
        derivative = verify.physical_time_derivative(poly, power, F(1, 4))
        for s in (F(1, 5), F(1)):
            expected = -value(w.exact.derivative(poly, 0), s, F(2, 7))/(F(1, 4)*power*s**(power-1))
            assert value(derivative, s, F(2, 7)) == expected
        assert isinstance(value(derivative, F(0), F(1, 3)), F)


def test_rejects_interface_jump_and_singular_time_derivative():
    c = candidate()
    polys = local.local_polynomials(c)
    polys[0] = w.exact.add(polys[0], {(2, 1): F(1), (2, 2): F(-1)})
    with pytest.raises(ValueError, match="C1"):
        verify.check_traces(c, polys)
    with pytest.raises(ValueError, match="Singular"):
        verify.physical_time_derivative({(1, 0): F(1)}, 2, F(1, 4))


def test_affine_substitution_exact():
    a = {(0, 0): F(2, 7), (3, 2): F(-11, 5), (2, 4): F(9)}
    b = verify.affine(verify.affine(a, 0, F(1, 4), F(1, 2)), 1, F(3, 8), F(1, 16))
    assert value(b, F(2, 3), F(3, 7)) == value(a, F(1, 4)+F(1, 2)*F(2, 3), F(3, 8)+F(1, 16)*F(3, 7))


def test_transformed_rational_residual_and_KKT_coverage():
    market = w.Market()
    up = {(0, 0): F(1), (2, 1): F(1, 20)}
    vp = {(0, 0): F(1), (2, 1): F(1, 40), (3, 2): F(-1, 80)}
    rect = (F(1, 4), F(3, 4), F(0), F(1))
    pt = verify.physical_time_derivative(vp, 2, market.horizon)
    arrays = [w.exact.bernstein_enclosure(verify.affine(poly, 0, rect[0], rect[1]-rect[0])) for poly in (up, vp, pt)]
    root, branches = verify.root_branches(*arrays, market, rect)
    for x in (F(0), F(1, 3), F(2, 3), F(1)):
        s = F(2, 5)
        theta = rect[0]+(rect[1]-rect[0])*s
        z = market.radius*(2*x-1)
        uv, vv = value(up, theta, x), value(vp, theta, x)
        uz = value(w.exact.derivative(up, 1), theta, x)/(2*market.radius)
        vz = value(w.exact.derivative(vp, 1), theta, x)/(2*market.radius)
        vzz = value(w.exact.derivative(w.exact.derivative(vp, 1), 1), theta, x)/(4*market.radius**2)
        mu = [market.mu0[i]+market.mu1[i]*z for i in range(2)]
        _, pi = w.qp_exact(tuple(mu[i]+market.cross[i]*uz/uv for i in range(2)), market)
        state = tuple(-1 if a == 0 else 1 if a == market.caps[i] else 0 for i, a in enumerate(pi))
        name = next(name for _, name, active in branches if active == state)
        h = sum(pi[i]*(mu[i]*vv+market.cross[i]*vz) for i in range(2))-vv*sum(pi[i]*market.sigma[i][j]*pi[j] for i in range(2) for j in range(2))
        expected = uv**2*(value(pt, theta, x)-z*vz+market.state_variance*vzz/2-h)
        lo, hi = enclosed_at(root[name], s, x)
        assert lo <= expected <= hi


@pytest.mark.parametrize("z", [F(0), F(1, 2)])
def test_finite_horizon_bound_against_independent_high_precision(z):
    mp.dps = 90
    result = finite.bound(w.Market(), z, partitions=(32,))
    zz = mp.mpf(z.numerator)/z.denominator
    total = mp.mpf(0)
    for j in range(1, 33):
        left, right = mp.mpf(j-1)/128, mp.mpf(j)/128
        level = mp.exp(left)*(2-mp.mpf(17)/200)+mp.mpf(17)/200-zz
        variance = mp.expm1(2*right)/8
        total += 2*mp.exp(-level**2/(2*variance))
    payoff = mp.exp(mp.mpf(976339)/3980000/4+mp.mpf(2)/25)+1
    expected = mp.exp(mp.mpf(33)/400)*payoff*total
    assert mp.mpf(result["exit_probability_upper"]) >= total
    assert mp.mpf(result["value_localization_upper"]) >= expected
    assert result["value_localization_upper"] < (1e-16 if not z else 2e-10)


def test_finite_exit_rejects_inapplicable_geometry():
    with pytest.raises(ValueError, match="interior"):
        finite.bound(w.Market(), F(2))
    with pytest.raises(ValueError, match="shifted"):
        finite.bound(w.Market(radius=F(1, 100)))
    with pytest.raises(ValueError, match="partition"):
        finite.bound(w.Market(), partitions=(0,))
    with pytest.raises(ValueError, match="successful"):
        finite.bounds(w.Market(), F(1), {"certified": False})


def test_cover_and_budget_fail_closed():
    c = {"u": ba.constant(1), "v": ba.constant(1), "r": ba.constant(0), "log_gradient_numerator": ba.constant(0)}
    branches = [([], "r", (0, 0))]
    left = (c, branches, (F(0), F(1), F(0), F(1, 2)), 0)
    right = (c, branches, (F(0), F(1), F(1, 2), F(1)), 0)
    assert verify.certify(iter([left, right]), w.Market(), 1e-6)["covered_area_exact"] == "1"
    with pytest.raises(ArithmeticError, match="Incomplete"):
        verify.certify(iter([left]), w.Market(), 1e-6)
    assert not verify.certify(iter([left, right]), w.Market(), 1e-6, max_cells=1)["certified"]


def test_cli_invalidates_stale_result(tmp_path, monkeypatch):
    c = candidate()
    c["policy_sha256"] = "invalid"
    path = tmp_path/"candidate.json"
    path.write_text(json.dumps(c))
    result = tmp_path/"candidate_certificate.json"
    result.write_text('{"certified": true}')
    monkeypatch.setattr(local, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify", "--candidate", "candidate.json"])
    with pytest.raises(ValueError, match="hash"):
        verify.main()
    assert not json.loads(result.read_text())["certified"]
