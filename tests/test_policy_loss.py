"""Independent rational checks of frozen-policy certification and its arithmetic."""
from fractions import Fraction as F
from math import comb
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from mpmath import mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import bernstein_arithmetic as ba
import policy_loss_certificate as p
import portfolio_welfare as w


def exact_value(poly, s, y):
    return sum((a*s**i*y**j for (i, j), a in poly.items()), F(0))


def inside(enclosure, rational):
    lower, upper = enclosure
    assert rational.shape == lower.shape == upper.shape
    for ij in np.ndindex(lower.shape):
        assert F(float(lower[ij])) <= rational[ij] <= F(float(upper[ij]))


@pytest.mark.parametrize("axis", [0, 1])
def test_rational_restriction_and_derivative(axis):
    poly = {(0, 0): F(1, 7), (3, 1): F(-17, 3), (1, 4): F(2, 11)}
    coefficients = ba.rational_bernstein(poly)
    parts = ba.rational_split(coefficients, axis)
    for side in (0, 1):
        for s, y in ((F(0), F(1)), (F(1, 3), F(2, 5)), (F(1), F(0))):
            n, m = (int(a)-1 for a in parts[side].shape)
            got = sum((parts[side][i, j]*comb(n, i)*s**i*(1-s)**(n-i)
                       *comb(m, j)*y**j*(1-y)**(m-j)
                       for i in range(n+1) for j in range(m+1)), F(0))
            coords = [s, y]
            coords[axis] = (coords[axis]+side)/2
            assert got == exact_value(poly, *coords)
    derivative = ba.derivative(ba.enclose_rationals(coefficients), axis)
    inside(derivative, ba.rational_bernstein(w.exact.derivative(poly, axis)))


def test_outward_product_sum_elevation_against_exact_coefficients():
    a = {(0, 0): F(3, 7), (1, 3): F(-4, 5), (2, 2): F(9, 11)}
    b = {(0, 1): F(8, 3), (3, 2): F(-5, 13)}
    aa, bb = w.exact.bernstein_enclosure(a), w.exact.bernstein_enclosure(b)
    inside(ba.multiply(aa, bb), ba.rational_bernstein(w.exact.mul(a, b)))
    inside(ba.add(aa, bb), ba.rational_bernstein(w.exact.add(a, b)))
    elevated = ba.elevate(aa, (4, 6))
    for s in (F(0), F(2, 5), F(1)):
        for y in (F(0), F(3, 7), F(1)):
            lo = hi = F(0)
            for i in range(4):
                for j in range(6):
                    weight = comb(3, i)*s**i*(1-s)**(3-i)*comb(5, j)*y**j*(1-y)**(5-j)
                    lo += F(float(elevated[0][i, j]))*weight
                    hi += F(float(elevated[1][i, j]))*weight
            assert lo <= exact_value(a, s, y) <= hi


def interval_at(enclosure, s, y):
    n, m = (int(a)-1 for a in enclosure[0].shape)
    values = []
    for endpoint in enclosure:
        values.append(sum((F(float(endpoint[i, j]))*comb(n, i)*s**i*(1-s)**(n-i)
                           *comb(m, j)*y**j*(1-y)**(m-j)
                           for i in range(n+1) for j in range(m+1)), F(0)))
    return values


@pytest.mark.parametrize("restricted", [False, True])
def test_frozen_branch_formula_and_quadratic_gap_including_active_changes(restricted):
    market = w.Market()
    u = {(0, 0): F(1), (1, 1): F(1, 20)}
    v = {(0, 0): F(1), (1, 2): -F(1, 30), (1, 1): F(1, 30)}
    rect = (F(0), F(1), F(0), F(1))
    def restricted_coefficients(poly):
        a = ba.rational_bernstein(poly)
        if restricted:
            a = ba.rational_split(ba.rational_split(a, 0)[0], 0)[1]
            a = ba.rational_split(ba.rational_split(a, 1)[1], 1)[0]
        return ba.enclose_rationals(a)
    if restricted:
        rect = (F(1, 4), F(1, 2), F(1, 2), F(3, 4))
    root, branches = p.branch_roots(restricted_coefficients(u), restricted_coefficients(v), market, rect)
    changed = False
    for s in (F(1, 3), F(1)):
        for k in range(21):
            y = F(k, 20)
            sg, yg = rect[0]+(rect[1]-rect[0])*s, rect[2]+(rect[3]-rect[2])*y
            z = market.radius*(2*yg-1)
            uv, vv = exact_value(u, sg, yg), exact_value(v, sg, yg)
            uz = exact_value(w.exact.derivative(u, 1), sg, yg)/(2*market.radius)
            vz = exact_value(w.exact.derivative(v, 1), sg, yg)/(2*market.radius)
            vzz = exact_value(w.exact.derivative(w.exact.derivative(v, 1), 1), sg, yg)/(4*market.radius**2)
            vt = -exact_value(w.exact.derivative(v, 0), sg, yg)/market.horizon
            mu = tuple(market.mu0[i]+market.mu1[i]*z for i in range(2))
            _, pi = w.qp_exact(tuple(mu[i]+market.cross[i]*uz/uv for i in range(2)), market)
            _, greedy = w.qp_exact(tuple(mu[i]+market.cross[i]*vz/vv for i in range(2)), market)
            changed |= any((pi[i] in (0, market.caps[i])) != (greedy[i] in (0, market.caps[i])) for i in range(2))
            h = lambda a: sum(a[i]*(mu[i]*vv+market.cross[i]*vz) for i in range(2))-vv*sum(a[i]*market.sigma[i][j]*a[j] for i in range(2) for j in range(2))
            expected = uv**2*(vt-market.kappa*z*vz+market.state_variance*vzz/2-h(pi))
            state = tuple(-1 if a == 0 else 1 if a == market.caps[i] else 0 for i, a in enumerate(pi))
            name = next(name for _, name, active in branches if active == state)
            lo, hi = interval_at(root[name], s, y)
            assert lo <= expected <= hi
            numerator = uv*vz-vv*uz
            lo, hi = interval_at(root["log_gradient_numerator"], s, y)
            assert lo <= numerator <= hi
            kappa = (market.cross[0]**2-2*market.sigma12*market.cross[0]*market.cross[1]+market.cross[1]**2)/(1-market.sigma12**2)
            assert 0 <= h(greedy)-h(pi) <= kappa*numerator*numerator/(4*uv*uv*vv)
    if not restricted:
        assert changed


def test_verifier_rejects_small_budget():
    market = w.Market()
    root, branches = p.branch_roots(ba.constant(1), ba.constant(1), market, (F(0), F(1), F(0), F(1)))
    result = p.certify([(root, 0)], branches, market, 1e-10, 1e-10, max_cells=1)
    assert not result["certified"]


def test_policy_hash_stays_fixed_and_evaluation_trace_is_exact():
    policy = json.loads(p.POLICY_PATH.read_text())
    evaluation = json.loads((p.OUT / "evaluation_32_64_6.json").read_text())
    assert evaluation["policy_updates"] == 0
    import hashlib
    assert evaluation["policy_sha256"] == hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest()
    poly = w.candidate_polynomial(evaluation)
    left, right = map(F, evaluation["trace_slopes_exact"])
    for t in (F(0), F(2, 7), F(1)):
        assert exact_value(poly, F(0), t) == 1
        assert exact_value(poly, t, F(0)) == 1+t*left
        assert exact_value(poly, t, F(1)) == 1+t*right


def test_utility_and_ce_bounds_use_interval_rounding():
    mp.dps = 80
    evaluation = json.loads((p.OUT / "evaluation_32_64_6.json").read_text())
    poly = w.candidate_polynomial(evaluation)
    certificate = {"frozen_residual_upper": 2e-5, "hamiltonian_gap_upper": 1e-8}
    b = p.verified_bounds(w.Market(), poly, certificate)
    phi = mp.expm1(mp.mpf(33)/400)/(mp.mpf(33)/100)
    loss = phi*(2*mp.mpf(2e-5)+mp.mpf(1e-8))+2*mp.mpf(b["value_localization_upper"])
    assert mp.mpf(b["unit_wealth_utility_loss_upper"]) >= loss
    assert mp.mpf(b["relative_ce_loss_bp_upper"]) >= 10000*loss/mp.mpf(b["whole_space_policy_factor_interval"][0])


def test_uniform_inner_cylinder_bound_and_invalid_radius():
    cert = {"frozen_residual_upper": 2e-5, "hamiltonian_gap_upper": 1e-8, "v_lower": 0.9324999999999999}
    result = p.uniform_inner_bounds(w.Market(), cert)
    assert 0.14 < result["relative_ce_loss_bp_upper"] < 0.15
    assert result["policy_factor_lower"] > 0.9324
    with pytest.raises(ValueError):
        p.uniform_inner_bounds(w.Market(), cert, F(2))


def test_cli_failure_invalidates_old_certificate_and_rejects_hash(tmp_path, monkeypatch):
    policy = p.POLICY_PATH.read_bytes()
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(policy)
    data = json.loads((p.OUT / "evaluation_16_48_4.json").read_text())
    data["policy_sha256"] = "incorrect hash"
    (tmp_path / "evaluation_16_48_4.json").write_text(json.dumps(data))
    result_path = tmp_path / "evaluation_16_48_4_certificate.json"
    result_path.write_text('{"certified": true}')
    monkeypatch.setattr(p, "POLICY_PATH", policy_path)
    monkeypatch.setattr(p, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify", "--degree-time", "16", "--degree-space", "48", "--reweights", "4", "--verify-only"])
    with pytest.raises(ValueError, match="hash mismatch"):
        p.main()
    assert not json.loads(result_path.read_text())["certified"]
