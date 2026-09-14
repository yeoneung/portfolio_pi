"""Independent algebra, model, interval, and fail-closed welfare checks."""
import contextlib
import io
import json
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from mpmath import mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import portfolio_welfare as w


def evaluate(poly, s, y):
    return sum((v*s**i*y**j for (i, j), v in poly.items()), F(0))


def test_joint_covariance_and_invalid_inputs():
    w.Market().validate()
    with pytest.raises(ValueError):
        replace(w.Market(), cross=(F(2), F(2))).validate()
    with pytest.raises(ValueError):
        replace(w.Market(), sigma12=F(1)).validate()
    with pytest.raises(ValueError):
        w.lyapunov_lambda(F(4), w.Market())
    with pytest.raises(ValueError):
        w.localization(w.Market(), F(2))


def test_sigma_qp_all_nine_active_sets():
    m = w.Market()
    for state in w.STATES:
        target = tuple(F(0) if s == -1 else m.caps[i] if s == 1 else m.caps[i]/2
                       for i, s in enumerate(state))
        linear = tuple(2*sum(m.sigma[i][j]*target[j] for j in range(2))+F(state[i])
                       for i in range(2))
        _, actual = w.qp_exact(linear, m)
        assert actual == target
        numeric = w.qp_numeric(np.array([linear], dtype=float), np.ones(1), m)[0]
        np.testing.assert_allclose(numeric, np.array(target, float), atol=1e-14)


def test_fixed_trace_and_terminal_corner_compatibility():
    data = json.loads((w.OUT / "greedy_candidate.json").read_text())
    m = w.Market.from_json(data["market"])
    poly = w.candidate_polynomial(data)
    left, right = w.trace_slopes(m)
    for k in range(11):
        z = F(k, 10)
        assert evaluate(poly, F(0), z) == 1
        assert evaluate(poly, z, F(0)) == 1+z*left
        assert evaluate(poly, z, F(1)) == 1+z*right
    for y in (F(0), F(1)):
        z = m.radius*(2*y-1)
        mu = tuple(m.mu0[i]+m.mu1[i]*z for i in range(2))
        h, _ = w.qp_exact(mu, m)
        assert -evaluate(w.exact.derivative(poly, 0), F(0), y)/m.horizon-h == 0
    bad = dict(data, trace_slopes_exact=["0", "0"])
    with pytest.raises(ValueError):
        w.candidate_polynomial(bad)


@pytest.mark.parametrize("mode", ["greedy", "myopic"])
def test_branch_residuals_against_exact_qp(mode):
    m = w.Market()
    poly = {(0, 0): F(1), (1, 1): -F(1, 100), (1, 2): F(1, 100)}
    polys, branches = w.branch_polynomials(poly, m, mode)
    for s in (F(0), F(2, 5), F(1)):
        for k in range(17):
            y = F(k, 16)
            u = evaluate(poly, s, y)
            us = evaluate(w.exact.derivative(poly, 0), s, y)
            uz = evaluate(w.exact.derivative(poly, 1), s, y)/(2*m.radius)
            uzz = evaluate(w.exact.derivative(w.exact.derivative(poly, 1), 1), s, y)/(4*m.radius**2)
            z = m.radius*(2*y-1)
            mu = tuple(m.mu0[i]+m.mu1[i]*z for i in range(2))
            linear = tuple(mu[i]+(m.cross[i]*uz/u if mode == "greedy" else 0) for i in range(2))
            _, a = w.qp_exact(linear, m)
            expected = -us/m.horizon+m.state_variance*uzz/2
            expected += (-m.kappa*z-sum(a[i]*m.cross[i] for i in range(2)))*uz
            expected += (-sum(a[i]*mu[i] for i in range(2))+
                         sum(a[i]*m.sigma[i][j]*a[j] for i in range(2) for j in range(2)))*u
            feasible = [name for conditions, name, _ in branches
                        if all(evaluate(polys[c], s, y) >= 0 for c in conditions)]
            assert feasible
            for name in feasible:
                assert evaluate(polys[name], s, y)/u == expected


def test_quadratic_weight_bound_for_every_policy():
    m = w.Market()
    for a in (F(1, 50), F(1), F(98, 25)):
        lam = w.lyapunov_lambda(a, m)
        for k in range(-40, 41):
            z = F(k, 4)
            for i in range(4):
                for j in range(4):
                    pi = (m.caps[0]*F(i, 3), m.caps[1]*F(j, 3))
                    beta = -m.kappa*z-sum(m.cross[i]*pi[i] for i in range(2))
                    c = -sum(pi[i]*(m.mu0[i]+m.mu1[i]*z) for i in range(2))
                    c += sum(pi[i]*m.sigma[i][j]*pi[j] for i in range(2) for j in range(2))
                    generator = m.state_variance*(2*a+4*a*a*z*z)/2+beta*2*a*z+c
                    assert generator <= lam


def test_interval_localization_and_ce_against_high_precision():
    mp.dps = 90
    m = w.Market()
    loc = w.localization(m)
    convert = lambda s: mp.mpf(F(s).numerator)/F(s).denominator
    a0, a1 = convert(loc["a0_exact"]), convert(loc["a1_exact"])
    lam0, lam1 = convert(loc["lambda0_exact"]), convert(loc["lambda1_exact"])
    expected = (mp.exp(lam0/4)+1)*mp.exp(lam1/4-(a1-a0)*4)
    assert mp.mpf(loc["value_localization_upper"]) >= expected
    data = json.loads((w.OUT / "greedy_candidate.json").read_text())
    result = json.loads((w.OUT / "greedy_results.json").read_text())
    bounds = w.welfare_bounds(m, w.candidate_polynomial(data), result["certificate"], "greedy")
    eta = mp.mpf(result["certificate"]["residual_sup_upper"])
    phi = mp.expm1(mp.mpf(33)/400)/(mp.mpf(33)/100)
    d = phi*eta+mp.mpf(bounds["value_localization_upper"])
    value = convert(bounds["candidate_at_initial_state_exact"])
    assert mp.mpf(bounds["relative_ce_loss_upper"]) >= 1-(value-d)/(value+d)
    assert 0 < bounds["relative_ce_loss_bp_upper"] < 2


def test_small_budget_fails_closed_and_clears_summary():
    data = (w.OUT / "greedy_candidate.json").read_text()
    with tempfile.TemporaryDirectory(prefix="welfare-failure-") as temp:
        out = Path(temp)
        table = out / "table.tex"
        (out / "greedy_candidate.json").write_text(data)
        table.write_text("old successful result")
        (out / "welfare_results.json").write_text('{"certified":true}')
        with patch.object(w, "OUT", out), patch.object(w, "TABLE", table), contextlib.redirect_stdout(io.StringIO()):
            code = w.main(["--verify-only", "--max-cells", "1"])
        assert code == 2
        assert not json.loads((out / "welfare_results.json").read_text())["certified"]
        assert "old successful" not in table.read_text()


def test_supplied_full_domain_certificate_is_reproducible():
    data = json.loads((w.OUT / "greedy_candidate.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        result = w.evaluate_candidate(data, 0.00025, 100000)
    assert result["certificate"]["certified"]
    assert result["certificate"]["covered_area_exact"] == "1"
    assert result["certificate"]["switching_cells"] > 0
    assert result["bounds"]["relative_ce_loss_bp_upper"] < 2
