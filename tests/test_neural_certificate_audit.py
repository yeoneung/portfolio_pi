"""Independent differentiation and fail-closed tests for the neural audit."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys

from mpmath import mp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
import neural_certificate_audit as audit


def small_checkpoint():
    return {"format": "hard_value_net_exact_binary_weights_v1", "gamma": 2,
            "risk_free_rate": .03, "horizon": 3., "domain_scale": 2.,
            "caps": [.1, .2], "config": {"depth": 1},
            "market": {"xbar": [.25], "eta": [.5], "k0": [.5], "K": [[2.]],
                       "Q": [[.25]], "Gamma": [[.1], [-.2]], "lam0": [.1, .2], "Lam": [[.25], [.125]]},
            "state_dict": {"center": [[.25]], "half_ax": [[1.]],
                           "net.0.weight": [[.5, -.25, .125], [-.125, .25, .375]],
                           "net.0.bias": [.0625, -.125],
                           "net.2.weight": [[.25, -.125]], "net.2.bias": [-.0625]}}


def real(x):
    x = F(x)
    return mp.mpf(x.numerator)/x.denominator


def reference_value(checkpoint, tau, z):
    """Independent scalar network; mp.diff supplies all derivatives."""
    state = checkpoint["state_dict"]
    y = (z-real(state["center"][0][0]))/real(state["half_ax"][0][0])
    features = [tau/real(checkpoint["horizon"]), y, 1-y*y]
    depth = checkpoint["config"]["depth"]
    for j in range(depth+1):
        features = [sum((real(w)*x for w, x in zip(row, features)), real(bias))
                    for row, bias in zip(state[f"net.{2*j}.weight"], state[f"net.{2*j}.bias"])]
        if j < depth:
            features = list(map(mp.tanh, features))
    return mp.exp(-real(checkpoint["risk_free_rate"])*tau+tau*(1-y*y)*features[0])


def reference_residual(c, tau, z):
    f = lambda t, x: reference_value(c, t, x)
    value = f(tau, z)
    gt, gz, gzz = (mp.diff(f, (tau, z), orders) for orders in ((1, 0), (0, 1), (0, 2)))
    m = c["market"]
    residual = gt-(real(m["k0"][0])-real(m["K"][0][0])*z)*gz-real(m["Q"][0][0])*gzz/2
    residual += real(c["risk_free_rate"])*value
    for i, cap in enumerate(c["caps"]):
        lam = real(m["lam0"][i])+real(m["Lam"][i][0])*z
        sig = lam+real(m["Gamma"][i][0])*gz/value
        pi = min(real(cap), max(-real(cap), sig/2))
        residual += (pi*lam-pi*pi)*value+pi*real(m["Gamma"][i][0])*gz
    return value, residual


@pytest.mark.parametrize("tau,z", [(F(0), F(5, 4)), (F(3), F(1, 4)), (F(1, 7), F(-2, 3))])
def test_neural_interval_against_independent_mpdiff(tau, z):
    mp.dps = 90
    audit.iv.dps = 50
    c = small_checkpoint()
    result = audit.evaluate(c, tau, z)
    v, residual = reference_residual(c, real(tau), real(z))
    for exact, key in ((v, "neural_value_interval"), (residual, "greedy_residual_interval")):
        lo, hi = result[key]
        assert real(lo) <= exact <= real(hi)


def test_actual_checkpoint_interval_against_independent_mpdiff():
    mp.dps = 90
    audit.iv.dps = 50
    c = json.loads((ROOT/"results/neural_certificate/seed_11_final.json").read_text())
    tau, z = F(3), F(c["market"]["xbar"][0])
    result = audit.evaluate(c, tau, z)
    v, residual = reference_residual(c, real(tau), real(z))
    for exact, key in ((v, "neural_value_interval"), (residual, "greedy_residual_interval")):
        lo, hi = result[key]
        assert real(lo) <= exact <= real(hi)


def test_corner_identity_and_barrier_against_exact_algebra():
    mp.dps = 90
    audit.iv.dps = 50
    c = small_checkpoint()
    z = F(5, 4)
    expected = sum((min(F(cap), max(-F(cap), (F(c["market"]["lam0"][i])+F(c["market"]["Lam"][i][0])*z)/2))
                    *(F(c["market"]["lam0"][i])+F(c["market"]["Lam"][i][0])*z)
                    -min(F(cap), max(-F(cap), (F(c["market"]["lam0"][i])+F(c["market"]["Lam"][i][0])*z)/2))**2
                    for i, cap in enumerate(c["caps"])), F(0))
    assert audit.ideal_corner_floor(c, z) == expected
    result = audit.evaluate(c, F(0), z)["greedy_residual_interval"]
    assert F(result[0]) <= expected <= F(result[1])
    cplus, phi = audit.scalar_barrier(c)
    expected_phi = mp.expm1(real(cplus)*3)/real(cplus)
    lo, hi = audit.endpoints(phi)
    assert real(lo) <= expected_phi <= real(hi)


def test_trace_error_accounts_for_unmodified_float_buffers():
    audit.iv.dps = 50
    c = small_checkpoint()
    assert audit.trace_mismatch(c)["lateral_trace_error_upper"] < 1e-45
    c["state_dict"]["center"][0][0] += 1e-6
    bound = audit.trace_mismatch(c)["lateral_trace_error_upper"]
    for z in (F(-3, 4), F(5, 4)):
        v = audit.evaluate(c, F(3), z)["neural_value_interval"]
        trace = audit.endpoints(audit.iv.exp(-3*audit.scalar(c["risk_free_rate"])))
        assert max(abs(v[0]-trace[1]), abs(v[1]-trace[0])) <= bound


def test_rejects_invalid_model():
    c = small_checkpoint()
    c["market"]["Q"] = [[.001]]
    with pytest.raises(ValueError, match="model"):
        audit.audit(c)


def test_cli_invalidates_old_success_on_bad_input(tmp_path, monkeypatch):
    checkpoint, output = tmp_path/"input.json", tmp_path/"audit.json"
    checkpoint.write_text('{"format":"wrong"}')
    output.write_text('{"welfare_certified":true}')
    monkeypatch.setattr(sys, "argv", ["audit", "--checkpoint", str(checkpoint), "--output", str(output)])
    with pytest.raises(ValueError, match="Unsupported"):
        audit.main()
    assert not json.loads(output.read_text())["welfare_certified"]
