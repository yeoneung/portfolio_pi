"""Exact-weight neural audit of a necessary scalar-barrier certificate condition.

This verifies point enclosures and a full-domain residual LOWER bound. It does
not turn point samples into an upper bound or certify neural policy welfare.
The network is the real function represented by the exported binary weights;
GPU evaluation roundoff is not included in that mathematical-policy definition.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from mpmath import iv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/neural_certificate"


def scalar(x):
    if isinstance(x, F):
        return iv.mpf(x.numerator)/x.denominator
    if isinstance(x, float):
        return scalar(F(x))
    return iv.mpf(x)


def endpoints(x):
    return [float(np.nextafter(float(x.a), -np.inf)),
            float(np.nextafter(float(x.b), np.inf))]


def jet(x, dt=0, dz=0, dzz=0):
    return tuple(scalar(a) for a in (x, dt, dz, dzz))


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def scale(a, b):
    b = scalar(b)
    return tuple(x*b for x in a)


def multiply(a, b):
    v, t, z, zz = a
    w, s, y, yy = b
    return v*w, t*w+v*s, z*w+v*y, zz*w+2*z*y+v*yy


def tanh(a):
    v, t, z, zz = a
    e = iv.exp(2*v)
    f = (e-1)/(e+1)
    df = 1-f*f
    return f, df*t, df*z, df*zz-2*f*df*z*z


def log_value_jet(checkpoint, tau, z):
    """Value and derivatives (tau,z,zz) in physical coordinates, all enclosed."""
    state = checkpoint["state_dict"]
    horizon = F(checkpoint["horizon"])
    center = F(state["center"][0][0])
    half = F(state["half_ax"][0][0])
    if half <= 0:
        raise ValueError("Invalid neural normalization")
    tj, zj = jet(tau, dt=1), jet(z, dz=1)
    normalized = scale(add(zj, jet(-center)), 1/half)
    distance = add(jet(1), scale(multiply(normalized, normalized), -1))
    features = [scale(tj, 1/horizon), normalized, distance]
    depth = checkpoint["config"]["depth"]
    for layer in range(depth+1):
        weights = state[f"net.{2*layer}.weight"]
        biases = state[f"net.{2*layer}.bias"]
        if len(weights) != len(biases) or any(len(w) != len(features) for w in weights):
            raise ValueError("Neural layer shape mismatch")
        output = []
        for row, bias in zip(weights, biases):
            value = jet(bias)
            for weight, feature in zip(row, features):
                if weight:
                    value = add(value, scale(feature, weight))
            output.append(tanh(value) if layer < depth else value)
        features = output
    if len(features) != 1:
        raise ValueError("Expected a scalar network output")
    q = F(1)-F(checkpoint["gamma"])
    return add(scale(tj, q*F(checkpoint["risk_free_rate"])),
               multiply(multiply(tj, distance), features[0]))


def clipped(signal, cap):
    cap = scalar(cap)
    return iv.mpf([max((-cap).a, min(cap.a, signal.a)),
                   max((-cap).b, min(cap.b, signal.b))])


def evaluate(checkpoint, tau, z):
    if checkpoint["gamma"] != 2 or len(checkpoint["market"]["xbar"]) != 1:
        raise ValueError("This audit is for the supplied gamma=2 scalar-state benchmark")
    h, ht, hz, hzz = log_value_jet(checkpoint, tau, z)
    m = checkpoint["market"]
    drift = scalar(m["k0"][0])-scalar(m["K"][0][0])*scalar(z)
    normalized = ht-drift*hz-scalar(m["Q"][0][0])/2*(hzz+hz*hz)
    normalized += scalar(checkpoint["risk_free_rate"])
    policies = []
    for i, cap in enumerate(checkpoint["caps"]):
        lam = scalar(m["lam0"][i])+scalar(m["Lam"][i][0])*scalar(z)
        signal = lam+scalar(m["Gamma"][i][0])*hz
        policy = clipped(signal/2, cap)
        normalized += policy*signal-policy*policy
        policies.append(endpoints(policy))
    value = iv.exp(h)
    return {"tau_exact": str(tau), "z_exact": str(z),
            "neural_value_interval": endpoints(value),
            "normalized_greedy_residual_interval": endpoints(normalized),
            "greedy_residual_interval": endpoints(value*normalized),
            "log_gradient_interval": endpoints(hz), "policy_intervals": policies}


def ideal_corner_floor(checkpoint, z):
    """Exact necessary residual at a compatible geometric terminal corner.

    Any C^{1,2} candidate with terminal value 1 and lateral exp(-r*tau)
    has this residual for the final neural policy, whose terminal log-gradient
    is zero. This does not assume the float32 buffers equal the domain inputs.
    """
    m = checkpoint["market"]
    ans = F(0)
    for i, cap in enumerate(checkpoint["caps"]):
        cap = F(cap)
        lam = F(m["lam0"][i])+F(m["Lam"][i][0])*z
        policy = min(cap, max(-cap, lam/2))
        ans += policy*lam-policy**2
    return ans


def scalar_barrier(checkpoint):
    m = checkpoint["market"]
    center = F(m["xbar"][0])
    half = F(checkpoint["domain_scale"])*F(m["eta"][0])
    r = F(checkpoint["risk_free_rate"])
    candidates = []
    for z in (center-half, center+half):
        # For the exposure box, the reaction is convex and separable in actions
        # and affine in the state. Its maximum is attained at these vertices.
        candidates.append(-r+sum((F(cap)**2+F(cap)*abs(F(m["lam0"][i])+F(m["Lam"][i][0])*z)
                                  for i, cap in enumerate(checkpoint["caps"])), F(0)))
    cplus = max(F(0), *candidates)
    horizon = scalar(F(checkpoint["horizon"]))
    phi = (iv.exp(scalar(cplus)*horizon)-1)/scalar(cplus) if cplus else horizon
    return cplus, phi


def trace_mismatch(checkpoint):
    """Allow the saved float32 normalization to differ from the model domain."""
    state, m = checkpoint["state_dict"], checkpoint["market"]
    center, half = F(m["xbar"][0]), F(checkpoint["domain_scale"])*F(m["eta"][0])
    nc, nh = F(state["center"][0][0]), F(state["half_ax"][0][0])
    depth = checkpoint["config"]["depth"]
    raw_bound = abs(F(state[f"net.{2*depth}.bias"][0]))+sum(
        (abs(F(w)) for w in state[f"net.{2*depth}.weight"][0]), F(0))
    distance_bound = max(abs(1-((z-nc)/nh)**2) for z in (center-half, center+half))
    log_error = F(checkpoint["horizon"])*distance_bound*raw_bound
    return {"terminal_trace_exact": True, "lateral_log_trace_error_upper": endpoints(scalar(log_error))[1],
            "lateral_trace_error_upper": endpoints(iv.exp(scalar(log_error))-1)[1],
            "normalization_buffers_unchanged": True}


def audit(checkpoint, precision=50):
    if precision < 30:
        raise ValueError("At least 30 decimal digits required")
    iv.dps = precision
    m = checkpoint["market"]
    n = len(checkpoint["caps"])
    if (checkpoint["gamma"] != 2 or checkpoint["risk_free_rate"] < 0
            or checkpoint["horizon"] <= 0 or checkpoint["domain_scale"] <= 0
            or len(m["xbar"]) != 1 or any(cap < 0 for cap in checkpoint["caps"])
            or any(len(m[k]) != n for k in ("lam0", "Lam", "Gamma"))
            or F(m["Q"][0][0]) <= sum((F(a[0])**2 for a in m["Gamma"]), F(0))):
        raise ValueError("Invalid or unsupported scalar financial model")
    center = F(m["xbar"][0])
    half = F(checkpoint["domain_scale"])*F(m["eta"][0])
    points = {}
    for name, tau, z in (("terminal_left", F(0), center-half),
                         ("terminal_right", F(0), center+half),
                         ("initial_center", F(checkpoint["horizon"]), center)):
        print("Enclosing actual neural network:", name, flush=True)
        points[name] = evaluate(checkpoint, tau, z)
    # Continuity up to the closure implies that boundary point values are
    # lower bounds on the essential supremum over the open cylinder as well.
    residual_floor = max(max(0., item["greedy_residual_interval"][0],
                             -item["greedy_residual_interval"][1]) for item in points.values())
    cplus, phi = scalar_barrier(checkpoint)
    budget_floor = endpoints(phi*scalar(residual_floor))[0]
    initial_upper = points["initial_center"]["neural_value_interval"][1]
    exact_floors = {name: ideal_corner_floor(checkpoint, z) for name, z in (
        ("left", center-half), ("right", center+half))}
    return {"audit_certified": True, "welfare_certified": False,
            "status": "scalar_barrier_inconclusive" if budget_floor >= initial_upper else "full_domain_upper_enclosure_still_required",
            "policy_updates_after_checkpoint": 0, "precision_decimal_digits": precision,
            "full_domain_residual_lower_bound": residual_floor,
            "full_domain_residual_upper_bound": None,
            "cplus_exact": str(cplus), "phi_interval": endpoints(phi),
            "minimum_required_scalar_error_budget": budget_floor,
            "initial_neural_value_upper": initial_upper,
            "scalar_positive_endpoint_condition_ruled_out": budget_floor >= initial_upper,
            "ideal_trace_corner_residuals_exact": {k: str(v) for k, v in exact_floors.items()},
            "ideal_trace_corner_residual_intervals": {k: endpoints(scalar(v)) for k, v in exact_floors.items()},
            "trace_audit": trace_mismatch(checkpoint),
            "points": points,
            "scope": "necessary-condition audit of the fixed neural candidate and its exact greedy policy; no CE upper certificate",
            "interpretation": "the lower bound is on the residual and the scalar certificate budget, not on actual value error or investment loss"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUT/"seed_11_final.json")
    parser.add_argument("--output", type=Path, default=OUT/"neural_audit.json")
    parser.add_argument("--precision", type=int, default=50)
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("Output must not overwrite the checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('{"audit_certified": false, "welfare_certified": false}\n')
    payload = args.checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checkpoint = json.loads(payload)
    if checkpoint["format"] != "hard_value_net_exact_binary_weights_v1":
        raise ValueError("Unsupported checkpoint")
    start = time.perf_counter()
    result = audit(checkpoint, args.precision)
    result.update({"checkpoint_sha256": digest, "audit_source_sha256": source_hash,
                   "seconds": time.perf_counter()-start})
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != digest or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash:
        raise ArithmeticError("Audit inputs changed during execution")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "points"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
