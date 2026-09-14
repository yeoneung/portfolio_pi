"""Evaluate a hash-fixed greedy policy in an enriched polynomial space.

Numerical fitting only proposes candidates. Verified residual and Hamiltonian
gap bounds are evaluated separately; sampled diagnostics are never certificates.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import portfolio_welfare as w
import bernstein_arithmetic as ba

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "policy_loss"
POLICY_PATH = ROOT / "results/portfolio_welfare/greedy_candidate.json"


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "experiments/policy_loss_certificate.py", "experiments/bernstein_arithmetic.py",
        "experiments/portfolio_welfare.py", "experiments/direct_residual_certificate.py")}


def coefficient_array(data):
    den = int(data["coefficient_denominator"])
    return np.array([[int(k)/den for k in row] for row in data["coefficient_numerators"]])


def frozen_policy(data, s, y):
    market = w.Market.from_json(data["market"])
    value, _, dz, _ = w.values(coefficient_array(data), s, y, market)
    if np.min(value) <= 0:
        raise ArithmeticError("Nonpositive policy-generating candidate")
    z = float(market.radius)*(2*y-1)
    mu = np.asarray(market.mu0, float)+z[:, None]*np.asarray(market.mu1, float)
    signal = mu*value[:, None]+dz[:, None]*np.asarray(market.cross, float)
    return w.qp_numeric(signal, value, market)


def fit_evaluation(policy, ds, dx, reweights=0):
    market = w.Market.from_json(policy["market"])
    sa = (1-np.cos(np.linspace(0, np.pi, 2*ds+9)))/2
    ya = (1-np.cos(np.linspace(0, np.pi, 3*dx+15)))/2
    ss, yy = np.meshgrid(sa, ya, indexing="ij")
    s, y = ss.ravel(), yy.ravel()
    z = float(market.radius)*(2*y-1)
    a = frozen_policy(policy, s, y)
    mu = np.asarray(market.mu0, float)+z[:, None]*np.asarray(market.mu1, float)
    sigma = np.asarray(market.sigma, float)
    drift = -float(market.kappa)*z-a@np.asarray(market.cross, float)
    reaction = -np.sum(a*mu, axis=1)+np.einsum("ni,ij,nj->n", a, sigma, a)
    b0, bs, by, byy = w.design(ds, dx, s, y)
    bz, bzz = by/(2*float(market.radius)), byy/(4*float(market.radius)**2)
    matrix = -bs/float(market.horizon)+float(market.state_variance)/2*bzz
    matrix += drift[:, None]*bz+reaction[:, None]*b0
    e, es, ez, ezz = w.baseline(s, y, market)
    base = -es/float(market.horizon)+float(market.state_variance)/2*ezz+drift*ez+reaction*e
    boundary_weight = np.where((y == 0) | (y == 1), 4., 1.)
    weights = boundary_weight.copy()
    history = []
    for iteration in range(reweights+1):
        coefficients = np.linalg.lstsq(matrix*weights[:, None], -base*weights, rcond=1e-12)[0]
        residual = matrix@coefficients+base
        history.append(float(np.max(np.abs(residual))))
        if iteration < reweights:
            target = np.maximum(np.abs(residual), 0.05*np.max(np.abs(residual)))
            weights = boundary_weight*np.sqrt(weights*target/np.mean(target))
            weights /= np.max(weights)
    den = 2**42
    numerators = np.rint(coefficients*den).astype(np.int64).reshape(ds+1, dx+1)
    return {"market": market.json(), "mode": "fixed_greedy_evaluation",
            "degree_time": ds, "degree_space": dx, "coefficient_denominator": str(den),
            "coefficient_numerators": [[str(int(n)) for n in row] for row in numerators],
            "trace_slopes_exact": [str(a) for a in w.trace_slopes(market)],
            "basis": policy["basis"], "sampled_fit_residual_history_NOT_certificate": history,
            "reweight_iterations": reweights, "policy_updates": 0}


def diagnostics(policy, evaluation, ns=65, ny=2001):
    market = w.Market.from_json(policy["market"])
    sa = (1-np.cos(np.linspace(0, np.pi, ns)))/2
    ss, yy = np.meshgrid(sa, np.linspace(0, 1, ny), indexing="ij")
    s, y = ss.ravel(), yy.ravel()
    eta = gap = quadratic = 0.
    minimum = np.inf
    for start in range(0, len(s), 1500):
        st, yt = s[start:start+1500], y[start:start+1500]
        a = frozen_policy(policy, st, yt)
        v, vs, vz, vzz = w.values(coefficient_array(evaluation), st, yt, market)
        u, _, uz, _ = w.values(coefficient_array(policy), st, yt, market)
        z = float(market.radius)*(2*yt-1)
        mu = np.asarray(market.mu0, float)+z[:, None]*np.asarray(market.mu1, float)
        sigma, cross = np.asarray(market.sigma, float), np.asarray(market.cross, float)
        linear = mu*v[:, None]+vz[:, None]*cross
        greedy = w.qp_numeric(linear, v, market)
        h = lambda pi: np.sum(pi*linear, axis=1)-v*np.einsum("ni,ij,nj->n", pi, sigma, pi)
        residual = -vs/float(market.horizon)+float(market.state_variance)/2*vzz
        residual += -float(market.kappa)*z*vz-h(a)
        eta = max(eta, float(np.max(np.abs(residual))))
        gap = max(gap, float(np.max(h(greedy)-h(a))))
        quadratic = max(quadratic, float(np.max(v/4*(cross@np.linalg.solve(sigma, cross))*(vz/v-uz/u)**2)))
        minimum = min(minimum, float(np.min(v)))
    return {"sampled_frozen_residual_sup_NOT_certificate": eta,
            "sampled_gap_sup_NOT_certificate": gap,
            "sampled_quadratic_gap_sup_NOT_certificate": quadratic,
            "sampled_positive_min_NOT_certificate": minimum, "grid": [ns, ny]}


def branch_roots(u, v, market, rectangle):
    """Residual numerator u^2 L^{G[u]}v, with all KKT branches retained."""
    sl, sh, yl, yh = rectangle
    uz = ba.scale(ba.derivative(u, 1), 1/(2*market.radius*(yh-yl)))
    vz = ba.scale(ba.derivative(v, 1), 1/(2*market.radius*(yh-yl)))
    vzz = ba.scale(ba.derivative(vz, 1), 1/(2*market.radius*(yh-yl)))
    z = w.exact.bernstein_enclosure({(0, 0): market.radius*(2*yl-1), (0, 1): 2*market.radius*(yh-yl)})
    mu = [ba.add(ba.constant(market.mu0[i]), ba.scale(z, market.mu1[i])) for i in range(2)]
    wu = [ba.add(ba.multiply(mu[i], u), ba.scale(uz, market.cross[i])) for i in range(2)]
    wv = [ba.add(ba.multiply(mu[i], v), ba.scale(vz, market.cross[i])) for i in range(2)]
    av = ba.sum_all([ba.scale(ba.derivative(v, 0), -1/(market.horizon*(sh-sl))),
                     ba.scale(vzz, market.state_variance/2), ba.scale(ba.multiply(z, vz), -market.kappa)])
    u2 = ba.multiply(u, u)
    base = ba.multiply(u2, av)
    root = {"u": u, "v": v,
            "log_gradient_numerator": ba.add(ba.multiply(u, vz), ba.scale(ba.multiply(v, uz), -1))}
    branches = []
    for k, state in enumerate(w.STATES):
        piu = [ba.constant(0), ba.constant(0)]
        fixed = [i for i in range(2) if state[i]]
        free = [i for i in range(2) if not state[i]]
        for i in fixed:
            piu[i] = ba.scale(u, market.caps[i] if state[i] == 1 else F(0))
        rhs = [ba.scale(a, F(1, 2)) for a in wu]
        if len(free) == 2:
            det = 1-market.sigma12**2
            piu = [ba.scale(ba.add(rhs[i], ba.scale(rhs[1-i], -market.sigma12)), 1/det) for i in range(2)]
        elif free:
            i = free[0]
            piu[i] = ba.add(rhs[i], ba.scale(piu[fixed[0]], -market.sigma12))
        gradient = [ba.add(wu[i], ba.scale(ba.sum_all(ba.scale(piu[j], market.sigma[i][j]) for j in range(2)), -2)) for i in range(2)]
        conditions = []
        for i in range(2):
            if not state[i]:
                conditions.extend([piu[i], ba.add(ba.scale(u, market.caps[i]), ba.scale(piu[i], -1))])
            else:
                conditions.append(ba.scale(gradient[i], F(state[i])))
        names = []
        for j, polynomial in enumerate(conditions):
            name = f"b{k}_c{j}"
            root[name] = polynomial
            names.append(name)
        linear = ba.sum_all(ba.multiply(piu[i], wv[i]) for i in range(2))
        quadratic = ba.sum_all(ba.scale(ba.multiply(piu[i], piu[j]), market.sigma[i][j]) for i in range(2) for j in range(2))
        residual = ba.add(ba.add(base, ba.scale(ba.multiply(u, linear), -1)), ba.multiply(v, quadratic))
        name = f"b{k}_residual"
        root[name] = residual
        branches.append((names, name, state))
    return root, branches


def polynomial_roots(policy, evaluation, prelevels=4):
    market = w.Market.from_json(policy["market"])
    if evaluation["market"] != policy["market"] or evaluation["mode"] != "fixed_greedy_evaluation":
        raise ValueError("Evaluation model or policy mode mismatch")
    print("Constructing exact candidate polynomials", flush=True)
    up, vp = w.candidate_polynomial(policy), w.candidate_polynomial(evaluation)
    print("Exact Bernstein conversion and coarse restriction", flush=True)
    todo = [(ba.rational_bernstein(up), ba.rational_bernstein(vp), (F(0), F(1), F(0), F(1)))]
    for level in range(prelevels):
        nxt = []
        axis = level % 2
        for u, v, rect in todo:
            us, vs = ba.rational_split(u, axis), ba.rational_split(v, axis)
            k = 2*axis
            midpoint = (rect[k]+rect[k+1])/2
            for j in (0, 1):
                bounds = list(rect)
                bounds[k+1-j] = midpoint
                nxt.append((us[j], vs[j], tuple(bounds)))
        todo = nxt
    roots = []
    for i, (u, v, rect) in enumerate(todo):
        root, branches = branch_roots(ba.enclose_rationals(u), ba.enclose_rationals(v), market, rect)
        roots.append((root, prelevels))
        print("Enclosed coarse patch", i+1, "of", len(todo), flush=True)
    return roots, branches, vp


def certify(roots, branches, market, eta_target, gap_target, max_cells=200000, max_depth=40):
    todo = list(roots)
    accepted = switching = visited = 0
    area = F(0)
    eta_max = gap_max = 0.
    min_u = min_v = np.inf
    kappa = (market.cross[0]**2-2*market.sigma12*market.cross[0]*market.cross[1]+market.cross[1]**2)/(1-market.sigma12**2)/4
    kappa_upper = ba.rational(kappa)[1]
    while todo:
        if visited >= max_cells:
            return {"certified": False, "reason": "subdivision budget exhausted", "visited_cells": visited}
        cell, depth = todo.pop()
        visited += 1
        ranges = {name: (float(lo.min()), float(hi.max())) for name, (lo, hi) in cell.items()}
        ul, uh = ranges["u"]
        vl, vh = ranges["v"]
        possible = [(name, state) for conditions, name, state in branches
                    if all(ranges[c][1] >= 0 for c in conditions)]
        if not possible:
            raise ArithmeticError("KKT branch coverage lost")
        eta = gap = np.inf
        if min(ul, vl) > 0:
            denom = (float(w.exact.down(ul*ul)), float(w.exact.up(uh*uh)))
            eta = max(max(abs(v) for v in w.exact.interval_divide(ranges[name], denom)) for name, _ in possible)
            e = max(abs(v) for v in ranges["log_gradient_numerator"])
            numerator = float(w.exact.up(kappa_upper*w.exact.up(e*e)))
            denominator = float(w.exact.down(denom[0]*vl))
            if denominator > 0:
                gap = float(w.exact.up(numerator/denominator))
        if eta <= eta_target and gap <= gap_target:
            accepted += 1
            switching += len(possible) > 1
            eta_max, gap_max = max(eta_max, eta), max(gap_max, gap)
            min_u, min_v = min(min_u, ul), min(min_v, vl)
            area += F(1, 2**depth)
        elif visited >= max_cells or depth >= max_depth:
            return {"certified": False, "reason": "subdivision budget or arithmetic resolution exhausted",
                    "visited_cells": visited, "failed_depth": depth,
                    "residual_upper": eta if np.isfinite(eta) else None,
                    "gap_upper": gap if np.isfinite(gap) else None}
        else:
            children = [{}, {}]
            for name, enclosure in cell.items():
                parts = w.exact.split_bernstein(enclosure, depth % 2)
                for j in (0, 1):
                    children[j][name] = parts[j]
            todo.extend((c, depth+1) for c in children)
        if visited % 5000 == 0:
            print("Verified cells visited:", visited, "accepted:", accepted, flush=True)
    if area != 1:
        raise ArithmeticError("Incomplete domain cover")
    return {"certified": True, "covered_area_exact": str(area), "visited_cells": visited,
            "accepted_cells": accepted, "switching_cells": switching,
            "frozen_residual_upper": eta_max, "hamiltonian_gap_upper": gap_max,
            "u_lower": min_u, "v_lower": min_v}


def uniform_inner_bounds(market, certificate, inner_radius=F(1, 2)):
    if not 0 <= inner_radius < market.radius:
        raise ValueError("Inner radius must be nonnegative and strictly inside D_R")
    w.iv.dps = 45
    cplus = w.coefficient_bounds(market)[3]
    c, t = w.mpiv(cplus), w.mpiv(market.horizon)
    phi = (w.iv.exp(c*t)-1)/c if cplus else t
    loc = w.localization(market, inner_radius)
    ell = w.mpiv(loc["value_localization_upper"])
    eta, gap = w.mpiv(certificate["frozen_residual_upper"]), w.mpiv(certificate["hamiltonian_gap_upper"])
    error = phi*eta+ell
    factor_lower = w.lower(w.mpiv(certificate["v_lower"])-error)
    if factor_lower <= 0:
        raise ArithmeticError("No positive factor lower bound over the inner cylinder")
    loss = phi*(2*eta+gap)+2*ell
    return {"scope": "all 0<=t<=T and |z|<=inner_radius, same whole-space deployed policy",
            "inner_radius_exact": str(inner_radius), "localization_upper": loc["value_localization_upper"],
            "policy_factor_lower": factor_lower,
            "relative_ce_loss_bp_upper": w.upper(10000*loss/w.mpiv(factor_lower))}


def verified_bounds(market, poly, certificate):
    w.iv.dps = 45
    cplus = w.coefficient_bounds(market)[3]
    c, t = w.mpiv(cplus), w.mpiv(market.horizon)
    phi = (w.iv.exp(c*t)-1)/c if cplus else t
    eta, gap = w.mpiv(certificate["frozen_residual_upper"]), w.mpiv(certificate["hamiltonian_gap_upper"])
    loc = w.localization(market)
    ell = w.mpiv(loc["value_localization_upper"])
    point_exact = w.exact_point(poly)
    point = w.mpiv(point_exact)
    evaluation_error = phi*eta+ell
    loss = phi*(2*eta+gap)+2*ell
    policy_lo, policy_hi = w.lower(point-evaluation_error), w.upper(point+evaluation_error)
    if policy_lo <= 0:
        raise ArithmeticError("Positive full-space factor bound missing")
    # q=-1: relative CE loss = (g^pi-g*)/g^pi.
    ce_loss = loss/w.mpiv(policy_lo)
    return {"scope": "same fixed whole-space greedy policy at t=0, z=0; gamma=2",
            "evaluation_initial_exact": str(point_exact), "phi_upper": w.upper(phi), **loc,
            "evaluation_factor_error_upper": w.upper(evaluation_error),
            "whole_space_policy_factor_interval": [policy_lo, policy_hi],
            "whole_space_policy_ce_interval": [w.lower(1/w.mpiv(policy_hi)), w.upper(1/w.mpiv(policy_lo))],
            "unit_wealth_utility_loss_upper": w.upper(loss),
            "relative_ce_loss_bp_upper": w.upper(10000*ce_loss)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--degree-time", type=int, default=8)
    p.add_argument("--degree-space", type=int, default=48)
    p.add_argument("--reweights", type=int, default=0)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--target-residual", type=float, default=0.00005)
    p.add_argument("--target-gap", type=float, default=1e-8)
    p.add_argument("--max-cells", type=int, default=200000)
    args = p.parse_args()
    if (min(args.degree_time, args.degree_space, args.reweights) < 0 or args.max_cells < 1
            or not all(math.isfinite(a) and a > 0 for a in (args.target_residual, args.target_gap))):
        p.error("Invalid degree, tolerance, or verification budget")
    OUT.mkdir(exist_ok=True)
    digest = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
    policy = json.loads(POLICY_PATH.read_text())
    name = f"evaluation_{args.degree_time}_{args.degree_space}_{args.reweights}"
    if args.verify_only:
        initial_sources = source_hashes()
        result_path = OUT / (name+"_certificate.json")
        result_path.write_text('{"certified": false, "reason": "verification incomplete"}\n')
        evaluation = json.loads((OUT / (name+".json")).read_text())
        if evaluation["policy_sha256"] != digest:
            raise ValueError("Frozen policy hash mismatch")
        start = time.perf_counter()
        roots, branches, poly = polynomial_roots(policy, evaluation)
        construction_seconds = time.perf_counter()-start
        print("Bernstein construction seconds:", construction_seconds, flush=True)
        market = w.Market.from_json(policy["market"])
        certificate = certify(roots, branches, market, args.target_residual, args.target_gap, args.max_cells)
        output = {"certified": certificate["certified"], "certificate": certificate,
                  "policy_sha256": digest, "evaluation_sha256": hashlib.sha256((OUT/(name+".json")).read_bytes()).hexdigest(),
                  "construction_seconds": construction_seconds, "verification_seconds": time.perf_counter()-start,
                  "source_sha256": source_hashes(), "target_residual": args.target_residual,
                  "target_gap": args.target_gap, "policy_updates": 0}
        if certificate["certified"]:
            output["bounds"] = verified_bounds(market, poly, certificate)
            output["uniform_inner_bounds"] = uniform_inner_bounds(market, certificate)
        if hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != digest:
            raise ArithmeticError("Frozen policy changed during verification")
        if source_hashes() != initial_sources:
            raise ArithmeticError("Verification sources changed during the run")
        result_path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
        print(json.dumps(output, indent=2), flush=True)
        return 0 if certificate["certified"] else 2
    start = time.perf_counter()
    evaluation = fit_evaluation(policy, args.degree_time, args.degree_space, args.reweights)
    fitting_seconds = time.perf_counter()-start
    evaluation["policy_sha256"] = digest
    (OUT / (name+".json")).write_text(json.dumps(evaluation, indent=2)+"\n")
    print("Fitted", name, "in", fitting_seconds, "seconds", flush=True)
    result = {"certified": False, "candidate": name+".json", "policy_sha256": digest,
              "fitting_seconds": fitting_seconds, **diagnostics(policy, evaluation)}
    if hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != digest:
        raise ArithmeticError("Frozen policy changed")
    (OUT / (name+"_diagnostics.json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
