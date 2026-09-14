"""Full-domain certification of a C1 local evaluation; the policy stays fixed."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import local_policy_evaluation as local
import portfolio_welfare as w
import policy_loss_certificate as p
import bernstein_arithmetic as ba
import finite_horizon_localization as finite

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        "experiments/local_policy_evaluation.py", "experiments/verify_local_policy.py",
        "experiments/finite_horizon_localization.py", "experiments/policy_loss_certificate.py",
        "experiments/portfolio_welfare.py", "experiments/bernstein_arithmetic.py",
        "experiments/direct_residual_certificate.py")}


def affine(poly, axis, offset, width):
    """Exact substitution x=offset+width*r into a sparse bivariate polynomial."""
    answer = {}
    for ij, value in poly.items():
        for j in range(ij[axis]+1):
            key = list(ij)
            key[axis] = j
            key = tuple(key)
            answer[key] = answer.get(key, F(0))+value*F(math.comb(ij[axis], j))*offset**(ij[axis]-j)*width**j
    return {key: a for key, a in answer.items() if a}


def edge(poly, x):
    result = {}
    for (i, j), a in poly.items():
        result[i] = result.get(i, F(0))+a*x**j
    return {i: a for i, a in result.items() if a}


def check_traces(candidate, polys):
    market, knots, _ = local.read_candidate(candidate)
    power = candidate["time_power"]
    for poly in polys:
        terminal = {j: a for (i, j), a in poly.items() if i == 0 and a}
        if terminal != {0: F(1)} or any(0 < i < power for (i, j) in poly):
            raise ValueError("Terminal trace or graded derivative compatibility lost")
    slopes = w.trace_slopes(market)
    for index, x, slope in ((0, F(0), slopes[0]), (-1, F(1), slopes[1])):
        expected = {0: F(1), power: slope}
        if edge(polys[index], x) != {i: a for i, a in expected.items() if a}:
            raise ValueError("Lateral trace mismatch")
    for j in range(len(polys)-1):
        if edge(polys[j], F(1)) != edge(polys[j+1], F(0)):
            raise ValueError("Discontinuous local evaluation")
        left = w.exact.scale(w.exact.derivative(polys[j], 1), 1/(knots[j+1]-knots[j]))
        right = w.exact.scale(w.exact.derivative(polys[j+1], 1), 1/(knots[j+2]-knots[j+1]))
        if edge(left, F(1)) != edge(right, F(0)):
            raise ValueError("Spatial C1 gluing failed")
    return {"terminal_trace_exact": True, "lateral_trace_exact": True,
            "spatial_C1_exact": True, "interfaces_checked": len(polys)-1,
            "graded_time_derivative_bounded": True}


def physical_time_derivative(poly, power, horizon):
    result = {}
    for (i, j), a in poly.items():
        if i:
            if i < power:
                raise ValueError("Singular physical-time derivative")
            result[i-power, j] = -a*i/(power*horizon)
    return result


def root_branches(u, v, vt, market, rect):
    sl, sh, yl, yh = rect
    dz = 1/(2*market.radius*(yh-yl))
    uz, vz = ba.scale(ba.derivative(u, 1), dz), ba.scale(ba.derivative(v, 1), dz)
    vzz = ba.scale(ba.derivative(vz, 1), dz)
    z = w.exact.bernstein_enclosure({(0, 0): market.radius*(2*yl-1), (0, 1): 2*market.radius*(yh-yl)})
    mu = [ba.add(ba.constant(market.mu0[i]), ba.scale(z, market.mu1[i])) for i in range(2)]
    wu = [ba.add(ba.multiply(mu[i], u), ba.scale(uz, market.cross[i])) for i in range(2)]
    wv = [ba.add(ba.multiply(mu[i], v), ba.scale(vz, market.cross[i])) for i in range(2)]
    av = ba.sum_all([vt, ba.scale(vzz, market.state_variance/2), ba.scale(ba.multiply(z, vz), -market.kappa)])
    base = ba.multiply(ba.multiply(u, u), av)
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
        # Exclude only a branch whose necessary KKT inequality is impossible
        # on the entire coarse patch; every remaining branch is retained.
        if any(float(hi.max()) < 0 for lo, hi in conditions):
            continue
        names = []
        for j, condition in enumerate(conditions):
            name = f"b{k}_c{j}"
            root[name] = condition
            names.append(name)
        linear = ba.sum_all(ba.multiply(piu[i], wv[i]) for i in range(2))
        quadratic = ba.sum_all(ba.scale(ba.multiply(piu[i], piu[j]), market.sigma[i][j]) for i in range(2) for j in range(2))
        numerator = ba.add(ba.add(base, ba.scale(ba.multiply(u, linear), -1)), ba.multiply(v, quadratic))
        name = f"b{k}_residual"
        root[name] = numerator
        branches.append((names, name, state))
    if not branches:
        raise ArithmeticError("Lost all possible KKT branches")
    return root, branches


def roots(policy, candidate, polys, time_levels=2):
    market, knots, _ = local.read_candidate(candidate)
    up = w.candidate_polynomial(policy)
    power = candidate["time_power"]
    up = {(i*power, j): a for (i, j), a in up.items()}
    for cell, vp in enumerate(polys):
        yl, yh = knots[cell], knots[cell+1]
        pu = affine(up, 1, yl, yh-yl)
        pt = physical_time_derivative(vp, power, market.horizon)
        todo = [(tuple(ba.rational_bernstein(poly) for poly in (pu, vp, pt)), F(0), F(1))]
        for _ in range(time_levels):
            nxt = []
            for arrays, sl, sh in todo:
                halves = [ba.rational_split(a, 0) for a in arrays]
                mid = (sl+sh)/2
                nxt.extend([(tuple(a[0] for a in halves), sl, mid), (tuple(a[1] for a in halves), mid, sh)])
            todo = nxt
        for arrays, sl, sh in todo:
            root, branches = root_branches(*(ba.enclose_rationals(a) for a in arrays), market, (sl, sh, yl, yh))
            yield root, branches, (sl, sh, yl, yh), 0
        if (cell+1) % 8 == 0 or cell+1 == len(polys):
            print("Constructed spatial patches", cell+1, "of", len(polys), flush=True)


def certify(root_iterator, market, target, gap_target=1e-8, max_cells=200000, max_depth=40):
    count = accepted = switching = 0
    area = F(0)
    eta_max = gap_max = 0.
    min_u = min_v = np.inf
    k = (market.cross[0]**2-2*market.sigma12*market.cross[0]*market.cross[1]+market.cross[1]**2)/(1-market.sigma12**2)/4
    ku = ba.rational(k)[1]
    for root in root_iterator:
        todo = [root]
        while todo:
            if count >= max_cells:
                return {"certified": False, "reason": "subdivision budget exhausted", "visited_cells": count}
            cell, branches, rect, depth = todo.pop()
            count += 1
            ranges = {name: (float(lo.min()), float(hi.max())) for name, (lo, hi) in cell.items()}
            ul, uh = ranges["u"]
            vl, vh = ranges["v"]
            possible = [name for conditions, name, state in branches if all(ranges[c][1] >= 0 for c in conditions)]
            if not possible:
                raise ArithmeticError("KKT cover lost")
            eta = gap = np.inf
            if min(ul, vl) > 0:
                den = (float(w.exact.down(ul*ul)), float(w.exact.up(uh*uh)))
                eta = max(max(abs(a) for a in w.exact.interval_divide(ranges[name], den)) for name in possible)
                e = max(abs(a) for a in ranges["log_gradient_numerator"])
                numerator = float(w.exact.up(ku*w.exact.up(e*e)))
                denominator = float(w.exact.down(den[0]*vl))
                if denominator > 0:
                    gap = float(w.exact.up(numerator/denominator))
            if eta <= target and gap <= gap_target:
                accepted += 1
                switching += len(possible) > 1
                eta_max, gap_max = max(eta_max, eta), max(gap_max, gap)
                min_u, min_v = min(min_u, ul), min(min_v, vl)
                area += (rect[1]-rect[0])*(rect[3]-rect[2])
            elif depth >= max_depth:
                return {"certified": False, "reason": "depth or arithmetic resolution exhausted",
                        "visited_cells": count, "failed_rectangle": [str(x) for x in rect],
                        "residual_upper": eta if np.isfinite(eta) else None,
                        "gap_upper": gap if np.isfinite(gap) else None}
            else:
                axis = depth % 2
                children = [{}, {}]
                for name, enclosure in cell.items():
                    parts = w.exact.split_bernstein(enclosure, axis)
                    for j in (0, 1):
                        children[j][name] = parts[j]
                midpoint = (rect[2*axis]+rect[2*axis+1])/2
                for j in (0, 1):
                    bounds = list(rect)
                    bounds[2*axis+1-j] = midpoint
                    todo.append((children[j], branches, tuple(bounds), depth+1))
    if area != 1:
        raise ArithmeticError("Incomplete exact cover")
    return {"certified": True, "covered_area_exact": str(area), "visited_cells": count,
            "accepted_cells": accepted, "switching_cells": switching,
            "frozen_residual_upper": eta_max, "hamiltonian_gap_upper": gap_max,
            "u_lower": min_u, "v_lower": min_v}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="evaluation_24_8_6_4_p2_b3.json")
    parser.add_argument("--target-residual", type=float, default=1e-6)
    parser.add_argument("--max-cells", type=int, default=200000)
    parser.add_argument("--time-levels", type=int, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.target_residual) or args.target_residual <= 0 or args.max_cells < 1 or not 0 <= args.time_levels <= 5:
        parser.error("Invalid verification parameters")
    path = (local.OUT/args.candidate).resolve()
    if path.parent != local.OUT.resolve() or path.suffix != ".json":
        parser.error("Candidate must be a JSON file in results/local_policy")
    result_path = path.with_name(path.stem+"_certificate.json")
    result_path.write_text('{"certified": false, "reason": "verification incomplete"}\n')
    digest = hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest()
    payload = path.read_bytes()
    evaluation_hash = hashlib.sha256(payload).hexdigest()
    initial_sources = source_hashes()
    candidate = json.loads(payload)
    policy = json.loads(p.POLICY_PATH.read_text())
    if candidate["policy_sha256"] != digest or candidate["market"] != policy["market"]:
        raise ValueError("Frozen policy hash or model mismatch")
    start = time.perf_counter()
    polys = local.local_polynomials(candidate)
    traces = check_traces(candidate, polys)
    print("Exact trace and C1 checks passed", flush=True)
    market, knots, _ = local.read_candidate(candidate)
    certificate = certify(roots(policy, candidate, polys, args.time_levels), market, args.target_residual, max_cells=args.max_cells)
    output = {"certified": certificate["certified"], "certificate": certificate, "traces": traces,
              "policy_sha256": digest, "evaluation_sha256": evaluation_hash, "source_sha256": source_hashes(),
              "policy_updates": 0, "verification_seconds": time.perf_counter()-start,
              "target_residual": args.target_residual, "time_levels": args.time_levels,
              "space_intervals": len(polys), "time_power": candidate["time_power"]}
    if certificate["certified"]:
        cell = next(j for j in range(len(polys)) if knots[j] <= F(1, 2) <= knots[j+1])
        point = w.exact_point(polys[cell], F(1), (F(1, 2)-knots[cell])/(knots[cell+1]-knots[cell]))
        output["bounds"] = finite.bounds(market, point, certificate)
        output["uniform_inner_bounds"] = finite.bounds(market, point, certificate, F(1, 2), certificate["v_lower"])
    if (hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest() != digest
            or hashlib.sha256(path.read_bytes()).hexdigest() != evaluation_hash or source_hashes() != initial_sources):
        raise ArithmeticError("Verification input changed during the run")
    result_path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps(output, indent=2), flush=True)
    return 0 if certificate["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
