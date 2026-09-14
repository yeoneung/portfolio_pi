"""C1 piecewise-polynomial evaluation of a fixed covariance-QP policy.

Fitting is numerical only. Shared nodal values and slopes preserve exact C1
gluing after dyadic rounding; terminal and lateral traces are imposed exactly.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import portfolio_welfare as w
import policy_loss_certificate as p
import bernstein_arithmetic as ba

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/local_policy"


def mesh(level, boundary_levels=0):
    if not 1 <= level <= 8:
        raise ValueError("Spatial level must be between 1 and 8")
    if not 0 <= boundary_levels <= 8:
        raise ValueError("Invalid boundary refinement level")
    knots = {F(i, 2**level) for i in range(2**level+1)}
    for j in range(1, boundary_levels+1):
        point = F(1, 2**(level+j))
        knots.update((point, 1-point))
    return sorted(knots)


def spatial_basis(knots, cell, degree):
    """Global coefficient index and exact polynomial in the cell coordinate."""
    if degree < 4:
        raise ValueError("Spatial degree must be at least four")
    n = len(knots)-1
    h = knots[cell+1]-knots[cell]
    ans = []
    if cell > 0:
        ans.append((cell-1, {(0, 0): F(1), (0, 2): F(-3), (0, 3): F(2)}))
    if cell+1 < n:
        ans.append((cell, {(0, 2): F(3), (0, 3): F(-2)}))
    ans.extend([
        (n-1+cell, w.exact.scale({(0, 1): F(1), (0, 2): F(-2), (0, 3): F(1)}, h)),
        (n+cell, w.exact.scale({(0, 2): F(-1), (0, 3): F(1)}, h)),
    ])
    bubble = {(0, 2): F(1), (0, 3): F(-2), (0, 4): F(1)}
    for k, cheb in enumerate(w.exact.shifted_chebyshev(degree-4, 1)):
        ans.append((2*n+cell*(degree-3)+k, w.exact.mul(bubble, cheb)))
    return ans


def time_basis(degree, s, power=1):
    values, derivatives = [], []
    for k in range(degree+1):
        cheb = np.polynomial.Chebyshev.basis(k)
        value = cheb(2*s-1)
        values.append(s**power*value)
        derivatives.append(value+2*s*cheb.deriv()(2*s-1)/power)
    return np.array(values).T, np.array(derivatives).T


def spatial_values(basis, x):
    answer = []
    for order in range(3):
        values = []
        for _, poly in basis:
            for k in range(order):
                poly = w.exact.derivative(poly, 1)
            coefficients = np.zeros(max((j for i, j in poly), default=0)+1)
            for (i, j), a in poly.items():
                coefficients[j] = float(a)
            values.append(np.polynomial.polynomial.polyval(x, coefficients))
        answer.append(np.array(values).T)
    return answer


def design_cell(policy, knots, cell, ds, degree, s, x, power=1):
    m = w.Market.from_json(policy["market"])
    y = float(knots[cell])+float(knots[cell+1]-knots[cell])*x
    z = float(m.radius)*(2*y-1)
    pi = p.frozen_policy(policy, s**power, y)
    mu = np.asarray(m.mu0, float)+z[:, None]*np.asarray(m.mu1, float)
    drift = -float(m.kappa)*z-pi@np.asarray(m.cross, float)
    reaction = -np.sum(pi*mu, axis=1)+np.einsum("ni,ij,nj->n", pi, np.asarray(m.sigma, float), pi)
    basis = spatial_basis(knots, cell, degree)
    bx, by, byy = spatial_values(basis, x)
    length = 2*float(m.radius)*(float(knots[cell+1]-knots[cell]))
    by, byy = by/length, byy/length**2
    bs, bt = time_basis(ds, s, power)
    matrix = -bt[:, :, None]*bx[:, None, :]/float(m.horizon)
    matrix += bs[:, :, None]*(float(m.state_variance)/2*byy+drift[:, None]*by+reaction[:, None]*bx)[:, None, :]
    e, es, ez, ezz = w.baseline(s**power, y, m)
    base = -es/float(m.horizon)+float(m.state_variance)/2*ezz+drift*ez+reaction*e
    return matrix.reshape(len(s), -1), base, [index for index, poly in basis]


def fit(policy, ds, degree, level, reweights, power=1, boundary_levels=0):
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import splu
    knots = mesh(level, boundary_levels)
    n = len(knots)-1
    nb = (degree-1)*n
    sa = (1-np.cos(np.linspace(0, np.pi, 2*ds+9)))/2
    xa = (1-np.cos(np.linspace(0, np.pi, 2*degree+9)))/2
    ss, xx = np.meshgrid(sa, xa, indexing="ij")
    s, x = ss.ravel(), xx.ravel()
    rows, columns, data, offsets = [], [], [], []
    for cell in range(n):
        matrix, base, indices = design_cell(policy, knots, cell, ds, degree, s, x, power)
        cols = (np.arange(ds+1)[:, None]*nb+np.array(indices)[None, :]).ravel()
        rows.append(np.repeat(np.arange(len(s))+cell*len(s), len(cols)))
        columns.append(np.tile(cols, len(s)))
        data.append(matrix.ravel())
        offsets.append(base)
    a = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                   shape=(n*len(s), (ds+1)*nb)).tocsr()
    b = np.concatenate(offsets)
    weights = np.ones(len(b))
    history = []
    for iteration in range(reweights+1):
        weighted = a.multiply(weights[:, None]).tocsr()
        norms = np.sqrt(np.asarray(weighted.power(2).sum(axis=0)).ravel())
        if not np.isfinite(norms).all() or np.min(norms) <= 0:
            raise ArithmeticError("Singular or nonfinite fitting columns")
        scaled = weighted@diags(1/norms)
        normal = (scaled.T@scaled).tocsc()
        factor = splu(normal)
        rhs = -b*weights
        solution = factor.solve(np.asarray(scaled.T@rhs).ravel())
        for _ in range(3):
            solution += factor.solve(np.asarray(scaled.T@(rhs-scaled@solution)).ravel())
        coefficients = solution/norms
        residual = a@coefficients+b
        maximum = float(np.max(abs(residual)))
        history.append(maximum)
        print("Fit pass", iteration, "sampled residual", maximum, flush=True)
        if iteration < reweights:
            target = np.maximum(abs(residual), .05*maximum)
            weights = np.sqrt(weights*target/np.mean(target))
            weights /= max(weights)
    denominator = 2**48
    nums = np.rint(coefficients*denominator).astype(np.int64).reshape(ds+1, nb)
    return {"market": policy["market"], "mode": "C1_fixed_greedy_evaluation",
            "knots_exact": [str(k) for k in knots], "degree_time": ds, "degree_space": degree,
            "level": level, "boundary_levels": boundary_levels, "time_power": power, "reweight_iterations": reweights, "policy_updates": 0,
            "coefficient_denominator": str(denominator),
            "coefficient_numerators": [[str(int(a)) for a in row] for row in nums],
            "trace_slopes_exact": policy["trace_slopes_exact"],
            "sampled_fit_history_NOT_certificate": history}


def read_candidate(candidate):
    m = w.Market.from_json(candidate["market"])
    knots = list(map(F, candidate["knots_exact"]))
    if knots[0] != 0 or knots[-1] != 1 or any(a >= b for a, b in zip(knots, knots[1:])):
        raise ValueError("Invalid complete spatial partition")
    ds, degree = candidate["degree_time"], candidate["degree_space"]
    if ds < 0 or degree < 4 or candidate["time_power"] not in (1, 2, 3, 4) or candidate["mode"] != "C1_fixed_greedy_evaluation":
        raise ValueError("Invalid local evaluation space")
    if candidate["trace_slopes_exact"] != [str(a) for a in w.trace_slopes(m)]:
        raise ValueError("Prescribed trace mismatch")
    den = int(candidate["coefficient_denominator"])
    if den <= 0:
        raise ValueError("Invalid coefficient denominator")
    coefficients = np.array([[F(int(a), den) for a in row] for row in candidate["coefficient_numerators"]], dtype=object)
    if coefficients.shape != (ds+1, (degree-1)*(len(knots)-1)):
        raise ValueError("Coefficient shape mismatch")
    return m, knots, coefficients


def local_polynomials(candidate):
    m, knots, coefficients = read_candidate(candidate)
    power = candidate["time_power"]
    chebs = [w.exact.mul({(power, 0): F(1)}, t) for t in w.exact.shifted_chebyshev(candidate["degree_time"], 0)]
    left, right = w.trace_slopes(m)
    polys = []
    for cell in range(len(knots)-1):
        h = knots[cell+1]-knots[cell]
        poly = {(0, 0): F(1), (power, 0): left+(right-left)*knots[cell], (power, 1): (right-left)*h}
        for index, basis in spatial_basis(knots, cell, candidate["degree_space"]):
            for i, tc in enumerate(chebs):
                poly = w.exact.add(poly, w.exact.scale(w.exact.mul(tc, basis), coefficients[i, index]))
        polys.append(poly)
    return polys


def diagnostics(policy, candidate):
    m, knots, exact_coefficients = read_candidate(candidate)
    coefficients = np.asarray(exact_coefficients, float)
    sa = (1-np.cos(np.linspace(0, np.pi, 81)))/2
    xa = np.linspace(0, 1, 101)
    ss, xx = np.meshgrid(sa, xa, indexing="ij")
    s, x = ss.ravel(), xx.ravel()
    result, peak = 0., None
    for cell in range(len(knots)-1):
        matrix, base, indices = design_cell(policy, knots, cell, candidate["degree_time"], candidate["degree_space"], s, x, candidate["time_power"])
        residual = matrix@coefficients[:, indices].ravel()+base
        k = int(np.argmax(abs(residual)))
        if abs(residual[k]) > result:
            result = float(abs(residual[k]))
            peak = {"cell": cell, "s": float(s[k]), "x": float(x[k])}
    return {"sampled_residual_sup_NOT_certificate": result, "peak": peak, "grid_per_cell": [81, 101]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-time", type=int, default=16)
    parser.add_argument("--degree-space", type=int, default=8)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--reweights", type=int, default=4)
    parser.add_argument("--time-power", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--boundary-levels", type=int, default=3)
    args = parser.parse_args()
    if args.degree_time < 0 or args.degree_space < 4 or args.reweights < 0:
        parser.error("Invalid approximation parameters")
    OUT.mkdir(exist_ok=True)
    digest = hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest()
    policy = json.loads(p.POLICY_PATH.read_text())
    stem = f"evaluation_{args.degree_time}_{args.degree_space}_{args.level}_{args.reweights}_p{args.time_power}_b{args.boundary_levels}"
    (OUT/(stem+"_certificate.json")).write_text('{"certified": false, "reason": "candidate fitting in progress"}\n')
    start = time.perf_counter()
    candidate = fit(policy, args.degree_time, args.degree_space, args.level, args.reweights, args.time_power, args.boundary_levels)
    candidate["policy_sha256"] = digest
    seconds = time.perf_counter()-start
    (OUT/(stem+".json")).write_text(json.dumps(candidate, indent=2)+"\n")
    result = {"certified": False, "policy_sha256": digest, "fitting_seconds": seconds, **diagnostics(policy, candidate)}
    if hashlib.sha256(p.POLICY_PATH.read_bytes()).hexdigest() != digest:
        raise ArithmeticError("Frozen policy changed")
    (OUT/(stem+"_diagnostics.json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
