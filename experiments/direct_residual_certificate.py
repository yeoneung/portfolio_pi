"""Continuous-PDE certificate, independent of PI initialization or grid accuracy.

Fit a hard-trace polynomial by frozen-policy least squares, then freeze its
dyadic coefficients.  Exact rational polynomial algebra and outward-rounded
Bernstein subdivision certify positivity and the *Bellman* residual on the
entire cylinder, including cells crossing active-set interfaces.  Collocation
and dense samples are diagnostics only; neither is a certificate premise.

Model: gamma=2, r=0, sigma=1, sigma_Z=1/2, C=1/10, b(z)=-z,
mu(z)=1/5+z/2, A=[0,3/10], T=1/4, z in [-1,1].
With s=4*tau and x=(z+1)/2, the prescribed trace is
E(s,x)=1-3*s*x/100.  The candidate is E+s*x*(1-x)*P(s,x).
The certificate concerns the continuous Dirichlet problem, not the original
whole-space problem and not the high-dimensional neural runs.
"""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import time
import platform

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from mpmath import iv

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "direct_certificate"
TABLE = HERE.parent / "manuscript" / "tables" / "tab_direct_certificate.tex"


def add(a, b):
    c = a.copy()
    for ij, value in b.items():
        c[ij] = c.get(ij, F(0)) + value
    return {ij: v for ij, v in c.items() if v}


def scale(a, k):
    return {ij: k * v for ij, v in a.items() if k * v}


def mul(a, b):
    c = {}
    for (i, j), v in a.items():
        for (k, l), w in b.items():
            ij = (i + k, j + l)
            c[ij] = c.get(ij, F(0)) + v * w
    return {ij: v for ij, v in c.items() if v}


def derivative(a, axis):
    c = {}
    for ij, value in a.items():
        if ij[axis]:
            kl = list(ij)
            kl[axis] -= 1
            c[tuple(kl)] = value * ij[axis]
    return c


def shifted_chebyshev(n, axis):
    one = {(0, 0): F(1)}
    ij = (1, 0) if axis == 0 else (0, 1)
    y = {(0, 0): F(-1), ij: F(2)}
    seq = [one, y]
    for k in range(2, n + 1):
        seq.append(add(scale(mul(y, seq[-1]), F(2)), scale(seq[-2], F(-1))))
    return seq[:n + 1]


def candidate_polynomial(coefficients):
    ds, dx = coefficients.shape
    ts, tx = shifted_chebyshev(ds - 1, 0), shifted_chebyshev(dx - 1, 1)
    p = {}
    for i in range(ds):
        for j in range(dx):
            p = add(p, scale(mul(ts[i], tx[j]), coefficients[i, j]))
    e = {(0, 0): F(1), (1, 1): -F(3, 100)}
    return add(e, mul({(1, 1): F(1), (1, 2): F(-1)}, p))


def residual_polynomials(u):
    # R = A - max_{0<=a<=cap} (a*w-a*a*u).
    us, ux = derivative(u, 0), derivative(u, 1)
    uxx = derivative(ux, 1)
    z = {(0, 0): F(-1), (0, 1): F(2)}
    mu = {(0, 0): -F(3, 10), (0, 1): F(1)}
    aa = add(add(scale(us, F(-4)), scale(uxx, F(1, 32))),
             scale(mul(z, ux), -F(1, 2)))
    w = add(mul(mu, u), scale(ux, F(1, 20)))
    cap = F(3, 10)
    upper = add(add(aa, scale(w, -cap)), scale(u, cap * cap))
    middle = add(scale(mul(u, aa), F(4)), scale(mul(w, w), F(-1)))
    switch = add(w, scale(u, -2 * cap))
    return {"u": u, "w": w, "switch": switch, "lower": aa,
            "middle_numerator": middle, "upper": upper}


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def bernstein_enclosure(poly):
    """Exact power-to-Bernstein conversion on [0,1]^2, then enclose doubles.

    Python's Fraction -> float is correctly rounded; adjacent binary64
    numbers therefore enclose each exact coefficient. No BLAS rounding is
    used in the certificate path.
    """
    ns = max((i for i, j in poly), default=0)
    nx = max((j for i, j in poly), default=0)
    first = [[F(0) for _ in range(nx + 1)] for _ in range(ns + 1)]
    for (i, j), value in poly.items():
        for k in range(i, ns + 1):
            first[k][j] += value * F(math.comb(k, i), math.comb(ns, i))
    exact = [[sum((first[k][j] * F(math.comb(l, j), math.comb(nx, j))
                   for j in range(l + 1)), F(0))
              for l in range(nx + 1)] for k in range(ns + 1)]
    value = np.asarray([[float(v) for v in row] for row in exact])
    if not np.isfinite(value).all():
        raise ArithmeticError("Nonfinite Bernstein coefficient")
    lower, upper = down(value), up(value)
    # Check enclosure against the exact rationals, not merely a rounding assumption.
    for i, row in enumerate(exact):
        for j, v in enumerate(row):
            if not F(float(lower[i, j])) <= v <= F(float(upper[i, j])):
                raise ArithmeticError("Initial rational enclosure failed")
    return lower, upper


def split_bernstein(enclosure, axis):
    """Outward de Casteljau bisection; convex weights are exactly 1/2."""
    lo, hi = (np.moveaxis(v, axis, 0).copy() for v in enclosure)
    n = len(lo)
    ll, lh, rl, rh = (np.empty_like(lo) for _ in range(4))
    ll[0], lh[0], rl[-1], rh[-1] = lo[0], hi[0], lo[-1], hi[-1]
    for k in range(1, n):
        lo = down(down(lo[:-1] + lo[1:]) * 0.5)
        hi = up(up(hi[:-1] + hi[1:]) * 0.5)
        ll[k], lh[k], rl[n - 1 - k], rh[n - 1 - k] = lo[0], hi[0], lo[-1], hi[-1]
    return ((np.moveaxis(ll, 0, axis), np.moveaxis(lh, 0, axis)),
            (np.moveaxis(rl, 0, axis), np.moveaxis(rh, 0, axis)))


def interval_divide(a, b):
    if b[0] <= 0:
        raise ArithmeticError("Denominator is not certified positive")
    values = np.array([a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1]])
    return float(down(values.min())), float(up(values.max()))


def cell_bounds(cell):
    bounds = {name: (float(lo.min()), float(hi.max()))
              for name, (lo, hi) in cell.items()}
    lower_u, upper_u = bounds["u"]
    if lower_u <= 0:
        return lower_u, np.inf, ("unknown",)
    possible = []
    # Include every branch that could occur, also at switching surfaces.
    if bounds["w"][0] <= 0:
        possible.append(("lower", bounds["lower"]))
    if bounds["w"][1] >= 0 and bounds["switch"][0] <= 0:
        denominator = (float(down(4 * lower_u)), float(up(4 * upper_u)))
        possible.append(("interior", interval_divide(bounds["middle_numerator"], denominator)))
    if bounds["switch"][1] >= 0:
        possible.append(("upper", bounds["upper"]))
    residual = max(max(abs(lo), abs(hi)) for _, (lo, hi) in possible)
    return lower_u, residual, tuple(name for name, _ in possible)


def certify(polynomials, target, max_cells=100000, max_depth=30):
    root = {name: bernstein_enclosure(p) for name, p in polynomials.items()}
    queue = deque([(root, 0)])
    visited = accepted = crossing = 0
    minimum_u, maximum_residual = np.inf, 0.0
    deepest = 0
    covered_area = F(0)
    branch_cells = {"lower": 0, "interior": 0, "upper": 0}
    while queue:
        cell, depth = queue.pop()
        visited += 1
        lower_u, residual, branches = cell_bounds(cell)
        if lower_u > 0 and residual <= target:
            accepted += 1
            covered_area += F(1, 2 ** depth)
            crossing += int(len(branches) > 1)
            for branch in branches:
                branch_cells[branch] += 1
            minimum_u = min(minimum_u, lower_u)
            maximum_residual = max(maximum_residual, residual)
            deepest = max(deepest, depth)
            continue
        if visited >= max_cells or depth >= max_depth:
            return {"certified": False, "reason": "subdivision budget exhausted",
                    "visited_cells": visited, "remaining_cells": len(queue) + 1,
                    "failed_cell_residual_upper": residual if np.isfinite(residual) else None,
                    "failed_cell_u_lower": lower_u}
        children = [{}, {}]
        for name, enclosure in cell.items():
            halves = split_bernstein(enclosure, depth % 2)
            for k in (0, 1):
                children[k][name] = halves[k]
        queue.extend((child, depth + 1) for child in children)
    if covered_area != 1:
        raise ArithmeticError("Accepted cells do not cover the full square")
    return {"certified": True, "covered_area_exact": str(covered_area),
            "visited_cells": visited, "accepted_cells": accepted,
            "switching_cells": crossing, "branch_cells": branch_cells,
            "max_depth": deepest, "u_lower": minimum_u,
            "bellman_residual_sup_upper": maximum_residual}


def basis_values(degree, points, spatial=False):
    values = [[], [], []]
    prefactor = Polynomial([0, 1, -1]) if spatial else Polynomial([0, 1])
    for i in range(degree + 1):
        p = Chebyshev.basis(i).convert(kind=Polynomial)(Polynomial([-1, 2])) * prefactor
        for k in range(3):
            values[k].append(p.deriv(k)(points))
    return [np.asarray(v).T for v in values]


def design(ds, dx, s, x):
    vs, vx = basis_values(ds, s), basis_values(dx, x, spatial=True)
    def product(a, b):
        return (a[:, :, None] * b[:, None, :]).reshape(len(s), -1)
    u = product(vs[0], vx[0])
    us = product(vs[1], vx[0])
    ux = product(vs[0], vx[1])
    uxx = product(vs[0], vx[2])
    e = 1 - 0.03 * s * x
    es, ex = -0.03 * x, -0.03 * s
    return (u, us, ux, uxx), (e, es, ex, np.zeros_like(e))


def diagnostics(coefficients, ds, dx, points=201):
    ss, xx = np.meshgrid(np.linspace(0, 1, points), np.linspace(0, 1, points), indexing="ij")
    s, x = ss.ravel(), xx.ravel()
    bb, ee = design(ds, dx, s, x)
    u, us, ux, uxx = [e + b @ coefficients.ravel() for e, b in zip(ee, bb)]
    a = np.clip(((x - 0.3) + 0.05 * ux / u) / 2, 0, 0.3)
    residual = -4 * us + uxx / 32 - (2 * x - 1 + 0.1 * a) * ux / 2
    residual += (-a * (x - 0.3) + a * a) * u
    return {"sampled_residual_sup_NOT_certificate": float(np.max(np.abs(residual))),
            "sampled_u_min_NOT_certificate": float(u.min()),
            "sampled_active_lower_fraction": float(np.mean(a == 0)),
            "sampled_active_upper_fraction": float(np.mean(a == 0.3))}


def fit_candidate(ds, dx, iterations=8):
    # The choice of samples affects approximation quality only, never validity.
    s_axis = (1 - np.cos(np.linspace(0, np.pi, 2 * ds + 7))) / 2
    x_axis = (1 - np.cos(np.linspace(0, np.pi, 2 * dx + 9))) / 2
    ss, xx = np.meshgrid(s_axis, x_axis, indexing="ij")
    s, x = ss.ravel(), xx.ravel()
    bb, ee = design(ds, dx, s, x)
    c = np.zeros((ds + 1) * (dx + 1))
    history = []
    for iteration in range(iterations):
        u, us, ux, uxx = [e + b @ c for e, b in zip(ee, bb)]
        if u.min() <= 0:
            raise ValueError("Nonpositive training candidate")
        a = np.clip(((x - 0.3) + 0.05 * ux / u) / 2, 0, 0.3)
        drift, reaction = -(2 * x - 1 + 0.1 * a) / 2, -a * (x - 0.3) + a * a
        matrix = -4 * bb[1] + bb[3] / 32 + drift[:, None] * bb[2] + reaction[:, None] * bb[0]
        base = -4 * ee[1] + drift * ee[2] + reaction * ee[0]
        c = np.linalg.lstsq(matrix, -base, rcond=1e-12)[0]
        history.append({"iteration": iteration + 1,
                        "sampled_frozen_residual_sup": float(np.max(np.abs(matrix @ c + base)))})
    # Certificate applies to these exact dyadic coefficients, not the unrounded fit.
    denominator = 2 ** 42
    numerators = np.rint(c * denominator).astype(np.int64).reshape(ds + 1, dx + 1)
    exact = np.asarray([[F(int(k), denominator) for k in row] for row in numerators], dtype=object)
    return exact, numerators, denominator, history


def explicit_bounds(residual_upper):
    # c=-a*mu+a^2 <= 9/50 on the specified rectangle. No PDE estimate constant.
    # mpmath interval exp supplies an outward bound also for the barrier factor.
    iv.dps = 40
    c = iv.mpf(9) / 50
    phi = (iv.exp(c / 4) - 1) / c
    eta = iv.mpf(float(residual_upper))
    value = phi * eta
    regret = 2 * value
    def upper(v):
        return float(np.nextafter(float(v.b), np.inf))
    return {"c_upper_exact": "9/50", "horizon_exact": "1/4",
            "trace_mismatch_exact": "0", "barrier_factor_upper": upper(phi),
            "value_factor_sup_error_upper": upper(value),
            "greedy_value_factor_suboptimality_upper": upper(regret),
            "utility_regret_at_unit_wealth_upper": upper(regret),
            "scope": "continuous bounded-domain Dirichlet problem only"}


def upper_sci(value, digits=4):
    """Round a positive certified bound upward also in the printed table."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive bound")
    with localcontext() as ctx:
        ctx.prec = 60
        number = Decimal.from_float(value)
        exponent = number.adjusted() - digits + 1
        rounded = number.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_CEILING)
        mantissa, exp = f"{rounded:.{digits - 1}E}".split("E")
        return mantissa + r"\times10^{" + str(int(exp)) + "}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-time", type=int, default=6)
    parser.add_argument("--degree-space", type=int, default=14)
    parser.add_argument("--target-residual", type=float, default=0.0002)
    parser.add_argument("--max-cells", type=int, default=100000)
    parser.add_argument("--verify-only", action="store_true", help="Verify supplied exact coefficients without refitting")
    args = parser.parse_args(argv)
    if (min(args.degree_time, args.degree_space) < 0 or
            not math.isfinite(args.target_residual) or args.target_residual <= 0 or args.max_cells < 1):
        parser.error("Degrees must be nonnegative and target residual positive")
    OUT.mkdir(parents=True, exist_ok=True)
    # Invalidate previous success before any fallible fitting or verification.
    TABLE.write_text("% Certification not completed.\n\\textit{No certificate returned.}\n")
    (OUT / "certificate_results.json").write_text(
        json.dumps({"certificate": {"certified": False, "reason": "run not completed"}}, indent=2) + "\n")
    start = time.perf_counter()
    candidate_path = OUT / "candidate.json"
    if args.verify_only:
        candidate = json.loads(candidate_path.read_text())
        ds, dx = candidate["degree_time"], candidate["degree_space"]
        denominator = int(candidate["coefficient_denominator"])
        exact = np.asarray([[F(int(k), denominator) for k in row]
                            for row in candidate["coefficient_numerators"]], dtype=object)
        history = candidate["training_history"]
    else:
        ds, dx = args.degree_time, args.degree_space
        exact, numerators, denominator, history = fit_candidate(ds, dx)
        candidate = {"degree_time": ds, "degree_space": dx,
                     "coefficient_denominator": str(denominator),
                     "coefficient_numerators": [[str(int(k)) for k in row] for row in numerators],
                     "training_history": history,
                     "basis": "1-3*s*x/100+s*x*(1-x)*sum c_ij T_i(2s-1)T_j(2x-1)"}
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    print("Frozen candidate:", diagnostics(np.asarray(exact, dtype=float), ds, dx), flush=True)
    polynomials = residual_polynomials(candidate_polynomial(exact))
    print("Certifying full cylinder with Bernstein subdivision...", flush=True)
    certificate = certify(polynomials, args.target_residual, args.max_cells)
    result = {"candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "target_residual": args.target_residual, "certificate": certificate,
              "diagnostics": diagnostics(np.asarray(exact, dtype=float), ds, dx),
              "training_history": history,
              "rounding": "exact Fraction algebra; outward binary64 Bernstein arithmetic; mpmath iv barrier"}
    if certificate["certified"]:
        result["bounds"] = explicit_bounds(certificate["bellman_residual_sup_upper"])
    result["seconds"] = time.perf_counter() - start
    (OUT / "certificate_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if certificate["certified"]:
        b = result["bounds"]
        table = ("% Generated by experiments/direct_residual_certificate.py; do not hand-edit.\n"
                 "\\begin{tabular}{lr}\\toprule\nQuantity & Verified upper bound\\\\\\midrule\n"
                 f"$\\|\\mathfrak Bu\\|_\\infty$ & ${upper_sci(certificate['bellman_residual_sup_upper'])}$\\\\\n"
                 f"$\\|u-g_R\\|_\\infty$ & ${upper_sci(b['value_factor_sup_error_upper'])}$\\\\\n"
                 f"$\\|g_R^{{G[u]}}-g_R\\|_\\infty$ & ${upper_sci(b['greedy_value_factor_suboptimality_upper'])}$\\\\\n"
                 "\\bottomrule\\end{tabular}\n")
    else:
        # Overwrite any older successful table so a failed rerun cannot leave a stale certificate.
        table = "% Certification failed: no numerical bound may be claimed.\n\\textit{No certificate returned.}\n"
    TABLE.write_text(table)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if certificate["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
