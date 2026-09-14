"""Verified two-asset constrained portfolio welfare on the whole state space.

The fixed Dirichlet trace matches the terminal-corner myopic Hamiltonian.
Fits are numerical proposals only; every
claimed bound is recomputed from exact dyadic coefficients, all Sigma-QP
active sets, outward Bernstein enclosures, and interval scalar barriers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from mpmath import iv

import direct_residual_certificate as exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "portfolio_welfare"
TABLE = ROOT / "manuscript" / "tables" / "tab_portfolio_welfare.tex"


@dataclass(frozen=True)
class Market:
    radius: F = F(2)
    horizon: F = F(1, 4)
    kappa: F = F(1)
    state_variance: F = F(1, 4)
    sigma12: F = F(1, 4)
    mu0: tuple = (F(1, 5), F(3, 20))
    mu1: tuple = (F(1, 2), -F(1, 4))
    cross: tuple = (F(1, 5), F(1, 10))
    caps: tuple = (F(3, 10), F(1, 4))

    @property
    def sigma(self):
        return ((F(1), self.sigma12), (self.sigma12, F(1)))

    def validate(self):
        if min(self.radius, self.horizon, self.kappa, self.state_variance, *self.caps) <= 0:
            raise ValueError("Positive geometry, variance, and caps required")
        det = 1 - self.sigma12 ** 2
        if det <= 0:
            raise ValueError("Asset covariance must be positive definite")
        c0, c1 = self.cross
        schur = self.state_variance - (c0*c0 - 2*self.sigma12*c0*c1 + c1*c1) / det
        if schur < 0:
            raise ValueError("Joint asset-state covariance is not positive semidefinite")

    def json(self):
        return {"gamma": "2", "risk_free_rate": "0", "radius": str(self.radius),
                "horizon": str(self.horizon), "kappa": str(self.kappa),
                "state_variance": str(self.state_variance), "sigma12": str(self.sigma12),
                **{name: [str(v) for v in getattr(self, name)] for name in ("mu0", "mu1", "cross", "caps")}}

    @classmethod
    def from_json(cls, data):
        if data.get("gamma") != "2" or data.get("risk_free_rate") != "0":
            raise ValueError("This verifier implements gamma=2, r=0 only")
        keys = ("radius", "horizon", "kappa", "state_variance", "sigma12")
        m = cls(**{k: F(data[k]) for k in keys},
                **{k: tuple(F(v) for v in data[k]) for k in ("mu0", "mu1", "cross", "caps")})
        m.validate()
        return m


STATES = tuple(product((-1, 0, 1), repeat=2))  # lower, free, upper


def qp_exact(w, m):
    best = None
    for state in STATES:
        a = [m.caps[i] if state[i] == 1 else F(0) for i in range(2)]
        free = [i for i in range(2) if state[i] == 0]
        if len(free) == 2:
            det = 1-m.sigma12**2
            a = [(w[0]-m.sigma12*w[1])/(2*det), (w[1]-m.sigma12*w[0])/(2*det)]
        elif free:
            i = free[0]
            a[i] = w[i]/2-m.sigma12*a[1-i]
        if all(0 <= a[i] <= m.caps[i] for i in range(2)):
            score = sum(a[i]*w[i] for i in range(2))-sum(a[i]*m.sigma[i][j]*a[j] for i in range(2) for j in range(2))
            if best is None or score > best[0]:
                best = score, tuple(a)
    if best is None:
        raise ArithmeticError("Exact QP has no feasible branch")
    return best


def trace_slopes(m):
    slopes = tuple(-m.horizon*qp_exact(tuple(m.mu0[i]+m.mu1[i]*z for i in range(2)), m)[0]
                   for z in (-m.radius, m.radius))
    if min(slopes) <= -1 or max(slopes) > 0:
        raise ValueError("Corner-compatible affine trace must be positive and at most one")
    return slopes


def baseline(s, y, m):
    left, right = map(float, trace_slopes(m))
    slope = left+(right-left)*y
    return (1+s*slope, slope, s*(right-left)/(2*float(m.radius)), np.zeros_like(s))


def qp_numeric(w, u, m):
    """Exact active-set enumeration of the covariance-metric box QP, in float."""
    sigma = np.asarray(m.sigma, dtype=float)
    caps = np.asarray(m.caps, dtype=float)
    answer = np.full_like(w, np.nan)
    best = np.full(len(u), -np.inf)
    for state in STATES:
        fixed = [j for j in range(2) if state[j]]
        free = [j for j in range(2) if not state[j]]
        a = np.zeros_like(w)
        for j in fixed:
            a[:, j] = caps[j] if state[j] == 1 else 0
        if free:
            rhs = w[:, free] / (2*u[:, None])
            if fixed:
                rhs -= a[:, fixed] @ sigma[np.ix_(fixed, free)]
            a[:, free] = np.linalg.solve(sigma[np.ix_(free, free)], rhs.T).T
        feasible = np.all((a >= -1e-12) & (a <= caps+1e-12), axis=1)
        score = np.sum(a*w, axis=1) - u*np.einsum("ni,ij,nj->n", a, sigma, a)
        take = feasible & (score > best)
        answer[take] = np.clip(a[take], 0, caps)
        best[take] = score[take]
    if not np.isfinite(answer).all():
        raise ArithmeticError("No feasible QP branch")
    return answer


def design(ds, dx, s, y):
    def basis(degree, x, spatial):
        p = x*(1-x) if spatial else x
        dp = 1-2*x if spatial else np.ones_like(x)
        ddp = -2*np.ones_like(x) if spatial else np.zeros_like(x)
        output = [[], [], []]
        for k in range(degree+1):
            cheb = np.polynomial.Chebyshev.basis(k)
            v, dv, ddv = cheb(2*x-1), 2*cheb.deriv()(2*x-1), 4*cheb.deriv(2)(2*x-1)
            output[0].append(p*v)
            output[1].append(dp*v+p*dv)
            output[2].append(ddp*v+2*dp*dv+p*ddv)
        return [np.asarray(a).T for a in output]
    bs = basis(ds, s, False)
    bx = basis(dx, y, True)
    def tensor(a, b):
        return (a[:, :, None]*b[:, None, :]).reshape(len(s), -1)
    return tuple(tensor(a, b) for a, b in ((bs[0], bx[0]), (bs[1], bx[0]),
                                          (bs[0], bx[1]), (bs[0], bx[2])))


def values(coefficients, s, y, m):
    b = design(coefficients.shape[0]-1, coefficients.shape[1]-1, s, y)
    u, us, uy, uyy = (a @ coefficients.ravel() for a in b)
    e, es, ez, ezz = baseline(s, y, m)
    return u+e, us+es, uy/(2*float(m.radius))+ez, uyy/(4*float(m.radius)**2)+ezz


def fit(m, mode, ds, dx, iterations):
    s_axis = (1-np.cos(np.linspace(0, np.pi, 2*ds+9)))/2
    y_axis = (1-np.cos(np.linspace(0, np.pi, 3*dx+15)))/2
    ss, yy = np.meshgrid(s_axis, y_axis, indexing="ij")
    s, y = ss.ravel(), yy.ravel()
    z = float(m.radius)*(2*y-1)
    b0, bs, by, byy = design(ds, dx, s, y)
    bz, bzz = by/(2*float(m.radius)), byy/(4*float(m.radius)**2)
    e, es, ez, ezz = baseline(s, y, m)
    mu = np.asarray(m.mu0, float)+z[:, None]*np.asarray(m.mu1, float)
    cross, sigma = np.asarray(m.cross, float), np.asarray(m.sigma, float)
    c = np.zeros((ds+1)*(dx+1))
    history = []
    for k in range(iterations if mode == "greedy" else 1):
        u, uz = e+b0@c, ez+bz@c
        if np.min(u) <= 0:
            raise ArithmeticError("Nonpositive fitting iterate")
        w = mu*u[:, None]
        if mode == "greedy":
            w += uz[:, None]*cross
        a = qp_numeric(w, u, m)
        drift = -float(m.kappa)*z - a@cross
        reaction = -np.sum(a*mu, axis=1)+np.einsum("ni,ij,nj->n", a, sigma, a)
        matrix = -bs/float(m.horizon)+0.5*float(m.state_variance)*bzz
        matrix += drift[:, None]*bz + reaction[:, None]*b0
        base = -es/float(m.horizon)+0.5*float(m.state_variance)*ezz+drift*ez+reaction*e
        weights = np.where((y == 0) | (y == 1), 4.0, 1.0)
        c = np.linalg.lstsq(matrix*weights[:, None], -base*weights, rcond=1e-12)[0]
        history.append({"iteration": k+1, "sampled_frozen_residual_sup": float(np.max(np.abs(matrix@c+base)))})
    denominator = 2**42
    numerators = np.rint(c*denominator).astype(np.int64).reshape(ds+1, dx+1)
    return {"market": m.json(), "mode": mode, "degree_time": ds, "degree_space": dx,
            "coefficient_denominator": str(denominator),
            "coefficient_numerators": [[str(int(n)) for n in row] for row in numerators],
            "basis": "E+s*y*(1-y)*sum c_ij T_i(2*s-1)T_j(2*y-1)",
            "trace_slopes_exact": [str(a) for a in trace_slopes(m)], "fitting_history": history}


def candidate_polynomial(data):
    m = Market.from_json(data["market"])
    left, right = trace_slopes(m)
    if data.get("trace_slopes_exact") != [str(left), str(right)]:
        raise ValueError("Candidate trace differs from the prescribed market trace")
    den = int(data["coefficient_denominator"])
    if den <= 0:
        raise ValueError("Invalid coefficient denominator")
    c = np.asarray([[F(int(k), den) for k in row] for row in data["coefficient_numerators"]], dtype=object)
    if c.shape != (data["degree_time"]+1, data["degree_space"]+1):
        raise ValueError("Coefficient shape mismatch")
    # Reuse the exact basis implementation, replacing its nonconstant trace.
    return exact.add(exact.candidate_polynomial(c), {(1, 0): left, (1, 1): right-left+F(3, 100)})


def sum_poly(items):
    ans = {}
    for item in items:
        ans = exact.add(ans, item)
    return ans


def branch_polynomials(u, m, mode):
    uz = exact.scale(exact.derivative(u, 1), 1/(2*m.radius))
    uzz = exact.scale(exact.derivative(uz, 1), 1/(2*m.radius))
    z = {(0, 0): -m.radius, (0, 1): 2*m.radius}
    mu = [exact.add({(0, 0): m.mu0[i]}, exact.scale(z, m.mu1[i])) for i in range(2)]
    wf = [exact.add(exact.mul(mu[i], u), exact.scale(uz, m.cross[i])) for i in range(2)]
    w = wf if mode == "greedy" else [exact.mul(mu[i], u) for i in range(2)]
    base = sum_poly([exact.scale(exact.derivative(u, 0), -1/m.horizon),
                     exact.scale(uzz, m.state_variance/2), exact.scale(exact.mul(z, uz), -m.kappa)])
    base_u = exact.mul(base, u)
    out = {"u": u}
    branches = []
    for k, state in enumerate(STATES):
        p = [{} for _ in range(2)]  # policy_i * u
        fixed = [i for i in range(2) if state[i]]
        free = [i for i in range(2) if not state[i]]
        for i in fixed:
            p[i] = exact.scale(u, m.caps[i] if state[i] == 1 else F(0))
        rhs = [exact.scale(w[i], F(1, 2)) for i in range(2)]
        if len(free) == 2:
            determinant = 1-m.sigma12**2
            p[0] = exact.scale(exact.add(rhs[0], exact.scale(rhs[1], -m.sigma12)), 1/determinant)
            p[1] = exact.scale(exact.add(rhs[1], exact.scale(rhs[0], -m.sigma12)), 1/determinant)
        elif len(free) == 1:
            i, j = free[0], fixed[0]
            p[i] = exact.add(rhs[i], exact.scale(p[j], -m.sigma[i][j]))
        grad = [exact.add(w[i], exact.scale(sum_poly(exact.scale(p[j], m.sigma[i][j]) for j in range(2)), -2)) for i in range(2)]
        conditions = []  # each polynomial must be nonnegative for branch feasibility
        for i in range(2):
            if not state[i]:
                conditions.extend([p[i], exact.add(exact.scale(u, m.caps[i]), exact.scale(p[i], -1))])
            else:
                conditions.append(exact.scale(grad[i], F(state[i])))
        names = []
        for j, condition in enumerate(conditions):
            name = f"b{k}_c{j}"
            out[name] = condition
            names.append(name)
        numerator = exact.add(base_u, exact.scale(sum_poly(exact.mul(wf[i], p[i]) for i in range(2)), -1))
        numerator = exact.add(numerator, sum_poly(exact.scale(exact.mul(p[i], p[j]), m.sigma[i][j]) for i in range(2) for j in range(2)))
        name = f"b{k}_residual"
        out[name] = numerator
        branches.append((names, name, state))
    return out, branches


def certify(polys, branches, target, max_cells=100000, max_depth=34):
    print("  exact power-to-Bernstein conversion", flush=True)
    root = {name: exact.bernstein_enclosure(poly) for name, poly in polys.items()}
    todo = [(root, 0)]
    minimum = math.inf
    maximum = 0.0
    accepted = switching = visited = 0
    area = F(0)
    active = set()
    while todo:
        cell, depth = todo.pop()
        visited += 1
        ranges = {name: (float(a.min()), float(b.max())) for name, (a, b) in cell.items()}
        lo, hi = ranges["u"]
        possible = [(name, state) for conditions, name, state in branches if all(ranges[c][1] >= 0 for c in conditions)]
        if not possible:
            raise ArithmeticError("QP branch coverage lost")
        bound = math.inf
        if lo > 0:
            bound = max(max(abs(v) for v in exact.interval_divide(ranges[name], (lo, hi))) for name, _ in possible)
        if lo > 0 and bound <= target:
            accepted += 1
            switching += len(possible) > 1
            active.update(state for _, state in possible)
            minimum, maximum = min(minimum, lo), max(maximum, bound)
            area += F(1, 2**depth)
        elif visited >= max_cells or depth >= max_depth:
            return {"certified": False, "reason": "subdivision budget exhausted", "visited_cells": visited,
                    "failed_residual_upper": bound if math.isfinite(bound) else None, "failed_depth": depth}
        else:
            children = [{}, {}]
            for name, enclosure in cell.items():
                parts = exact.split_bernstein(enclosure, depth % 2)
                for j in (0, 1):
                    children[j][name] = parts[j]
            todo.extend((child, depth+1) for child in children)
    assert area == 1
    return {"certified": True, "covered_area_exact": str(area), "accepted_cells": accepted,
            "switching_cells": switching, "visited_cells": visited, "u_lower": minimum,
            "residual_sup_upper": maximum, "possible_active_sets": [list(s) for s in sorted(active)]}


def mpiv(x):
    if isinstance(x, F):
        return iv.mpf(x.numerator)/x.denominator
    return iv.mpf(x)


def upper(x):
    return float(np.nextafter(float(x.b), np.inf))


def lower(x):
    return float(np.nextafter(float(x.a), -np.inf))


def coefficient_bounds(m):
    vertices = list(product(*[(F(0), cap) for cap in m.caps]))
    c0 = max(sum(a[i]*m.sigma[i][j]*a[j] for i in range(2) for j in range(2))-sum(a[i]*m.mu0[i] for i in range(2)) for a in vertices)
    c1 = max(abs(sum(a[i]*m.mu1[i] for i in range(2))) for a in vertices)
    drift = max(abs(sum(a[i]*m.cross[i] for i in range(2))) for a in vertices)
    cplus = max(F(0), *(sum(a[i]*m.sigma[i][j]*a[j] for i in range(2) for j in range(2))-sum(a[i]*(m.mu0[i]+m.mu1[i]*z) for i in range(2)) for a in vertices for z in (-m.radius, m.radius)))
    return c0, c1, drift, cplus


def lyapunov_lambda(a, m):
    c0, c1, drift, _ = coefficient_bounds(m)
    damping = 2*a*m.kappa-2*m.state_variance*a*a
    if damping <= 0:
        raise ValueError("Quadratic weight outside the dissipative range")
    return a*m.state_variance+c0+(2*a*drift+c1)**2/(4*damping)


def localization(m, initial_z=F(0)):
    """Uniform stopped-payoff bound; every candidate weight is checked exactly."""
    iv.dps = 45
    if abs(initial_z) >= m.radius:
        raise ValueError("Initial state must be strictly interior")
    limit = m.kappa/m.state_variance
    best = None
    # Optimize a certified upper bound over a finite rational grid, not its validity.
    for j in range(1, 16):
        a0 = limit*F(j, 200)
        lam0 = lyapunov_lambda(a0, m)
        kg = iv.exp(mpiv(lam0*m.horizon))
        for k in range(20, 100):
            a1 = limit*F(k, 100)
            if a1 <= a0:
                continue
            lam1 = lyapunov_lambda(a1, m)
            exponent = lam1*m.horizon+a1*initial_z**2-(a1-a0)*m.radius**2
            bound = upper((kg+1)*iv.exp(mpiv(exponent)))
            if best is None or bound < best["value_localization_upper"]:
                best = {"value_localization_upper": bound, "a0_exact": str(a0), "a1_exact": str(a1),
                        "lambda0_exact": str(lam0), "lambda1_exact": str(lam1),
                        "initial_state_exact": str(initial_z), "weight": "exp(a*z^2)"}
    return best


def exact_point(poly, s=F(1), y=F(1, 2)):
    return sum((v*s**i*y**j for (i, j), v in poly.items()), F(0))


def welfare_bounds(m, poly, certificate, mode):
    iv.dps = 45
    _, _, _, cplus = coefficient_bounds(m)
    c, horizon = mpiv(cplus), mpiv(m.horizon)
    phi = (iv.exp(c*horizon)-1)/c if cplus else horizon
    value_error = phi*mpiv(certificate["residual_sup_upper"])
    loc = localization(m)
    ell = mpiv(loc["value_localization_upper"])
    point_exact = exact_point(poly)
    point = mpiv(point_exact)
    total = value_error+ell
    lo, hi = lower(point-total), upper(point+total)
    if lo <= 0:
        raise ArithmeticError("Positive whole-space factor lower bound not obtained")
    result = {"scope": "whole-space CRRA model at t=0, z=0", "mode": mode,
              "candidate_at_initial_state_exact": str(point_exact), "cplus_exact": str(cplus),
              "barrier_factor_upper": upper(phi), "dirichlet_value_error_upper": upper(value_error),
              **loc, "whole_space_factor_interval": [lo, hi],
              "whole_space_ce_interval": [lower(1/mpiv(hi)), upper(1/mpiv(lo))]}
    if mode == "greedy":
        # Both exact optimal and deployed factors lie in this same interval.
        # gamma=2 gives CE_pi/CE_star = g_star/g_pi.
        regret = 2*total
        relative_loss = 1-mpiv(lo)/mpiv(hi)
        result.update({"whole_space_utility_regret_upper": upper(regret),
                       "relative_ce_loss_upper": upper(relative_loss),
                       "relative_ce_loss_bp_upper": upper(10000*relative_loss),
                       "deployment": "exact Sigma-QP greedy policy inside D_R; zero risky weights outside D_R"})
    else:
        result["deployment"] = "same exact Sigma-QP myopic policy on the whole state space"
    return result


def evaluate_candidate(data, target, max_cells):
    m = Market.from_json(data["market"])
    mode = data["mode"]
    if mode not in ("greedy", "myopic"):
        raise ValueError("Unknown evaluation policy")
    poly = candidate_polynomial(data)
    polys, branches = branch_polynomials(poly, m, mode)
    certificate = certify(polys, branches, target, max_cells)
    result = {"market": m.json(), "mode": mode, "certificate": certificate}
    if certificate["certified"]:
        result["bounds"] = welfare_bounds(m, poly, certificate, mode)
    return result


def write_table(results):
    rows = []
    for r in results:
        b, c = r["bounds"], r["certificate"]
        rows.append(f"{r['mode'].capitalize()} & ${exact.upper_sci(c['residual_sup_upper'])}$ & "
                    f"${exact.upper_sci(b['dirichlet_value_error_upper'])}$ & "
                    f"${exact.upper_sci(b['value_localization_upper'])}$ & "+
                    (f"${exact.upper_sci(b['relative_ce_loss_bp_upper'])}$" if r["mode"] == "greedy" else "---")+r"\\")
    TABLE.write_text("% Generated by experiments/portfolio_welfare.py. All upper bounds rounded upward.\n"
                     r"\begin{tabular}{lrrrr}\toprule"+"\n"
                     r"Policy & Residual bound & Dirichlet error & Localization & CE loss (bp)\\\midrule"+"\n"
                     +"\n".join(rows)+"\n"+r"\bottomrule\end{tabular}"+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--degree-time", type=int, default=8)
    parser.add_argument("--degree-space", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--target-residual", type=float, default=0.00025)
    parser.add_argument("--max-cells", type=int, default=100000)
    args = parser.parse_args(argv)
    if min(args.degree_time, args.degree_space) < 0 or args.iterations < 1 or args.max_cells < 1 or not math.isfinite(args.target_residual) or args.target_residual <= 0:
        parser.error("Invalid degree, iteration budget, or tolerance")
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("% Welfare certification incomplete.\n\\textit{No certificate returned.}\n")
    summary_path = OUT / "welfare_results.json"
    summary_path.write_text(json.dumps({"certified": False, "reason": "run incomplete"})+"\n")
    results = []
    start = time.perf_counter()
    for mode in ("greedy", "myopic"):
        candidate_path = OUT / f"{mode}_candidate.json"
        if args.verify_only:
            data = json.loads(candidate_path.read_text())
        else:
            print(f"Fitting {mode}", flush=True)
            data = fit(Market(), mode, args.degree_time, args.degree_space, args.iterations)
            candidate_path.write_text(json.dumps(data, indent=2)+"\n")
        if data["mode"] != mode:
            raise ValueError("Candidate policy type mismatch")
        print(f"Verifying {mode}", flush=True)
        result = evaluate_candidate(data, args.target_residual, args.max_cells)
        result["candidate_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        results.append(result)
        (OUT / f"{mode}_results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["certificate"]["certified"]:
            summary_path.write_text(json.dumps({"certified": False, "results": results}, indent=2)+"\n")
            return 2
    a, b = [r["bounds"]["whole_space_ce_interval"] for r in results]
    raw_difference = [lower(mpiv(a[0])-mpiv(b[1])), upper(mpiv(a[1])-mpiv(b[0]))]
    if raw_difference[1] < 0:
        raise ArithmeticError("Certified intervals contradict optimality")
    difference = [max(0.0, raw_difference[0]), raw_difference[1]]
    output = {"certified": True, "results": results, "optimal_minus_myopic_ce_interval": difference,
              "raw_ce_difference_interval": raw_difference,
              "strict_dynamic_advantage_certified": difference[0] > 0,
              "seconds": time.perf_counter()-start,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "arithmetic_core_sha256": hashlib.sha256(Path(exact.__file__).read_bytes()).hexdigest()}
    summary_path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    write_table(results)
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
