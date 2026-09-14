"""Exact reduced policy iteration as a matrix ODE recursion (Prop. benchmark).

Implements the per-iterate coefficient ODEs of Proposition~\ref{prop:benchmark}
in the numerical code's convention (the deterministic factor exp((1-gamma) r tau)
is factored out and re-applied only at evaluation):

    g_n(tau, z) = exp((1-gamma) r tau) * exp(a_n(tau) + b_n(tau).z + z'C_n(tau)z/2)

with terminal data a_n(0)=b_n(0)=C_n(0)=0 and greedy update

    vartheta_{n+1}(tau, z) = (phi0 + Gamma b_n(tau))/gamma
                             + ((Lam + Gamma C_n(tau))/gamma) z.

Outputs, per supplied benchmark market (N=30, M in {1,3,5,10}):
  * Riccati finiteness:      max_n sup_tau ||C_n(tau)||_2
  * coefficient convergence: sup_tau ||C_n - C*||, ||b_n - b*||, |a_n - a*|
  * exact-PI Newton rate:    log-log regression of e_{n+1} on e_n, where e_n is
                             the relative L2 value error of iterate n against
                             the converged closed-form reference on a fixed
                             held-out cloud (the same eval convention as the
                             supplied metrics.json)
  * verification integrability: discretized Fredholm condition
                             lambda_max(2 K^{1/2} Q_op K^{1/2}) < 1 for the
                             converged policy at t=0 (the widest window)

Markets are loaded from the supplied market_params.npz snapshots so all
diagnostics refer to exactly the runs behind the reported tables.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Provide the POSIX user-id helper on Windows for imported dependencies.
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0  # type: ignore[attr-defined]
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "mpl"))

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code" / "pipinn_nemerical"
OUTS = CODE / "outputs" / "outputs" / "pipinn"
RESULTS = HERE.parent / "results" / "exact_pi_ode"

sys.path.insert(0, str(CODE))

GAMMA = 2.0
R = 0.03
TAU_MAX = 3.0
N_TAU_GRID = 1201          # dense tau grid for coefficient storage
N_ITER = 12                # PI iterations
EVAL_POINTS = 20_000
EVAL_SEED = 727            # same seed family as the archive metrics


def rk4_backward_coefficients(market, v_fun, Th_fun, tau_grid):
    """Integrate the frozen-policy coefficient ODEs on tau_grid (tau increasing).

    v_fun(tau) -> (N,), Th_fun(tau) -> (N, M) give the frozen affine policy.
    Returns arrays a (T,), b (T, M), C (T, M, M) with zero data at tau=0.
    """
    K, k0, Q, Gam, lam0, Lam = (
        market["K"], market["k0"], market["Q"], market["Gamma"],
        market["lam0"], market["Lam"],
    )
    q = 1.0 - GAMMA
    M = K.shape[0]
    n = len(tau_grid)
    a = np.zeros(n)
    b = np.zeros((n, M))
    C = np.zeros((n, M, M))

    def rhs(tau, a_v, b_v, C_v):
        v = v_fun(tau)
        Th = Th_fun(tau)
        ThG = Th.T @ Gam            # (M, M)
        GtV = Gam.T @ v             # (M,)
        dC = (
            -(K.T @ C_v + C_v @ K)
            + C_v @ Q @ C_v
            + q * (Th.T @ Lam + Lam.T @ Th)
            - q * GAMMA * (Th.T @ Th)
            + q * (ThG @ C_v + C_v @ ThG.T)
        )
        db = (
            C_v @ k0
            - K.T @ b_v
            + C_v @ Q @ b_v
            + q * (Th.T @ lam0 + Lam.T @ v - GAMMA * (Th.T @ v))
            + q * (ThG @ b_v + C_v @ GtV)
        )
        da = (
            k0 @ b_v
            + 0.5 * np.trace(Q @ C_v)
            + 0.5 * b_v @ Q @ b_v
            + q * (v @ lam0 - 0.5 * GAMMA * (v @ v) + v @ Gam @ b_v)
        )
        return da, db, 0.5 * (dC + dC.T)

    for i in range(1, n):
        h = tau_grid[i] - tau_grid[i - 1]
        t0 = tau_grid[i - 1]
        k1 = rhs(t0, a[i - 1], b[i - 1], C[i - 1])
        k2 = rhs(t0 + h / 2, a[i - 1] + h / 2 * k1[0], b[i - 1] + h / 2 * k1[1], C[i - 1] + h / 2 * k1[2])
        k3 = rhs(t0 + h / 2, a[i - 1] + h / 2 * k2[0], b[i - 1] + h / 2 * k2[1], C[i - 1] + h / 2 * k2[2])
        k4 = rhs(t0 + h, a[i - 1] + h * k3[0], b[i - 1] + h * k3[1], C[i - 1] + h * k3[2])
        a[i] = a[i - 1] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        b[i] = b[i - 1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        C[i] = C[i - 1] + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if not np.isfinite(C[i]).all():
            raise FloatingPointError(f"Riccati blow-up at tau={tau_grid[i]:.4f}")
    return a, b, C


def interp_coeff(tau_grid, arr):
    """Return a linear interpolant over tau for a (T, ...) coefficient array."""
    def f(tau):
        idx = np.searchsorted(tau_grid, tau)
        idx = np.clip(idx, 1, len(tau_grid) - 1)
        t0, t1 = tau_grid[idx - 1], tau_grid[idx]
        w = 0.0 if t1 == t0 else (tau - t0) / (t1 - t0)
        return (1 - w) * arr[idx - 1] + w * arr[idx]
    return f


def eval_g(a, b, C, tau_grid, tau, z):
    """Evaluate log g (without the exp(q r tau) lift) at points (tau, z)."""
    ai = interp_many(tau_grid, a, tau)
    bi = interp_many(tau_grid, b, tau)
    Ci = interp_many(tau_grid, C, tau)
    quad = 0.5 * np.einsum("bi,bij,bj->b", z, Ci, z)
    return ai + np.einsum("bi,bi->b", bi, z) + quad


def interp_many(tau_grid, arr, tau):
    idx = np.clip(np.searchsorted(tau_grid, tau), 1, len(tau_grid) - 1)
    t0, t1 = tau_grid[idx - 1], tau_grid[idx]
    w = np.where(t1 == t0, 0.0, (tau - t0) / np.maximum(t1 - t0, 1e-300))
    if arr.ndim == 1:
        return (1 - w) * arr[idx - 1] + w * arr[idx]
    shape = (len(tau),) + (1,) * (arr.ndim - 1)
    w = w.reshape(shape)
    return (1 - w) * arr[idx - 1] + w * arr[idx]


def fredholm_margin(market, Th_star, tau_grid, n_grid=400):
    """Discretized Fredholm check at t=0 for the converged affine policy.

    Y^0 solves dY = K_n Y ds + sigma_Z dB on [0, T]; the quadratic potential is
    Q_n(s).  We form the covariance operator of Y on an n_grid time grid and
    check lambda_max(2 K^{1/2} Q_op K^{1/2}) < 1.  Time-dependence of Theta* is
    kept through the grid.
    """
    K, Q, Gam, Lam, SigmaX = (
        market["K"], market["Q"], market["Gamma"], market["Lam"], market["SigmaX"],
    )
    q = 1.0 - GAMMA
    M = K.shape[0]
    T = tau_grid[-1]
    s_grid = np.linspace(0.0, T, n_grid)  # forward time; tau = T - s
    h = s_grid[1] - s_grid[0]

    # frozen matrices at each forward time s (policy evaluated at tau = T - s)
    Kn = np.empty((n_grid, M, M))
    Qn = np.empty((n_grid, M, M))
    for i, s in enumerate(s_grid):
        Th = Th_star(T - s)
        # Kn = -kappa + q sigma_Z rho' Theta = -K + q Gamma' Theta  (Gamma' = sigma_Z rho')
        Kn[i] = -K + q * (Gam.T @ Th)
        Qi = q * (0.5 * (Th.T @ Lam + Lam.T @ Th) - 0.5 * GAMMA * (Th.T @ Th))
        Qn[i] = 0.5 * (Qi + Qi.T)

    # fundamental matrices F(s_i) with F(0) = I, dF/ds = Kn F
    F = np.empty((n_grid, M, M))
    F[0] = np.eye(M)
    for i in range(1, n_grid):
        A = Kn[i - 1]
        F[i] = F[i - 1] + h * (A @ F[i - 1])
        # midpoint correction
        Fm = 0.5 * (F[i] + F[i - 1])
        F[i] = F[i - 1] + h * (0.5 * (Kn[i - 1] + Kn[i]) @ Fm)

    Finv = np.linalg.inv(F)
    SS = SigmaX @ SigmaX.T

    # Cov(Y_si, Y_sj) = F(si) [ int_0^{min} Finv(u) SS Finv(u)' du ] F(sj)'
    G = np.empty((n_grid, M, M))  # running integral
    G[0] = np.zeros((M, M))
    for i in range(1, n_grid):
        mid = 0.5 * (
            Finv[i - 1] @ SS @ Finv[i - 1].T + Finv[i] @ SS @ Finv[i].T
        )
        G[i] = G[i - 1] + h * mid

    big = np.empty((n_grid * M, n_grid * M))
    for i in range(n_grid):
        for j in range(n_grid):
            k = min(i, j)
            big[i * M:(i + 1) * M, j * M:(j + 1) * M] = F[i] @ G[k] @ F[j].T
    big = 0.5 * (big + big.T)

    # operator for the potential integral: int <Y_s, Qn(s) Y_s> ds  ~ h * blockdiag(Qn)
    Qop = np.zeros_like(big)
    for i in range(n_grid):
        Qop[i * M:(i + 1) * M, i * M:(i + 1) * M] = h * Qn[i]

    # lambda_max(2 K^{1/2} Qop K^{1/2}) = lambda_max(2 Qop^{1/2}... ) use
    # generalized eigenvalue: eig(2 * big @ Qop) has same nonzero spectrum.
    vals = np.linalg.eigvals(2.0 * big @ Qop)
    return float(np.max(vals.real))


def run_market(cell: Path, label: str):
    snap = np.load(cell / "market_params.npz", allow_pickle=True)
    market = {k: np.asarray(snap[k], dtype=float)
              for k in ("K", "k0", "Q", "Gamma", "lam0", "Lam", "SigmaX",
                        "xbar", "eta", "X_min", "X_max")}
    M = market["K"].shape[0]
    tau_grid = np.linspace(0.0, TAU_MAX, N_TAU_GRID)

    # --- converged reference (same solver as the archive uses) --------------
    from Liu_nd_pinn_homogeneous import solve_closed_form_ode, closed_form_coefficients

    sol = solve_closed_form_ode(
        TAU_MAX, GAMMA, market["K"], market["xbar"], market["SigmaX"],
        market["rho"] if "rho" in market else np.asarray(snap["rho"], dtype=float),
        market["lam0"], market["Lam"],
    )
    a_s, b_s, C_s = closed_form_coefficients(tau_grid, sol, M)
    a_star, b_star, C_star = a_s.reshape(-1), b_s, C_s

    # --- held-out evaluation cloud (archive convention: uniform box) --------
    rng = np.random.default_rng(EVAL_SEED)
    tau_pts = rng.uniform(0.0, TAU_MAX, EVAL_POINTS)
    z_pts = rng.uniform(market["X_min"], market["X_max"], (EVAL_POINTS, M))
    log_g_star = eval_g(a_star, b_star, C_star, tau_grid, tau_pts, z_pts)
    g_star = np.exp(log_g_star)

    # policy reference for policy-error metric
    b_i_star = interp_many(tau_grid, b_star, tau_pts)
    C_i_star = interp_many(tau_grid, C_star, tau_pts)
    grad_log_star = b_i_star + np.einsum("bij,bj->bi", C_i_star, z_pts)
    lam_x = market["lam0"][None, :] + z_pts @ market["Lam"].T
    vth_star = lam_x / GAMMA + (grad_log_star @ market["Gamma"].T) / GAMMA

    # --- PI recursion from several initial policies -------------------------
    # The benchmark contraction is so strong that from the myopic start a
    # single greedy step already reaches the ODE floor.  To fit a log-log
    # slope we run exact PI from progressively worse initial policies and
    # pool all pre-floor (e_n, e_{n+1}) pairs.
    def run_pi(v_init, Th_init, n_iter):
        v_fun = lambda tau: v_init
        Th_fun = lambda tau: Th_init
        sup_C_local, ev, ep, gaps = [], [], [], []
        for n in range(n_iter):
            a_n, b_n, C_n = rk4_backward_coefficients(market, v_fun, Th_fun, tau_grid)
            sup_C_local.append(float(np.max(np.linalg.norm(C_n, ord=2, axis=(1, 2)))))
            log_g_n = eval_g(a_n, b_n, C_n, tau_grid, tau_pts, z_pts)
            ev.append(float(np.linalg.norm(np.exp(log_g_n) - g_star) / np.linalg.norm(g_star)))
            b_i = interp_many(tau_grid, b_n, tau_pts)
            C_i = interp_many(tau_grid, C_n, tau_pts)
            grad_log = b_i + np.einsum("bij,bj->bi", C_i, z_pts)
            vth = lam_x / GAMMA + (grad_log @ market["Gamma"].T) / GAMMA
            ep.append(float(np.linalg.norm(vth - vth_star) / np.linalg.norm(vth_star)))
            gaps.append({
                "iter": n,
                "sup_dC": float(np.max(np.linalg.norm(C_n - C_star, ord=2, axis=(1, 2)))),
                "sup_db": float(np.max(np.linalg.norm(b_n - b_star, axis=1))),
                "sup_da": float(np.max(np.abs(a_n - a_star))),
            })
            bf = interp_coeff(tau_grid, b_n)
            Cf = interp_coeff(tau_grid, C_n)
            v_fun = (lambda bf: lambda tau: (market["lam0"] + market["Gamma"] @ bf(tau)) / GAMMA)(bf)
            Th_fun = (lambda Cf: lambda tau: (market["Lam"] + market["Gamma"] @ Cf(tau)) / GAMMA)(Cf)
        return sup_C_local, ev, ep, gaps

    v_myo = market["lam0"] / GAMMA
    Th_myo = market["Lam"] / GAMMA
    inits = {
        "myopic": (v_myo, Th_myo, N_ITER),
        "zero": (np.zeros_like(v_myo), np.zeros_like(Th_myo), 6),
        "half_myopic": (0.5 * v_myo, 0.5 * Th_myo, 6),
        "double_myopic": (2.0 * v_myo, 2.0 * Th_myo, 6),
        "triple_myopic": (3.0 * v_myo, 3.0 * Th_myo, 6),
    }
    runs = {}
    for name, (v0i, Th0i, n_it) in inits.items():
        runs[name] = run_pi(v0i, Th0i, n_it)

    sup_C, errs_value, errs_policy, coeff_gap = runs["myopic"]

    # --- Newton-rate fit pooled over initializations ------------------------
    pairs = []
    floor = max(errs_value[-1], 1e-13)
    for name, (_, ev, _, _) in runs.items():
        for i in range(len(ev) - 1):
            if ev[i + 1] > 5 * floor and ev[i] > 5 * floor and ev[i] < 0.5:
                pairs.append((ev[i], ev[i + 1], name, i))
    slope = intercept = float("nan")
    if len(pairs) >= 2:
        lx = np.log([p[0] for p in pairs])
        ly = np.log([p[1] for p in pairs])
        slope, intercept = np.polyfit(lx, ly, 1)
    e = np.array(errs_value)
    sat_index = next((i for i in range(len(e)) if e[i] <= 5 * floor), len(e) - 1)

    # --- Fredholm integrability margin for the converged policy -------------
    bfs = interp_coeff(tau_grid, b_star)
    Cfs = interp_coeff(tau_grid, C_star)
    Th_star = lambda tau: (market["Lam"] + market["Gamma"] @ Cfs(tau)) / GAMMA
    market_f = dict(market)
    margin = fredholm_margin(market_f, Th_star, tau_grid,
                             n_grid=200 if M <= 3 else 80)

    # sign of the quadratic potential at the converged policy: if Q_n <= 0 the
    # Gaussian quadratic functional is bounded above and integrability holds
    # with infinite margin (expected for gamma > 1, where q = 1-gamma < 0).
    qq = 1.0 - GAMMA
    max_eig_Qn = -np.inf
    for tau in np.linspace(0.0, TAU_MAX, 61):
        Th = Th_star(tau)
        Qn = qq * (0.5 * (Th.T @ market["Lam"] + market["Lam"].T @ Th)
                   - 0.5 * GAMMA * (Th.T @ Th))
        max_eig_Qn = max(max_eig_Qn, float(np.max(np.linalg.eigvalsh(0.5 * (Qn + Qn.T)))))

    out = {
        "label": label,
        "M": M,
        "gamma": GAMMA,
        "r": R,
        "tau_max": TAU_MAX,
        "n_iterations": N_ITER,
        "eval_points": EVAL_POINTS,
        "eval_seed": EVAL_SEED,
        "riccati_sup_C_per_iter": sup_C,
        "riccati_max_sup_C": max(sup_C),
        "converged_sup_C_star": float(np.max(np.linalg.norm(C_star, ord=2, axis=(1, 2)))),
        "value_rel_l2_per_iter": errs_value,
        "policy_rel_l2_per_iter": errs_policy,
        "coefficient_gaps": coeff_gap,
        "value_errors_by_init": {name: rv[1] for name, rv in runs.items()},
        "newton_loglog_slope": float(slope),
        "newton_loglog_intercept": float(intercept),
        "newton_fit_pairs": [
            {"e_n": p[0], "e_n1": p[1], "init": p[2], "n": p[3]} for p in pairs
        ],
        "saturation_index": int(sat_index),
        "fredholm_lambda_max_2KQK": margin,
        "fredholm_condition_met": bool(margin < 1.0),
        "max_eig_Qn_star": max_eig_Qn,
        "potential_nonpositive": bool(max_eig_Qn <= 1e-12),
    }
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_out = {}
    for L in (1, 3, 5, 10):
        cell = OUTS / f"N_30_M{L}"
        print(f"=== market N_30_M{L} ===")
        res = run_market(cell, f"N_30_M{L}")
        all_out[f"M{L}"] = res
        print(f"  max_n sup_tau ||C_n|| = {res['riccati_max_sup_C']:.6f}")
        print(f"  value  errors: {['%.3e' % v for v in res['value_rel_l2_per_iter']]}")
        print(f"  policy errors: {['%.3e' % v for v in res['policy_rel_l2_per_iter']]}")
        print(f"  newton slope = {res['newton_loglog_slope']:.3f}  "
              f"saturation at n = {res['saturation_index']}")
        print(f"  fredholm lambda_max(2KQK) = {res['fredholm_lambda_max_2KQK']:.4f} "
              f"(<1: {res['fredholm_condition_met']})")
    with open(RESULTS / "exact_pi_ode_results.json", "w") as fh:
        json.dump(all_out, fh, indent=2)
    print(f"\nsaved -> {RESULTS / 'exact_pi_ode_results.json'}")


if __name__ == "__main__":
    main()
