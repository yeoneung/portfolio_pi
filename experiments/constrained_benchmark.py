# -*- coding: utf-8 -*-
"""Constrained benchmark with active constraints and a finite-difference reference.

Setting (L = 1 state, supplied N = 30 market):
  * compact convex constraint: a box |vartheta_i| <= c_i on the whitened
    exposure, with c_i = 0.7 * max over [0,T] x D_r of |vartheta_i^{unc}|
    computed from the unconstrained affine ODE solution, so the constraint
    binds on roughly half of the interior state-time region;
  * bounded domain: the same ellipsoid (interval) D_R with R = 2.0 and the
    same hard Dirichlet traces psi_0 as the trace-exact experiment, so the
    FDM reference and the neural solver solve the *same* bounded-domain
    problem (matched domain, trace data, and objective);
  * Euclidean projection in whitened coordinates = componentwise clamp
    (= the Sigma-metric projection of the manuscript).

Outputs:
  1. FDM constrained policy iteration on a fine grid: per-iterate errors
     against its own converged limit -> exact-iteration contraction with
     active constraints (no neural error involved).
  2. Grid-accuracy check: the same FDM solver run *unconstrained* against
     the affine ODE closed form.
  3. Constrained PINN-PI with the hard-trace architecture: per-outer
     held-out errors against the converged FDM reference -> the neural
     evaluation floor, separated from the exact-iteration behavior.
  4. Active-set statistics and active-set agreement between the neural and
     FDM policies.

Everything is written to results/constrained_benchmark/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

if not hasattr(os, "getuid"):
    os.getuid = lambda: 0  # type: ignore[attr-defined]
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "mpl"))

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code" / "pipinn_nemerical"
RESULTS = HERE.parent / "results" / "constrained_benchmark"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from pipinn_hard_certificate import (  # noqa: E402
    DEVICE, DTYPE, T, HardValueNet, frozen_residual, sample_ellipsoid_uniform,
)
from Liu_nd_pinn_homogeneous import (  # noqa: E402
    closed_form_coefficients, solve_closed_form_ode,
)

GAMMA = 2.0
R_RATE = 0.03
TAU_MAX = 3.0
Q_EXP = 1.0 - GAMMA
R_DOM = 2.0          # domain radius in units of eta
R_INT = 1.0          # interior evaluation region
CAP_FRAC = 0.7       # c_i = CAP_FRAC * max |unconstrained exposure| over Q_R

N_Z = 1201
N_TAU = 1201
CAP_HOLDER = {}
PI_TOL = 1e-11
PI_MAX_ITERS = 30
NN_EVAL_SEED = 2027
NN_EVAL_SIZE = 20_000
DEFAULT_NN_SEEDS = (11, 12, 13, 14, 15)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Matched constrained FDM/PINN-PI benchmark.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_NN_SEEDS),
        help="neural training seeds (default: 11 12 13 14 15)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Optional exact-weight JSON export for each final network")
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.seeds):
        parser.error("all seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    return args


def load_market():
    snap = np.load(CODE / "outputs" / "outputs" / "pipinn" / "N_30_M1" /
                   "market_params.npz", allow_pickle=True)
    return {k: np.asarray(snap[k], dtype=float)
            for k in ("K", "k0", "Q", "Gamma", "lam0", "Lam", "SigmaX",
                      "rho", "xbar", "eta")}


# ---------------------------------------------------------------------------
# unconstrained ODE reference (for the cap and the grid-accuracy check)
# ---------------------------------------------------------------------------
def ode_solution(market):
    sol = solve_closed_form_ode(TAU_MAX, GAMMA, market["K"], market["xbar"],
                                market["SigmaX"], market["rho"],
                                market["lam0"], market["Lam"])
    return sol


def ode_g_and_policy(market, sol, tau, z):
    """g (with psi0 lift) and unconstrained exposure on flat points."""
    M = 1
    a, b, C = closed_form_coefficients(np.asarray(tau, dtype=float).reshape(-1), sol, M)
    z2 = np.asarray(z, dtype=float).reshape(-1, 1)
    expo = Q_EXP * R_RATE * np.asarray(tau).reshape(-1) + a.reshape(-1)
    expo = expo + (b[:, 0] * z2[:, 0]) + 0.5 * C[:, 0, 0] * z2[:, 0] ** 2
    g = np.exp(expo)
    grad_log = b[:, 0] + C[:, 0, 0] * z2[:, 0]
    lam_x = market["lam0"][None, :] + z2 @ market["Lam"].T
    vth = lam_x / GAMMA + np.outer(grad_log, market["Gamma"][:, 0]) / GAMMA
    return g, vth


# ---------------------------------------------------------------------------
# 1-D finite-difference frozen-policy solver with Dirichlet psi0 traces
# ---------------------------------------------------------------------------
class FDM1D:
    def __init__(self, market, cap):
        self.m = market
        self.cap = cap                       # (N,) or None for unconstrained
        self.zc = market["xbar"][0]
        self.hz = R_DOM * market["eta"][0]
        self.z = np.linspace(self.zc - self.hz, self.zc + self.hz, N_Z)
        self.dz = self.z[1] - self.z[0]
        self.tau = np.linspace(0.0, TAU_MAX, N_TAU)
        self.dt = self.tau[1] - self.tau[0]
        self.a_diff = 0.5 * float(market["Q"][0, 0])
        self.phi = market["lam0"][None, :] + self.z[:, None] * market["Lam"].T[0][None, :]
        self.Gam = market["Gamma"][:, 0]     # (N,)
        self.drift0 = float(market["k0"][0]) - float(market["K"][0, 0]) * self.z

    def greedy(self, g):
        """Projected greedy policy on the (tau, z) grid from a g grid."""
        gz = np.gradient(g, self.dz, axis=1)
        grad_log = gz / g
        vth = (self.phi[None, :, :] + grad_log[:, :, None] * self.Gam[None, None, :]) / GAMMA
        if self.cap is not None:
            vth = np.clip(vth, -self.cap[None, None, :], self.cap[None, None, :])
        return vth

    def evaluate(self, vth):
        """Solve the frozen-policy linear PDE for the (tau, z, N) policy grid."""
        g = np.empty((N_TAU, N_Z))
        g[0] = 1.0
        lower = np.empty(N_Z)
        diag = np.empty(N_Z)
        upper = np.empty(N_Z)
        for k in range(1, N_TAU):
            v = vth[k]                                    # (N_Z, N)
            drift = self.drift0 + Q_EXP * (v @ self.Gam)
            reac = Q_EXP * (R_RATE + np.einsum("ji,ji->j", v, self.phi)
                            - 0.5 * GAMMA * np.einsum("ji,ji->j", v, v))
            # implicit Euler: (I - dt L) g_k = g_{k-1}
            A = self.a_diff / self.dz ** 2
            B = drift / (2.0 * self.dz)
            lower[:] = -self.dt * (A - B)
            diag[:] = 1.0 - self.dt * (-2.0 * A + reac)
            upper[:] = -self.dt * (A + B)
            rhs = g[k - 1].copy()
            # Dirichlet traces (row 0 and row N_Z-1 become identity rows)
            diag[0] = 1.0; upper[0] = 0.0; rhs[0] = np.exp(Q_EXP * R_RATE * self.tau[k])
            diag[-1] = 1.0; lower[-1] = 0.0; rhs[-1] = np.exp(Q_EXP * R_RATE * self.tau[k])
            # solve_banded layout: ab[0, i+1] = A[i, i+1], ab[2, i] = A[i+1, i]
            ab = np.zeros((3, N_Z))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]
            g[k] = solve_banded((1, 1), ab, rhs)
        return g

    def policy_iteration(self, verbose=True):
        """Run projected PI from the myopic start; return per-iterate grids.

        The stopping rule uses the policy change on the interior region only:
        at the near-boundary nodes the clamped greedy target can oscillate
        between the box corners (the Dirichlet boundary layer has a large
        log-gradient) without affecting the interior evaluation."""
        mask = np.abs(self.z - self.zc) <= R_INT * self.m["eta"][0]
        vth0 = np.broadcast_to(self.phi[None, :, :] / GAMMA,
                               (N_TAU, N_Z, self.phi.shape[1])).copy()
        if self.cap is not None:
            vth0 = np.clip(vth0, -self.cap, self.cap)
        g_list = []
        vth = vth0
        for n in range(PI_MAX_ITERS):
            g = self.evaluate(vth)
            g_list.append(g)
            vth_new = self.greedy(g)
            diff_int = float(np.max(np.abs((vth_new - vth)[:, mask, :])))
            diff_all = float(np.max(np.abs(vth_new - vth)))
            if verbose:
                print(f"    fdm outer {n+1:2d}: policy change interior "
                      f"{diff_int:.3e} (full {diff_all:.3e})")
            vth = vth_new
            if diff_int < PI_TOL:
                break
        return g_list, vth


def interior_mask(fdm):
    return np.abs(fdm.z - fdm.zc) <= R_INT * fdm.m["eta"][0]


def rel_l2(a, b, mask):
    return float(np.linalg.norm((a - b)[:, mask]) / np.linalg.norm(b[:, mask]))


# ---------------------------------------------------------------------------
# constrained PINN-PI on the same Dirichlet problem
# ---------------------------------------------------------------------------
def greedy_clamped(model, tau, z, tensors, cap_t):
    g = model(tau, z)
    g_z = torch.autograd.grad(g, z, grad_outputs=torch.ones_like(g))[0]
    lam_x = tensors["lam0"][None, :] + z @ tensors["Lam"].T
    vth = (lam_x + (g_z / g) @ tensors["Gamma"].T) / GAMMA
    return torch.clamp(vth, -cap_t, cap_t)


def seed_neural_run(seed):
    """Reset every RNG used by a neural run before model construction."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_heldout_data(market, fdm_ref):
    """Build the common held-out cloud and evaluate the FDM reference once."""
    center, eta = market["xbar"], market["eta"]
    ev_rng = np.random.default_rng(NN_EVAL_SEED)
    z_int = sample_ellipsoid_uniform(
        NN_EVAL_SIZE, 1, center, R_INT * eta, ev_rng)
    tau_int = ev_rng.uniform(0.0, TAU_MAX, (NN_EVAL_SIZE, 1))
    g_ref, vth_ref, act_ref = fdm_ref(
        tau_int.reshape(-1), z_int.reshape(-1))
    return {
        "tau": tau_int,
        "z": z_int,
        "g_ref": g_ref,
        "vth_ref": vth_ref,
        "act_ref": act_ref,
    }


def train_constrained_pipinn(market, cap, heldout_data, cfg, seed=11, checkpoint_dir=None):
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    seed_neural_run(seed)
    center, eta = market["xbar"], market["eta"]
    half = R_DOM * eta
    tensors = {k: T(market[k]) for k in ("K", "k0", "Q", "Gamma", "lam0", "Lam")}
    cap_t = T(cap)
    model = HardValueNet(1, center, half, hidden=cfg["hidden"], depth=cfg["depth"])
    model.to(DEVICE)
    rng = np.random.default_rng(seed)

    # Every seed is evaluated on the same cloud and cached FDM values.
    tau_int = heldout_data["tau"]
    z_int = heldout_data["z"]
    g_ref = heldout_data["g_ref"]
    vth_ref = heldout_data["vth_ref"]
    act_ref = heldout_data["act_ref"]

    def heldout():
        tau_e = T(tau_int).requires_grad_(True)
        z_e = T(z_int).requires_grad_(True)
        g = model(tau_e, z_e).detach().cpu().numpy().reshape(-1)
        vth = greedy_clamped(model, tau_e, z_e, tensors, cap_t).detach().cpu().numpy()
        ev = np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref)
        ep = np.linalg.norm(vth - vth_ref) / np.linalg.norm(vth_ref)
        # float32-safe activity test: an entry is active when it sits at the
        # box boundary up to a relative tolerance
        act_net = np.abs(vth) >= cap[None, :] * (1.0 - 1e-4)
        agree = float((act_net == act_ref).mean())
        return float(ev), float(ep), agree

    frozen = None
    history = []
    n_shell = cfg["batch"] // 4
    for outer in range(cfg["outer_iters"]):
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["inner_epochs"], eta_min=1e-4)
        for _ in range(cfg["inner_epochs"]):
            z_u = sample_ellipsoid_uniform(cfg["batch"] - n_shell, 1, center, half, rng)
            sgn = rng.standard_normal((n_shell, 1))
            sgn /= np.abs(sgn)
            rad = rng.uniform(0.8, 1.0, (n_shell, 1))
            z_b = center[None, :] + rad * sgn * half[None, :]
            z_np = np.concatenate([z_u, z_b])
            tau_np = rng.uniform(0.0, TAU_MAX, (cfg["batch"], 1))
            tau = T(tau_np).requires_grad_(True)
            z = T(z_np).requires_grad_(True)
            if frozen is None:
                lam_x = tensors["lam0"][None, :] + z @ tensors["Lam"].T
                policy = torch.clamp(lam_x / GAMMA, -cap_t, cap_t).detach()
            else:
                policy = greedy_clamped(frozen, tau, z, tensors, cap_t).detach()
            res, _ = frozen_residual(model, policy, tau, z, tensors)
            loss = res.square().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        ev, ep, agree = heldout()
        history.append({"outer": outer + 1, "train_loss": float(loss.item()),
                        "value_rel_l2_vs_fdm": ev, "policy_rel_l2_vs_fdm": ep,
                        "active_set_agreement": agree})
        print(f"  [nn] outer {outer+1:2d}: loss={loss.item():.3e} value={ev:.3e} "
              f"policy={ep:.3e} active-agree={agree:.4f} "
              f"({time.perf_counter()-t0:.0f}s)")
        frozen = HardValueNet(1, center, half, hidden=cfg["hidden"], depth=cfg["depth"])
        frozen.load_state_dict(model.state_dict())
        frozen.to(DEVICE)
        for p in frozen.parameters():
            p.requires_grad_(False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "hard_value_net_exact_binary_weights_v1", "seed": seed,
            "config": cfg, "gamma": GAMMA, "risk_free_rate": R_RATE,
            "horizon": TAU_MAX, "domain_scale": R_DOM,
            "market": {k: np.asarray(v).tolist() for k, v in market.items()},
            "caps": np.asarray(cap).tolist(),
            "state_dict": {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
            "training_dtype": str(DTYPE), "history": history,
            "policy_definition": "exact covariance-metric greedy map of this frozen neural value, with the stored market and caps",
        }
        (checkpoint_dir / f"seed_{seed}_final.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return history, time.perf_counter() - t0


FINAL_NN_METRICS = (
    "value_rel_l2_vs_fdm",
    "policy_rel_l2_vs_fdm",
    "active_set_agreement",
)


def aggregate_final_metrics(nn_runs):
    ddof = 1 if len(nn_runs) > 1 else 0
    summary = {"n_seeds": len(nn_runs), "sd_ddof": ddof}
    for metric in FINAL_NN_METRICS:
        values = np.asarray(
            [run["history"][-1][metric] for run in nn_runs], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=ddof)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main(argv=None):
    global N_Z, N_TAU
    args = parse_args(argv)
    program_t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    market = load_market()
    sol = ode_solution(market)
    out = {"R": R_DOM, "interior_r": R_INT, "cap_frac": CAP_FRAC,
           "grid": {"n_z": N_Z, "n_tau": N_TAU},
           "nn_seeds": args.seeds,
           "nn_eval": {"seed": NN_EVAL_SEED, "n_points": NN_EVAL_SIZE}}

    # ---- cap from the unconstrained ODE policy over the *interior* --------
    # c_i = CAP_FRAC * max_{[0,T] x D_r} |vartheta_i^{unc}|, so the box binds
    # on a substantial part of the region where guarantees are examined.
    zc, hz = market["xbar"][0], R_INT * market["eta"][0]
    rng = np.random.default_rng(3)
    z_cloud = rng.uniform(zc - hz, zc + hz, 40_000)
    tau_cloud = rng.uniform(0.0, TAU_MAX, 40_000)
    _, vth_unc = ode_g_and_policy(market, sol, tau_cloud, z_cloud)
    cap = CAP_FRAC * np.max(np.abs(vth_unc), axis=0)
    CAP_HOLDER["cap"] = cap
    out["cap_summary"] = {"min": float(cap.min()), "max": float(cap.max())}

    # ---- discretization check 1: myopic frozen-policy solve ---------------
    # A single linear evaluation of the myopic policy has an ODE-exact
    # whole-space counterpart (the per-iterate coefficient system).  The
    # interior gap between the Dirichlet FDM solve and the whole-space ODE
    # evaluation combines grid error with the Dirichlet localization gap.
    print("== discretization/localization checks ==")
    from exact_pi_ode import rk4_backward_coefficients, eval_g
    fdm0 = FDM1D(market, cap=None)
    mask = interior_mask(fdm0)
    vth_myo = np.broadcast_to(fdm0.phi[None, :, :] / GAMMA,
                              (N_TAU, N_Z, fdm0.phi.shape[1])).copy()
    g_myo_fdm = fdm0.evaluate(vth_myo)
    tau_grid = np.linspace(0.0, TAU_MAX, 1201)
    a_c, b_c, C_c = rk4_backward_coefficients(
        market, lambda t: market["lam0"] / GAMMA,
        lambda t: market["Lam"] / GAMMA, tau_grid)
    tg, zg = np.meshgrid(fdm0.tau, fdm0.z, indexing="ij")
    lg = eval_g(a_c, b_c, C_c, tau_grid, tg.reshape(-1),
                zg.reshape(-1, 1))
    g_myo_ode = np.exp(Q_EXP * R_RATE * tg.reshape(-1) + lg).reshape(N_TAU, N_Z)
    gap = rel_l2(g_myo_fdm, g_myo_ode, mask)
    out["fdm_myopic_dirichlet_vs_wholespace_ode_interior"] = gap
    print(f"  myopic eval: Dirichlet FDM vs whole-space ODE (interior): {gap:.3e}")

    # ---- discretization check 2: grid self-convergence (constrained) ------
    fine = (N_Z, N_TAU)
    N_Z, N_TAU = fine[0] // 2 + 1, fine[1] // 2 + 1
    fdm_coarse = FDM1D(market, cap=CAP_HOLDER["cap"])
    g_coarse, _ = fdm_coarse.policy_iteration(verbose=False)
    N_Z, N_TAU = fine

    # ---- constrained FDM PI: exact-iteration behavior ---------------------
    print("== constrained FDM policy iteration ==")
    fdm_c = FDM1D(market, cap=cap)
    g_list_c, vth_c = fdm_c.policy_iteration()
    g_star = g_list_c[-1]
    errs = [rel_l2(g, g_star, mask) for g in g_list_c[:-1]]
    out["fdm_constrained_iteration_errors"] = errs
    print("  exact-iteration value errors vs converged:",
          ["%.3e" % e for e in errs])

    # active-set statistics of the converged constrained policy
    active = np.abs(np.abs(vth_c) - cap[None, None, :]) < 1e-12
    out["active_fraction_all_entries"] = float(active.mean())
    out["active_fraction_any_asset"] = float(active.any(axis=2).mean())
    int_cols = mask
    out["active_fraction_any_asset_interior"] = float(
        active[:, int_cols, :].any(axis=2).mean())
    print(f"  active fraction: any-asset {out['active_fraction_any_asset']:.3f} "
          f"(interior {out['active_fraction_any_asset_interior']:.3f}), "
          f"per-entry {out['active_fraction_all_entries']:.3f}")

    # grid self-convergence: interpolate the coarse constrained solution
    from scipy.interpolate import RegularGridInterpolator as RGI
    itp_coarse = RGI((fdm_coarse.tau, fdm_coarse.z), g_coarse[-1])
    tgf, zgf = np.meshgrid(fdm_c.tau, fdm_c.z, indexing="ij")
    g_coarse_on_fine = itp_coarse(
        np.stack([tgf.reshape(-1), zgf.reshape(-1)], axis=1)).reshape(N_TAU, N_Z)
    selfconv = rel_l2(g_coarse_on_fine, g_star, mask)
    out["fdm_grid_self_convergence_rel_l2"] = selfconv
    print(f"  grid self-convergence (coarse vs fine, interior): {selfconv:.3e}")

    # how binding is the cap: constrained FDM policy vs unconstrained ODE
    # policy on the interior grid
    _, vth_ode_grid = ode_g_and_policy(market, sol, tgf.reshape(-1), zgf.reshape(-1))
    vth_ode_grid = vth_ode_grid.reshape(N_TAU, N_Z, -1)
    dpol = float(np.linalg.norm(vth_c[:, mask, :] - vth_ode_grid[:, mask, :]) /
                 np.linalg.norm(vth_ode_grid[:, mask, :]))
    out["constrained_vs_unconstrained_policy_rel_l2_interior"] = dpol
    print(f"  constrained FDM vs unconstrained ODE policy rel L2 (interior): {dpol:.3e}")

    # ---- reference interpolator for the neural experiment -----------------
    from scipy.interpolate import RegularGridInterpolator
    g_itp = RegularGridInterpolator((fdm_c.tau, fdm_c.z), g_star)
    gz_grid = np.gradient(g_star, fdm_c.dz, axis=1)
    gz_itp = RegularGridInterpolator((fdm_c.tau, fdm_c.z), gz_grid)

    def fdm_ref(tau_f, z_f):
        pts = np.stack([tau_f, z_f], axis=1)
        g = g_itp(pts)
        grad_log = gz_itp(pts) / g
        lam_x = market["lam0"][None, :] + np.outer(z_f, market["Lam"].T[0])
        vth = (lam_x + np.outer(grad_log, market["Gamma"][:, 0])) / GAMMA
        vth = np.clip(vth, -cap[None, :], cap[None, :])
        act = np.abs(vth) >= cap[None, :] * (1.0 - 1e-9)
        return g, vth, act

    heldout_data = make_heldout_data(market, fdm_ref)
    out["nn_eval"]["reference_active_fraction"] = float(
        heldout_data["act_ref"].mean())
    shared_seconds = time.perf_counter() - program_t0

    # ---- constrained PINN-PI ---------------------------------------------
    print("== constrained PINN-PI (hard traces, same Dirichlet problem) ==")
    cfg = {"hidden": 128, "depth": 3, "batch": 4096, "inner_epochs": 2500,
           "outer_iters": 8, "lr": 3e-3}
    out["nn_config"] = cfg
    nn_runs = []
    nn_loop_t0 = time.perf_counter()
    for seed in args.seeds:
        print(f"-- neural seed {seed} --")
        history, training_seconds = train_constrained_pipinn(
            market, cap, heldout_data, cfg, seed=seed, checkpoint_dir=args.checkpoint_dir)
        nn_runs.append({
            "seed": seed,
            "training_seconds": training_seconds,
            "history": history,
        })
        print(f"  seed {seed} completed in {training_seconds:.3f}s")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    nn_loop_seconds = time.perf_counter() - nn_loop_t0
    aggregate = aggregate_final_metrics(nn_runs)
    out["nn_runs"] = nn_runs
    out["nn_final_aggregate"] = aggregate
    # Expose the first seed's trajectory alongside the multi-seed results.
    out["nn_history_seed"] = nn_runs[0]["seed"]
    out["nn_history"] = nn_runs[0]["history"]
    out["runtime"] = {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "gpu_name": (torch.cuda.get_device_name(DEVICE)
                     if DEVICE.type == "cuda" else None),
        "shared_setup_fdm_eval_seconds": shared_seconds,
        "neural_runs_sum_seconds": float(sum(
            run["training_seconds"] for run in nn_runs)),
        "neural_loop_wall_seconds": nn_loop_seconds,
        "total_before_result_write_seconds": time.perf_counter() - program_t0,
    }

    print("== final neural aggregate (fixed outer-8 budget) ==")
    for metric in FINAL_NN_METRICS:
        stats = aggregate[metric]
        print(f"  {metric}: mean={stats['mean']:.9e} sd={stats['sd']:.9e} "
              f"min={stats['min']:.9e} max={stats['max']:.9e}")
    print(f"  shared setup/FDM/eval: {shared_seconds:.3f}s")
    print(f"  neural loop wall: {nn_loop_seconds:.3f}s")
    print(f"  total before result write: "
          f"{out['runtime']['total_before_result_write_seconds']:.3f}s")

    with open(args.output_dir / "constrained_benchmark_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"saved -> {args.output_dir / 'constrained_benchmark_results.json'}")


if __name__ == "__main__":
    main()
