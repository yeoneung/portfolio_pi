"""Trace-exact neural residual and radius diagnostics (not a verified certificate).

Implements exactly the architecture stated in the manuscript (eq:hard-network):

    g_hat(tau, z) = psi0(tau) * exp( tau * d_R(z) * N(tau, z) ),
    psi0(tau)     = exp((1-gamma) r tau),
    d_R(z)        = 1 - sum_i ((z_i - zbar_i) / (R eta_i))^2   (Lyapunov ellipsoid,
                                                                P = diag(1/eta_i^2)),

so positivity, the terminal condition g(0,.) = 1, and the lateral condition
g = psi0 on the ellipsoid boundary hold by construction (no terminal or
boundary penalty is trained).

For one modest state dimension L (1 and 3) and the supplied benchmark
market (N=30), the script reports the three diagnostics promised in
Section "Held-out residual and radius sensitivity":

 1. Independent validation of every frozen-policy evaluation residual: an
    importance-weighted L^p estimate on a fresh sample from the exactly
    normalized mixture
    nu = alpha * Unif(Q_R) + (1-alpha) * Unif((0,T) x D_{kappa R}),
    with p > L+2, together with a Hoeffding plug-in diagnostic at level delta
    using an empirically scanned envelope B_R, not a verified global bound.
    The inner-ellipsoid
    component concentrates validation points near the mean while retaining
    closed-form, exact importance weights.
 2. The greedy first-order-condition gap on the same sample (the benchmark is
    unconstrained, so the analytic greedy map is exact; the reported gap is a
    machine-precision check).
 3. Radius sensitivity: the same specification retrained at R' = 1.5 R and the
    sup distance of the recovered whitened exposures over a common interior
    held-out cloud D_r (r = 0.5 R, about 1.9 stationary standard deviations).

It also records the per-outer-iteration held-out value/policy error against
the converged affine ODE reference, giving the inexact-Newton saturation
diagnostic of Section "Numerical verification of the structural results".
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(os, "getuid"):
    os.getuid = lambda: 0  # type: ignore[attr-defined]
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "mpl"))

import torch
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code" / "pipinn_nemerical"
RESULTS = HERE.parent / "results" / "hard_certificate"
sys.path.insert(0, str(CODE))

from Liu_nd_pinn_homogeneous import closed_form_coefficients, solve_closed_form_ode

GAMMA = 2.0
R_RATE = 0.03
TAU_MAX = 3.0
Q_EXP = 1.0 - GAMMA

# GPU if available (float32 there: consumer Blackwell FP64 is ~1/64 rate);
# float64 on CPU.  Override with PIPINN_DEVICE=cpu / cuda.
_dev = os.environ.get("PIPINN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(_dev)
DTYPE = torch.float32 if DEVICE.type == "cuda" else torch.float64
torch.set_default_dtype(DTYPE)


def T(x):
    """numpy -> tensor on the run's device/dtype."""
    return torch.as_tensor(x, device=DEVICE, dtype=DTYPE)


# --------------------------------------------------------------------------
# hard-constraint architecture
# --------------------------------------------------------------------------
class HardValueNet(torch.nn.Module):
    """psi0(tau) * exp(tau * d_R(z) * N(tau, z)) on an axis-aligned ellipsoid.

    The inner network sees normalized coordinates (tau/T, (z-c)/h) plus the
    boundary function d_R as an extra feature, and its head is zero-initialized
    so training starts from g = psi0.  Both choices are needed to optimize the
    multiplicative hard-constraint form reliably (with raw inputs and a random
    head the residual stalls at its initialization level)."""

    def __init__(self, M: int, center, half_axes, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.M = M
        self.register_buffer("center", T(center).reshape(1, M))
        self.register_buffer("half_ax", T(half_axes).reshape(1, M))
        layers = []
        in_dim = M + 2
        for _ in range(depth):
            layers.append(torch.nn.Linear(in_dim, hidden))
            layers.append(torch.nn.Tanh())
            in_dim = hidden
        layers.append(torch.nn.Linear(in_dim, 1))
        self.net = torch.nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def d_R(self, z: torch.Tensor) -> torch.Tensor:
        u = (z - self.center) / self.half_ax
        return (1.0 - u.square().sum(dim=1, keepdim=True))

    def forward(self, tau: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u = (z - self.center) / self.half_ax
        dR = 1.0 - u.square().sum(dim=1, keepdim=True)
        N = self.net(torch.cat((tau / TAU_MAX, u, dR), dim=1))
        return torch.exp(Q_EXP * R_RATE * tau + tau * dR * N)


def g_derivatives(model, tau, z, Q):
    g = model(tau, z)
    ones = torch.ones_like(g)
    g_tau = torch.autograd.grad(g, tau, grad_outputs=ones, create_graph=True,
                                retain_graph=True)[0]
    g_z = torch.autograd.grad(g, z, grad_outputs=ones, create_graph=True,
                              retain_graph=True)[0]
    trace = torch.zeros_like(g)
    M = z.shape[1]
    for row in range(M):
        second = torch.autograd.grad(
            g_z[:, row: row + 1], z, grad_outputs=torch.ones_like(g),
            create_graph=True, retain_graph=True)[0]
        trace = trace + (second * Q[row][None, :]).sum(dim=1, keepdim=True)
    return g, g_tau, g_z, trace


def frozen_residual(model, policy, tau, z, tensors):
    """Residual of the frozen-policy reduced equation (archive convention)."""
    g, g_tau, g_z, q_trace = g_derivatives(model, tau, z, tensors["Q"])
    drift = tensors["k0"][None, :] - z @ tensors["K"].T
    lam_x = tensors["lam0"][None, :] + z @ tensors["Lam"].T
    gamma_grad = g_z @ tensors["Gamma"].T
    pol_lam = (policy * lam_x).sum(dim=1, keepdim=True)
    pol_sq = policy.square().sum(dim=1, keepdim=True)
    pol_cross = (policy * gamma_grad).sum(dim=1, keepdim=True)
    rhs = (drift * g_z).sum(dim=1, keepdim=True) + 0.5 * q_trace
    rhs = rhs + Q_EXP * (R_RATE * g + pol_lam * g - 0.5 * GAMMA * pol_sq * g + pol_cross)
    return g_tau - rhs, g


def greedy_policy(model, tau, z, tensors, create_graph=False):
    g = model(tau, z)
    g_z = torch.autograd.grad(g, z, grad_outputs=torch.ones_like(g),
                              create_graph=create_graph,
                              retain_graph=create_graph)[0]
    lam_x = tensors["lam0"][None, :] + z @ tensors["Lam"].T
    return (lam_x + (g_z / g) @ tensors["Gamma"].T) / GAMMA


def policy_used_for_evaluation(policy_source, tau, z, tensors):
    """Return the policy frozen while the corresponding value was trained.

    ``policy_source is None`` denotes the initial myopic policy.  Otherwise
    ``policy_source`` is the previous evaluated value network and the policy
    is ``G[policy_source]``.  It must not be replaced by ``G[model]`` when
    checking the residual of the newly evaluated network.
    """
    if policy_source is None:
        lam_x = tensors["lam0"][None, :] + z @ tensors["Lam"].T
        return (lam_x / GAMMA).detach()
    return greedy_policy(policy_source, tau, z, tensors).detach()


def fixed_copy(model):
    """Detached snapshot used as a frozen policy/value iterate.

    Constructing the copy before loading its state intentionally preserves the
    original experiment's torch-RNG consumption, so adding the audit trail does
    not change the initialization of the subsequent radius/dimension run.
    """
    linear_layers = [layer for layer in model.net
                     if isinstance(layer, torch.nn.Linear)]
    hidden = linear_layers[0].out_features
    depth = sum(isinstance(layer, torch.nn.Tanh) for layer in model.net)
    snapshot = HardValueNet(
        model.M,
        model.center.detach().cpu().numpy().reshape(-1),
        model.half_ax.detach().cpu().numpy().reshape(-1),
        hidden=hidden,
        depth=depth,
    )
    snapshot.load_state_dict(model.state_dict())
    snapshot.to(DEVICE).eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


# --------------------------------------------------------------------------
# sampling on the ellipsoid domain
# --------------------------------------------------------------------------
def sample_ellipsoid_uniform(n, M, center, half_axes, rng, shrink=1.0):
    """Uniform points in the (shrunk) axis-aligned ellipsoid."""
    x = rng.standard_normal((n, M))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    radius = rng.uniform(0.0, 1.0, (n, 1)) ** (1.0 / M)
    return center[None, :] + shrink * radius * x * half_axes[None, :]


def sample_boundary_shell(n, M, center, half_axes, rng, dmin=0.8, dmax=1.0):
    """Points in the outer radial shell of the ellipsoid (training only;
    the boundary layer forced by the psi0 lateral data must be resolved or
    the optimizer stalls on the near-boundary residual)."""
    x = rng.standard_normal((n, M))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    radius = rng.uniform(dmin, dmax, (n, 1))
    return center[None, :] + radius * x * half_axes[None, :]


def log_unif_density(M, half_axes, shrink=1.0):
    """log density of Unif(ellipsoid) x Unif(0, TAU_MAX)."""
    log_ball = (M / 2) * math.log(math.pi) - math.lgamma(M / 2 + 1)
    log_vol = log_ball + np.sum(np.log(shrink * half_axes))
    return -(log_vol + math.log(TAU_MAX))


def make_validation_sample(n, M, center, half_axes, alpha, inner_scale, rng):
    """Mixture sample on Q_R with exact importance weights w = mu / q.

    mu = Unif((0,T) x D_R) is the population L^p reference measure and

        q = alpha * mu + (1-alpha) * Unif((0,T) x D_{inner_scale R}).

    The second component has density inner_scale**(-M) relative to mu on the
    inner ellipsoid and zero outside.  Thus no estimated Gaussian truncation
    probability enters either sampling or the weights.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0.0 < inner_scale < 1.0:
        raise ValueError("inner_scale must lie strictly between zero and one")

    # Independent Bernoulli component labels make the returned points iid q.
    use_outer = rng.random(n) < alpha
    z = np.empty((n, M), dtype=float)
    n_outer = int(use_outer.sum())
    n_inner = n - n_outer
    z[use_outer] = sample_ellipsoid_uniform(
        n_outer, M, center, half_axes, rng)
    z[~use_outer] = sample_ellipsoid_uniform(
        n_inner, M, center, half_axes, rng, shrink=inner_scale)
    tau = rng.uniform(0.0, TAU_MAX, (n, 1))

    normalized_radius_sq = np.sum(
        ((z - center[None, :]) / half_axes[None, :]) ** 2, axis=1)
    in_inner = normalized_radius_sq < inner_scale ** 2
    inner_density_ratio = inner_scale ** (-M)
    density_ratio = alpha + (1.0 - alpha) * inner_density_ratio * in_inner
    w = 1.0 / density_ratio                 # bounded by 1/alpha
    return tau, z, w, {
        "inner_scale": inner_scale,
        "inner_volume_ratio": inner_scale ** M,
        "inner_density_ratio": inner_density_ratio,
        "outer_component_count": n_outer,
        "inner_component_count": n_inner,
        "w_max_bound": 1.0 / alpha,
        "weights_are_exact": True,
    }


# --------------------------------------------------------------------------
# ODE reference
# --------------------------------------------------------------------------
def ode_reference(market, tau_pts, z_pts):
    M = market["K"].shape[0]
    sol = solve_closed_form_ode(TAU_MAX, GAMMA, market["K"], market["xbar"],
                                market["SigmaX"], market["rho"], market["lam0"],
                                market["Lam"])
    a, b, C = closed_form_coefficients(tau_pts.reshape(-1), sol, M)
    Cx = np.einsum("bij,bj->bi", C, z_pts)
    expo = Q_EXP * R_RATE * tau_pts.reshape(-1) + a.reshape(-1)
    expo = expo + np.einsum("bi,bi->b", b, z_pts) + 0.5 * np.einsum("bi,bi->b", z_pts, Cx)
    g = np.exp(expo)
    lam_x = market["lam0"][None, :] + z_pts @ market["Lam"].T
    vth = lam_x / GAMMA + ((b + Cx) @ market["Gamma"].T) / GAMMA
    return g, vth


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_pipinn(market, domain_scale, cfg, rng, eval_cloud, tag):
    M = market["K"].shape[0]
    center = market["xbar"]
    half = domain_scale * market["eta"]
    tensors = {k: T(market[k]) for k in ("K", "k0", "Q", "Gamma", "lam0", "Lam")}

    model = HardValueNet(M, center, half, hidden=cfg["hidden"], depth=cfg["depth"])
    model.to(DEVICE)

    tau_e = T(eval_cloud[0].reshape(-1, 1))
    z_e = T(eval_cloud[1])
    g_star, vth_star = eval_cloud[2], eval_cloud[3]

    def heldout_errors():
        tau_r = tau_e.clone().requires_grad_(True)
        z_r = z_e.clone().requires_grad_(True)
        g = model(tau_r, z_r)
        vth = greedy_policy(model, tau_r, z_r, tensors)
        g_np = g.detach().cpu().numpy().reshape(-1)
        vth_np = vth.detach().cpu().numpy()
        ev = np.linalg.norm(g_np - g_star) / np.linalg.norm(g_star)
        ep = np.linalg.norm(vth_np - vth_star) / np.linalg.norm(vth_star)
        return float(ev), float(ep)

    history = []
    iterate_pairs = []
    frozen_model = None
    t0 = time.time()
    for outer in range(cfg["outer_iters"]):
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["inner_epochs"], eta_min=1e-4)
        n_shell = cfg["batch"] // 4
        for epoch in range(cfg["inner_epochs"]):
            z_np = np.concatenate([
                sample_ellipsoid_uniform(cfg["batch"] - n_shell, M, center, half, rng),
                sample_boundary_shell(n_shell, M, center, half, rng),
            ])
            tau_np = rng.uniform(0.0, TAU_MAX, (cfg["batch"], 1))
            tau = T(tau_np).requires_grad_(True)
            z = T(z_np).requires_grad_(True)
            with torch.enable_grad():
                policy = policy_used_for_evaluation(
                    frozen_model, tau, z, tensors)
            res, _ = frozen_residual(model, policy, tau, z, tensors)
            loss = res.square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        ev, ep = heldout_errors()
        history.append({"outer": outer + 1, "train_loss": float(loss.item()),
                        "heldout_value_rel_l2": ev, "heldout_policy_rel_l2": ep})
        print(f"  [{tag}] outer {outer+1:2d}: loss={loss.item():.3e} "
              f"value={ev:.3e} policy={ep:.3e} ({time.time()-t0:.0f}s)")
        # Preserve the actual policy/value pair used by this evaluation.
        # After the update, G[evaluated_snapshot] is the next policy, not the
        # frozen policy whose linear PDE the snapshot approximately solves.
        evaluated_snapshot = fixed_copy(model)
        iterate_pairs.append((evaluated_snapshot, frozen_model))
        frozen_model = evaluated_snapshot
    return model, tensors, history, iterate_pairs


def residual_on_points(model, policy_source, tensors, tau_np, z_np, chunk=8192):
    """Frozen-policy evaluation residual for one stored outer iterate."""
    out = []
    for start in range(0, len(tau_np), chunk):
        tau = T(tau_np[start:start + chunk]).requires_grad_(True)
        z = T(z_np[start:start + chunk]).requires_grad_(True)
        policy = policy_used_for_evaluation(policy_source, tau, z, tensors)
        res, _ = frozen_residual(model, policy, tau, z, tensors)
        out.append(res.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def bellman_residual_on_points(model, tensors, tau_np, z_np, chunk=8192):
    """Bellman residual B[g] = L^{G[g]}g, kept as a separate diagnostic."""
    out = []
    for start in range(0, len(tau_np), chunk):
        tau = T(tau_np[start:start + chunk]).requires_grad_(True)
        z = T(z_np[start:start + chunk]).requires_grad_(True)
        policy = greedy_policy(model, tau, z, tensors).detach()
        res, _ = frozen_residual(model, policy, tau, z, tensors)
        out.append(res.detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def foc_gap_on_points(model, tensors, tau_np, z_np, chunk=8192):
    """|gamma * vartheta_greedy - (phi + Gamma grad log g)| — analytic-greedy check."""
    gaps = []
    for start in range(0, len(tau_np), chunk):
        tau = T(tau_np[start:start + chunk]).requires_grad_(True)
        z = T(z_np[start:start + chunk]).requires_grad_(True)
        g = model(tau, z)
        g_z = torch.autograd.grad(g, z, grad_outputs=torch.ones_like(g))[0]
        lam_x = tensors["lam0"][None, :] + z @ tensors["Lam"].T
        target = lam_x + (g_z / g) @ tensors["Gamma"].T
        vth = target / GAMMA
        gap = (GAMMA * vth - target).abs().max().item()
        gaps.append(gap)
    return max(gaps)


def run_dimension(L, cfg):
    print(f"\n================ L = {L} ================")
    snap = np.load(CODE / "outputs" / "outputs" / "pipinn" / f"N_30_M{L}" /
                   "market_params.npz", allow_pickle=True)
    market = {k: np.asarray(snap[k], dtype=float)
              for k in ("K", "k0", "Q", "Gamma", "lam0", "Lam", "SigmaX",
                        "rho", "xbar", "eta")}
    M = L
    center, eta = market["xbar"], market["eta"]
    rng = np.random.default_rng(2026)

    # radii in units of the state-diffusion scale eta (= sigma_Z here):
    # computational domains R = 2.0, R' = 3.0; guarantees are examined on the
    # interior region D_r with r = 1.0 (about 1.9 stationary stds).
    R_BASE, R_LARGE, R_INT = 2.0, 3.0, 1.0

    z_int = sample_ellipsoid_uniform(20_000, M, center, R_INT * eta, rng)
    tau_int = rng.uniform(0.0, TAU_MAX, (20_000, 1))
    g_star, vth_star = ode_reference(market, tau_int, z_int)
    eval_cloud = (tau_int, z_int, g_star, vth_star)

    results = {"L": L, "p": L + 3, "config": cfg,
               "domain": {"center": center.tolist(), "eta": eta.tolist(),
                          "R_base": R_BASE, "R_large": R_LARGE,
                          "interior_r": R_INT}}
    models = {}
    for scale, key in ((R_BASE, "R"), (R_LARGE, "Rprime")):
        print(f"--- domain scale {scale} ---")
        model, tensors, hist, iterate_pairs = train_pipinn(
            market, scale, cfg, np.random.default_rng(11 + int(10 * scale)),
            eval_cloud, f"L{L} s{scale}")
        models[key] = (model, tensors, iterate_pairs)
        results[f"history_{key}"] = hist

    model_R, tensors, iterate_pairs_R = models["R"]
    model_Rp, _, _ = models["Rprime"]

    # ---- diagnostic 1: independent frozen-policy L^p residual validation
    p = L + 3          # p > L + 2
    alpha = 0.5
    inner_scale = 0.5
    delta = 0.05
    M_val = 50_000
    M_env = 200_000
    num_outer = len(iterate_pairs_R)
    half_axes = R_BASE * eta
    q_r_volume = math.exp(-log_unif_density(M, half_axes))
    volume_lp_scale = q_r_volume ** (1.0 / p)
    validation_iterations = []
    final_validation_cloud = None

    for outer_index, (evaluated_model, policy_source) in enumerate(
            iterate_pairs_R, start=1):
        # Every iterate gets a fresh sample independent of training and of the
        # samples used for all other iterates.
        val_rng = np.random.default_rng(
            np.random.SeedSequence([90210, L, outer_index]))
        tau_v, z_v, w_v, mix_info = make_validation_sample(
            M_val, M, center, half_axes, alpha, inner_scale, val_rng)
        res_v = residual_on_points(
            evaluated_model, policy_source, tensors, tau_v, z_v)

        # This is an empirical envelope scan only.  It remains deliberately
        # distinct from a verified interval/Lipschitz envelope.
        env_rng = np.random.default_rng(
            np.random.SeedSequence([777, L, outer_index]))
        z_env = sample_ellipsoid_uniform(
            M_env, M, center, half_axes, env_rng)
        tau_env = env_rng.uniform(0.0, TAU_MAX, (M_env, 1))
        res_env = residual_on_points(
            evaluated_model, policy_source, tensors, tau_env, z_env)
        empirical_S_R = 1.5 * float(np.max(np.abs(res_env)))

        mean_wp = float(np.mean(w_v * np.abs(res_v) ** p))
        w_max = mix_info["w_max_bound"]
        empirical_variable_envelope = w_max * empirical_S_R ** p
        standalone_hoeff = empirical_variable_envelope * math.sqrt(
            math.log(1.0 / delta) / (2 * M_val))
        simultaneous_hoeff = empirical_variable_envelope * math.sqrt(
            math.log(num_outer / delta) / (2 * M_val))
        lp_point = mean_wp ** (1.0 / p)
        lp_standalone = (mean_wp + standalone_hoeff) ** (1.0 / p)
        lp_simultaneous = (mean_wp + simultaneous_hoeff) ** (1.0 / p)
        source_label = ("myopic"
                        if policy_source is None
                        else f"greedy_from_outer_{outer_index - 1}")
        iteration_result = {
            "outer": outer_index,
            "evaluated_policy_source": source_label,
            "residual_operator": "frozen_policy_evaluation",
            "M": M_val,
            "sample_weight_mean": float(np.mean(w_v)),
            "sample_weight_max": float(np.max(w_v)),
            "empirical_residual_envelope_S_R": empirical_S_R,
            "empirical_variable_envelope_w_abs_r_p": empirical_variable_envelope,
            "envelope_scan_points": M_env,
            "normalized_Lp_point_estimate": lp_point,
            "normalized_Lp_standalone_95_bound_if_envelope_holds": lp_standalone,
            "normalized_Lp_simultaneous_95_bound_if_envelopes_hold": lp_simultaneous,
            "Lebesgue_Lp_point_estimate": volume_lp_scale * lp_point,
            "Lebesgue_Lp_standalone_95_bound_if_envelope_holds": (
                volume_lp_scale * lp_standalone),
            "Lebesgue_Lp_simultaneous_95_bound_if_envelopes_hold": (
                volume_lp_scale * lp_simultaneous),
            "max_abs_residual_on_validation": float(np.max(np.abs(res_v))),
        }
        validation_iterations.append(iteration_result)
        print(
            f"  validation outer {outer_index}: frozen L^{p} "
            f"point={lp_point:.4e} standalone95={lp_standalone:.4e} "
            f"simultaneous95={lp_simultaneous:.4e} S_R={empirical_S_R:.3e}")
        if outer_index == num_outer:
            final_validation_cloud = (tau_v, z_v, w_v)

    final_validation = validation_iterations[-1]
    tau_v, z_v, w_v = final_validation_cloud
    final_bellman_residual = bellman_residual_on_points(
        model_R, tensors, tau_v, z_v)
    bellman_lp_point = float(
        np.mean(w_v * np.abs(final_bellman_residual) ** p) ** (1.0 / p))
    results["validation"] = {
        "target_measure": "normalized uniform measure on Q_R",
        "Q_R_Lebesgue_volume": q_r_volume,
        "M_per_outer_iterate": M_val,
        "p": p,
        "familywise_delta": delta,
        "number_of_validated_outer_iterates": num_outer,
        "alpha_uniform": alpha,
        "inner_ellipsoid_scale": inner_scale,
        "mixture": (
            "0.5 Unif((0,T)xD_R) + 0.5 Unif((0,T)xD_{0.5R})"),
        "importance_weights_are_exact": True,
        "importance_weight_formula": (
            "1 / [alpha + (1-alpha)*inner_scale^(-L)*1{z in D_inner}]"),
        "w_max": mix_info["w_max_bound"],
        "inner_volume_ratio": mix_info["inner_volume_ratio"],
        "empirical_envelope_is_not_verified": True,
        "iterations": validation_iterations,
        # Convenience aliases below always refer to the final *frozen-policy*
        # pair for downstream result readers.
        "weighted_Lp_point_estimate": final_validation[
            "normalized_Lp_point_estimate"],
        "weighted_Lp_high_prob_bound": final_validation[
            "normalized_Lp_standalone_95_bound_if_envelope_holds"],
        "weighted_Lp_simultaneous_high_prob_bound": final_validation[
            "normalized_Lp_simultaneous_95_bound_if_envelopes_hold"],
        "envelope_B_R": final_validation[
            "empirical_residual_envelope_S_R"],
        "max_abs_residual_on_validation": final_validation[
            "max_abs_residual_on_validation"],
        "final_bellman_Lp_point_estimate_separate_diagnostic": bellman_lp_point,
    }
    print(
        f"  final comparison: frozen point="
        f"{final_validation['normalized_Lp_point_estimate']:.4e}; "
        f"Bellman point={bellman_lp_point:.4e}")

    # ---- diagnostic 2: greedy gap (analytic unconstrained maximizer)
    foc = foc_gap_on_points(model_R, tensors, tau_v[:20_000], z_v[:20_000])
    results["greedy_gap"] = {
        "constraint_set": "unbounded (A = R^N); greedy map analytic",
        "max_foc_residual": foc,
    }
    print(f"  greedy FOC residual: {foc:.2e}")

    # ---- diagnostic 3: radius sensitivity on the common interior cloud
    tau_t = T(tau_int).requires_grad_(True)
    z_t = T(z_int).requires_grad_(True)
    vth_R = greedy_policy(model_R, tau_t, z_t, tensors).detach().cpu().numpy()
    tau_t2 = T(tau_int).requires_grad_(True)
    z_t2 = T(z_int).requires_grad_(True)
    vth_Rp = greedy_policy(model_Rp, tau_t2, z_t2, tensors).detach().cpu().numpy()
    sup_diff = float(np.max(np.linalg.norm(vth_R - vth_Rp, axis=1)))
    mean_diff = float(np.mean(np.linalg.norm(vth_R - vth_Rp, axis=1)))
    sup_pol = float(np.max(np.linalg.norm(vth_star, axis=1)))
    results["radius_sensitivity"] = {
        "sup_policy_diff_interior": sup_diff,
        "mean_policy_diff_interior": mean_diff,
        "sup_reference_policy_norm": sup_pol,
        "relative_sup": sup_diff / sup_pol,
    }
    print(f"  radius sensitivity: sup|pi_R' - pi_R| = {sup_diff:.4e} "
          f"(rel {sup_diff / sup_pol:.3e})")

    # ---- Newton saturation from the per-outer history
    ev = [h["heldout_value_rel_l2"] for h in results["history_R"]]
    floor = min(ev)
    sat = next((i for i, e in enumerate(ev) if e <= 3 * floor), len(ev) - 1)
    results["newton_saturation"] = {"heldout_value_errors": ev,
                                    "evaluation_floor": floor,
                                    "saturation_outer_index": sat + 1}
    return results


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    all_res = {}
    cfgs = {
        1: {"hidden": 128, "depth": 3, "batch": 4096, "inner_epochs": 2500,
            "outer_iters": 8, "lr": 3e-3},
        3: {"hidden": 128, "depth": 3, "batch": 4096, "inner_epochs": 3000,
            "outer_iters": 8, "lr": 3e-3},
    }
    for L in (1, 3):
        all_res[f"L{L}"] = run_dimension(L, cfgs[L])
        with open(RESULTS / "hard_certificate_results.json", "w") as fh:
            json.dump(all_res, fh, indent=2)
    print(f"\nsaved -> {RESULTS / 'hard_certificate_results.json'}")


if __name__ == "__main__":
    main()
