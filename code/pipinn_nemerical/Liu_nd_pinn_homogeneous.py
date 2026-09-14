"""Homogeneous PINN for the multi-state Kim--Omberg benchmark.

The solver uses a Tanh MLP, a soft terminal penalty, and a closed-form
Riccati reference under the CRRA homogeneity reduction

    V(tau, w, z) = w**(1-gamma)/(1-gamma) * g(tau, z).

The network therefore learns the positive scalar multiplier ``g`` from
``(tau, z)`` only.  The unconstrained normalized policy is

    vartheta = theta / w
             = [lambda(z) + Gamma grad_z log g] / gamma.

The market generator's joint-correlation blocks are converted to canonical
identity-shock cross correlation in ``joint_market_setup_dirichlet.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mor-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F

from joint_market_setup_dirichlet import (
    JointMarketParams,
    canonical_cross_correlation,
    generate_joint_market_params,
)


METHOD_NAME = "pinn_homogeneous"
SCHEMA_NAME = "homogeneous_canonical_v1"
PUBLICATION_SLICE_SCHEMA = "homogeneous_publication_slice_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Homogeneous PINN for the multidimensional Kim--Omberg PDE"
    )

    problem = parser.add_argument_group("problem")
    problem.add_argument("--n-assets", type=int, default=30)
    problem.add_argument("--m-states", type=int, default=10)
    problem.add_argument("--seed", type=int, default=12)
    problem.add_argument("--market-seed", type=int, default=None)
    problem.add_argument("--tau-max", type=float, default=3.0)
    problem.add_argument("--gamma", type=float, default=2.0)
    problem.add_argument("--r", type=float, default=0.03)
    problem.add_argument("--lambda0", type=float, default=0.1)
    problem.add_argument("--x-range-scale", type=float, default=1.0)
    problem.add_argument("--dirichlet-concentration", type=float, default=1.0)
    problem.add_argument("--alpha-scale", type=float, default=0.25)

    network = parser.add_argument_group("network")
    network.add_argument("--value-hidden", type=int, default=256)
    network.add_argument("--value-depth", type=int, default=3)
    network.add_argument("--g-floor", type=float, default=1e-6)

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=40_000)
    training.add_argument("--batch-size", type=int, default=10_000)
    training.add_argument("--terminal-frac", type=float, default=0.5)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--resample-every", type=int, default=2_000)
    training.add_argument("--w-terminal", type=float, default=20.0)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--print-every", type=int, default=1_000)
    training.add_argument(
        "--lr-schedule", choices=("plateau", "fixed"), default="plateau"
    )
    training.add_argument("--scheduler-patience", type=int, default=3_000)
    training.add_argument("--scheduler-factor", type=float, default=0.5)
    training.add_argument("--scheduler-min-lr", type=float, default=1e-5)

    heldout = parser.add_argument_group("heldout monitoring and diagnostic best")
    heldout.add_argument("--val-points", type=int, default=50_000)
    heldout.add_argument("--val-terminal-points", type=int, default=10_000)
    heldout.add_argument("--val-every", type=int, default=25)
    heldout.add_argument("--val-chunk", type=int, default=4_096)
    heldout.add_argument("--pres-target", type=float, default=None)

    evaluation = parser.add_argument_group("evaluation and plots")
    evaluation.add_argument("--test-points", type=int, default=100_000)
    evaluation.add_argument("--eval-margin", type=float, default=0.10)
    evaluation.add_argument("--eval-chunk", type=int, default=4_096)
    evaluation.add_argument("--n-tau", type=int, default=100)
    evaluation.add_argument("--n-x", type=int, default=100)
    evaluation.add_argument("--max-plot-assets", type=int, default=1)
    evaluation.add_argument("--figure-dpi", type=int, default=300)
    evaluation.add_argument(
        "--figure-output-mode",
        choices=("separate", "combined", "both"),
        default="separate",
        help=(
            "value/policy figure layout: individual panels, composite panels, "
            "or both"
        ),
    )
    evaluation.add_argument("--skip-figures", action="store_true")
    evaluation.add_argument("--skip-eval", action="store_true")

    runtime = parser.add_argument_group("runtime and output")
    runtime.add_argument("--device", type=str, default="auto")
    runtime.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    runtime.add_argument("--output-root", type=Path, default=Path("outputs/pinn_homogeneous"))
    runtime.add_argument("--weight-root", type=Path, default=Path("weights/pinn_homogeneous"))
    runtime.add_argument("--run-tag", type=str, default="auto")
    runtime.add_argument("--eval-only", action="store_true")
    runtime.add_argument("--checkpoint", type=Path, default=None)
    runtime.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the exact existing run-tag directories in place",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    finite_fields = (
        "tau_max",
        "gamma",
        "r",
        "lambda0",
        "x_range_scale",
        "dirichlet_concentration",
        "alpha_scale",
        "g_floor",
        "terminal_frac",
        "lr",
        "w_terminal",
        "grad_clip",
        "eval_margin",
    )
    for name in finite_fields:
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"{name.replace('_', '-')} must be finite")
    if args.pres_target is not None and not math.isfinite(args.pres_target):
        raise ValueError("pres-target must be finite when supplied")
    if args.seed < 0 or (args.market_seed is not None and args.market_seed < 0):
        raise ValueError("seed and market-seed must be nonnegative")
    if args.n_assets <= 0 or args.m_states <= 0:
        raise ValueError("n-assets and m-states must be positive")
    if args.tau_max <= 0.0:
        raise ValueError("tau-max must be positive")
    if args.gamma <= 0.0 or math.isclose(args.gamma, 1.0):
        raise ValueError("gamma must be positive and different from one")
    if args.value_hidden <= 0 or args.value_depth <= 0:
        raise ValueError("value-hidden and value-depth must be positive")
    if args.x_range_scale <= 0.0 or args.dirichlet_concentration <= 0.0:
        raise ValueError("x-range-scale and dirichlet-concentration must be positive")
    if args.alpha_scale < 0.0:
        raise ValueError("alpha-scale must be nonnegative")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 < args.terminal_frac <= 1.0:
        raise ValueError("terminal-frac must lie in (0, 1]")
    if args.lr <= 0.0 or args.w_terminal < 0.0:
        raise ValueError("lr must be positive and w-terminal nonnegative")
    if args.resample_every < 0:
        raise ValueError("resample-every must be nonnegative")
    if args.val_points <= 0 or args.val_terminal_points <= 0 or args.val_every <= 0:
        raise ValueError("validation sizes and val-every must be positive")
    if not 0.0 <= args.eval_margin < 0.5:
        raise ValueError("eval-margin must lie in [0, 0.5)")
    if not 0.0 < args.g_floor < 1.0:
        raise ValueError("g-floor must lie in (0, 1) so g(0,z)=1 remains attainable")
    if args.grad_clip < 0.0 or args.print_every <= 0:
        raise ValueError("grad-clip must be nonnegative and print-every positive")
    if args.lr_schedule == "plateau":
        if args.scheduler_patience < 0:
            raise ValueError("scheduler-patience must be nonnegative")
        if not 0.0 < args.scheduler_factor < 1.0:
            raise ValueError("scheduler-factor must lie in (0, 1)")
        if not 0.0 < args.scheduler_min_lr <= args.lr:
            raise ValueError("scheduler-min-lr must lie in (0, lr]")
    if args.val_chunk <= 0 or args.eval_chunk <= 0 or args.test_points <= 0:
        raise ValueError("validation/evaluation chunk sizes and test-points must be positive")
    if (
        args.n_tau < 2
        or args.n_x < 2
        or args.max_plot_assets <= 0
        or args.figure_dpi <= 0
    ):
        raise ValueError(
            "plot grid sizes must be at least two; max-plot-assets and figure-dpi "
            "must be positive"
        )
    if args.pres_target is not None and args.pres_target <= 0.0:
        raise ValueError("pres-target must be positive when supplied")
    if args.eval_only and args.checkpoint is None:
        raise ValueError("eval-only requires --checkpoint")


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {spec}")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def automatic_run_tag(args: argparse.Namespace) -> str:
    """Return a readable fallback tag for direct (non-launcher) runs.

    ``run_parallel.sh`` supplies its own tag and appends only hyperparameters
    explicitly written on a job line.  A standalone argparse invocation does
    not retain which values were explicitly supplied, so its safe readable
    fallback is the dimension pair alone.  Full configuration/source hashes
    remain in checkpoints and provenance files rather than directory names.
    """
    return f"N_{args.n_assets}_M{args.m_states}"


def prepare_run_directories(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    tag = automatic_run_tag(args) if args.run_tag == "auto" else args.run_tag
    if (
        not tag
        or len(tag) > 240
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in tag)
    ):
        raise ValueError(
            "run-tag must be 1--240 ASCII letters, digits, underscores, or hyphens"
        )
    output_dir = args.output_root / tag
    weight_dir = args.weight_root / tag

    # Eval-only always writes to an isolated child directory.  Creation is
    # deliberately deferred until after checkpoint/config/market validation,
    # so a rejected evaluation leaves no misleading output directory behind.
    if args.eval_only:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = output_dir / f"eval_only_{timestamp}"
        return output_dir, weight_dir, tag

    run_paths = []
    seen_paths = set()
    for path, root in (
        (output_dir, args.output_root),
        (weight_dir, args.weight_root),
    ):
        path_key = str(path.absolute())
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"refusing invalid run directory: {path}")
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        if resolved_path.parent != resolved_root or resolved_path.name != tag:
            raise ValueError(f"refusing unexpected run path: {path}")
        run_paths.append(path)

    existing_files = any(path.exists() and any(path.iterdir()) for path in run_paths)
    if existing_files and not args.eval_only and not args.overwrite:
        raise FileExistsError(
            f"run directory already contains files for tag={tag!r}; "
            "choose another --run-tag or pass --overwrite explicitly"
        )
    if existing_files and args.overwrite:
        # Validation above is deliberately complete before the first removal,
        # so an invalid second path cannot leave a half-replaced direct run.
        for path in run_paths:
            if not path.exists() or not any(path.iterdir()):
                continue
            shutil.rmtree(path)
            print(f"Replacing existing run directory in place: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, weight_dir, tag


def print_market_report(params: JointMarketParams, rho: np.ndarray) -> None:
    diagnostics = params.diag["joint_corr"]
    rho_norm = float(np.linalg.svd(rho, compute_uv=False)[0])
    print(f"\n[Joint market] assets={params.n}, states={params.k}")
    print(
        "  raw joint C: "
        f"cond={diagnostics['cond']:.3f}, "
        f"min_eig={diagnostics['min_eig']:.3e}"
    )
    print(f"  canonical rho spectral norm={rho_norm:.6f}")
    print(
        "  convention=canonical_identity_shocks "
        "(SigmaX=diag(eta), Q=SigmaX SigmaX^T)"
    )


def build_market(
    args: argparse.Namespace,
) -> Tuple[JointMarketParams, Dict[str, np.ndarray]]:
    market_seed = args.seed if args.market_seed is None else args.market_seed
    params = generate_joint_market_params(
        n=args.n_assets,
        k=args.m_states,
        seed=market_seed,
        sample_alpha=True,
        alpha_dist="dirichlet",
        dirichlet_concentration=args.dirichlet_concentration,
        alpha_scale=args.alpha_scale,
    )
    if any(value is None for value in (params.K, params.theta, params.SigmaZ, params.alpha)):
        raise RuntimeError("market generator did not return the required OU/loadings arrays")

    # Updated numerical convention: C supplies a feasible canonical cross
    # block, while the HJB itself has identity within-block Brownian shocks.
    rho = canonical_cross_correlation(params)
    sigma_x = np.asarray(params.SigmaZ, dtype=float)
    K = np.asarray(params.K, dtype=float)
    xbar = np.asarray(params.theta, dtype=float)
    Lam = np.asarray(params.alpha, dtype=float)
    Q = sigma_x @ sigma_x.T
    Gamma = rho @ sigma_x.T
    k0 = K @ xbar
    lam0 = np.full(args.n_assets, args.lambda0, dtype=float)
    eta = np.asarray(params.eta, dtype=float)
    x_min = xbar - args.x_range_scale * eta
    x_max = xbar + args.x_range_scale * eta

    arrays = {
        "K": K,
        "xbar": xbar,
        "SigmaX": sigma_x,
        "rho": rho,
        "Lam": Lam,
        "Q": Q,
        "Gamma": Gamma,
        "k0": k0,
        "lam0": lam0,
        "eta": eta,
        "X_min": x_min,
        "X_max": x_max,
        "market_seed": np.array([market_seed], dtype=np.int64),
        "market_schema": np.array([SCHEMA_NAME]),
        "rho_spectral_norm": np.array([np.linalg.norm(rho, ord=2)], dtype=float),
    }
    return params, arrays


def save_market_snapshot(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def market_fingerprint(arrays: Dict[str, np.ndarray]) -> str:
    """Hash the exact HJB market primitives stored in a checkpoint."""
    digest = hashlib.sha256()
    for name in ("K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0", "lam0", "X_min", "X_max"):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def solve_closed_form_ode(
    T: float,
    gamma: float,
    K: np.ndarray,
    xbar: np.ndarray,
    SigmaX: np.ndarray,
    rho: np.ndarray,
    lam0: np.ndarray,
    Lam: np.ndarray,
):
    """Solve the affine-quadratic closed-form Riccati system accurately."""
    M = K.shape[0]
    Q = SigmaX @ SigmaX.T
    Gamma = rho @ SigmaX.T
    k0 = K @ xbar
    alpha = (1.0 - gamma) / gamma

    def rhs(_tau: float, y: np.ndarray) -> np.ndarray:
        b = y[1 : 1 + M]
        C = y[1 + M :].reshape(M, M)
        C = 0.5 * (C + C.T)
        A = Lam + Gamma @ C
        m = lam0 + Gamma @ b
        dC = -(K.T @ C + C @ K) + C @ Q @ C + alpha * (A.T @ A)
        db = C @ k0 - K.T @ b + C @ Q @ b + alpha * (A.T @ m)
        da = (
            k0.T @ b
            + 0.5 * np.trace(Q @ C)
            + 0.5 * (b.T @ Q @ b)
            + 0.5 * alpha * (m.T @ m)
        )
        return np.concatenate(([da], db, dC.reshape(-1)))

    y0 = np.zeros(1 + M + M * M, dtype=float)
    t_eval = np.linspace(0.0, T, 8_001)
    solution = solve_ivp(
        rhs,
        (0.0, T),
        y0,
        t_eval=t_eval,
        dense_output=True,
        method="RK45",
        rtol=1e-12,
        atol=1e-14,
    )
    if not solution.success:
        raise RuntimeError(f"closed-form ODE solve failed: {solution.message}")
    return solution


def closed_form_coefficients(tau: np.ndarray, solution, M: int):
    tau = np.asarray(tau, dtype=float).reshape(-1)
    if solution.sol is not None:
        values = np.asarray(solution.sol(tau), dtype=float)
    else:
        values = np.vstack(
            [np.interp(tau, solution.t, solution.y[row]) for row in range(solution.y.shape[0])]
        )
    a = values[0]
    b = values[1 : 1 + M].T
    C = values[1 + M :].T.reshape(-1, M, M)
    C = 0.5 * (C + np.swapaxes(C, 1, 2))
    return a, b, C


def closed_form_batch(
    tau: np.ndarray,
    x: np.ndarray,
    solution,
    gamma: float,
    r: float,
    lam0: np.ndarray,
    Lam: np.ndarray,
    Gamma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    M = x.shape[1]
    a, b, C = closed_form_coefficients(tau, solution, M)
    Cx = np.einsum("bij,bj->bi", C, x)
    exponent = (1.0 - gamma) * r * tau + a + np.einsum("bi,bi->b", b, x)
    exponent += 0.5 * np.einsum("bi,bi->b", x, Cx)
    g = np.exp(exponent)
    grad_log_g = b + Cx
    lam_x = lam0[None, :] + x @ Lam.T
    myopic = lam_x / gamma
    hedging = (grad_log_g @ Gamma.T) / gamma
    vartheta = myopic + hedging
    return g, vartheta, myopic, hedging


class HomogeneousValueNet(nn.Module):
    """Tanh MLP mapping ``(tau, z)`` to a positive multiplier ``g``."""

    def __init__(self, M: int, hidden: int = 256, depth: int = 3, g_floor: float = 1e-6):
        super().__init__()
        self.M = M
        self.g_floor = float(g_floor)
        layers = []
        in_dim = M + 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raw_g = self.net(torch.cat((tau, x), dim=1))
        return F.softplus(raw_g) + self.g_floor


def sample_interior(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    X_min: np.ndarray,
    X_max: np.ndarray,
    tau_max: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M = len(X_min)
    eps_tau = min(1e-3, 0.01 * tau_max)
    if rng is None:
        tau = eps_tau + torch.rand(n, 1, device=device, dtype=dtype) * (tau_max - eps_tau)
        low = torch.as_tensor(X_min, device=device, dtype=dtype)
        high = torch.as_tensor(X_max, device=device, dtype=dtype)
        x = low + torch.rand(n, M, device=device, dtype=dtype) * (high - low)
    else:
        tau_np = rng.uniform(eps_tau, tau_max, size=(n, 1))
        x_np = rng.uniform(X_min, X_max, size=(n, M))
        tau = torch.as_tensor(tau_np, device=device, dtype=dtype)
        x = torch.as_tensor(x_np, device=device, dtype=dtype)
    tau.requires_grad_(True)
    x.requires_grad_(True)
    return tau, x


def sample_terminal(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    X_min: np.ndarray,
    X_max: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M = len(X_min)
    if rng is None:
        low = torch.as_tensor(X_min, device=device, dtype=dtype)
        high = torch.as_tensor(X_max, device=device, dtype=dtype)
        x = low + torch.rand(n, M, device=device, dtype=dtype) * (high - low)
    else:
        x_np = rng.uniform(X_min, X_max, size=(n, M))
        x = torch.as_tensor(x_np, device=device, dtype=dtype)
    tau = torch.zeros(n, 1, device=device, dtype=dtype)
    return tau, x


def g_derivatives(
    model: HomogeneousValueNet,
    tau: torch.Tensor,
    x: torch.Tensor,
    Q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return g, g_tau, grad_x g, and tr(Q Hess_x g)."""
    g = model(tau, x)
    ones = torch.ones_like(g)
    g_tau = torch.autograd.grad(
        g, tau, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]
    g_x = torch.autograd.grad(
        g, x, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    q_hessian_trace = torch.zeros_like(g)
    M = x.shape[1]
    for row in range(M):
        second_row = torch.autograd.grad(
            g_x[:, row : row + 1],
            x,
            grad_outputs=torch.ones_like(g_x[:, row : row + 1]),
            create_graph=True,
            retain_graph=True,
        )[0]
        q_hessian_trace = q_hessian_trace + torch.sum(
            second_row * Q[row : row + 1, :], dim=1, keepdim=True
        )
    return g, g_tau, g_x, q_hessian_trace


def homogeneous_hjb_residual(
    model: HomogeneousValueNet,
    tau: torch.Tensor,
    x: torch.Tensor,
    gamma: float,
    r: float,
    K: torch.Tensor,
    k0: torch.Tensor,
    Q: torch.Tensor,
    Gamma: torch.Tensor,
    lam0: torch.Tensor,
    Lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual ``g_tau - RHS`` of the reduced nonlinear HJB."""
    g, g_tau, g_x, q_hessian_trace = g_derivatives(model, tau, x, Q)
    drift = k0[None, :] - x @ K.T
    lam_x = lam0[None, :] + x @ Lam.T
    cross = g_x @ Gamma.T
    driver_numerator = lam_x * g + cross

    rhs = torch.sum(drift * g_x, dim=1, keepdim=True)
    rhs = rhs + 0.5 * q_hessian_trace
    rhs = rhs + (1.0 - gamma) * r * g
    rhs = rhs + ((1.0 - gamma) / (2.0 * gamma)) * torch.sum(
        driver_numerator.square(), dim=1, keepdim=True
    ) / g
    return g_tau - rhs, g


def compute_normalized_policy(
    model: HomogeneousValueNet,
    tau: torch.Tensor,
    x: torch.Tensor,
    gamma: float,
    Gamma: torch.Tensor,
    lam0: torch.Tensor,
    Lam: torch.Tensor,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = model(tau, x)
    g_x = torch.autograd.grad(
        g,
        x,
        grad_outputs=torch.ones_like(g),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    lam_x = lam0[None, :] + x @ Lam.T
    myopic = lam_x / gamma
    hedging = (g_x @ Gamma.T) / (gamma * g)
    vartheta = myopic + hedging
    return g, vartheta, myopic, hedging


def tensor_market(
    arrays: Dict[str, np.ndarray], device: torch.device, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    names = ("K", "k0", "Q", "Gamma", "lam0", "Lam")
    return {
        name: torch.as_tensor(arrays[name], device=device, dtype=dtype)
        for name in names
    }


def heldout_residual_score(
    model: HomogeneousValueNet,
    interior: Tuple[torch.Tensor, torch.Tensor],
    terminal: Tuple[torch.Tensor, torch.Tensor],
    gamma: float,
    r: float,
    tensors: Dict[str, torch.Tensor],
    chunk: int,
) -> Dict[str, float]:
    """Evaluate the fixed unseen collocation set without closed-form leakage."""
    was_training = model.training
    model.eval()
    tau_all, x_all = interior
    residual_squared_sum = 0.0
    residual_count = 0
    with torch.enable_grad():
        for start in range(0, tau_all.shape[0], chunk):
            tau = tau_all[start : start + chunk].detach().clone().requires_grad_(True)
            x = x_all[start : start + chunk].detach().clone().requires_grad_(True)
            residual, _ = homogeneous_hjb_residual(
                model,
                tau,
                x,
                gamma,
                r,
                tensors["K"],
                tensors["k0"],
                tensors["Q"],
                tensors["Gamma"],
                tensors["lam0"],
                tensors["Lam"],
            )
            residual_squared_sum += float(residual.detach().square().sum().cpu())
            residual_count += residual.numel()

    tau_terminal, x_terminal = terminal
    terminal_squared_sum = 0.0
    terminal_count = 0
    with torch.no_grad():
        for start in range(0, tau_terminal.shape[0], chunk):
            g_terminal = model(
                tau_terminal[start : start + chunk], x_terminal[start : start + chunk]
            )
            error = g_terminal - 1.0
            terminal_squared_sum += float(error.square().sum().cpu())
            terminal_count += error.numel()

    pde_rms = math.sqrt(residual_squared_sum / max(residual_count, 1))
    terminal_rms = math.sqrt(terminal_squared_sum / max(terminal_count, 1))
    if not (math.isfinite(pde_rms) and math.isfinite(terminal_rms)):
        raise FloatingPointError(
            f"non-finite heldout residual: pde_rms={pde_rms}, "
            f"terminal_rms={terminal_rms}"
        )
    if was_training:
        model.train()
    return {
        "pde_rms": pde_rms,
        "terminal_rms": terminal_rms,
        "p_res": pde_rms + terminal_rms,
    }


def save_checkpoint(
    path: Path,
    model: HomogeneousValueNet,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    epoch: int,
    heldout: Dict[str, float],
    args: argparse.Namespace,
    tag: str,
    market_hash: str,
) -> None:
    payload = {
        "method": METHOD_NAME,
        "schema": SCHEMA_NAME,
        "run_tag": tag,
        "market_fingerprint": market_hash,
        "epoch": int(epoch),
        "heldout": dict(heldout),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "config": json_ready(vars(args)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def validate_eval_checkpoint(
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    expected_market_hash: str,
) -> None:
    if checkpoint.get("method") != METHOD_NAME:
        raise ValueError(
            f"checkpoint method mismatch: expected {METHOD_NAME!r}, "
            f"found {checkpoint.get('method')!r}"
        )
    if checkpoint.get("schema") != SCHEMA_NAME:
        raise ValueError(
            f"checkpoint schema mismatch: expected {SCHEMA_NAME!r}, "
            f"found {checkpoint.get('schema')!r}"
        )
    if checkpoint.get("market_fingerprint") != expected_market_hash:
        raise ValueError("checkpoint market fingerprint does not match current market parameters")
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint does not contain its training config")
    critical = (
        "n_assets",
        "m_states",
        "market_seed",
        "tau_max",
        "gamma",
        "r",
        "lambda0",
        "x_range_scale",
        "dirichlet_concentration",
        "alpha_scale",
        "value_hidden",
        "value_depth",
        "g_floor",
        "dtype",
    )
    mismatches = []
    current = vars(args)
    for key in critical:
        left = saved.get(key)
        right = current.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-14)
        else:
            equal = left == right
        if not equal:
            mismatches.append(f"{key}: checkpoint={left!r}, current={right!r}")
    if mismatches:
        raise ValueError("eval-only critical config mismatch:\n  " + "\n  ".join(mismatches))


def write_history_csv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    fieldnames: Optional[Iterable[str]] = None,
) -> None:
    rows = list(rows)
    fields = list(fieldnames) if fieldnames is not None else []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_pinn(
    model: HomogeneousValueNet,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    output_dir: Path,
    weight_dir: Path,
    tag: str,
) -> Tuple[list[Dict[str, float]], list[Dict[str, float]], Dict[str, Any]]:
    market_hash = market_fingerprint(arrays)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            min_lr=args.scheduler_min_lr,
        )

    terminal_batch = max(1, int(round(args.batch_size * args.terminal_frac)))
    validation_seed = int(arrays["market_seed"][0]) * 1_000_003 + 911
    validation_rng = np.random.default_rng(validation_seed)
    val_interior = sample_interior(
        args.val_points,
        device,
        dtype,
        arrays["X_min"],
        arrays["X_max"],
        args.tau_max,
        rng=validation_rng,
    )
    val_terminal = sample_terminal(
        args.val_terminal_points,
        device,
        dtype,
        arrays["X_min"],
        arrays["X_max"],
        rng=validation_rng,
    )

    train_history: list[Dict[str, float]] = []
    val_history: list[Dict[str, float]] = []
    best_score = float("inf")
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_path = weight_dir / "value_net_best_heldout.pt"
    final_path = weight_dir / "value_net_final.pt"

    initial_metrics = heldout_residual_score(
        model,
        val_interior,
        val_terminal,
        args.gamma,
        args.r,
        tensors,
        args.val_chunk,
    )
    best_score = initial_metrics["p_res"]
    best_metrics = initial_metrics
    val_history.append({"epoch": 0, **initial_metrics})
    save_checkpoint(
        best_path,
        model,
        optimizer,
        scheduler,
        0,
        initial_metrics,
        args,
        tag,
        market_hash,
    )

    print("\n" + "=" * 72)
    print(
        f"Training homogeneous PINN: N={args.n_assets}, M={args.m_states}, "
        f"input_dim={args.m_states + 1}"
    )
    print(
        f"initial heldout: pde_rms={initial_metrics['pde_rms']:.3e}, "
        f"terminal_rms={initial_metrics['terminal_rms']:.3e}, "
        f"p_res={initial_metrics['p_res']:.3e}"
    )
    print("=" * 72)

    tau_train: Optional[torch.Tensor] = None
    x_train: Optional[torch.Tensor] = None
    tau_terminal: Optional[torch.Tensor] = None
    x_terminal: Optional[torch.Tensor] = None
    stopped_early = bool(
        args.pres_target is not None and initial_metrics["p_res"] <= args.pres_target
    )
    if stopped_early:
        print(
            "Initial model already reaches p_res target: "
            f"{initial_metrics['p_res']:.3e} <= {args.pres_target:.3e}; "
            "skipping optimizer steps"
        )
    last_validation_epoch = 0
    maximum_epoch = 0 if stopped_early else args.epochs

    for epoch in range(1, maximum_epoch + 1):
        # Correct block-start resampling.  resample_every=0 means one fixed set.
        if tau_train is None or (
            args.resample_every > 0 and (epoch - 1) % args.resample_every == 0
        ):
            tau_train, x_train = sample_interior(
                args.batch_size,
                device,
                dtype,
                arrays["X_min"],
                arrays["X_max"],
                args.tau_max,
            )
            tau_terminal, x_terminal = sample_terminal(
                terminal_batch,
                device,
                dtype,
                arrays["X_min"],
                arrays["X_max"],
            )

        optimizer.zero_grad(set_to_none=True)
        residual, _ = homogeneous_hjb_residual(
            model,
            tau_train,
            x_train,
            args.gamma,
            args.r,
            tensors["K"],
            tensors["k0"],
            tensors["Q"],
            tensors["Gamma"],
            tensors["lam0"],
            tensors["Lam"],
        )
        pde_loss = residual.square().mean()
        terminal_prediction = model(tau_terminal, x_terminal)
        terminal_loss = (terminal_prediction - 1.0).square().mean()
        total_loss = pde_loss + args.w_terminal * terminal_loss
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch}: "
                f"pde={float(pde_loss.detach().cpu())}, "
                f"terminal={float(terminal_loss.detach().cpu())}"
            )
        total_loss.backward()
        if args.grad_clip > 0.0:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise FloatingPointError(
                    f"non-finite aggregate gradient at epoch {epoch}"
                ) from error
            grad_norm_value = float(grad_norm.detach().cpu())
        else:
            component_norms = [
                parameter.grad.detach().to(torch.float64).norm(2)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            grad_norm = (
                torch.stack(component_norms).norm(2)
                if component_norms
                else torch.zeros((), device=device, dtype=torch.float64)
            )
            if not bool(torch.isfinite(grad_norm).detach().cpu()):
                raise FloatingPointError(
                    f"non-finite aggregate gradient at epoch {epoch}"
                )
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()
        # Input coordinates are leaf tensors because autograd differentiates
        # the PDE with respect to them.  They are not optimization variables,
        # so discard any accumulated leaf gradients between epochs.
        tau_train.grad = None
        x_train.grad = None
        # Keep the updated PINN optimization contract: the LR scheduler observes
        # the current training objective.  The fixed heldout set monitors the
        # run and supplies a diagnostic best checkpoint, but the paper protocol
        # reports the final training state.
        if scheduler is not None:
            scheduler.step(float(total_loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "total": float(total_loss.detach().cpu()),
            "pde": float(pde_loss.detach().cpu()),
            "terminal": float(terminal_loss.detach().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm_value,
        }
        train_history.append(row)

        should_validate = epoch % args.val_every == 0 or epoch == args.epochs
        if should_validate:
            metrics = heldout_residual_score(
                model,
                val_interior,
                val_terminal,
                args.gamma,
                args.r,
                tensors,
                args.val_chunk,
            )
            last_validation_epoch = epoch
            val_history.append({"epoch": epoch, **metrics})
            if metrics["p_res"] < best_score:
                best_score = metrics["p_res"]
                best_epoch = epoch
                best_metrics = metrics
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics,
                    args,
                    tag,
                    market_hash,
                )
            if args.pres_target is not None and metrics["p_res"] <= args.pres_target:
                print(
                    f"Reached p_res target at epoch {epoch}: "
                    f"{metrics['p_res']:.3e} <= {args.pres_target:.3e}"
                )
                stopped_early = True

        # Validation cadence controls measurement only; it must not silently
        # override the user-facing print cadence.  At a routine print, report
        # the most recently measured fixed-heldout score together with the
        # epoch at which that score was computed (it can therefore be stale).
        should_print = (
            epoch == 1
            or epoch % args.print_every == 0
            or epoch == args.epochs
            or stopped_early
        )
        if should_print:
            latest_heldout = val_history[-1]
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_total={row['total']:.3e} "
                f"train_pde={row['pde']:.3e} "
                f"train_tc={row['terminal']:.3e} "
                f"lr={row['lr']:.3e} "
                f"heldout_p_res={latest_heldout['p_res']:.3e} "
                f"heldout_epoch={int(latest_heldout['epoch'])}"
            )
        if stopped_early:
            break

    final_epoch = train_history[-1]["epoch"] if train_history else 0
    if last_validation_epoch != final_epoch:
        metrics = heldout_residual_score(
            model,
            val_interior,
            val_terminal,
            args.gamma,
            args.r,
            tensors,
            args.val_chunk,
        )
        val_history.append({"epoch": final_epoch, **metrics})
        if metrics["p_res"] < best_score:
            best_score = metrics["p_res"]
            best_epoch = int(final_epoch)
            best_metrics = metrics
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                int(final_epoch),
                metrics,
                args,
                tag,
                market_hash,
            )

    final_metrics = val_history[-1]
    final_heldout = {
        "pde_rms": final_metrics["pde_rms"],
        "terminal_rms": final_metrics["terminal_rms"],
        "p_res": final_metrics["p_res"],
    }
    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        int(final_epoch),
        final_heldout,
        args,
        tag,
        market_hash,
    )
    write_history_csv(
        output_dir / "train_history.csv",
        train_history,
        fieldnames=("epoch", "total", "pde", "terminal", "lr", "grad_norm"),
    )
    write_history_csv(output_dir / "heldout_history.csv", val_history)

    # FINAL-ITERATE POLICY (matching the updated PINN): evaluation and figures
    # use the final training state.  The minimum-heldout checkpoint is retained
    # strictly as a diagnostic artifact and is never restored automatically.
    official = load_checkpoint(final_path, device)
    model.load_state_dict(official["model_state_dict"])
    selection = {
        "official_checkpoint": str(final_path),
        "official_rule": "final_training_state",
        "official_epoch": int(final_epoch),
        "official_heldout": final_heldout,
        "diagnostic_best_checkpoint": str(best_path),
        "best_epoch": int(best_epoch),
        "best_heldout": best_metrics,
        "final_epoch": int(final_epoch),
        "stopped_early": stopped_early,
        "validation_seed": int(validation_seed),
        "market_fingerprint": market_hash,
    }
    write_json(output_dir / "selection.json", selection)
    return train_history, val_history, selection


def relative_l2(
    prediction: np.ndarray, reference: np.ndarray
) -> Optional[float]:
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    denominator = np.linalg.norm(reference)
    if denominator <= 1e-15:
        return None
    return float(np.linalg.norm(prediction - reference) / denominator)


def error_metrics(
    prediction: np.ndarray, reference: np.ndarray, prefix: str
) -> Dict[str, Any]:
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not (np.all(np.isfinite(prediction)) and np.all(np.isfinite(reference))):
        raise FloatingPointError(f"non-finite values encountered in {prefix} evaluation")
    error = prediction - reference
    rmse = float(np.sqrt(np.mean(np.square(error))))
    reference_std = float(np.std(reference))
    rel_l2 = relative_l2(prediction, reference)
    return {
        f"RelL2_{prefix}": rel_l2,
        f"RelL2Defined_{prefix}": rel_l2 is not None,
        f"RMSE_{prefix}": rmse,
        f"MAE_{prefix}": float(np.mean(np.abs(error))),
        f"MaxAbs_{prefix}": float(np.max(np.abs(error))),
        f"StdNRMSE_{prefix}": (
            float(rmse / reference_std) if reference_std > 1e-15 else None
        ),
        f"StdNRMSEDefined_{prefix}": reference_std > 1e-15,
    }


def evaluate_model_points(
    model: HomogeneousValueNet,
    tau_np: np.ndarray,
    x_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    tensors: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    was_training = model.training
    model.eval()
    g_values = []
    vartheta_values = []
    myopic_values = []
    hedging_values = []
    with torch.enable_grad():
        for start in range(0, len(tau_np), args.eval_chunk):
            tau = torch.as_tensor(
                tau_np[start : start + args.eval_chunk], device=device, dtype=dtype
            ).reshape(-1, 1)
            x = torch.as_tensor(
                x_np[start : start + args.eval_chunk], device=device, dtype=dtype
            ).requires_grad_(True)
            g, vartheta, myopic, hedging = compute_normalized_policy(
                model,
                tau,
                x,
                args.gamma,
                tensors["Gamma"],
                tensors["lam0"],
                tensors["Lam"],
                create_graph=False,
            )
            g_values.append(g.detach().cpu().numpy())
            vartheta_values.append(vartheta.detach().cpu().numpy())
            myopic_values.append(myopic.detach().cpu().numpy())
            hedging_values.append(hedging.detach().cpu().numpy())
    if was_training:
        model.train()
    return (
        np.concatenate(g_values, axis=0).reshape(-1),
        np.concatenate(vartheta_values, axis=0),
        np.concatenate(myopic_values, axis=0),
        np.concatenate(hedging_values, axis=0),
    )


def full_dimensional_evaluation(
    model: HomogeneousValueNet,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    solution,
    output_dir: Path,
) -> Dict[str, Any]:
    margin = args.eval_margin
    x_low = arrays["X_min"] + margin * (arrays["X_max"] - arrays["X_min"])
    x_high = arrays["X_max"] - margin * (arrays["X_max"] - arrays["X_min"])
    # Match the updated numerical protocol: margin shrinks the state window,
    # while tau always spans the full interior.  A method-independent seed
    # gives PINN and PI-PINN corresponding unit-cube test points.
    evaluation_seed = 727
    rng = np.random.default_rng(evaluation_seed)
    eps_tau = min(1e-3, 0.01 * args.tau_max)
    tau = rng.uniform(eps_tau, args.tau_max, size=args.test_points)
    x = rng.uniform(x_low, x_high, size=(args.test_points, args.m_states))

    g_pinn, vartheta_pinn, _myopic_pinn, hedging_pinn = evaluate_model_points(
        model, tau, x, args, device, dtype, tensors
    )
    g_cf, vartheta_cf, _myopic_cf, hedging_cf = closed_form_batch(
        tau,
        x,
        solution,
        args.gamma,
        args.r,
        arrays["lam0"],
        arrays["Lam"],
        arrays["Gamma"],
    )

    metrics: Dict[str, Any] = {
        "test_points": int(args.test_points),
        "eval_margin": float(args.eval_margin),
        "eval_seed": int(evaluation_seed),
    }
    metrics.update(error_metrics(g_pinn, g_cf, "g"))
    metrics.update(error_metrics(vartheta_pinn, vartheta_cf, "vartheta"))
    metrics.update(error_metrics(hedging_pinn, hedging_cf, "hedging"))
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in metrics.items():
            writer.writerow((key, value))
    return metrics


def evaluate_tau_x_slice(
    model: HomogeneousValueNet,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    solution,
    dim_x: int = 0,
) -> Dict[str, np.ndarray]:
    tau_values = np.linspace(0.0, args.tau_max, args.n_tau)
    x_span = arrays["X_max"] - arrays["X_min"]
    x_low = arrays["X_min"][dim_x] + args.eval_margin * x_span[dim_x]
    x_high = arrays["X_max"][dim_x] - args.eval_margin * x_span[dim_x]
    x_values = np.linspace(x_low, x_high, args.n_x)
    x_grid, tau_grid = np.meshgrid(x_values, tau_values, indexing="xy")
    x_full = np.tile(arrays["xbar"], (tau_grid.size, 1))
    x_full[:, dim_x] = x_grid.reshape(-1)
    tau_flat = tau_grid.reshape(-1)

    g_pinn, vartheta_pinn, myopic_pinn, hedging_pinn = evaluate_model_points(
        model, tau_flat, x_full, args, device, dtype, tensors
    )
    g_cf, vartheta_cf, myopic_cf, hedging_cf = closed_form_batch(
        tau_flat,
        x_full,
        solution,
        args.gamma,
        args.r,
        arrays["lam0"],
        arrays["Lam"],
        arrays["Gamma"],
    )
    grid_shape = (args.n_tau, args.n_x)
    policy_shape = (args.n_tau, args.n_x, args.n_assets)
    return {
        "tau_grid": tau_grid,
        "x_grid": x_grid,
        "g_pinn": g_pinn.reshape(grid_shape),
        "g_cf": g_cf.reshape(grid_shape),
        "vartheta_pinn": vartheta_pinn.reshape(policy_shape),
        "vartheta_cf": vartheta_cf.reshape(policy_shape),
        "myopic_pinn": myopic_pinn.reshape(policy_shape),
        "myopic_cf": myopic_cf.reshape(policy_shape),
        "hedging_pinn": hedging_pinn.reshape(policy_shape),
        "hedging_cf": hedging_cf.reshape(policy_shape),
    }


def save_publication_slice(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    args: argparse.Namespace,
    method: str,
    run_tag: str,
    market_hash: str,
    state_dimension: int = 0,
) -> Dict[str, Any]:
    """Persist the common NumPy-only input contract for paired paper figures."""
    if method not in {"pinn", "pipinn"}:
        raise ValueError(f"unsupported publication-slice method: {method!r}")
    reference_ranges = np.ptp(data["vartheta_cf"], axis=(0, 1))
    asset_indices = np.argsort(reference_ranges)[::-1][
        : min(args.max_plot_assets, args.n_assets)
    ].astype(np.int64)
    arrays = {
        "tau_grid": np.asarray(data["tau_grid"]),
        "x_grid": np.asarray(data["x_grid"]),
        "g_pred": np.asarray(data["g_pinn"]),
        "g_exact": np.asarray(data["g_cf"]),
        "vartheta_pred": np.asarray(data["vartheta_pinn"]),
        "vartheta_exact": np.asarray(data["vartheta_cf"]),
        "myopic": np.asarray(data["myopic_cf"]),
        "hedging_pred": np.asarray(data["hedging_pinn"]),
        "hedging_exact": np.asarray(data["hedging_cf"]),
    }
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                f"publication slice contains non-finite values in {name}"
            )

    npz_path = output_dir / "publication_slice.npz"
    temporary_npz = npz_path.with_name(
        f".{npz_path.stem}.tmp.{os.getpid()}{npz_path.suffix}"
    )
    try:
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_npz, npz_path)
    finally:
        temporary_npz.unlink(missing_ok=True)

    identity = {
        "n_assets": int(args.n_assets),
        "m_states": int(args.m_states),
        "seed": int(args.seed),
        "market_seed": int(args.market_seed),
        "tau_max": float(args.tau_max),
        "gamma": float(args.gamma),
        "r": float(args.r),
        "lambda0": float(args.lambda0),
        "x_range_scale": float(args.x_range_scale),
        "dirichlet_concentration": float(args.dirichlet_concentration),
        "alpha_scale": float(args.alpha_scale),
        "eval_margin": float(args.eval_margin),
        "n_tau": int(args.n_tau),
        "n_x": int(args.n_x),
        "state_dimension": int(state_dimension),
        "market_fingerprint": str(market_hash),
    }
    identity_digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": PUBLICATION_SLICE_SCHEMA,
        "status": "COMPLETE",
        "method": method,
        "run_tag": run_tag,
        "n_assets": int(args.n_assets),
        "m_states": int(args.m_states),
        "config_identity": identity,
        "config_identity_sha256": identity_digest,
        "asset_indices": [int(asset) for asset in asset_indices],
        "npz_file": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
    }
    write_json(output_dir / "publication_slice.json", metadata)
    return metadata


def heat_basic(
    axis,
    values: np.ndarray,
    title: Optional[str],
    extent,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="bilinear",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xlabel(r"state $z_1$")
    axis.set_ylabel(r"$\tau$")
    if title:
        axis.set_title(title)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def heat_diverging(
    axis,
    values: np.ndarray,
    title: Optional[str],
    extent,
) -> None:
    abs_max = max(float(np.max(np.abs(values))), 1e-12)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="bilinear",
        cmap="RdBu_r",
        norm=norm,
    )
    axis.set_xlabel(r"state $z_1$")
    axis.set_ylabel(r"$\tau$")
    if title:
        axis.set_title(title)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_g_comparison(
    data: Dict[str, np.ndarray],
    path_stem: Path,
    dpi: int,
    show: bool = False,
    method: str = "pinn",
    output_mode: str = "separate",
) -> None:
    if method not in {"pinn", "pipinn"}:
        raise ValueError(f"unsupported plotting method: {method!r}")
    if output_mode not in {"separate", "combined", "both"}:
        raise ValueError(f"unsupported figure output mode: {output_mode!r}")
    extent = [
        data["x_grid"].min(),
        data["x_grid"].max(),
        data["tau_grid"].min(),
        data["tau_grid"].max(),
    ]
    vmin = min(float(data["g_pinn"].min()), float(data["g_cf"].min()))
    vmax = max(float(data["g_pinn"].max()), float(data["g_cf"].max()))
    method_label = "PINN" if method == "pinn" else "PI-PINN"
    panels = (
        ("pred", data["g_pinn"], False, f"g ({method_label})"),
        ("closed_form", data["g_cf"], False, "g (closed form)"),
        ("diff", data["g_pinn"] - data["g_cf"], True, "g difference"),
    )
    if output_mode in {"separate", "both"}:
        for panel, values, is_difference, _ in panels:
            fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
            if is_difference:
                heat_diverging(axis, values, None, extent)
            else:
                heat_basic(axis, values, None, extent, vmin, vmax)
            path = path_stem.with_name(
                f"{path_stem.name}_{method}_{panel}.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    if output_mode in {"combined", "both"}:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for axis, (_, values, is_difference, title) in zip(axes, panels):
            if is_difference:
                heat_diverging(axis, values, title, extent)
            else:
                heat_basic(axis, values, title, extent, vmin, vmax)
        path = path_stem.with_suffix(".png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def plot_policy_comparison(
    data: Dict[str, np.ndarray],
    path_stem: Path,
    max_assets: int,
    dpi: int,
    show: bool = False,
    method: str = "pinn",
    output_mode: str = "separate",
) -> None:
    if method not in {"pinn", "pipinn"}:
        raise ValueError(f"unsupported plotting method: {method!r}")
    if output_mode not in {"separate", "combined", "both"}:
        raise ValueError(f"unsupported figure output mode: {output_mode!r}")
    extent = [
        data["x_grid"].min(),
        data["x_grid"].max(),
        data["tau_grid"].min(),
        data["tau_grid"].max(),
    ]
    reference = data["vartheta_cf"]
    ranges = np.ptp(reference, axis=(0, 1))
    asset_indices = np.argsort(ranges)[::-1][:max_assets]
    for rank, asset in enumerate(asset_indices, start=1):
        component_panels = []
        for name, label in (
            ("vartheta", "total"),
            ("myopic", "myopic"),
            ("hedging", "hedging"),
        ):
            predicted = data[f"{name}_pinn"][:, :, asset]
            exact = data[f"{name}_cf"][:, :, asset]
            vmin = min(float(predicted.min()), float(exact.min()))
            vmax = max(float(predicted.max()), float(exact.max()))
            panels = (
                ("pred", predicted, False),
                ("closed_form", exact, False),
                ("diff", predicted - exact, True),
            )
            component_panels.append((label, panels, vmin, vmax))
            if output_mode in {"separate", "both"}:
                for panel, values, is_difference in panels:
                    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
                    if is_difference:
                        heat_diverging(axis, values, None, extent)
                    else:
                        heat_basic(axis, values, None, extent, vmin, vmax)
                    path = path_stem.with_name(
                        f"{path_stem.name}_rank{rank:02d}_asset{asset}_"
                        f"{method}_{label}_{panel}.png"
                    )
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    if show:
                        plt.show()
                    plt.close(fig)

        if output_mode in {"combined", "both"}:
            method_label = "PINN" if method == "pinn" else "PI-PINN"
            column_titles = (method_label, "closed form", "difference")
            fig, axes = plt.subplots(
                3, 3, figsize=(15, 12), constrained_layout=True
            )
            for row, (label, panels, vmin, vmax) in enumerate(component_panels):
                for column, ((_, values, is_difference), column_title) in enumerate(
                    zip(panels, column_titles)
                ):
                    title = f"{label} ({column_title})"
                    if is_difference:
                        heat_diverging(axes[row, column], values, title, extent)
                    else:
                        heat_basic(
                            axes[row, column], values, title, extent, vmin, vmax
                        )
            path = path_stem.with_name(
                f"{path_stem.name}_rank{rank:02d}_asset{asset}.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)


def plot_loss_history(
    train_history: list[Dict[str, float]],
    val_history: list[Dict[str, float]],
    path: Path,
    dpi: int,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    epochs = np.asarray([row["epoch"] for row in train_history])
    for key, label in (("total", "total"), ("pde", "PDE"), ("terminal", "terminal")):
        axes[0].semilogy(epochs, [row[key] for row in train_history], label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    val_epochs = [row["epoch"] for row in val_history]
    axes[1].semilogy(val_epochs, [row["p_res"] for row in val_history], label="p_res")
    axes[1].semilogy(val_epochs, [row["pde_rms"] for row in val_history], label="PDE RMS")
    axes[1].semilogy(
        val_epochs, [row["terminal_rms"] for row in val_history], label="terminal RMS"
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("heldout residual")
    axes[1].set_title("Fixed heldout monitoring")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.market_seed is None:
        args.market_seed = args.seed
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    set_seed(args.seed)
    output_dir, weight_dir, tag = prepare_run_directories(args)

    print(f"Device: {device}; dtype: {args.dtype}; run_tag: {tag}")
    params, arrays = build_market(args)
    current_market_hash = market_fingerprint(arrays)
    checkpoint = None
    if args.eval_only:
        checkpoint = load_checkpoint(args.checkpoint, device)
        validate_eval_checkpoint(checkpoint, args, current_market_hash)

    config = {
        **vars(args),
        "method": METHOD_NAME,
        "schema": SCHEMA_NAME,
        "resolved_run_tag": tag,
        "resolved_output_dir": str(output_dir),
        "market_fingerprint": current_market_hash,
    }
    config_name = "config_eval.json" if args.eval_only else "config.json"
    write_json(output_dir / config_name, config)
    print_market_report(params, arrays["rho"])
    save_market_snapshot(output_dir / "market_params.npz", arrays)
    print(
        f"state domain: min={np.array2string(arrays['X_min'], precision=4)}, "
        f"max={np.array2string(arrays['X_max'], precision=4)}"
    )

    print("\nSolving tight closed-form Riccati reference...")
    solution = solve_closed_form_ode(
        args.tau_max,
        args.gamma,
        arrays["K"],
        arrays["xbar"],
        arrays["SigmaX"],
        arrays["rho"],
        arrays["lam0"],
        arrays["Lam"],
    )
    print(f"ODE success with {len(solution.t)} stored time points")

    tensors = tensor_market(arrays, device, dtype)
    model = HomogeneousValueNet(
        M=args.m_states,
        hidden=args.value_hidden,
        depth=args.value_depth,
        g_floor=args.g_floor,
    ).to(device=device, dtype=dtype)

    train_history: list[Dict[str, float]] = []
    val_history: list[Dict[str, float]] = []
    started = time.perf_counter()
    if args.eval_only:
        model.load_state_dict(checkpoint["model_state_dict"])
        selection = {
            "official_checkpoint": str(args.checkpoint),
            "official_rule": "user-specified eval-only checkpoint",
            "official_epoch": int(checkpoint.get("epoch", -1)),
            "official_heldout": checkpoint.get("heldout", {}),
            "market_fingerprint": current_market_hash,
        }
    else:
        train_history, val_history, selection = train_pinn(
            model,
            args,
            device,
            dtype,
            arrays,
            tensors,
            output_dir,
            weight_dir,
            tag,
        )
    training_seconds = time.perf_counter() - started

    metrics: Dict[str, Any] = {}
    if not args.skip_eval:
        print("\nRunning independent full-dimensional evaluation...")
        metrics = full_dimensional_evaluation(
            model,
            args,
            device,
            dtype,
            arrays,
            tensors,
            solution,
            output_dir,
        )
        def display_metric(value: Optional[float]) -> str:
            return "undefined" if value is None else f"{value:.3e}"

        print(
            f"RelL2_g={display_metric(metrics['RelL2_g'])}, "
            f"RelL2_vartheta={display_metric(metrics['RelL2_vartheta'])}, "
            f"RelL2_hedging={display_metric(metrics['RelL2_hedging'])}"
        )

    if not args.skip_figures:
        if train_history:
            plot_loss_history(
                train_history,
                val_history,
                output_dir / "loss_history.png",
                args.figure_dpi,
            )
        slice_data = evaluate_tau_x_slice(
            model,
            args,
            device,
            dtype,
            arrays,
            tensors,
            solution,
        )
        save_publication_slice(
            slice_data,
            output_dir,
            args,
            method="pinn",
            run_tag=tag,
            market_hash=current_market_hash,
        )
        plot_g_comparison(
            slice_data,
            output_dir / "g_tau_state",
            args.figure_dpi,
            method="pinn",
            output_mode=args.figure_output_mode,
        )
        plot_policy_comparison(
            slice_data,
            output_dir / "policy_tau_state",
            args.max_plot_assets,
            args.figure_dpi,
            method="pinn",
            output_mode=args.figure_output_mode,
        )

    status = {
        "status": "COMPLETE",
        "method": METHOD_NAME,
        "schema": SCHEMA_NAME,
        "run_tag": tag,
        "market_fingerprint": current_market_hash,
        "device": str(device),
        "dtype": args.dtype,
        "training_seconds": training_seconds,
        "selection": selection,
        "metrics": metrics,
    }
    write_json(output_dir / "result.json", status)
    print(f"\nCompleted in {training_seconds:.2f}s")
    print(f"outputs: {output_dir}")
    print(f"weights: {weight_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
