"""Homogeneous PI-PINN for the multidimensional Kim--Omberg benchmark.

The code shares the validated market, network, Riccati reference, evaluation,
and plotting utilities from ``Liu_nd_pinn_homogeneous.py``.  It changes only
the training core: every outer iteration freezes the previous normalized
policy, solves the resulting linear PDE, restores that outer solve's fixed-
heldout inner-best model and Adam state, and then performs the analytic greedy
policy update.

The official PI-PINN model is the restored inner-best state of the *last*
completed outer iteration.  Scores from different outer iterations solve
different frozen-policy PDEs and are therefore never used for cross-outer
model selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mor-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Liu_nd_pinn_homogeneous import (
    SCHEMA_NAME,
    HomogeneousValueNet,
    build_market,
    compute_normalized_policy,
    evaluate_tau_x_slice,
    full_dimensional_evaluation,
    g_derivatives,
    heldout_residual_score,
    json_ready,
    load_checkpoint,
    market_fingerprint,
    plot_g_comparison,
    plot_policy_comparison,
    prepare_run_directories,
    print_market_report,
    resolve_device,
    resolve_dtype,
    sample_interior,
    sample_terminal,
    save_publication_slice,
    save_market_snapshot,
    set_seed,
    solve_closed_form_ode,
    tensor_market,
    write_json,
)


METHOD_NAME = "pipinn_homogeneous"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Homogeneous PI-PINN for the multidimensional Kim--Omberg PDE"
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

    training = parser.add_argument_group("policy iteration and inner training")
    training.add_argument("--outer-iters", type=int, default=20)
    training.add_argument(
        "--inner-epochs",
        "--eval-epochs",
        dest="inner_epochs",
        type=int,
        default=2_000,
    )
    training.add_argument("--batch-size", type=int, default=10_000)
    training.add_argument("--terminal-frac", type=float, default=0.5)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument(
        "--pe-resample-every",
        type=int,
        default=0,
        help="inner collocation resampling period; 0 keeps one batch per outer",
    )
    training.add_argument("--w-terminal", type=float, default=20.0)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument(
        "--theta-init-method", choices=("myopic", "zero"), default="myopic"
    )
    training.add_argument("--theta-init-scale", type=float, default=1.0)
    training.add_argument("--print-every-outer", type=int, default=1)
    training.add_argument(
        "--print-every-eval",
        type=int,
        default=0,
        help="inner progress period; 0 suppresses routine inner prints",
    )

    scheduler = parser.add_argument_group("optimizer and scheduler")
    scheduler.add_argument(
        "--lr-schedule",
        choices=("carry_plateau", "inner_plateau", "fixed"),
        default="carry_plateau",
    )
    scheduler.add_argument("--adam-reset", choices=("keep", "full"), default="keep")
    scheduler.add_argument("--scheduler-patience", type=int, default=20)
    scheduler.add_argument("--scheduler-factor", type=float, default=0.5)
    scheduler.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    scheduler.add_argument("--carry-lr-min", type=float, default=1e-5)
    scheduler.add_argument("--carry-lr-max", type=float, default=3e-4)

    selection = parser.add_argument_group("fixed heldout sets")
    selection.add_argument("--sel-points", type=int, default=10_000)
    selection.add_argument("--sel-terminal-points", type=int, default=2_000)
    selection.add_argument("--sel-every", type=int, default=50)
    selection.add_argument(
        "--sel-patience",
        type=int,
        default=0,
        help="non-improving selection checks before stopping an inner solve; 0 disables",
    )
    selection.add_argument("--val-points", type=int, default=50_000)
    selection.add_argument("--val-terminal-points", type=int, default=10_000)
    selection.add_argument(
        "--val-every",
        type=int,
        default=25,
        help="large validation cadence used only when --pres-target is supplied",
    )
    selection.add_argument("--val-chunk", type=int, default=4_096)
    selection.add_argument("--pres-target", type=float, default=None)

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
    runtime.add_argument(
        "--output-root", type=Path, default=Path("outputs/pipinn_homogeneous")
    )
    runtime.add_argument(
        "--weight-root", type=Path, default=Path("weights/pipinn_homogeneous")
    )
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
        "theta_init_scale",
        "scheduler_factor",
        "scheduler_min_lr",
        "carry_lr_min",
        "carry_lr_max",
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
    if not 0.0 < args.g_floor < 1.0:
        raise ValueError("g-floor must lie in (0,1)")
    if args.x_range_scale <= 0.0 or args.dirichlet_concentration <= 0.0:
        raise ValueError("x-range-scale and dirichlet-concentration must be positive")
    if args.alpha_scale < 0.0:
        raise ValueError("alpha-scale must be nonnegative")
    if args.outer_iters <= 0 or args.inner_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("outer-iters, inner-epochs, and batch-size must be positive")
    if not 0.0 < args.terminal_frac <= 1.0:
        raise ValueError("terminal-frac must lie in (0,1]")
    if args.lr <= 0.0 or args.w_terminal < 0.0 or args.grad_clip < 0.0:
        raise ValueError("lr must be positive; loss weight and grad-clip nonnegative")
    if args.pe_resample_every < 0:
        raise ValueError("pe-resample-every must be nonnegative")
    if args.theta_init_scale < 0.0:
        raise ValueError("theta-init-scale must be nonnegative")
    if args.print_every_outer <= 0 or args.print_every_eval < 0:
        raise ValueError("print periods must be nonnegative, with outer period positive")
    if args.lr_schedule != "fixed":
        if args.scheduler_patience < 0:
            raise ValueError("scheduler-patience must be nonnegative")
        if not 0.0 < args.scheduler_factor < 1.0:
            raise ValueError("scheduler-factor must lie in (0,1)")
        if args.scheduler_min_lr <= 0.0 or args.scheduler_min_lr > args.lr:
            raise ValueError("scheduler-min-lr must lie in (0,lr]")
    if args.lr_schedule == "carry_plateau":
        if not 0.0 < args.carry_lr_min <= args.carry_lr_max:
            raise ValueError("carry LR bounds must be positive and ordered")
        if args.carry_lr_min > args.lr:
            raise ValueError("carry-lr-min cannot exceed the initial lr")
        if max(args.carry_lr_min, args.scheduler_min_lr) > args.carry_lr_max:
            raise ValueError(
                "the effective carry LR floor cannot exceed carry-lr-max"
            )
    if (
        args.sel_points <= 0
        or args.sel_terminal_points <= 0
        or args.sel_every <= 0
        or args.sel_patience < 0
        or args.val_points <= 0
        or args.val_terminal_points <= 0
        or args.val_every <= 0
        or args.val_chunk <= 0
    ):
        raise ValueError("heldout sizes/cadences must be positive; patience nonnegative")
    if args.pres_target is not None and args.pres_target <= 0.0:
        raise ValueError("pres-target must be positive when supplied")
    if not 0.0 <= args.eval_margin < 0.5:
        raise ValueError("eval-margin must lie in [0,0.5)")
    if args.test_points <= 0 or args.eval_chunk <= 0:
        raise ValueError("test-points and eval-chunk must be positive")
    if (
        args.n_tau < 2
        or args.n_x < 2
        or args.max_plot_assets <= 0
        or args.figure_dpi <= 0
    ):
        raise ValueError("plot grid sizes and figure settings must be positive")
    if args.eval_only and args.checkpoint is None:
        raise ValueError("eval-only requires --checkpoint")


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def state_dict_hash(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_hash(model: HomogeneousValueNet) -> str:
    return state_dict_hash(model.state_dict())


def analytic_policy_source_hash(args: argparse.Namespace) -> str:
    kind = f"analytic:{args.theta_init_method}"
    return hashlib.sha256(
        (
            f"{kind}|scale={args.theta_init_scale:.17g}|"
            f"gamma={args.gamma:.17g}"
        ).encode("utf-8")
    ).hexdigest()


def torch_save_atomic(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze_policy_source(model: HomogeneousValueNet) -> HomogeneousValueNet:
    source = copy.deepcopy(model)
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    return source


def evaluate_frozen_policy(
    source: Optional[HomogeneousValueNet],
    tau: torch.Tensor,
    x: torch.Tensor,
    args: argparse.Namespace,
    tensors: Dict[str, torch.Tensor],
    chunk: int,
) -> torch.Tensor:
    """Evaluate the outer-frozen normalized policy without building train graphs."""
    values = []
    if source is None:
        lam_x = tensors["lam0"][None, :] + x.detach() @ tensors["Lam"].T
        if args.theta_init_method == "myopic":
            policy = args.theta_init_scale * lam_x / args.gamma
        else:
            policy = torch.zeros_like(lam_x)
        if not bool(torch.isfinite(policy).all().detach().cpu()):
            raise FloatingPointError("non-finite analytic initial policy")
        return policy.detach()

    was_training = source.training
    source.eval()
    with torch.enable_grad():
        for start in range(0, x.shape[0], chunk):
            tau_chunk = tau[start : start + chunk].detach()
            x_chunk = x[start : start + chunk].detach().clone().requires_grad_(True)
            _, vartheta, _, _ = compute_normalized_policy(
                source,
                tau_chunk,
                x_chunk,
                args.gamma,
                tensors["Gamma"],
                tensors["lam0"],
                tensors["Lam"],
                create_graph=False,
            )
            values.append(vartheta.detach())
    if was_training:
        source.train()
    policy = torch.cat(values, dim=0)
    if not bool(torch.isfinite(policy).all().detach().cpu()):
        raise FloatingPointError("non-finite frozen policy")
    return policy


def frozen_policy_residual(
    model: HomogeneousValueNet,
    policy: torch.Tensor,
    tau: torch.Tensor,
    x: torch.Tensor,
    gamma: float,
    r: float,
    tensors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual ``g_tau - L^policy g`` for one fixed normalized policy."""
    g, g_tau, g_x, q_hessian_trace = g_derivatives(
        model, tau, x, tensors["Q"]
    )
    drift = tensors["k0"][None, :] - x @ tensors["K"].T
    lam_x = tensors["lam0"][None, :] + x @ tensors["Lam"].T
    gamma_grad = g_x @ tensors["Gamma"].T

    policy_dot_lam = torch.sum(policy * lam_x, dim=1, keepdim=True)
    policy_norm_sq = torch.sum(policy.square(), dim=1, keepdim=True)
    policy_dot_cross = torch.sum(policy * gamma_grad, dim=1, keepdim=True)

    rhs = torch.sum(drift * g_x, dim=1, keepdim=True)
    rhs = rhs + 0.5 * q_hessian_trace
    rhs = rhs + (1.0 - gamma) * (
        r * g
        + policy_dot_lam * g
        - 0.5 * gamma * policy_norm_sq * g
        + policy_dot_cross
    )
    return g_tau - rhs, g


def build_fixed_set(
    n_interior: int,
    n_terminal: int,
    seed: int,
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    tau, x = sample_interior(
        n_interior,
        device,
        dtype,
        arrays["X_min"],
        arrays["X_max"],
        args.tau_max,
        rng=rng,
    )
    tau_terminal, x_terminal = sample_terminal(
        n_terminal,
        device,
        dtype,
        arrays["X_min"],
        arrays["X_max"],
        rng=rng,
    )
    return (tau.detach(), x.detach()), (tau_terminal.detach(), x_terminal.detach())


def frozen_heldout_score(
    model: HomogeneousValueNet,
    policy: torch.Tensor,
    coordinates: Tuple[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
    args: argparse.Namespace,
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    (tau_all, x_all), (tau_terminal, x_terminal) = coordinates
    residual_sum = 0.0
    residual_count = 0
    with torch.enable_grad():
        for start in range(0, tau_all.shape[0], args.val_chunk):
            tau = tau_all[start : start + args.val_chunk].detach().clone()
            tau.requires_grad_(True)
            x = x_all[start : start + args.val_chunk].detach().clone()
            x.requires_grad_(True)
            residual, _ = frozen_policy_residual(
                model,
                policy[start : start + args.val_chunk],
                tau,
                x,
                args.gamma,
                args.r,
                tensors,
            )
            residual_sum += float(residual.detach().square().sum().cpu())
            residual_count += residual.numel()

    terminal_sum = 0.0
    terminal_count = 0
    with torch.no_grad():
        for start in range(0, tau_terminal.shape[0], args.val_chunk):
            prediction = model(
                tau_terminal[start : start + args.val_chunk],
                x_terminal[start : start + args.val_chunk],
            )
            error = prediction - 1.0
            terminal_sum += float(error.square().sum().cpu())
            terminal_count += error.numel()
    if was_training:
        model.train()

    pde_rms = math.sqrt(residual_sum / max(residual_count, 1))
    terminal_rms = math.sqrt(terminal_sum / max(terminal_count, 1))
    if not (math.isfinite(pde_rms) and math.isfinite(terminal_rms)):
        raise FloatingPointError(
            f"non-finite frozen heldout residual: {pde_rms=}, {terminal_rms=}"
        )
    return {
        "pde_rms": pde_rms,
        "terminal_rms": terminal_rms,
        "p_res": pde_rms + terminal_rms,
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    if args.lr_schedule == "fixed":
        return None
    minimum = args.scheduler_min_lr
    if args.lr_schedule == "carry_plateau":
        minimum = max(minimum, args.carry_lr_min)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        min_lr=minimum,
    )


def prepare_optimizer_for_outer(
    model: HomogeneousValueNet,
    optimizer: torch.optim.Optimizer,
    outer_iter: int,
    args: argparse.Namespace,
) -> Tuple[
    torch.optim.Optimizer,
    Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    float,
]:
    carried_lr = float(optimizer.param_groups[0]["lr"])
    if args.lr_schedule == "carry_plateau":
        start_lr = args.lr if outer_iter == 1 else min(
            args.carry_lr_max,
            max(args.carry_lr_min, args.scheduler_min_lr, carried_lr),
        )
    else:
        start_lr = args.lr

    if outer_iter > 1 and args.adam_reset == "full":
        optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = start_lr
    scheduler = make_scheduler(optimizer, args)
    return optimizer, scheduler, float(start_lr)


def capture_inner_state(
    model: HomogeneousValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    epoch: int,
    train_metrics: Dict[str, Any],
    sel_metrics: Dict[str, float],
    qres_metrics: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    state = cpu_tree(model.state_dict())
    return {
        "epoch": int(epoch),
        "model": state,
        "model_hash": state_dict_hash(state),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scheduler": None if scheduler is None else cpu_tree(scheduler.state_dict()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        # Keep the accepted training row in the same immutable snapshot as
        # the selected model/optimizer state.  This prevents an outer summary
        # from accidentally reporting the literal last inner step instead of
        # the state that was actually restored and accepted.
        "train": {
            "total": (
                None
                if train_metrics.get("total") is None
                else float(train_metrics["total"])
            ),
            "pde": (
                None
                if train_metrics.get("pde") is None
                else float(train_metrics["pde"])
            ),
            "terminal": (
                None
                if train_metrics.get("terminal") is None
                else float(train_metrics["terminal"])
            ),
        },
        "sel": dict(sel_metrics),
        "qres": None if qres_metrics is None else dict(qres_metrics),
    }


def restore_inner_state(
    snapshot: Dict[str, Any],
    model: HomogeneousValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
) -> None:
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    if scheduler is not None and snapshot["scheduler"] is not None:
        scheduler.load_state_dict(snapshot["scheduler"])


def train_one_outer(
    model: HomogeneousValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    source: Optional[HomogeneousValueNet],
    source_hash: str,
    qsel: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    qres: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    policy_sel: torch.Tensor,
    policy_res: torch.Tensor,
    outer_iter: int,
    global_step_base: int,
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    terminal_batch = max(1, int(round(args.batch_size * args.terminal_frac)))
    rows: list[Dict[str, Any]] = []
    fallback_best: Optional[Dict[str, Any]] = None
    qualified_best: Optional[Dict[str, Any]] = None
    bad_checks = 0
    target_crossed = False
    selection_checks = 0
    qres_checks = 0
    tau_train: Optional[torch.Tensor] = None
    x_train: Optional[torch.Tensor] = None
    tau_terminal: Optional[torch.Tensor] = None
    x_terminal: Optional[torch.Tensor] = None
    policy_train: Optional[torch.Tensor] = None
    accepted_train_cache: Dict[int, Dict[str, Optional[float]]] = {}

    def accepted_training_metrics(epoch: int) -> Dict[str, Optional[float]]:
        """Re-evaluate the objective at the exact post-update snapshot state.

        The ordinary inner-history row is intentionally recorded before its
        optimizer update, matching the original training loop.  Selection is
        performed after that update, however, so publication/outer summaries
        must not label the pre-update row as the restored model's loss.
        """
        if epoch in accepted_train_cache:
            return dict(accepted_train_cache[epoch])
        if epoch == 0:
            metrics: Dict[str, Optional[float]] = {
                "total": None,
                "pde": None,
                "terminal": None,
            }
            accepted_train_cache[epoch] = metrics
            return dict(metrics)
        if any(
            value is None
            for value in (
                tau_train,
                x_train,
                tau_terminal,
                x_terminal,
                policy_train,
            )
        ):
            raise RuntimeError("accepted-state training batch is unavailable")
        with torch.enable_grad():
            tau_snapshot = tau_train.detach().clone().requires_grad_(True)
            x_snapshot = x_train.detach().clone().requires_grad_(True)
            residual_snapshot, _ = frozen_policy_residual(
                model,
                policy_train.detach(),
                tau_snapshot,
                x_snapshot,
                args.gamma,
                args.r,
                tensors,
            )
            pde_snapshot = residual_snapshot.square().mean()
            terminal_snapshot = (
                model(tau_terminal.detach(), x_terminal.detach()) - 1.0
            ).square().mean()
            total_snapshot = pde_snapshot + args.w_terminal * terminal_snapshot
        values = (total_snapshot, pde_snapshot, terminal_snapshot)
        if not all(bool(torch.isfinite(value).detach().cpu()) for value in values):
            raise FloatingPointError(
                f"non-finite accepted-state loss at outer={outer_iter}, epoch={epoch}"
            )
        metrics = {
            "total": float(total_snapshot.detach().cpu()),
            "pde": float(pde_snapshot.detach().cpu()),
            "terminal": float(terminal_snapshot.detach().cpu()),
        }
        accepted_train_cache[epoch] = metrics
        return dict(metrics)

    def check_selection(epoch: int, row: Dict[str, Any]) -> Dict[str, float]:
        nonlocal fallback_best, bad_checks, selection_checks
        selection_checks += 1
        sel_metrics = frozen_heldout_score(
            model, policy_sel, qsel, args, tensors
        )
        if args.lr_schedule == "carry_plateau" and scheduler is not None:
            scheduler.step(sel_metrics["p_res"])

        improved = (
            fallback_best is None
            or sel_metrics["p_res"] < fallback_best["sel"]["p_res"]
        )
        if improved:
            fallback_best = capture_inner_state(
                model,
                optimizer,
                scheduler,
                epoch,
                accepted_training_metrics(epoch),
                sel_metrics,
                None,
            )
            bad_checks = 0
        else:
            bad_checks += 1
        row.update(
            {
                "sel_pde_rms": sel_metrics["pde_rms"],
                "sel_terminal_rms": sel_metrics["terminal_rms"],
                "sel_p_res": sel_metrics["p_res"],
                "selection_check": True,
                "selection_improved": improved,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        return sel_metrics

    def check_target(
        epoch: int,
        row: Dict[str, Any],
        sel_metrics: Optional[Dict[str, float]],
    ) -> bool:
        nonlocal qualified_best, target_crossed, qres_checks
        if args.pres_target is None:
            return False
        qres_checks += 1
        qres_metrics = frozen_heldout_score(
            model, policy_res, qres, args, tensors
        )
        current_qualified = qres_metrics["p_res"] <= args.pres_target
        if current_qualified:
            target_crossed = True
            # A target crossing may occur between regular Q_sel checks.  Rank
            # and save that state without advancing the Q_sel scheduler or
            # selection-patience counter.
            if sel_metrics is None:
                sel_metrics = frozen_heldout_score(
                    model, policy_sel, qsel, args, tensors
                )
                row.update(
                    {
                        "sel_pde_rms": sel_metrics["pde_rms"],
                        "sel_terminal_rms": sel_metrics["terminal_rms"],
                        "sel_p_res": sel_metrics["p_res"],
                    }
                )
            qualified_best = capture_inner_state(
                model,
                optimizer,
                scheduler,
                epoch,
                accepted_training_metrics(epoch),
                sel_metrics,
                qres_metrics,
            )
        row.update(
            {
                "qres_check": True,
                "qres_pde_rms": qres_metrics["pde_rms"],
                "qres_terminal_rms": qres_metrics["terminal_rms"],
                "qres_p_res": qres_metrics["p_res"],
                "target_qualified": current_qualified,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        return current_qualified

    epoch_zero = {
        "outer_iter": outer_iter,
        "inner_epoch": 0,
        "global_step": None,
        "plot_step": global_step_base,
        "synthetic": True,
        "total": None,
        "pde": None,
        "terminal": None,
        "grad_norm": None,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    initial_sel = check_selection(0, epoch_zero)
    stop_now = check_target(0, epoch_zero, initial_sel)
    rows.append(epoch_zero)

    epochs_used = 0
    stop_reason = "pres_target_epoch0" if stop_now else "budget"
    for epoch in range(1, 0 if stop_now else args.inner_epochs + 1):
        if tau_train is None or (
            args.pe_resample_every > 0
            and (epoch - 1) % args.pe_resample_every == 0
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
            policy_train = evaluate_frozen_policy(
                source, tau_train, x_train, args, tensors, args.val_chunk
            )
            if policy_train.requires_grad or policy_train.grad_fn is not None:
                raise RuntimeError("frozen training policy unexpectedly tracks gradients")

        optimizer.zero_grad(set_to_none=True)
        residual, _ = frozen_policy_residual(
            model,
            policy_train,
            tau_train,
            x_train,
            args.gamma,
            args.r,
            tensors,
        )
        pde_loss = residual.square().mean()
        terminal_prediction = model(tau_terminal, x_terminal)
        terminal_loss = (terminal_prediction - 1.0).square().mean()
        total_loss = pde_loss + args.w_terminal * terminal_loss
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            raise FloatingPointError(
                f"non-finite inner loss at outer={outer_iter}, epoch={epoch}"
            )
        total_loss.backward()

        if args.grad_clip > 0.0:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise FloatingPointError(
                    f"non-finite gradient at outer={outer_iter}, epoch={epoch}"
                ) from error
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
                    f"non-finite gradient at outer={outer_iter}, epoch={epoch}"
                )
        optimizer.step()
        tau_train.grad = None
        x_train.grad = None
        if args.lr_schedule == "inner_plateau" and scheduler is not None:
            scheduler.step(float(total_loss.detach().cpu()))

        epochs_used = epoch
        row = {
            "outer_iter": outer_iter,
            "inner_epoch": epoch,
            "global_step": global_step_base + epoch,
            "plot_step": global_step_base + epoch,
            "synthetic": False,
            "total": float(total_loss.detach().cpu()),
            "pde": float(pde_loss.detach().cpu()),
            "terminal": float(terminal_loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        should_select = epoch % args.sel_every == 0 or epoch == args.inner_epochs
        should_validate_target = bool(
            args.pres_target is not None
            and (epoch % args.val_every == 0 or epoch == args.inner_epochs)
        )
        selected_metrics = check_selection(epoch, row) if should_select else None
        stop_now = (
            check_target(epoch, row, selected_metrics)
            if should_validate_target
            else False
        )
        rows.append(row)

        patience_active = args.pres_target is None or qualified_best is not None
        patience_stop = bool(
            args.sel_patience > 0
            and patience_active
            and bad_checks >= args.sel_patience
        )
        if args.print_every_eval > 0 and (
            epoch == 1
            or epoch % args.print_every_eval == 0
            or epoch == args.inner_epochs
            or stop_now
            or patience_stop
        ):
            suffix = (
                f", sel={row['sel_p_res']:.3e}"
                if row.get("sel_p_res") is not None
                else ""
            )
            print(
                f"    inner={epoch}/{args.inner_epochs} total={row['total']:.3e} "
                f"pde={row['pde']:.3e} tc={row['terminal']:.3e} "
                f"lr={row['lr']:.3e}{suffix}"
            )

        if stop_now:
            stop_reason = "pres_target"
            break
        if patience_stop:
            stop_reason = "selection_patience"
            break

    selected = qualified_best if target_crossed and qualified_best is not None else fallback_best
    if selected is None:
        raise RuntimeError("inner selection produced no checkpoint")
    end_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    restore_inner_state(selected, model, optimizer, scheduler)
    restored_lr_before_carry = float(optimizer.param_groups[0]["lr"])
    if args.lr_schedule == "carry_plateau":
        floor = max(args.scheduler_min_lr, args.carry_lr_min)
        for group, end_lr in zip(optimizer.param_groups, end_lrs):
            group["lr"] = max(floor, min(float(group["lr"]), end_lr))

    post_sel = frozen_heldout_score(model, policy_sel, qsel, args, tensors)
    post_qres = frozen_heldout_score(model, policy_res, qres, args, tensors)
    if model_hash(model) != selected["model_hash"]:
        raise RuntimeError("model hash mismatch after inner-best restore")
    if source is not None and model_hash(source) != source_hash:
        raise RuntimeError("outer-frozen policy source changed during inner training")

    info = {
        "outer_iter": int(outer_iter),
        "epochs_used": int(epochs_used),
        "selection_checks": int(selection_checks),
        "qres_checks": int(qres_checks),
        "stop_reason": stop_reason,
        "selected_inner_epoch": int(selected["epoch"]),
        "selected_model_hash": selected["model_hash"],
        "selected_train": dict(selected["train"]),
        "selected_sel": selected["sel"],
        "selected_qres": selected["qres"],
        "post_restore_sel": post_sel,
        "post_restore_frozen_qres": post_qres,
        "target_crossed_during_training": bool(target_crossed),
        "target_reached_post_restore": bool(
            args.pres_target is not None
            and post_qres["p_res"] <= args.pres_target
        ),
        "lr_end_before_restore": end_lrs[0],
        "lr_selected_checkpoint": restored_lr_before_carry,
        "lr_after_restore": float(optimizer.param_groups[0]["lr"]),
        "source_hash_verified": True,
    }
    return rows, info


def policy_mse_on_fixed_points(
    model: HomogeneousValueNet,
    frozen_policy: torch.Tensor,
    coordinates: Tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    tensors: Dict[str, torch.Tensor],
) -> float:
    tau, x = coordinates
    improved = evaluate_frozen_policy(
        model, tau, x, args, tensors, args.val_chunk
    )
    value = float((improved - frozen_policy).square().mean().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite policy-iteration difference")
    return value


def save_pipinn_checkpoint(
    path: Path,
    model: HomogeneousValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    args: argparse.Namespace,
    tag: str,
    market_hash: str,
    selection: Dict[str, Any],
    policy_source_state: Optional[Dict[str, torch.Tensor]],
    policy_source_kind: str,
) -> None:
    payload = {
        "method": METHOD_NAME,
        "schema": SCHEMA_NAME,
        "run_tag": tag,
        "market_fingerprint": market_hash,
        "final_outer": int(selection["final_outer"]),
        "selection": json_ready(selection),
        "model_state_dict": cpu_tree(model.state_dict()),
        "model_state_hash": model_hash(model),
        "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": (
            None if scheduler is None else cpu_tree(scheduler.state_dict())
        ),
        "final_policy_source_kind": policy_source_kind,
        "final_policy_source_state_dict": policy_source_state,
        "final_policy_source_hash": selection["final_policy_source_hash"],
        "config": json_ready(vars(args)),
    }
    torch_save_atomic(payload, path)


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
        raise ValueError("checkpoint market fingerprint does not match current market")
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
        "theta_init_method",
        "theta_init_scale",
        "val_points",
        "val_terminal_points",
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
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("checkpoint is missing model_state_dict")
    stored_model_hash = checkpoint.get("model_state_hash")
    if not isinstance(stored_model_hash, str):
        raise ValueError("checkpoint is missing model_state_hash")
    if state_dict_hash(checkpoint["model_state_dict"]) != stored_model_hash:
        raise ValueError("checkpoint model_state_dict does not match its stored hash")
    if checkpoint.get("final_policy_source_kind") == "network":
        source_state = checkpoint.get("final_policy_source_state_dict")
        source_hash = checkpoint.get("final_policy_source_hash")
        if not isinstance(source_state, dict) or not isinstance(source_hash, str):
            raise ValueError("checkpoint is missing its frozen network policy source")
        if state_dict_hash(source_state) != source_hash:
            raise ValueError(
                "checkpoint frozen policy-source state does not match its stored hash"
            )
    if int(checkpoint.get("final_outer", 0)) <= 0:
        raise ValueError("checkpoint has no completed final outer iteration")


def run_policy_iteration(
    model: HomogeneousValueNet,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    output_dir: Path,
    weight_dir: Path,
    tag: str,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    market_hash = market_fingerprint(arrays)
    qres_seed = int(arrays["market_seed"][0]) * 1_000_003 + 911
    qsel_seed = int(arrays["market_seed"][0]) * 1_000_003 + 2_027
    qres = build_fixed_set(
        args.val_points,
        args.val_terminal_points,
        qres_seed,
        args,
        arrays,
        device,
        dtype,
    )
    qsel = build_fixed_set(
        args.sel_points,
        args.sel_terminal_points,
        qsel_seed,
        args,
        arrays,
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
    inner_history: list[Dict[str, Any]] = []
    outer_history: list[Dict[str, Any]] = []
    global_step = 0
    final_source_state: Optional[Dict[str, torch.Tensor]] = None
    final_source_kind = ""
    final_source_hash = ""

    print("\n" + "=" * 76)
    print(
        f"Training homogeneous PI-PINN: N={args.n_assets}, M={args.m_states}, "
        f"outer={args.outer_iters}, inner={args.inner_epochs}"
    )
    print(
        f"policy init={args.theta_init_method}, schedule={args.lr_schedule}, "
        f"Adam reset={args.adam_reset}"
    )
    print("official rule: final outer after within-outer fixed-heldout restore")
    print("=" * 76)

    for outer_iter in range(1, args.outer_iters + 1):
        if outer_iter == 1:
            source = None
            source_kind = f"analytic:{args.theta_init_method}"
            source_hash = analytic_policy_source_hash(args)
            source_state = None
        else:
            source = freeze_policy_source(model)
            source_kind = "network"
            source_hash = model_hash(source)
            source_state = cpu_tree(source.state_dict())

        policy_sel = evaluate_frozen_policy(
            source, qsel[0][0], qsel[0][1], args, tensors, args.val_chunk
        )
        policy_res = evaluate_frozen_policy(
            source, qres[0][0], qres[0][1], args, tensors, args.val_chunk
        )
        if policy_sel.requires_grad or policy_res.requires_grad:
            raise RuntimeError("heldout frozen policy unexpectedly requires gradients")

        optimizer, scheduler, outer_start_lr = prepare_optimizer_for_outer(
            model, optimizer, outer_iter, args
        )
        rows, info = train_one_outer(
            model,
            optimizer,
            scheduler,
            source,
            source_hash,
            qsel,
            qres,
            policy_sel,
            policy_res,
            outer_iter,
            global_step,
            args,
            arrays,
            tensors,
            device,
            dtype,
        )
        inner_history.extend(rows)
        global_step += int(info["epochs_used"])
        theta_diff = policy_mse_on_fixed_points(
            model, policy_sel, qsel[0], args, tensors
        )
        last_real = next(
            (row for row in reversed(rows) if not row.get("synthetic")), None
        )
        outer_row = {
            "outer_iter": outer_iter,
            "policy_source_kind": source_kind,
            "policy_source_hash": source_hash,
            "outer_start_lr": outer_start_lr,
            "inner_epochs_used": info["epochs_used"],
            "selected_inner_epoch": info["selected_inner_epoch"],
            "selection_checks": info["selection_checks"],
            "qres_checks": info["qres_checks"],
            "stop_reason": info["stop_reason"],
            "last_train_total": None if last_real is None else last_real["total"],
            "last_train_pde": None if last_real is None else last_real["pde"],
            "last_train_terminal": None if last_real is None else last_real["terminal"],
            "accepted_train_total": info["selected_train"]["total"],
            "accepted_train_pde": info["selected_train"]["pde"],
            "accepted_train_terminal": info["selected_train"]["terminal"],
            "inner_best_sel_p_res": info["post_restore_sel"]["p_res"],
            "post_restore_frozen_policy_pde_rms": info[
                "post_restore_frozen_qres"
            ]["pde_rms"],
            "post_restore_terminal_rms": info["post_restore_frozen_qres"][
                "terminal_rms"
            ],
            "post_restore_frozen_policy_p_res": info[
                "post_restore_frozen_qres"
            ]["p_res"],
            "theta_diff_mse": theta_diff,
            "target_crossed_during_training": info[
                "target_crossed_during_training"
            ],
            "target_reached_post_restore": info["target_reached_post_restore"],
            "lr_end_before_restore": info["lr_end_before_restore"],
            "lr_selected_checkpoint": info["lr_selected_checkpoint"],
            "lr_after_restore": info["lr_after_restore"],
            "accepted_model_hash": info["selected_model_hash"],
            "source_hash_verified": info["source_hash_verified"],
        }
        outer_history.append(outer_row)
        final_source_state = source_state
        final_source_kind = source_kind
        final_source_hash = source_hash

        if (
            outer_iter == 1
            or outer_iter % args.print_every_outer == 0
            or outer_iter == args.outer_iters
        ):
            def metric_text(value: Optional[float]) -> str:
                return "undefined" if value is None else f"{value:.3e}"

            print(
                f"outer={outer_iter}/{args.outer_iters} "
                f"inner_best={info['selected_inner_epoch']} "
                f"train_total="
                f"{metric_text(outer_row['accepted_train_total'])} "
                f"train_pde="
                f"{metric_text(outer_row['accepted_train_pde'])} "
                f"train_terminal="
                f"{metric_text(outer_row['accepted_train_terminal'])} "
                f"val_pde_rmse="
                f"{outer_row['post_restore_frozen_policy_pde_rms']:.3e} "
                f"val_terminal_rmse="
                f"{outer_row['post_restore_terminal_rms']:.3e} "
                f"val_p_res={outer_row['post_restore_frozen_policy_p_res']:.3e} "
                f"theta_diff={theta_diff:.3e} lr={info['lr_after_restore']:.3e}"
            )

    final_nonlinear = None
    if not args.skip_eval:
        final_nonlinear = heldout_residual_score(
            model,
            qres[0],
            qres[1],
            args.gamma,
            args.r,
            tensors,
            args.val_chunk,
        )

    final_info = outer_history[-1]
    selection = {
        "official_rule": "final_outer_after_inner_fixed_heldout_restore",
        "official_checkpoint": str(weight_dir / "value_net_final_outer.pt"),
        "final_outer": int(args.outer_iters),
        "final_inner_best_epoch": int(final_info["selected_inner_epoch"]),
        "final_inner_best_sel_p_res": final_info["inner_best_sel_p_res"],
        "final_post_restore_frozen_policy_p_res": final_info[
            "post_restore_frozen_policy_p_res"
        ],
        "final_nonlinear_hjb": final_nonlinear,
        "final_model_hash": model_hash(model),
        "final_policy_source_kind": final_source_kind,
        "final_policy_source_hash": final_source_hash,
        "pres_target": args.pres_target,
        "final_target_reached_post_restore": (
            None
            if args.pres_target is None
            else bool(final_info["target_reached_post_restore"])
        ),
        "all_outer_targets_reached_post_restore": (
            None
            if args.pres_target is None
            else all(bool(row["target_reached_post_restore"]) for row in outer_history)
        ),
        "total_inner_optimizer_steps": int(global_step),
        "qsel_seed": int(qsel_seed),
        "qres_seed": int(qres_seed),
        "test_seed": 727,
        "market_fingerprint": market_hash,
    }
    checkpoint_path = weight_dir / "value_net_final_outer.pt"
    save_pipinn_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        args,
        tag,
        market_hash,
        selection,
        final_source_state,
        final_source_kind,
    )
    write_rows(output_dir / "inner_history.csv", inner_history)
    write_rows(output_dir / "outer_history.csv", outer_history)
    write_json(output_dir / "selection.json", selection)
    return inner_history, outer_history, selection


def reconstruct_policy_source(
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[HomogeneousValueNet]:
    kind = checkpoint.get("final_policy_source_kind")
    state = checkpoint.get("final_policy_source_state_dict")
    if isinstance(kind, str) and kind.startswith("analytic:"):
        if state is not None:
            raise ValueError("analytic final policy source unexpectedly has network state")
        if kind != f"analytic:{args.theta_init_method}":
            raise ValueError("analytic policy-source kind does not match current config")
        if checkpoint.get("final_policy_source_hash") != analytic_policy_source_hash(args):
            raise ValueError("analytic final policy-source hash mismatch")
        return None
    if kind != "network" or not isinstance(state, dict):
        raise ValueError("checkpoint is missing its final frozen policy source")
    source = HomogeneousValueNet(
        M=args.m_states,
        hidden=args.value_hidden,
        depth=args.value_depth,
        g_floor=args.g_floor,
    ).to(device=device, dtype=dtype)
    source.load_state_dict(state)
    source = freeze_policy_source(source)
    expected = checkpoint.get("final_policy_source_hash")
    if model_hash(source) != expected:
        raise ValueError("final frozen policy source hash mismatch")
    return source


def eval_only_frozen_audit(
    model: HomogeneousValueNet,
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    arrays: Dict[str, np.ndarray],
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    source = reconstruct_policy_source(checkpoint, args, device, dtype)
    qres_seed = int(arrays["market_seed"][0]) * 1_000_003 + 911
    qres = build_fixed_set(
        args.val_points,
        args.val_terminal_points,
        qres_seed,
        args,
        arrays,
        device,
        dtype,
    )
    policy = evaluate_frozen_policy(
        source, qres[0][0], qres[0][1], args, tensors, args.val_chunk
    )
    measured = frozen_heldout_score(model, policy, qres, args, tensors)
    recorded = checkpoint.get("selection", {}).get(
        "final_post_restore_frozen_policy_p_res"
    )
    if recorded is None or not math.isfinite(float(recorded)):
        raise ValueError("checkpoint has no finite recorded final frozen-policy p_res")
    difference = abs(measured["p_res"] - float(recorded))
    if args.dtype == "float64":
        relative_tolerance, absolute_tolerance = 1e-10, 1e-12
    else:
        relative_tolerance, absolute_tolerance = 5e-6, 5e-7
    if not math.isclose(
        measured["p_res"],
        float(recorded),
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise ValueError(
            "eval-only final frozen-policy residual does not reproduce the "
            f"checkpoint: measured={measured['p_res']:.12g}, "
            f"recorded={float(recorded):.12g}"
        )
    return {
        "qres_seed": qres_seed,
        "measured": measured,
        "recorded_p_res": recorded,
        "absolute_p_res_difference": difference,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def positive_for_log(value: Any) -> float:
    if value is None or not math.isfinite(float(value)):
        return np.nan
    return max(float(value), 1e-30)


def plot_pi_convergence(
    outer_history: list[Dict[str, Any]], path: Path, dpi: int
) -> None:
    iterations = np.asarray([row["outer_iter"] for row in outer_history])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].semilogy(
        iterations,
        [positive_for_log(row["post_restore_frozen_policy_p_res"]) for row in outer_history],
        linewidth=1.6,
    )
    axes[0].set_title("Post-restore frozen-policy residual")
    axes[0].set_xlabel("outer iteration")
    axes[0].set_ylabel(r"$p_{res}$")

    axes[1].semilogy(
        iterations,
        [positive_for_log(row["theta_diff_mse"]) for row in outer_history],
        linewidth=1.6,
    )
    axes[1].set_title("Successive normalized-policy MSE")
    axes[1].set_xlabel("outer iteration")
    axes[1].set_ylabel(r"MSE$(\vartheta_{n+1},\vartheta_n)$")

    axes[2].semilogy(
        iterations,
        [positive_for_log(row["lr_after_restore"]) for row in outer_history],
        linewidth=1.6,
    )
    axes[2].set_title("Accepted learning rate")
    axes[2].set_xlabel("outer iteration")
    axes[2].set_ylabel("learning rate")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_inner_history(
    inner_history: list[Dict[str, Any]], path: Path, dpi: int
) -> None:
    real = [row for row in inner_history if not row.get("synthetic")]
    sel_checks = [row for row in inner_history if row.get("sel_p_res") is not None]
    qres_checks = [row for row in inner_history if row.get("qres_p_res") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    if real:
        steps = [row["global_step"] for row in real]
        for key, label in (("total", "total"), ("pde", "PDE"), ("terminal", "terminal")):
            axes[0].semilogy(
                steps,
                [positive_for_log(row[key]) for row in real],
                label=label,
                linewidth=1.2,
            )
    axes[0].set_title("Inner policy-evaluation losses")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    if real:
        axes[0].legend()
    axes[0].grid(alpha=0.3)

    if sel_checks:
        positions = [row["plot_step"] for row in sel_checks]
        axes[1].semilogy(
            positions,
            [positive_for_log(row["sel_p_res"]) for row in sel_checks],
            label="Q_sel frozen p_res",
            linewidth=1.2,
        )
    if qres_checks:
        axes[1].semilogy(
            [row["plot_step"] for row in qres_checks],
            [positive_for_log(row["qres_p_res"]) for row in qres_checks],
            label="Q_res frozen p_res",
            linewidth=1.2,
        )
    axes[1].set_title("Within-outer heldout selection")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel(r"$p_{res}$")
    if sel_checks or qres_checks:
        axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
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

    # Build and, for eval-only, fully validate all numerical checkpoint state
    # before creating any output files.  Rejected evaluations therefore leave
    # no partial config, market snapshot, or result directory behind.
    tensors = tensor_market(arrays, device, dtype)
    model = HomogeneousValueNet(
        M=args.m_states,
        hidden=args.value_hidden,
        depth=args.value_depth,
        g_floor=args.g_floor,
    ).to(device=device, dtype=dtype)
    pre_eval_frozen_audit: Optional[Dict[str, Any]] = None
    if args.eval_only:
        model.load_state_dict(checkpoint["model_state_dict"])
        if model_hash(model) != checkpoint.get("model_state_hash"):
            raise ValueError("loaded model state hash does not match checkpoint")
        # Source provenance is structural and is checked even when the user
        # skips numerical post-training evaluation.
        reconstruct_policy_source(checkpoint, args, device, dtype)
        if not args.skip_eval:
            pre_eval_frozen_audit = eval_only_frozen_audit(
                model, checkpoint, args, arrays, tensors, device, dtype
            )

    config = {
        **vars(args),
        "method": METHOD_NAME,
        "schema": SCHEMA_NAME,
        "resolved_run_tag": tag,
        "resolved_output_dir": str(output_dir),
        "market_fingerprint": current_market_hash,
    }
    write_json(output_dir / ("config_eval.json" if args.eval_only else "config.json"), config)
    print_market_report(params, arrays["rho"])
    save_market_snapshot(output_dir / "market_params.npz", arrays)
    print(
        f"state domain: min={np.array2string(arrays['X_min'], precision=4)}, "
        f"max={np.array2string(arrays['X_max'], precision=4)}"
    )

    solution = None
    if not args.skip_eval or not args.skip_figures:
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

    inner_history: list[Dict[str, Any]] = []
    outer_history: list[Dict[str, Any]] = []
    started = time.perf_counter()
    if args.eval_only:
        selection = dict(checkpoint.get("selection", {}))
        selection["official_checkpoint"] = str(args.checkpoint)
        selection["official_rule"] = "user-specified final-outer checkpoint"
        if pre_eval_frozen_audit is not None:
            selection["eval_only_frozen_audit"] = pre_eval_frozen_audit
    else:
        inner_history, outer_history, selection = run_policy_iteration(
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
        if inner_history:
            plot_inner_history(
                inner_history, output_dir / "inner_loss_history.png", args.figure_dpi
            )
        if outer_history:
            plot_pi_convergence(
                outer_history, output_dir / "pi_convergence.png", args.figure_dpi
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
            method="pipinn",
            run_tag=tag,
            market_hash=current_market_hash,
        )
        plot_g_comparison(
            slice_data,
            output_dir / "g_tau_state",
            args.figure_dpi,
            method="pipinn",
            output_mode=args.figure_output_mode,
        )
        plot_policy_comparison(
            slice_data,
            output_dir / "policy_tau_state",
            args.max_plot_assets,
            args.figure_dpi,
            method="pipinn",
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
