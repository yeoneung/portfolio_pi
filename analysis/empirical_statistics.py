"""Portfolio performance, paired block bootstrap, and Newey-West inference.

Monthly total returns determine growth, volatility, drawdown, and CRRA CER.
Sharpe uses excess returns. Mean-variance statistics are reported separately.
"""
from __future__ import annotations

import math
import numpy as np

Z_975 = 1.959963984540054

def _metrics(
    r: np.ndarray,
    turnover: np.ndarray | None = None,
    *,
    gamma: float = 5.0,
) -> dict[str, float]:
    """Excess-return metrics, including a mean-variance CER approximation.

    The exact plug-in CRRA CER is computed separately by _crra_cer.
    """
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return {k: float("nan") for k in (
            "ann_geometric_excess_return", "ann_volatility", "excess_sharpe",
            "mean_variance_cer", "max_drawdown", "avg_turnover",
        )}
    if np.any(1.0 + r <= 0.0):
        ann_ret = float("nan")
    else:
        ann_ret = float(np.expm1(12.0 * np.mean(np.log1p(r))))
    std = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
    var = float(std * std) if np.isfinite(std) else 0.0
    wealth = np.cumprod(1.0 + r)
    peaks = np.maximum.accumulate(wealth)
    return {
        "ann_geometric_excess_return": ann_ret,
        "ann_volatility": float(math.sqrt(12.0) * std),
        "excess_sharpe": float(math.sqrt(12.0) * np.mean(r) / std) if std > 0 else float("nan"),
        "mean_variance_cer": float(12.0 * (np.mean(r) - 0.5 * gamma * var)),
        "max_drawdown": float(np.min(wealth / peaks - 1.0)),
        "avg_turnover": float(np.mean(turnover)) if turnover is not None else float("nan"),
    }


def _crra_cer(total_return: np.ndarray, gamma: float) -> float:
    """Annualized CRRA CER in the manuscript, using monthly total returns."""
    total_return = np.asarray(total_return, dtype=float)
    gross = 1.0 + total_return
    if len(gross) == 0 or np.any(gross <= 0.0):
        return float("nan")
    if abs(gamma - 1.0) < 1.0e-12:
        return float(np.expm1(12.0 * np.mean(np.log(gross))))
    moment = float(np.mean(np.power(gross, 1.0 - gamma)))
    return float(moment ** (12.0 / (1.0 - gamma)) - 1.0)


def _total_metrics(
    total_return: np.ndarray,
    excess_return: np.ndarray,
    turnover: np.ndarray | None,
    *,
    gamma: float,
) -> dict[str, float]:
    """Total-return metrics while retaining an excess-return Sharpe."""
    total_return = np.asarray(total_return, dtype=float)
    excess_return = np.asarray(excess_return, dtype=float)
    if len(total_return) == 0:
        return {k: float("nan") for k in (
            "ann_geometric_total_return", "ann_total_volatility", "excess_sharpe",
            "mean_variance_total_cer", "crra_total_cer", "max_drawdown_total",
            "avg_turnover",
        )}
    ann_ret = (
        float(np.expm1(12.0 * np.mean(np.log1p(total_return))))
        if np.all(1.0 + total_return > 0.0) else float("nan")
    )
    total_std = float(np.std(total_return, ddof=1)) if len(total_return) > 1 else float("nan")
    excess_std = float(np.std(excess_return, ddof=1)) if len(excess_return) > 1 else float("nan")
    total_var = total_std * total_std if np.isfinite(total_std) else 0.0
    wealth = np.cumprod(1.0 + total_return)
    peaks = np.maximum.accumulate(wealth)
    return {
        "ann_geometric_total_return": ann_ret,
        "ann_total_volatility": float(math.sqrt(12.0) * total_std),
        "excess_sharpe": (
            float(math.sqrt(12.0) * np.mean(excess_return) / excess_std)
            if excess_std > 0 else float("nan")
        ),
        "mean_variance_total_cer": float(
            12.0 * (np.mean(total_return) - 0.5 * gamma * total_var)
        ),
        "crra_total_cer": _crra_cer(total_return, gamma),
        "max_drawdown_total": float(np.min(wealth / peaks - 1.0)),
        "avg_turnover": float(np.mean(turnover)) if turnover is not None else float("nan"),
    }


def _metrics_matrix(r: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    """Metrics for a (replications x months) matrix."""
    mean = np.mean(r, axis=1)
    std = np.std(r, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = math.sqrt(12.0) * mean / std
    log_growth = np.mean(np.log1p(r), axis=1)
    return {
        "ann_geometric_excess_return": np.expm1(12.0 * log_growth),
        "ann_volatility": math.sqrt(12.0) * std,
        "excess_sharpe": sharpe,
        "mean_variance_cer": 12.0 * (mean - 0.5 * gamma * std * std),
    }


def _total_metrics_matrix(total: np.ndarray, excess: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    """Metrics for (replications x months) matrices."""
    total_mean = np.mean(total, axis=1)
    total_std = np.std(total, axis=1, ddof=1)
    excess_mean = np.mean(excess, axis=1)
    excess_std = np.std(excess, axis=1, ddof=1)
    gross = 1.0 + total
    with np.errstate(divide="ignore", invalid="ignore"):
        ann_return = np.expm1(12.0 * np.mean(np.log(gross), axis=1))
        sharpe = math.sqrt(12.0) * excess_mean / excess_std
        if abs(gamma - 1.0) < 1.0e-12:
            crra = ann_return
        else:
            crra = np.power(
                np.mean(np.power(gross, 1.0 - gamma), axis=1),
                12.0 / (1.0 - gamma),
            ) - 1.0
    return {
        "ann_geometric_total_return": ann_return,
        "ann_total_volatility": math.sqrt(12.0) * total_std,
        "excess_sharpe": sharpe,
        "mean_variance_total_cer": 12.0 * (
            total_mean - 0.5 * gamma * total_std * total_std
        ),
        "crra_total_cer": crra,
    }


def _circular_block_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    reps: int,
    block_length: int,
    rng: np.random.Generator,
    gamma: float,
    chunk_size: int = 1000,
) -> dict[str, np.ndarray]:
    n = len(a)
    n_blocks = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)
    names = tuple(_metrics_matrix(a.reshape(1, -1), gamma))
    out = {name: np.empty(reps, dtype=float) for name in names}
    written = 0
    while written < reps:
        size = min(chunk_size, reps - written)
        starts = rng.integers(0, n, size=(size, n_blocks))
        idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        ma = _metrics_matrix(a[idx], gamma)
        mb = _metrics_matrix(b[idx], gamma)
        for name in names:
            out[name][written:written + size] = ma[name] - mb[name]
        written += size
    return out


def _circular_block_bootstrap_total(
    total_a: np.ndarray,
    excess_a: np.ndarray,
    total_b: np.ndarray,
    excess_b: np.ndarray,
    *,
    reps: int,
    block_length: int,
    rng: np.random.Generator,
    gamma: float,
    chunk_size: int = 1000,
) -> dict[str, np.ndarray]:
    n = len(total_a)
    n_blocks = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)
    names = tuple(_total_metrics_matrix(
        total_a.reshape(1, -1), excess_a.reshape(1, -1), gamma,
    ))
    out = {name: np.empty(reps, dtype=float) for name in names}
    written = 0
    while written < reps:
        size = min(chunk_size, reps - written)
        starts = rng.integers(0, n, size=(size, n_blocks))
        idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        ma = _total_metrics_matrix(total_a[idx], excess_a[idx], gamma)
        mb = _total_metrics_matrix(total_b[idx], excess_b[idx], gamma)
        for name in names:
            out[name][written:written + size] = ma[name] - mb[name]
        written += size
    return out


def _newey_west_long_run_cov(x: np.ndarray, lags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    centered = x - np.mean(x, axis=0, keepdims=True)
    cov = centered.T @ centered / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma_l = centered[lag:].T @ centered[:-lag] / n
        cov += weight * (gamma_l + gamma_l.T)
    return cov


def _hac_rows(universe: str, a: np.ndarray, b: np.ndarray, gamma: float, lags: int) -> list[dict]:
    n = len(a)
    diff = a - b
    long_run_diff = _newey_west_long_run_cov(diff[:, None], lags)[0, 0]
    mean_est = float(12.0 * np.mean(diff))
    mean_se = float(12.0 * math.sqrt(max(long_run_diff, 0.0) / n))

    moments = np.column_stack((a, b, a * a, b * b))
    means = np.mean(moments, axis=0)
    ma, mb, m2a, m2b = means
    variance_scale = n / (n - 1.0) if n > 1 else 1.0
    gradient = np.array((12.0 * (1.0 + gamma * variance_scale * ma),
                         12.0 * (-1.0 - gamma * variance_scale * mb),
                         -6.0 * gamma * variance_scale,
                         6.0 * gamma * variance_scale))
    lrcov = _newey_west_long_run_cov(moments, lags)
    cer_se = float(math.sqrt(max(float(gradient @ lrcov @ gradient) / n, 0.0)))
    cer_est = (
        _metrics(a, gamma=gamma)["mean_variance_cer"]
        - _metrics(b, gamma=gamma)["mean_variance_cer"]
    )

    return [
        {
            "universe": universe,
            "metric": "annualized_arithmetic_excess_mean",
            "estimate": mean_est,
            "hac_se": mean_se,
            "ci_low": mean_est - Z_975 * mean_se,
            "ci_high": mean_est + Z_975 * mean_se,
            "months": n,
            "newey_west_lags": lags,
        },
        {
            "universe": universe,
            "metric": "mean_variance_cer",
            "estimate": cer_est,
            "hac_se": cer_se,
            "ci_low": cer_est - Z_975 * cer_se,
            "ci_high": cer_est + Z_975 * cer_se,
            "months": n,
            "newey_west_lags": lags,
        },
    ]


def _hac_total_rows(
    universe: str,
    total_a: np.ndarray,
    excess_a: np.ndarray,
    total_b: np.ndarray,
    excess_b: np.ndarray,
    gamma: float,
    lags: int,
) -> list[dict]:
    """Delta-method Newey-West intervals for total-return metrics."""
    del excess_a, excess_b  # Sharpe inference is intentionally not linearized here.
    n = len(total_a)
    diff = total_a - total_b
    long_run_diff = _newey_west_long_run_cov(diff[:, None], lags)[0, 0]
    mean_est = float(12.0 * np.mean(diff))
    mean_se = float(12.0 * math.sqrt(max(long_run_diff, 0.0) / n))

    moments = np.column_stack((total_a, total_b, total_a * total_a, total_b * total_b))
    ma, mb, m2a, m2b = np.mean(moments, axis=0)
    variance_scale = n / (n - 1.0) if n > 1 else 1.0
    gradient = np.array((
        12.0 * (1.0 + gamma * variance_scale * ma),
        12.0 * (-1.0 - gamma * variance_scale * mb),
        -6.0 * gamma * variance_scale,
        6.0 * gamma * variance_scale,
    ))
    lrcov = _newey_west_long_run_cov(moments, lags)
    mv_cer_se = float(math.sqrt(max(float(gradient @ lrcov @ gradient) / n, 0.0)))
    mv_cer_est = (
        _total_metrics(total_a, total_a, None, gamma=gamma)["mean_variance_total_cer"]
        - _total_metrics(total_b, total_b, None, gamma=gamma)["mean_variance_total_cer"]
    )

    gross_a = 1.0 + total_a
    gross_b = 1.0 + total_b
    if np.any(gross_a <= 0.0) or np.any(gross_b <= 0.0):
        crra_est = crra_se = float("nan")
    elif abs(gamma - 1.0) < 1.0e-12:
        log_moments = np.column_stack((np.log(gross_a), np.log(gross_b)))
        lma, lmb = np.mean(log_moments, axis=0)
        crra_est = float(np.expm1(12.0 * lma) - np.expm1(12.0 * lmb))
        crra_gradient = np.array((
            12.0 * np.exp(12.0 * lma),
            -12.0 * np.exp(12.0 * lmb),
        ))
        crra_lrcov = _newey_west_long_run_cov(log_moments, lags)
        crra_se = float(math.sqrt(max(
            float(crra_gradient @ crra_lrcov @ crra_gradient) / n, 0.0,
        )))
    else:
        power_a = np.power(gross_a, 1.0 - gamma)
        power_b = np.power(gross_b, 1.0 - gamma)
        power_moments = np.column_stack((power_a, power_b))
        pma, pmb = np.mean(power_moments, axis=0)
        exponent = 12.0 / (1.0 - gamma)
        crra_gradient = np.array((
            exponent * pma ** (exponent - 1.0),
            -exponent * pmb ** (exponent - 1.0),
        ))
        crra_lrcov = _newey_west_long_run_cov(power_moments, lags)
        crra_se = float(math.sqrt(max(
            float(crra_gradient @ crra_lrcov @ crra_gradient) / n, 0.0,
        )))
        crra_est = _crra_cer(total_a, gamma) - _crra_cer(total_b, gamma)

    rows = []
    for metric, estimate, se in (
        ("annualized_arithmetic_total_mean", mean_est, mean_se),
        ("mean_variance_total_cer", mv_cer_est, mv_cer_se),
        ("crra_total_cer", crra_est, crra_se),
    ):
        rows.append({
            "universe": universe,
            "metric": metric,
            "estimate": estimate,
            "hac_se": se,
            "ci_low": estimate - Z_975 * se,
            "ci_high": estimate + Z_975 * se,
            "months": n,
            "newey_west_lags": lags,
        })
    return rows


def _cost_coefficients(r: np.ndarray, turnover: np.ndarray, gamma: float) -> tuple[float, float, float]:
    r = np.asarray(r, dtype=float)
    turnover = np.asarray(turnover, dtype=float)
    var_r = float(np.var(r, ddof=1))
    var_t = float(np.var(turnover, ddof=1))
    cov_rt = float(np.cov(r, turnover, ddof=1)[0, 1])
    return (
        12.0 * (float(np.mean(r)) - 0.5 * gamma * var_r),
        12.0 * (-float(np.mean(turnover)) + gamma * cov_rt),
        -6.0 * gamma * var_t,
    )


def _first_positive_root(coeff: tuple[float, float, float]) -> float:
    c0, c1, c2 = coeff
    if abs(c2) < 1.0e-15:
        if abs(c1) < 1.0e-15:
            return float("nan")
        root = -c0 / c1
        return float(root) if root >= 0 else float("nan")
    roots = np.roots([c2, c1, c0])
    valid = sorted(float(x.real) for x in roots if abs(x.imag) < 1.0e-10 and x.real >= 0)
    return valid[0] if valid else float("nan")


def _crra_break_even_cost(
    total_a: np.ndarray,
    turnover_a: np.ndarray,
    total_b: np.ndarray,
    turnover_b: np.ndarray,
    gamma: float,
    *,
    max_bps: float = 1000.0,
) -> float:
    """First cost where a positive zero-cost CRRA-CER advantage disappears."""
    def objective(unit_cost: float) -> float:
        return (
            _crra_cer(total_a - unit_cost * turnover_a, gamma)
            - _crra_cer(total_b - unit_cost * turnover_b, gamma)
        )

    f0 = objective(0.0)
    if not np.isfinite(f0) or f0 <= 0.0:
        return float("nan")
    grid = np.linspace(0.0, max_bps / 10000.0, 2001)
    previous_x = 0.0
    for x in grid[1:]:
        fx = objective(float(x))
        if not np.isfinite(fx):
            break
        if fx <= 0.0:
            lo, hi = previous_x, float(x)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if objective(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        previous_x = float(x)
    return float("nan")
