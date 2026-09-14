import numpy as np
import torch

from dynalloc_v2.pipinn_backend import (
    _qp_optimality_diagnostics,
    _solve_qp_long_only_budget_full,
)


def test_common_sigma_qp_matches_active_set_solution_and_reports_kkt_residual():
    sigma_np = np.array(
        [[0.50, 0.08, 0.02], [0.08, 0.35, 0.04], [0.02, 0.04, 0.20]],
        dtype=float,
    )
    signal_np = np.array([0.24, 0.18, -0.03], dtype=float)
    gamma = 5.0
    cap = 0.65
    sigma = torch.tensor(sigma_np, dtype=torch.float64)
    signal = torch.tensor(signal_np[None, :], dtype=torch.float64)
    weights = _solve_qp_long_only_budget_full(
        sigma,
        signal,
        gamma,
        cap,
        iters=300,
        tol=1.0e-10,
    )
    diag = _qp_optimality_diagnostics(sigma, signal, weights, gamma, cap)

    # The third coordinate is inactive; the first two solve the unrestricted
    # stationarity equations and their sum is below the budget cap.
    reference = np.zeros(3)
    determinant = sigma_np[0, 0] * sigma_np[1, 1] - sigma_np[0, 1] ** 2
    reference[0] = (
        sigma_np[1, 1] * signal_np[0] - sigma_np[0, 1] * signal_np[1]
    ) / (gamma * determinant)
    reference[1] = (
        sigma_np[0, 0] * signal_np[1] - sigma_np[0, 1] * signal_np[0]
    ) / (gamma * determinant)
    assert reference.sum() < cap
    assert (gamma * sigma_np[2] @ reference - signal_np[2]) > 0.0
    np.testing.assert_allclose(weights.numpy().ravel(), reference, atol=2.0e-8, rtol=0.0)
    assert diag["qp_projected_kkt_inf"] <= 1.1e-10
    assert diag["qp_nonneg_violation"] == 0.0
    assert diag["qp_budget_violation"] <= 1.0e-12


def test_common_sigma_qp_binding_budget():
    sigma = torch.eye(2, dtype=torch.float64)
    signal = torch.ones((1, 2), dtype=torch.float64)
    weights = _solve_qp_long_only_budget_full(
        sigma,
        signal,
        gamma=1.0,
        cap=1.0,
        iters=300,
        tol=1.0e-10,
    )
    np.testing.assert_allclose(weights.numpy().ravel(), [0.5, 0.5], atol=2.0e-10, rtol=0.0)
    diag = _qp_optimality_diagnostics(sigma, signal, weights, gamma=1.0, cap=1.0)
    assert diag["qp_projected_kkt_inf"] <= 2.0e-10
    assert diag["qp_nonneg_violation"] == 0.0
    assert diag["qp_budget_violation"] <= 1.0e-12
