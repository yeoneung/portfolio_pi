"""Policy-uniform finite-horizon exit/payoff bounds for the scalar predictor.

The tilted state need not be Gaussian: its adapted additional drift is bounded.
Exponential-martingale maxima on a finite time partition control its exit.
Every scalar operation contributing to a reported bound uses interval arithmetic.
"""
from fractions import Fraction as F

import portfolio_welfare as w


def bound(market, initial_z=F(0), partitions=(1, 2, 4, 8, 16, 32, 64)):
    market.validate()
    if abs(initial_z) >= market.radius:
        raise ValueError("Initial point must be strictly interior")
    _, _, drift, cplus = w.coefficient_bounds(market)
    if market.radius < drift/market.kappa:
        raise ValueError("Radius must dominate the maximal shifted equilibrium")
    if not partitions or any(type(n) is not int or n < 1 for n in partitions):
        raise ValueError("Positive integer partition counts required")
    w.iv.dps = 50
    a0 = market.kappa/market.state_variance/F(200)
    lam0 = w.lyapunov_lambda(a0, market)
    # The prescribed affine trace is positive and <=1, including its corners.
    w.trace_slopes(market)
    payoff = w.iv.exp(w.mpiv(lam0*market.horizon+a0*market.radius**2))+1
    potential = w.iv.exp(w.mpiv(cplus*market.horizon))
    best = None
    for n in partitions:
        total = w.mpiv(0)
        for j in range(1, n+1):
            left, right = market.horizon*F(j-1, n), market.horizon*F(j, n)
            level = (w.iv.exp(w.mpiv(market.kappa*left))*w.mpiv(market.radius-drift/market.kappa)
                     +w.mpiv(drift/market.kappa-abs(initial_z)))
            variance = w.mpiv(market.state_variance/(2*market.kappa))*(w.iv.exp(w.mpiv(2*market.kappa*right))-1)
            total += 2*w.iv.exp(-level**2/(2*variance))
        probability = min(1., w.upper(total))
        value = w.upper(potential*payoff*w.mpiv(probability))
        if best is None or value < best["value_localization_upper"]:
            best = {"value_localization_upper": value, "exit_probability_upper": probability,
                    "partition_count": n, "initial_state_exact": str(initial_z),
                    "a0_exact": str(a0), "lambda0_exact": str(lam0),
                    "drift_bound_exact": str(drift), "cplus_exact": str(cplus),
                    "continuation_discrepancy_upper": w.upper(payoff),
                    "method": "finite-partition exponential-martingale exit bound"}
    return best


def bounds(market, point_exact, certificate, initial_z=F(0), factor_lower=None, localizer=bound):
    """The q=-1 fixed-policy theorem, with the finite-horizon localization."""
    if not certificate.get("certified", False):
        raise ValueError("A successful full-cylinder certificate is required")
    w.iv.dps = 50
    cplus = w.coefficient_bounds(market)[3]
    phi = ((w.iv.exp(w.mpiv(cplus*market.horizon))-1)/w.mpiv(cplus)
           if cplus else w.mpiv(market.horizon))
    loc = localizer(market, initial_z)
    eta, gap = w.mpiv(certificate["frozen_residual_upper"]), w.mpiv(certificate["hamiltonian_gap_upper"])
    ell = w.mpiv(loc["value_localization_upper"])
    error, loss = phi*eta+ell, phi*(2*eta+gap)+2*ell
    value = w.mpiv(point_exact if factor_lower is None else factor_lower)
    lower = w.lower(value-error)
    if lower <= 0:
        raise ArithmeticError("No positive deployed-factor lower bound")
    result = {**loc, "evaluation_factor_error_upper": w.upper(error),
              "unit_wealth_utility_loss_upper": w.upper(loss), "policy_factor_lower": lower,
              "relative_ce_loss_bp_upper": w.upper(10000*loss/w.mpiv(lower)),
              "budget_bp": {"evaluation": w.upper(10000*2*phi*eta/w.mpiv(lower)),
                            "improvement_gap": w.upper(10000*phi*gap/w.mpiv(lower)),
                            "localization": w.upper(10000*2*ell/w.mpiv(lower))}}
    if factor_lower is None:
        upper = w.upper(value+error)
        result.update({"evaluation_initial_exact": str(point_exact),
                       "whole_space_policy_factor_interval": [lower, upper],
                       "whole_space_policy_ce_interval": [w.lower(1/w.mpiv(upper)), w.upper(1/w.mpiv(lower))]})
    else:
        result["scope"] = "all 0<=t<=T and |z|<=the stated initial-state radius"
    return result
