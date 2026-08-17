"""
optimizer.py — Constrained mean-variance portfolio optimization.

This is the heart of the project: it replaces the old strategy's arbitrary
hand-picked weight caps with a principled optimiser that chooses weights by
trading off expected return against risk. The max-weight cap does not disappear
— it changes job, from being the whole allocation rule to being a documented
risk guardrail sitting on top of real optimisation.

The problem solved each week:

    maximise    wᵀμ − (λ/2) · wᵀΣw
    subject to  Σ w = 1            (fully invested, no idle cash)
                0 ≤ wᵢ ≤ max_weight (long-only, per-name cap)

where μ is the per-week expected-return vector (from signal.py) and Σ is the
shrunk covariance matrix (from covariance.py). λ (risk_aversion) sets the
return-vs-risk tradeoff.

This is a quadratic program (quadratic objective, linear constraints). We solve
it numerically with scipy.optimize.minimize (SLSQP), which handles equality
constraints and bounds directly and adds no dependency beyond scipy. Validated
against the closed-form answer in the special cases where one exists (see tests).

Two honest failure modes are handled explicitly:
  * The solver can fail to converge — we check the success flag and RAISE rather
    than return unconverged garbage weights.
  * Some names have no signal (NaN) in a given week — we optimise only over the
    names we can actually value, and re-check the cap is still feasible for that
    smaller universe.

Solver-minimises note: scipy minimises, so we minimise the NEGATIVE of the
objective. Maximising (return − risk) == minimising (risk − return).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import OptimizerConfig


class OptimizationError(Exception):
    """Raised when the optimiser cannot return trustworthy weights.

    Covers solver non-convergence and infeasible problems (e.g. a per-name cap
    too small for the universe to sum to 1). Raised rather than returning
    best-effort weights so a failure is visible and located, never silently
    folded into the backtest as if it were a real allocation.
    """


@dataclass(frozen=True)
class OptimizationResult:
    """The optimiser's output plus the diagnostics that tell you if λ is sane.

    weights:
        Series indexed by ticker, summing to 1, each in [0, max_weight]. Only
        includes the names that had a valid signal this week.
    expected_return:
        The optimised portfolio's expected weekly return, wᵀμ.
    expected_vol:
        The optimised portfolio's expected weekly volatility, sqrt(wᵀΣw).
    n_binding:
        How many names hit the max-weight cap. Persistently high => λ too low
        (optimiser is jamming into the cap, signal dominating the risk model).
    drift_from_equal_weight:
        Sum of absolute differences between the optimised weights and equal
        weight. ~0 => optimiser barely tilts (λ too high, signal unused); large
        => strong tilt. The two diagnostics together are how λ is tuned by
        observation rather than by argument.
    """

    weights: pd.Series
    expected_return: float
    expected_vol: float
    n_binding: int
    drift_from_equal_weight: float


def _objective(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, lam: float) -> float:
    """Negative mean-variance utility (scipy minimises, so we negate).

    Returns  −( wᵀμ − (λ/2)·wᵀΣw )  =  (λ/2)·wᵀΣw − wᵀμ.
    """
    port_return = w @ mu
    port_variance = w @ cov @ w
    return 0.5 * lam * port_variance - port_return


def _objective_gradient(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    """Analytic gradient of the objective, d/dw [ (λ/2)wᵀΣw − wᵀμ ] = λΣw − μ.

    Supplying the exact gradient makes the solver faster and more reliable than
    letting it estimate one numerically.
    """
    return lam * (cov @ w) - mu


def optimize_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    config: OptimizerConfig,
) -> OptimizationResult:
    """Solve the constrained mean-variance problem for one rebalance.

    expected_returns:
        Per-week expected returns by ticker (from signal.py). Names with NaN are
        dropped — we optimise only over what we can value this week.
    covariance:
        Shrunk covariance matrix by ticker (from covariance.py). Must cover at
        least the valued names.
    config:
        OptimizerConfig carrying long_only, max_weight, fully_invested,
        risk_aversion.
    """
    # --- Align and drop names with no signal this week ---
    mu_series = expected_returns.dropna()
    valued = list(mu_series.index)
    if len(valued) == 0:
        raise OptimizationError("No names have a valid signal this week; cannot optimise.")

    # Restrict covariance to the valued names, in the SAME order as mu.
    cov_df = covariance.loc[valued, valued]
    mu = mu_series.to_numpy(dtype=float)
    cov = cov_df.to_numpy(dtype=float)
    n = len(valued)

    # --- Feasibility check: can n weights, each ≤ cap, sum to 1? ---
    # If max_weight * n < 1, no long-only fully-invested solution exists.
    if config.fully_invested and config.max_weight * n < 1.0 - 1e-12:
        raise OptimizationError(
            f"Infeasible: {n} names each capped at {config.max_weight} cannot sum "
            f"to 1 (max achievable {config.max_weight * n:.3f}). Raise max_weight "
            f"or widen the universe."
        )

    # --- Bounds: long-only lower bound 0 (or -inf if shorts allowed), cap upper ---
    lower = 0.0 if config.long_only else -np.inf
    bounds = [(lower, config.max_weight) for _ in range(n)]

    # --- Equality constraint: weights sum to 1 ---
    constraints = []
    if config.fully_invested:
        constraints.append({
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones_like(w),
        })

    # --- Start from equal weight (a neutral, feasible starting point) ---
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        _objective,
        w0,
        args=(mu, cov, config.risk_aversion),
        method="SLSQP",
        jac=_objective_gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    # --- Fail loud on non-convergence ---
    if not result.success:
        raise OptimizationError(
            f"Solver did not converge: {result.message}"
        )

    w = result.x
    # Clean tiny numerical dust: clip to bounds and renormalise to sum 1.
    w = np.clip(w, 0.0 if config.long_only else -np.inf, config.max_weight)
    if config.fully_invested:
        w = w / w.sum()

    weights = pd.Series(w, index=valued)

    # --- Diagnostics ---
    port_return = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    n_binding = int(np.sum(w >= config.max_weight - 1e-6))
    equal_weight = np.full(n, 1.0 / n)
    drift = float(np.sum(np.abs(w - equal_weight)))

    return OptimizationResult(
        weights=weights,
        expected_return=port_return,
        expected_vol=port_vol,
        n_binding=n_binding,
        drift_from_equal_weight=drift,
    )
