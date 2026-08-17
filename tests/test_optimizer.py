"""
test_optimizer.py — Tests for the constrained mean-variance optimizer.

The crown-jewel test here is test_matches_min_variance_closed_form: when the
signal is flat and the cap is not binding, the mean-variance solution must equal
the analytical minimum-variance portfolio. We deliberately build a covariance
matrix whose unconstrained min-variance weights are already non-negative, so the
long-only constraint is slack and the solver MUST reproduce the closed form.
That is an independent mathematical truth, not the code checking itself.
"""

import numpy as np
import pandas as pd
import pytest

from oil_optimizer.config import OptimizerConfig
from oil_optimizer import optimizer as opt


TICKERS = ["XOM", "CVX", "SHEL", "TTE", "BP", "COP",
           "OXY", "CNQ", "SU", "FANG", "EOG", "PBR"]


def diagonal_cov(seed: int = 0) -> pd.DataFrame:
    """A DIAGONAL covariance (uncorrelated names, different variances).

    For a diagonal Sigma the unconstrained min-variance weights are proportional
    to 1/variance — always strictly positive — so the long-only constraint is
    guaranteed slack and the solver must match the closed form exactly.
    """
    rng = np.random.default_rng(seed)
    variances = rng.uniform(0.02, 0.08, len(TICKERS))
    Sigma = np.diag(variances)
    return pd.DataFrame(Sigma, index=TICKERS, columns=TICKERS)


def dense_cov(seed: int = 1) -> pd.DataFrame:
    """A valid dense positive-definite covariance for constraint tests."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (len(TICKERS), len(TICKERS)))
    Sigma = A @ A.T / len(TICKERS) + np.eye(len(TICKERS)) * 0.02
    return pd.DataFrame(Sigma, index=TICKERS, columns=TICKERS)


def test_matches_min_variance_closed_form():
    """Independent truth: flat signal + non-binding cap => the mean-variance
    optimum IS the minimum-variance portfolio, whose closed form is
    w = Σ⁻¹1 / (1ᵀΣ⁻¹1). Built on a diagonal Σ so long-only is slack."""
    cov = diagonal_cov()
    Sigma = cov.to_numpy()
    inv = np.linalg.inv(Sigma)
    ones = np.ones(len(TICKERS))
    w_closed_form = inv @ ones / (ones @ inv @ ones)

    mu_flat = pd.Series(np.zeros(len(TICKERS)), index=TICKERS)
    # Cap off (1.0), high risk aversion so it is purely min-variance.
    config = OptimizerConfig(max_weight=1.0, risk_aversion=50.0)
    result = opt.optimize_weights(mu_flat, cov, config)

    assert np.allclose(result.weights.to_numpy(), w_closed_form, atol=1e-4)


def test_weights_sum_to_one():
    """Independent truth: fully-invested means weights sum to exactly 1."""
    mu = pd.Series(np.random.default_rng(0).normal(0.002, 0.003, len(TICKERS)), index=TICKERS)
    result = opt.optimize_weights(mu, dense_cov(), OptimizerConfig())
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_long_only_respected():
    """Independent truth: long-only means every weight is non-negative."""
    mu = pd.Series(np.random.default_rng(1).normal(0.002, 0.003, len(TICKERS)), index=TICKERS)
    result = opt.optimize_weights(mu, dense_cov(), OptimizerConfig())
    assert (result.weights >= -1e-9).all()


def test_max_weight_cap_respected():
    """Independent truth: no weight may exceed the configured cap."""
    mu = pd.Series(np.random.default_rng(2).normal(0.005, 0.005, len(TICKERS)), index=TICKERS)
    cap = 0.30
    result = opt.optimize_weights(mu, dense_cov(), OptimizerConfig(max_weight=cap))
    assert (result.weights <= cap + 1e-6).all()


def test_infeasible_cap_raises():
    """Independent truth: n names each capped below 1/n cannot sum to 1, so the
    problem is infeasible and must raise rather than return bogus weights."""
    mu = pd.Series(np.zeros(len(TICKERS)), index=TICKERS)
    # 12 names capped at 0.05 -> max achievable 0.60 < 1 -> infeasible.
    with pytest.raises(opt.OptimizationError):
        opt.optimize_weights(mu, dense_cov(), OptimizerConfig(max_weight=0.05))


def test_nan_signal_names_dropped():
    """Independent truth: names with no signal (NaN) must be excluded from the
    optimisation, not fed in as zeros. Result covers only the valued names."""
    mu = pd.Series(np.random.default_rng(3).normal(0.002, 0.003, len(TICKERS)), index=TICKERS)
    mu[["OXY", "PBR"]] = np.nan
    result = opt.optimize_weights(mu, dense_cov(), OptimizerConfig())
    assert len(result.weights) == len(TICKERS) - 2
    assert "OXY" not in result.weights.index
    assert "PBR" not in result.weights.index
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_lower_lambda_tilts_more_or_equal():
    """Independent truth: a LOWER risk aversion cares less about variance, so it
    should tilt at least as far from equal weight as a higher one (weakly more
    aggressive). Checks the sign of lambda's effect is correct."""
    mu = pd.Series(np.random.default_rng(5).normal(0.003, 0.004, len(TICKERS)), index=TICKERS)
    cov = dense_cov()
    aggressive = opt.optimize_weights(mu, cov, OptimizerConfig(risk_aversion=0.1))
    cautious = opt.optimize_weights(mu, cov, OptimizerConfig(risk_aversion=50.0))
    assert aggressive.drift_from_equal_weight >= cautious.drift_from_equal_weight - 1e-6
