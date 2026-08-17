"""
test_covariance.py — Tests for the covariance module.

Each test states, in its docstring, the INDEPENDENT truth it checks — the
property that must hold regardless of how the code computes things. A test that
just re-runs the code's own logic would prove nothing; these assert against
mathematical properties, hand-known facts, and a separately-written reference
implementation.

Run from the project root with:  pytest
"""

import numpy as np
import pandas as pd
import pytest

from oil_optimizer.config import CovarianceConfig
from oil_optimizer import covariance as cov


# ------------------------------------------------------------------
# A shared helper to build realistic correlated-oil-basket returns.
# Not a test — just Arrange machinery the tests reuse.
# ------------------------------------------------------------------
TICKERS = ["XOM", "CVX", "SHEL", "TTE", "BP", "COP",
           "OXY", "CNQ", "SU", "FANG", "EOG", "PBR"]


def make_returns(seed: int = 0, n_weeks: int = 104, factors: int = 3,
                 idio: float = 0.03) -> pd.DataFrame:
    """Simulate weekly returns driven by 1-3 common factors plus noise.

    More factors + bigger idio => more dispersed correlations => a non-degenerate
    shrinkage problem (the target does not fit perfectly, so delta is interior).
    """
    rng = np.random.default_rng(seed)
    p = len(TICKERS)
    X = rng.normal(0, 0.045, n_weeks)[:, None] * rng.uniform(0.8, 1.3, p)
    if factors >= 2:
        X = X + rng.normal(0, 0.020, n_weeks)[:, None] * rng.uniform(-0.5, 0.5, p)
    if factors >= 3:
        X = X + rng.normal(0, 0.015, n_weeks)[:, None] * rng.uniform(-0.3, 0.4, p)
    X = X + rng.normal(0, idio, (n_weeks, p))
    idx = pd.date_range("2024-01-05", periods=n_weeks, freq="W-FRI")
    return pd.DataFrame(X, index=idx, columns=TICKERS)


# ------------------------------------------------------------------
# INDEPENDENT REFERENCE: a loop-form transcription of Ledoit-Wolf (2004)
# constant-correlation shrinkage. Written differently from the production
# code (explicit loops, no vectorization) so that agreement between the two
# is real evidence, not a copy of one implementation checking itself.
# ------------------------------------------------------------------
def reference_delta(values: np.ndarray) -> float:
    """Slow, explicit Ledoit-Wolf constant-correlation shrinkage intensity."""
    t, p = values.shape
    x = values - values.mean(axis=0)
    S = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            S[i, j] = np.mean(x[:, i] * x[:, j])
    v = np.sqrt(np.diag(S))
    corr = np.array([[S[i, j] / (v[i] * v[j]) for j in range(p)] for i in range(p)])
    rbar = np.mean([corr[i, j] for i in range(p) for j in range(p) if i != j])
    F = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            F[i, j] = v[i] * v[j] if i == j else rbar * v[i] * v[j]
    pi = 0.0
    for i in range(p):
        for j in range(p):
            pi += np.mean((x[:, i] * x[:, j] - S[i, j]) ** 2)
    rho = 0.0
    for i in range(p):
        rho += np.mean((x[:, i] ** 2 - S[i, i]) ** 2)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            th_ii = np.mean((x[:, i] ** 2 - S[i, i]) * (x[:, i] * x[:, j] - S[i, j]))
            th_jj = np.mean((x[:, j] ** 2 - S[j, j]) * (x[:, i] * x[:, j] - S[i, j]))
            rho += (rbar / 2.0) * ((v[j] / v[i]) * th_ii + (v[i] / v[j]) * th_jj)
    gamma = 0.0
    for i in range(p):
        for j in range(p):
            gamma += (F[i, j] - S[i, j]) ** 2
    return max(0.0, min(1.0, ((pi - rho) / gamma) / t))


# ==================================================================
# Tests
# ==================================================================
def test_refusal_on_short_window():
    """Independent truth: 40 clean rows < min_observations (60), so the
    estimator MUST refuse rather than produce a matrix from too little data."""
    short = make_returns(n_weeks=40)
    config = CovarianceConfig()
    with pytest.raises(cov.InsufficientDataError):
        cov.sample_covariance(short, config)
    with pytest.raises(cov.InsufficientDataError):
        cov.shrink_covariance(short, config)


def test_has_sufficient_data_predicate():
    """Independent truth: the cheap predicate must agree with the floor —
    True for a full window, False for a 40-row window."""
    config = CovarianceConfig()
    assert cov.has_sufficient_data(make_returns(n_weeks=104), config) is True
    assert cov.has_sufficient_data(make_returns(n_weeks=40), config) is False


def test_shrunk_matrix_is_valid_covariance():
    """Independent truth: ANY legitimate covariance matrix is symmetric and
    positive semi-definite (no negative eigenvalues). If shrinkage ever breaks
    this, the optimizer breaks — high-value guard."""
    result = cov.shrink_covariance(make_returns(), CovarianceConfig())
    M = result.covariance.to_numpy()
    assert np.allclose(M, M.T), "covariance must be symmetric"
    eigenvalues = np.linalg.eigvalsh(M)
    assert np.all(eigenvalues >= -1e-10), "covariance must be PSD"


def test_shrinkage_improves_conditioning():
    """Independent truth: regularizing toward a well-conditioned target cannot
    make conditioning WORSE. Shrunk condition number <= sample condition number."""
    diag = cov.covariance_diagnostics(make_returns(), CovarianceConfig())
    assert diag.condition_shrunk <= diag.condition_sample


def test_delta_in_unit_interval():
    """Independent truth: delta is a blend weight; it is meaningless outside
    [0, 1], so it must always be clamped into that range."""
    result = cov.shrink_covariance(make_returns(), CovarianceConfig())
    assert 0.0 <= result.shrinkage_delta <= 1.0


def test_delta_matches_reference():
    """Independent truth (the crown-jewel test): the production delta must match
    a SEPARATELY-WRITTEN loop-form reference implementation of the same paper's
    formulas. Two implementations written differently agreeing = real evidence
    the math is correct, not a copy checking itself."""
    returns = make_returns(seed=7)
    values = returns.dropna().to_numpy()
    values = values - values.mean(axis=0)  # match production's centering path
    # production path (re-centered inside), so feed raw and let it center:
    production = cov.shrink_covariance(returns, CovarianceConfig()).shrinkage_delta
    reference = reference_delta(returns.to_numpy())
    assert production == pytest.approx(reference, abs=1e-9)
