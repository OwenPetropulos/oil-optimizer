"""
test_backtest.py — The no-look-ahead proof, plus backtest/metrics sanity tests.

The centrepiece is test_no_lookahead: it proves the backtest's decisions cannot
see the future. The method — scramble all data AFTER a given week and assert
every weight decision up to that week is byte-for-byte unchanged — is a direct
demonstration, not a spot-check. If a decision ever depended on future data,
scrambling the future would change it, and the test would fail.
"""

import numpy as np
import pandas as pd
import pytest

from oil_optimizer.config import Config, CovarianceConfig
from oil_optimizer import backtest as bt
from oil_optimizer import metrics as met


TICKERS = ["XOM", "CVX", "SHEL", "TTE", "BP", "COP",
           "OXY", "CNQ", "SU", "FANG", "EOG", "PBR"]


def make_synthetic(seed: int = 0, n_weeks: int = 180):
    """Synthetic prices and point-in-time fair values for the whole universe.

    Prices follow a correlated random walk (one crude factor + idio). Fair values
    are prices perturbed by a persistent mispricing, so there is a mild, real
    signal — enough for the loop to act on without being trivially perfect.
    """
    rng = np.random.default_rng(seed)
    p = len(TICKERS)
    crude = rng.normal(0, 0.03, n_weeks)
    idio = rng.normal(0, 0.02, (n_weeks, p))
    log_rets = crude[:, None] * rng.uniform(0.8, 1.2, p) + idio
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    idx = pd.date_range("2022-01-07", periods=n_weeks, freq="W-FRI")
    prices = pd.DataFrame(prices, index=idx, columns=TICKERS)

    # Fair values = price * (1 + persistent mispricing) — a mild signal.
    mispricing = rng.normal(0, 0.15, (n_weeks, p))
    fair_values = prices * (1.0 + mispricing)
    return prices, fair_values


def small_config():
    """A config with a short covariance window so tests run on modest data."""
    cfg = Config()
    # Shrink the window and floor so a ~180-week synthetic set exercises the loop.
    object.__setattr__(cfg, "covariance",
                       CovarianceConfig(window_weeks=52, min_observations=30))
    return cfg


# ==================================================================
# THE no-look-ahead proof
# ==================================================================
def test_no_lookahead():
    """Independent truth: decisions at/through week k use only data up to k, so
    corrupting all data AFTER k must leave every weight decision through k
    unchanged. If any decision peeked forward, scrambling the future would move
    it and this assertion would fail."""
    prices, fair_values = make_synthetic(seed=1)
    config = small_config()

    # Baseline run on clean data.
    base = bt.run_backtest(prices, fair_values, config)

    # Pick a cut point partway through, scramble everything strictly after it.
    cut = prices.index[len(prices) // 2]
    rng = np.random.default_rng(999)

    prices_corrupt = prices.copy()
    fv_corrupt = fair_values.copy()
    future = prices.index > cut
    # Replace all future prices and fair values with random garbage.
    prices_corrupt.loc[future] = rng.uniform(1, 500, prices_corrupt.loc[future].shape)
    fv_corrupt.loc[future] = rng.uniform(1, 500, fv_corrupt.loc[future].shape)

    corrupt = bt.run_backtest(prices_corrupt, fv_corrupt, config)

    # Every weight decision made AT OR BEFORE the cut must be identical.
    base_w = base.weights_history.loc[base.weights_history.index <= cut]
    corrupt_w = corrupt.weights_history.loc[corrupt.weights_history.index <= cut]

    assert base_w.shape == corrupt_w.shape, "decision count changed — timing bug"
    assert np.allclose(base_w.to_numpy(), corrupt_w.to_numpy(), atol=1e-12), \
        "a pre-cut decision changed when the future was scrambled — LOOK-AHEAD BUG"


# ==================================================================
# Backtest sanity
# ==================================================================
def test_backtest_runs_and_shapes():
    """Independent truth: the loop produces one net return per earned week, and
    weights each week sum to ~1 (fully invested) when a position is held."""
    prices, fair_values = make_synthetic(seed=2)
    result = bt.run_backtest(prices, fair_values, small_config())
    assert len(result.weekly_returns) > 0
    # Rows that actually hold a position sum to ~1.
    wsum = result.weights_history.sum(axis=1)
    held = wsum[wsum > 1e-6]
    assert np.allclose(held.to_numpy(), 1.0, atol=1e-6)


def test_transaction_costs_reduce_returns():
    """Independent truth: charging a cost per unit turnover can only LOWER net
    returns versus a zero-cost run — never raise them."""
    prices, fair_values = make_synthetic(seed=3)
    from oil_optimizer.config import BacktestConfig

    cfg_cost = small_config()
    cfg_free = small_config()
    object.__setattr__(cfg_free, "backtest", BacktestConfig(cost_per_turnover=0.0))

    with_cost = bt.run_backtest(prices, fair_values, cfg_cost).weekly_returns.sum()
    no_cost = bt.run_backtest(prices, fair_values, cfg_free).weekly_returns.sum()
    assert with_cost <= no_cost + 1e-12


# ==================================================================
# Metrics sanity
# ==================================================================
def test_sharpe_of_constant_positive_series():
    """Independent truth: a constant positive return has zero volatility, so the
    metrics must not divide by zero — Sharpe is defined as 0 in that guard."""
    r = pd.Series([0.01] * 20)
    m = met.score(r)
    assert m.annualised_volatility == pytest.approx(0.0)
    assert m.sharpe_ratio == 0.0


def test_max_drawdown_is_negative_on_a_fall():
    """Independent truth: a series that rises then falls must report a negative
    max drawdown equal to the peak-to-trough fall."""
    # +10% then -20%: wealth 1.0 -> 1.1 -> 0.88, drawdown from peak = 0.88/1.1-1.
    r = pd.Series([0.10, -0.20])
    m = met.score(r)
    assert m.max_drawdown == pytest.approx(0.88 / 1.10 - 1.0, abs=1e-9)
