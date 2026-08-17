"""
backtest.py — Weekly walk-forward backtest with transaction costs.

This is where the modules connect and run. Walking forward one week at a time,
at each rebalance date t the loop:

  1. Reads the prices and point-in-time fair values available AS OF t.
  2. Builds the expected-return signal from them (signal.py).
  3. Estimates the shrunk covariance from the trailing return window ENDING at t
     (covariance.py).
  4. If the window is too thin, HOLDS last week's weights (no trade, no cost).
  5. Otherwise optimises to get target weights (optimizer.py).
  6. Charges transaction cost on turnover (sum of |Δ weight|) vs last week.
  7. Earns the portfolio return from t to t+1 — data the strategy could NOT see
     when it chose weights.

No-look-ahead discipline is structural, not incidental. The DECISION at t uses
only data up to and including t; the RETURN is measured from t to t+1. These are
separate, clearly-marked steps, and test_no_lookahead.py proves that shuffling
future data cannot change any decision.

Data contract (match this when wiring real data):
  prices:       DataFrame, index = weekly dates, columns = tickers, values =
                prices. Must be sorted ascending by date.
  fair_values:  DataFrame, same index/columns, values = point-in-time DCF fair
                values. ALREADY lagged so each row contains only what was public
                that week — the backtest trusts this and does not re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import Config
from . import covariance as cov
from . import signal as sig
from . import optimizer as opt


@dataclass(frozen=True)
class BacktestResult:
    """Everything the backtest produces.

    weekly_returns:
        Realised NET (after cost) weekly portfolio returns, indexed by the week
        in which they were earned. The series metrics.py scores.
    weights_history:
        DataFrame of target weights chosen at each rebalance (dates × tickers).
    turnover_history:
        Series of turnover (sum |Δ weight|) at each rebalance.
    cost_history:
        Series of transaction costs charged at each rebalance.
    n_holds:
        How many weeks the loop held (thin data or optimiser failure) instead of
        re-optimising. A high count is a data-coverage warning worth surfacing.
    """

    weekly_returns: pd.Series
    weights_history: pd.DataFrame
    turnover_history: pd.Series
    cost_history: pd.Series
    n_holds: int


def _turnover(new_w: pd.Series, old_w: pd.Series, universe: List[str]) -> float:
    """Sum of absolute weight changes across the union of names.

    Reindexed onto the full universe (missing = 0 weight) so a name entering or
    leaving the book counts its full weight as turnover, which is what actually
    gets traded.
    """
    a = new_w.reindex(universe).fillna(0.0)
    b = old_w.reindex(universe).fillna(0.0)
    return float(np.abs(a - b).sum())


def run_backtest(
    prices: pd.DataFrame,
    fair_values: pd.DataFrame,
    config: Config,
) -> BacktestResult:
    """Run the weekly walk-forward backtest.

    Returns realised net weekly returns plus weights/turnover/cost history.
    """
    # --- Align inputs: same dates, same tickers, sorted ascending ---
    prices = prices.sort_index()
    fair_values = fair_values.sort_index()
    dates = prices.index
    universe = list(prices.columns)

    # Precompute the weekly simple returns of every name (used for both the
    # covariance window and the realised portfolio return).
    asset_returns = sig_ret = prices.pct_change()  # first row NaN by construction

    window = config.covariance.window_weeks

    weekly_returns: dict = {}
    weights_history: dict = {}
    turnover_history: dict = {}
    cost_history: dict = {}
    n_holds = 0

    prev_weights = pd.Series(dtype=float)  # empty = no position yet

    # We can only start once we have a full covariance window of returns behind
    # us, and we stop one short of the end (need t+1 to measure the return).
    for i in range(window, len(dates) - 1):
        t = dates[i]
        t_next = dates[i + 1]

        # ---------- DECISION at t: uses ONLY data up to and including t ----------
        # Trailing return window ending at t (rows i-window+1 .. i inclusive).
        window_returns = asset_returns.iloc[i - window + 1 : i + 1]

        target_weights: Optional[pd.Series] = None

        if cov.has_sufficient_data(window_returns, config.covariance):
            # Signal from point-in-time fair values and prices AS OF t.
            fv_t = fair_values.loc[t]
            px_t = prices.loc[t]
            mu = sig.expected_returns(fv_t, px_t, config.signal)

            # Shrunk covariance from the trailing window.
            try:
                shrunk = cov.shrink_covariance(window_returns, config.covariance)
                result = opt.optimize_weights(mu, shrunk.covariance, config.optimizer)
                target_weights = result.weights
            except (cov.InsufficientDataError, opt.OptimizationError):
                target_weights = None  # fall through to hold

        if target_weights is None:
            # ---------- HOLD: thin data or optimiser failure ----------
            n_holds += 1
            if prev_weights.empty:
                # No position yet and cannot form one — sit in cash this week.
                weekly_returns[t_next] = 0.0
                continue
            target_weights = prev_weights.copy()
            turnover = 0.0  # held: no trade
        else:
            turnover = _turnover(target_weights, prev_weights, universe)

        # ---------- COST charged on turnover ----------
        cost = turnover * config.backtest.cost_per_turnover

        # ---------- RETURN earned from t to t+1 (unseeable at decision time) ----------
        next_returns = asset_returns.loc[t_next]
        gross = float((target_weights.reindex(universe).fillna(0.0)
                       * next_returns.reindex(universe).fillna(0.0)).sum())
        net = gross - cost

        weekly_returns[t_next] = net
        weights_history[t] = target_weights.reindex(universe).fillna(0.0)
        turnover_history[t] = turnover
        cost_history[t] = cost
        prev_weights = target_weights

    return BacktestResult(
        weekly_returns=pd.Series(weekly_returns).sort_index(),
        weights_history=pd.DataFrame(weights_history).T.sort_index(),
        turnover_history=pd.Series(turnover_history).sort_index(),
        cost_history=pd.Series(cost_history).sort_index(),
        n_holds=n_holds,
    )
