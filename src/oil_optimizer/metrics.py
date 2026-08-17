"""
metrics.py — Performance scoring for a strategy's weekly return series.

Turns a series of realised weekly portfolio returns into the standard numbers
used to judge a strategy: annualised return, annualised volatility, Sharpe
ratio, and maximum drawdown. Kept deliberately small and pure — it receives a
return series and computes; it knows nothing about how the returns were made.

Annualisation note: these are WEEKLY returns, and there are ~52 weeks a year.
Returns scale with time (×52); volatility scales with the SQUARE ROOT of time
(×√52), because variance is additive over independent periods and volatility is
its square root. This is why a Sharpe ratio annualises by √52, not 52.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class PerformanceMetrics:
    """The scorecard for a return series.

    annualised_return:
        Mean weekly return scaled to a year (× 52). What the strategy makes per
        year on average, in the same units as the input returns (decimal).
    annualised_volatility:
        Standard deviation of weekly returns scaled to a year (× √52). The risk.
    sharpe_ratio:
        (annualised_return − annualised risk-free) / annualised_volatility.
        Return earned per unit of risk taken. The honest headline metric: it
        does not reward taking more risk to get more return.
    max_drawdown:
        The worst peak-to-trough decline of cumulative wealth, as a negative
        number (e.g. -0.23 = a 23% fall from a prior high). What the investor
        would actually have had to endure.
    n_periods:
        Number of weekly returns scored (context for the above).
    """

    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    n_periods: int


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of cumulative wealth.

    Builds the wealth curve (compounding the returns), tracks the running peak,
    and finds the deepest fall below a prior peak. Returned as a negative number.
    """
    if len(returns) == 0:
        return 0.0
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def score(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full scorecard from a weekly return series.

    returns:
        Series of realised weekly returns (decimal, e.g. 0.004 = +0.4%).
    risk_free_annual:
        Annual risk-free rate to subtract in the Sharpe numerator. Default 0.0
        (excess-over-cash can be layered in later); kept explicit so the choice
        is visible rather than hidden.
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0)

    mean_weekly = float(r.mean())
    vol_weekly = float(r.std(ddof=1)) if n > 1 else 0.0

    ann_return = mean_weekly * WEEKS_PER_YEAR
    ann_vol = vol_weekly * np.sqrt(WEEKS_PER_YEAR)

    # Sharpe: excess annual return per unit annual volatility. Guard the
    # zero-vol case (a constant or near-constant series) rather than dividing by
    # a vanishingly small number, which would produce a meaningless huge Sharpe.
    # We treat volatility below a tiny epsilon as effectively zero.
    if ann_vol > 1e-12:
        sharpe = (ann_return - risk_free_annual) / ann_vol
    else:
        sharpe = 0.0

    return PerformanceMetrics(
        annualised_return=ann_return,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown(r),
        n_periods=n,
    )
