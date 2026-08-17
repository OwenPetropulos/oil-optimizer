"""
signal.py — Turn DCF fair values into per-week expected-return signals.

The optimiser needs two inputs: a covariance matrix (from covariance.py) and a
vector of expected returns. This module produces the second one, honestly.

The value premise: a stock trading below its DCF fair value is expected to rise
toward that value, so its expected return is positive; above fair value, the
reverse. The raw measure of mispricing is the fractional gap:

    raw_gap = (fair_value - price) / price

But that raw gap is NOT a usable expected return, for two reasons this module
handles explicitly:

  1. The extremes are the least trustworthy. An 80% gap on a mature large-cap
     oil name is almost always a bad DCF input, not a real 80% opportunity. We
     compress the gap through a scaled tanh so it stays near-linear in the
     believable range but smoothly saturates in the implausible tail — extreme
     gaps still rank high but contribute far less than their raw magnitude. This
     is a smooth regularisation, NOT a hard cap (no discontinuity).

  2. The raw gap is a TOTAL expected move with no time scale, while the optimiser
     works against a WEEKLY covariance matrix. We divide the compressed gap by an
     assumed correction horizon (in weeks) so expected return and risk share one
     time scale. Without this the optimiser's risk/return tradeoff is on
     mismatched units and is silently meaningless.

Both the compression scale and the correction horizon are documented research
choices living in SignalConfig, meant to be swept in a sensitivity analysis
rather than trusted as exact truths. Neither changes the RANKING of names; they
set how much conviction and what time scale the signal carries.

No-look-ahead note: this module is pure. It receives the fair values and prices
the caller hands it for a single point in time and transforms them. It fetches
nothing and reaches for no future data; point-in-time discipline lives in the
caller that assembles these inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SignalConfig


# ------------------------------------------------------------------
# 1. Raw price-to-fair-value gap
# ------------------------------------------------------------------
def raw_gap(
    fair_values: pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """Fractional gap between fair value and price, per ticker.

    (fair_value - price) / price. Positive = cheap (price below fair value).

    fair_values, prices:
        Series indexed by ticker, aligned to the same universe. A missing fair
        value or a missing/zero price yields NaN for that name — an HONEST
        'no signal', never a fake 0.0. A fake zero would assert 'I expect exactly
        no return' about a name we actually know nothing about, and the optimiser
        would treat that assertion as real information.
    """
    fv = fair_values.astype(float)
    px = prices.astype(float)

    # Guard: non-positive prices are meaningless denominators -> NaN, not inf.
    safe_px = px.where(px > 0.0, other=np.nan)

    return (fv - safe_px) / safe_px


# ------------------------------------------------------------------
# 2. Smooth compression of the gap (tanh)
# ------------------------------------------------------------------
def compress_gap(
    gap: pd.Series,
    config: SignalConfig,
) -> pd.Series:
    """Compress the raw gap with a scaled tanh.

    compressed = scale * tanh(gap / scale)

    Near-linear for |gap| << scale (the trusted range keeps its honest
    proportions), smoothly flattening for |gap| >> scale (implausible gaps are
    tamed). NaNs pass through untouched as 'no signal'. This is the smooth
    alternative to a hard cap: an 80% and a 40% gap stay distinct and correctly
    ordered, but the marginal conviction of extra gap shrinks the further out
    you go.
    """
    scale = config.compression_scale
    return scale * np.tanh(gap / scale)


# ------------------------------------------------------------------
# 3. Horizon scaling -> per-week expected return
# ------------------------------------------------------------------
def to_weekly_expected_return(
    compressed_gap: pd.Series,
    config: SignalConfig,
) -> pd.Series:
    """Convert a compressed TOTAL expected move into a per-week expected return.

    Divides by the assumed correction horizon (weeks) so the signal is on the
    same weekly footing as the weekly covariance matrix. This is a pure scale
    change — every name is divided by the same number, so the ranking of names
    is untouched; only the overall magnitude (and thus optimiser aggressiveness)
    changes.
    """
    return compressed_gap / float(config.correction_horizon_weeks)


# ------------------------------------------------------------------
# Full pipeline: fair values + prices -> weekly expected returns
# ------------------------------------------------------------------
def expected_returns(
    fair_values: pd.Series,
    prices: pd.Series,
    config: SignalConfig,
) -> pd.Series:
    """End-to-end signal: raw gap -> tanh compression -> per-week expected return.

    Returns a Series indexed by ticker, aligned to the input universe, carrying
    NaN for any name with a missing fair value or non-positive/missing price.
    The caller (optimiser/backtest) decides how to treat NaN names — typically
    by excluding them from that week's optimisation. Producing NaN rather than
    a silent 0 keeps that decision explicit.
    """
    gap = raw_gap(fair_values, prices)
    compressed = compress_gap(gap, config)
    return to_weekly_expected_return(compressed, config)
