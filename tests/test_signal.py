"""
test_signal.py — Tests for the signal module.

As in test_covariance, each test states the INDEPENDENT truth it checks.
"""

import numpy as np
import pandas as pd
import pytest

from oil_optimizer.config import SignalConfig
from oil_optimizer import signal as sig


TICKERS = ["XOM", "CVX", "SHEL", "TTE", "BP", "COP",
           "OXY", "CNQ", "SU", "FANG", "EOG", "PBR"]


def test_compression_is_monotonic():
    """Independent truth: tanh is monotonic, so a bigger raw gap must never
    produce a SMALLER compressed signal. If ordering ever inverts, the value
    premise (cheaper => higher expected return) is broken."""
    config = SignalConfig()
    gaps = pd.Series(np.linspace(-0.5, 1.0, 25))
    compressed = sig.compress_gap(gaps, config)
    diffs = np.diff(compressed.to_numpy())
    assert np.all(diffs >= 0), "compression must preserve order of gaps"


def test_compression_tames_extremes():
    """Independent truth: an 80% gap must compress to LESS than 80% (it saturates),
    while a small gap stays close to linear. This is the whole point of tanh."""
    config = SignalConfig()  # scale 0.30
    extreme = sig.compress_gap(pd.Series([0.80]), config).iloc[0]
    small = sig.compress_gap(pd.Series([0.05]), config).iloc[0]
    assert extreme < 0.80, "extreme gap must be compressed below its raw value"
    assert small == pytest.approx(0.05, abs=0.002), "small gap stays near-linear"


def test_horizon_preserves_ranking():
    """Independent truth: horizon scaling is division by a constant, so it CANNOT
    reorder names — only rescale magnitudes. The ranking at horizon=13 must be
    identical to the ranking at horizon=52."""
    fv = pd.Series([110, 95, 120, 105, 130, 101], index=TICKERS[:6], dtype=float)
    px = pd.Series([88, 95, 105, 100, 72, 100], index=TICKERS[:6], dtype=float)
    er52 = sig.expected_returns(fv, px, SignalConfig(correction_horizon_weeks=52))
    er13 = sig.expected_returns(fv, px, SignalConfig(correction_horizon_weeks=13))
    assert list(er52.rank().values) == list(er13.rank().values)


def test_broken_inputs_yield_nan_not_zero():
    """Independent truth (honesty property): a missing fair value or a
    non-positive price means 'no signal' and MUST be NaN, never a fake 0.0 that
    would falsely assert 'I expect exactly zero return' about an unknown name."""
    fv = pd.Series([110.0, np.nan, 100.0], index=["A", "B", "C"])
    px = pd.Series([88.0, 95.0, 0.0], index=["A", "B", "C"])  # C price = 0
    er = sig.expected_returns(fv, px, SignalConfig())
    assert not np.isnan(er["A"]), "valid name must have a real signal"
    assert np.isnan(er["B"]), "missing fair value must be NaN"
    assert np.isnan(er["C"]), "zero price must be NaN, not a fake zero"


def test_fair_valued_name_gives_zero_signal():
    """Independent truth: a stock trading exactly at fair value has zero
    mispricing, so its expected excess return must be ~0."""
    fv = pd.Series([100.0], index=["X"])
    px = pd.Series([100.0], index=["X"])
    er = sig.expected_returns(fv, px, SignalConfig())
    assert er["X"] == pytest.approx(0.0, abs=1e-12)


def test_cheap_name_positive_expensive_name_negative():
    """Independent truth: below fair value => expect a rise (positive signal);
    above fair value => expect a fall (negative signal)."""
    fv = pd.Series([120.0, 80.0], index=["cheap", "expensive"])
    px = pd.Series([100.0, 100.0], index=["cheap", "expensive"])
    er = sig.expected_returns(fv, px, SignalConfig())
    assert er["cheap"] > 0
    assert er["expensive"] < 0
