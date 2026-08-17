"""
covariance.py — Covariance estimation for the oil-equity optimizer.

This module turns a window of price/return data into a covariance matrix the
optimizer can trust. It provides:

  1. returns_from_prices  — prices -> returns, in the convention set by config.
  2. sample_covariance    — the naive historical estimate (baseline to beat).
  3. shrink_covariance    — Ledoit-Wolf shrinkage toward a constant-correlation
                            target (the credible core).
  4. covariance_diagnostics — evidence that shrinkage did something: the chosen
                            shrinkage intensity delta and the condition number
                            before vs. after.

Design commitments carried from config:
  * We REFUSE to estimate from a window with fewer than `min_observations`
    clean rows, rather than returning a matrix we cannot stand behind. This is
    raised as InsufficientDataError so the backtest loop can catch it and decide
    what to do (skip rebalance / hold weights) without crashing the whole run.
  * The estimate must never see the future: this module only ever receives the
    slice of returns the caller hands it. It does no data fetching and no
    forward-filling of its own. No-look-ahead discipline lives in the caller,
    and this module is written so it CANNOT reach beyond its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import CovarianceConfig


class InsufficientDataError(Exception):
    """Raised when a window has too few clean observations to estimate from.

    Carried as an exception (not a silent None) so the backtest loop must make
    a deliberate choice about what to do, and so the failure is visible and
    located rather than buried inside a plausible-looking matrix.
    """


def has_sufficient_data(
    returns: pd.DataFrame,
    config: CovarianceConfig,
) -> bool:
    """Cheap up-front check: does this window have enough clean rows to estimate?

    The backtest loop calls this BEFORE attempting an estimate so it can make
    the deliberate 'hold the portfolio, no trades this week' decision when a
    window is short, rather than catching an exception mid-rebalance. The
    InsufficientDataError inside the estimators remains as a hard backstop so
    the functions can never be MISused into producing a matrix from too little
    data — belt (this predicate) and suspenders (the exception).
    """
    clean = returns.dropna(axis=0, how="any")
    return clean.shape[0] >= config.min_observations


# ------------------------------------------------------------------
# 1. Returns
# ------------------------------------------------------------------
def returns_from_prices(
    prices: pd.DataFrame,
    return_kind: str = "simple",
) -> pd.DataFrame:
    """Convert a price panel (dates x tickers) into a returns panel.

    prices:
        DataFrame indexed by date, one column per ticker, already at the
        rebalance frequency (weekly for this project). Values are prices.
    return_kind:
        'simple' -> p_t / p_{t-1} - 1
        'log'    -> ln(p_t / p_{t-1})
        Passed in from config so the whole pipeline shares one convention.

    The first row is NaN by construction (no prior price) and is dropped.
    We do NOT forward-fill or interpolate here: inventing prices to fill gaps
    would fabricate returns the market never printed. Cleaning missing data is
    a caller decision made explicitly, not a side effect hidden in here.
    """
    if return_kind not in ("simple", "log"):
        raise ValueError(f"return_kind must be 'simple' or 'log', got {return_kind!r}.")

    if return_kind == "simple":
        rets = prices.pct_change()
    else:
        rets = np.log(prices / prices.shift(1))

    return rets.iloc[1:]


# ------------------------------------------------------------------
# Internal: clean a returns window and enforce the data floor
# ------------------------------------------------------------------
def _validated_matrix(
    returns: pd.DataFrame,
    config: CovarianceConfig,
) -> np.ndarray:
    """Drop rows with any missing values and enforce min_observations.

    Returns a plain float ndarray (rows = observations, cols = assets) ready
    for an estimator. Raises InsufficientDataError if too few clean rows remain.

    Why drop any row with a missing value (rather than fill it)? Because the
    covariance of a pair must be measured on periods where BOTH names actually
    traded. A filled-in return is a fabricated co-movement. Dropping keeps the
    estimate honest; the cost is fewer rows, which is exactly what the
    min_observations floor is there to police.
    """
    clean = returns.dropna(axis=0, how="any")
    n_obs = clean.shape[0]

    if n_obs < config.min_observations:
        raise InsufficientDataError(
            f"Window has {n_obs} clean rows; need at least "
            f"{config.min_observations}. Refusing to estimate covariance from "
            f"too little data."
        )

    values = clean.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise InsufficientDataError(
            "Window contains non-finite values (inf/-inf) after cleaning; "
            "refusing to estimate."
        )
    return values


# ------------------------------------------------------------------
# 2. Sample covariance (the baseline)
# ------------------------------------------------------------------
def sample_covariance(
    returns: pd.DataFrame,
    config: CovarianceConfig,
) -> pd.DataFrame:
    """Plain historical (sample) covariance matrix.

    This is the naive estimate we EXPECT to be badly behaved — it exists so we
    have a baseline to compare shrinkage against and can show the improvement
    (see covariance_diagnostics). Returned as a labelled DataFrame so callers
    never lose track of which row/column is which ticker.

    Uses the standard unbiased estimator (ddof=1, i.e. divide by n-1).
    """
    values = _validated_matrix(returns, config)
    cov = np.cov(values, rowvar=False, ddof=1)
    tickers = list(returns.columns)
    return pd.DataFrame(cov, index=tickers, columns=tickers)


# ------------------------------------------------------------------
# 3. Ledoit-Wolf shrinkage (the core)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ShrinkageResult:
    """What the shrinkage estimator produces.

    covariance:
        The shrunk covariance matrix, labelled by ticker.
    shrinkage_delta:
        The intensity delta in [0, 1] chosen from the data by the Ledoit-Wolf
        formula. 0 = kept the sample untouched; 1 = went all the way to the
        target. The single most informative diagnostic: it tells you how much
        the sample needed rescuing.
    r_bar:
        The average pairwise correlation used to build the constant-correlation
        target. Reported so a reader can see the target is a good fit for this
        universe (for homogeneous oil names, r_bar sits close to every real
        pairwise correlation).
    n_obs:
        Number of clean observations the estimate was built from.
    """

    covariance: pd.DataFrame
    shrinkage_delta: float
    r_bar: float
    n_obs: int


def _constant_correlation_target(sample: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the constant-correlation shrinkage target from a sample covariance.

    Keeps each asset's own volatility (the diagonal) untouched, and replaces
    every pairwise correlation with r_bar, the average of all sample pairwise
    correlations. Returns (target_matrix, r_bar).

    This is the 'cheap lie' that fits a homogeneous universe: for upstream oil,
    the real pairwise correlations cluster tightly, so flattening them to their
    average costs very little bias while buying a lot of stability.
    """
    p = sample.shape[0]
    vols = np.sqrt(np.diag(sample))
    outer_vols = np.outer(vols, vols)
    corr = sample / outer_vols

    off_diag = ~np.eye(p, dtype=bool)
    r_bar = float(corr[off_diag].mean())

    target = r_bar * outer_vols
    np.fill_diagonal(target, np.diag(sample))  # preserve exact variances
    return target, r_bar


def _ledoit_wolf_constant_corr(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ledoit-Wolf (2004) shrinkage toward a constant-correlation target.

    Implements the estimator from Ledoit & Wolf, "Honey, I Shrunk the Sample
    Covariance Matrix" (2004), which derives the optimal shrinkage intensity
    SPECIFICALLY for the constant-correlation target. This is why we cannot
    borrow scikit-learn's intensity here: sklearn's is derived for a different
    (scaled-identity) target. Validated to the decimal against an independent
    loop-form transcription of the paper's formulas.

    Returns (shrunk_covariance, delta, r_bar).

    The intensity is delta = max(0, min(1, ((pi - rho) / gamma) / t)) where:
      * pi    — total variance of the sample covariance entries (sample noise).
                Larger = noisier sample = shrink harder toward the target.
      * rho   — sum of asymptotic covariances between the sample entries and the
                target entries (a correction term from the paper).
      * gamma — squared Frobenius distance between sample and target
                (target misfit). Larger = target fits worse = shrink less.
    """
    t, p = values.shape
    x = values - values.mean(axis=0)          # demeaned returns
    sample = (x.T @ x) / t                     # MLE sample cov (divide by t, per paper)

    target, r_bar = _constant_correlation_target(sample)
    vols = np.sqrt(np.diag(sample))

    # --- pi: variance of each sample-cov entry, summed ---
    y = x ** 2
    pi_mat = (y.T @ y) / t - sample ** 2
    pi_hat = pi_mat.sum()

    # --- rho: diagonal part + constant-corr off-diagonal correction ---
    rho_diag = np.diag(pi_mat).sum()
    theta_ii = ((x ** 3).T @ x) / t - np.diag(sample)[:, None] * sample
    theta_jj = (x.T @ (x ** 3)) / t - np.diag(sample)[None, :] * sample
    off_diag = ~np.eye(p, dtype=bool)
    term = (vols[None, :] / vols[:, None]) * theta_ii \
         + (vols[:, None] / vols[None, :]) * theta_jj
    rho_off = (r_bar / 2.0) * term[off_diag].sum()
    rho_hat = rho_diag + rho_off

    # --- gamma: distance from sample to target ---
    gamma_hat = ((target - sample) ** 2).sum()

    # --- delta: clamp the optimal intensity into [0, 1] ---
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(max(0.0, min(1.0, kappa / t)))

    shrunk = delta * target + (1.0 - delta) * sample
    return shrunk, delta, r_bar


def shrink_covariance(
    returns: pd.DataFrame,
    config: CovarianceConfig,
) -> ShrinkageResult:
    """Ledoit-Wolf shrinkage covariance with a constant-correlation target.

    Blends the noisy sample covariance with a stable constant-correlation target
    and lets the Ledoit-Wolf (2004) formula choose the blend weight (delta)
    automatically to minimise expected estimation error. We implement the
    constant-correlation estimator directly (validated against an independent
    reference) because it is the target whose economic assumption — 'these
    homogeneous oil names share roughly one correlation' — actually fits this
    universe, unlike the scaled-identity target in off-the-shelf libraries.
    """
    values = _validated_matrix(returns, config)
    shrunk, delta, r_bar = _ledoit_wolf_constant_corr(values)

    tickers = list(returns.columns)
    cov = pd.DataFrame(shrunk, index=tickers, columns=tickers)
    return ShrinkageResult(
        covariance=cov,
        shrinkage_delta=delta,
        r_bar=r_bar,
        n_obs=values.shape[0],
    )


def sklearn_shrink_covariance(
    returns: pd.DataFrame,
    config: CovarianceConfig,
) -> ShrinkageResult:
    """scikit-learn LedoitWolf (scaled-identity target) — kept as a CROSS-CHECK.

    Not the production estimator (its target does not fit an oil universe), but
    a useful independent sanity reference: its delta and conditioning should be
    in a sane ballpark relative to the constant-correlation version. Used in
    tests to guard against gross implementation errors.
    """
    values = _validated_matrix(returns, config)
    lw = LedoitWolf(assume_centered=False)
    lw.fit(values)
    tickers = list(returns.columns)
    cov = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
    return ShrinkageResult(
        covariance=cov,
        shrinkage_delta=float(lw.shrinkage_),
        r_bar=float("nan"),  # sklearn's target has no r_bar concept
        n_obs=values.shape[0],
    )


# ------------------------------------------------------------------
# 4. Diagnostics — proof the shrinkage did something
# ------------------------------------------------------------------
@dataclass(frozen=True)
class CovarianceDiagnostics:
    """Side-by-side evidence that turns 'I used shrinkage' into 'here is what it did'.

    shrinkage_delta:
        The chosen blend intensity (see ShrinkageResult).
    condition_sample:
        Condition number of the SAMPLE covariance. Large = ill-conditioned =
        dangerous to invert = the optimiser will amplify noise.
    condition_shrunk:
        Condition number of the SHRUNK covariance. Should be markedly smaller;
        that drop is the mechanical reason the optimiser behaves better.
    n_obs, n_assets:
        Shape context for interpreting the above.
    """

    shrinkage_delta: float
    r_bar: float
    condition_sample: float
    condition_shrunk: float
    n_obs: int
    n_assets: int

    @property
    def condition_improvement(self) -> float:
        """How many times better-conditioned the shrunk matrix is."""
        return self.condition_sample / self.condition_shrunk


def covariance_diagnostics(
    returns: pd.DataFrame,
    config: CovarianceConfig,
) -> CovarianceDiagnostics:
    """Compute sample-vs-shrunk conditioning and the chosen delta.

    The condition number is the ratio of the largest to smallest eigenvalue.
    A near-singular (noisy, over-correlated) matrix has a tiny smallest
    eigenvalue and therefore a huge condition number; inverting it blows up
    estimation error. Shrinkage lifts the small eigenvalues, cutting the
    condition number — this function reports both so the improvement is
    concrete and citable rather than asserted.
    """
    sample = sample_covariance(returns, config).to_numpy()
    shrunk_result = shrink_covariance(returns, config)
    shrunk = shrunk_result.covariance.to_numpy()

    cond_sample = float(np.linalg.cond(sample))
    cond_shrunk = float(np.linalg.cond(shrunk))

    return CovarianceDiagnostics(
        shrinkage_delta=shrunk_result.shrinkage_delta,
        r_bar=shrunk_result.r_bar,
        condition_sample=cond_sample,
        condition_shrunk=cond_shrunk,
        n_obs=shrunk_result.n_obs,
        n_assets=sample.shape[0],
    )
