"""
config.py — Central, validated configuration for the oil-equity optimizer.

Every assumption in the project lives here and nowhere else. Downstream modules
(covariance, optimizer, backtest, metrics) read their settings from a single
Config instance rather than hardcoding constants. This gives us one place to
look, one place to change, and — via __post_init__ validation — one place that
refuses to let a nonsensical setting silently corrupt a result.

Design rule: a bad assumption should crash loudly at construction time, not
produce a plausible-looking wrong answer twelve steps later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ------------------------------------------------------------------
# The investable universe: 12 pure upstream / integrated oil names.
# Deliberately excludes refiners (MPC, VLO, PSX), midstream (ET), and
# oilfield services / drillers (RIG, VAL) — those do not ride the crude
# factor the same way (refiners are partly anti-correlated with crude),
# so mixing them would muddy both the DCF signal and the covariance
# structure. Kept as a module-level constant so it is easy to cite.
# ------------------------------------------------------------------
UPSTREAM_OIL_UNIVERSE: List[str] = [
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    "SHEL",  # Shell
    "TTE",   # TotalEnergies
    "BP",    # BP
    "COP",   # ConocoPhillips
    "OXY",   # Occidental
    "CNQ",   # Canadian Natural Resources
    "SU",    # Suncor
    "FANG",  # Diamondback Energy
    "EOG",   # EOG Resources
    "PBR",   # Petrobras
]


@dataclass(frozen=True)
class CovarianceConfig:
    """Settings the covariance module reads from.

    tickers:
        The universe the covariance matrix is estimated over. Defaults to the
        upstream oil list. Kept as its own field (not a global) so tests can
        construct a Config over a tiny 2- or 3-name universe without touching
        module state.

    window_weeks:
        Number of weekly return observations that feed each covariance estimate.
        This is the core bias/variance dial: a short window reacts fast but is
        noisy; a long window is stable but stale. 104 weeks = ~2 years, a common
        default that gives enough rows to estimate a 12-name matrix without
        reaching so far back that the covariance structure has changed.

    min_observations:
        Hard floor on usable return rows in a window. If a rolling window has
        fewer clean (finite, non-missing) rows than this, the estimator refuses
        to produce a matrix rather than fitting garbage to too little data.
        Must be <= window_weeks (you cannot require more rows than the window
        can hold) and should comfortably exceed the number of assets.

    return_kind:
        'simple' or 'log'. Documented once here so the WHOLE pipeline uses one
        convention. Mixing simple and log returns across modules is a classic
        silent bug — the numbers look fine and the results are subtly wrong.
    """

    tickers: List[str] = field(default_factory=lambda: list(UPSTREAM_OIL_UNIVERSE))
    window_weeks: int = 104
    min_observations: int = 60
    return_kind: str = "simple"

    def __post_init__(self) -> None:
        # --- tickers ---
        if len(self.tickers) == 0:
            raise ValueError("CovarianceConfig.tickers is empty.")
        if len(self.tickers) != len(set(self.tickers)):
            dupes = sorted({t for t in self.tickers if self.tickers.count(t) > 1})
            raise ValueError(f"CovarianceConfig.tickers has duplicates: {dupes}")

        # --- window / observations ---
        if self.window_weeks <= 0:
            raise ValueError(
                f"window_weeks must be positive, got {self.window_weeks}."
            )
        if self.min_observations <= 0:
            raise ValueError(
                f"min_observations must be positive, got {self.min_observations}."
            )
        if self.min_observations > self.window_weeks:
            raise ValueError(
                f"min_observations ({self.min_observations}) cannot exceed "
                f"window_weeks ({self.window_weeks}); a window can never supply "
                f"more rows than it holds."
            )
        # A covariance matrix estimated from fewer rows than assets is singular
        # by construction. Warn the user loudly at config time rather than
        # letting the optimizer choke on an uninvertible matrix later.
        if self.min_observations <= len(self.tickers):
            raise ValueError(
                f"min_observations ({self.min_observations}) must exceed the "
                f"number of assets ({len(self.tickers)}); otherwise the sample "
                f"covariance is singular and shrinkage is patching over a "
                f"structurally broken estimate."
            )

        # --- return convention ---
        if self.return_kind not in ("simple", "log"):
            raise ValueError(
                f"return_kind must be 'simple' or 'log', got {self.return_kind!r}."
            )


@dataclass(frozen=True)
class SignalConfig:
    """Settings the signal module reads from.

    Turns DCF fair values into per-week expected-return signals. Two documented
    research choices live here; both are deliberately visible and both should be
    swept in a sensitivity analysis rather than treated as ground truth.

    compression_scale:
        The tanh saturation point for the raw price-to-fair-value gap. The
        signal is compressed as scale * tanh(gap / scale): near-linear (trusted)
        for gaps well below `scale`, smoothly flattening for gaps beyond it, so
        implausibly large gaps (usually DCF input errors) still rank high but
        contribute far less than their raw magnitude. 0.30 means gaps up to
        ~30% are treated near-linearly and the tail beyond is tamed. NOT a hard
        cap — no discontinuity. Documented judgment; sweep it.

    correction_horizon_weeks:
        Assumed number of weeks for a typical mispricing to close. The compressed
        gap is a TOTAL expected move; dividing by this horizon puts it on the
        same per-week footing as the weekly covariance matrix, so the optimiser's
        risk and return inputs share one time scale. 52 = one year, a central
        estimate for mature large-cap oil names whose mispricings play out over
        a few earnings cycles. Scales overall aggressiveness, not the ranking
        of names. Documented judgment; sweep it.
    """

    compression_scale: float = 0.30
    correction_horizon_weeks: int = 52

    def __post_init__(self) -> None:
        if self.compression_scale <= 0:
            raise ValueError(
                f"compression_scale must be positive, got {self.compression_scale}."
            )
        if self.correction_horizon_weeks <= 0:
            raise ValueError(
                f"correction_horizon_weeks must be positive, got "
                f"{self.correction_horizon_weeks}."
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Settings the optimizer module will read from (used later).

    Defined now so the project's assumptions live in one contract, but kept in
    its own dataclass so nothing covariance-related is entangled with optimizer
    choices we have not built yet.

    long_only:
        No short positions. True for this strategy — the DCF ranks names by
        cheapness and we tilt toward the cheapest; there is no short thesis.

    max_weight:
        Per-name cap. This is NOT the old arbitrary strategy — the point is that
        the rest of the weight vector is now chosen by optimization, and the cap
        is a documented risk constraint, not the whole allocation rule. A cap
        near 1.0 effectively disables it; 0.30 keeps any single name to 30%.

    fully_invested:
        Weights sum to 1.0 (no cash). True for a long-only equity sleeve being
        compared against an equal-weight benchmark on the same names.

    risk_aversion:
        Lambda in the objective  maximize  wᵀμ − (λ/2)·wᵀΣw.  The dial between
        chasing return (low λ) and minimising variance (high λ). Set to a
        BALANCED-leaning-tolerant default: high enough to let the DCF signal
        actually tilt the book away from equal-weight (otherwise optimisation
        adds nothing), but not so low that the optimiser piles into the
        strongest-signal names — dangerous here because the 12 names are highly
        correlated (one crude factor), so concentration is a single-factor bet,
        not a diversified one. Tuned empirically by watching the cap-binding and
        drift-from-equal-weight diagnostics, not fixed by philosophy. The
        max_weight cap is the backstop that bounds the worst case if λ is a
        touch aggressive. Documented judgment; sweep it.
    """

    long_only: bool = True
    max_weight: float = 0.30
    fully_invested: bool = True
    risk_aversion: float = 5.0

    def __post_init__(self) -> None:
        if not (0.0 < self.max_weight <= 1.0):
            raise ValueError(
                f"max_weight must be in (0, 1], got {self.max_weight}."
            )
        if self.risk_aversion <= 0:
            raise ValueError(
                f"risk_aversion (lambda) must be positive, got {self.risk_aversion}."
            )
        # A cap below equal-weight (1/n) makes fully-invested + long-only
        # infeasible: you cannot sum n weights to 1 if each is capped below 1/n.
        # We cannot know n here (that is the per-week universe), but we can catch
        # the obviously-broken case of a cap at or below a small floor.
        if self.max_weight <= 0.0:
            raise ValueError("max_weight must be positive.")


@dataclass(frozen=True)
class BacktestConfig:
    """Settings the backtest module will read from (used later).

    cost_per_turnover:
        Transaction cost charged per unit of turnover, in decimal. Turnover is
        the sum of absolute weight changes at a rebalance; 0.0010 = 10 basis
        points. This is what makes results reflect real trading rather than
        paper trading. Lives here from the start so we never retrofit costs.
    """

    cost_per_turnover: float = 0.0010  # 10 bps

    def __post_init__(self) -> None:
        if self.cost_per_turnover < 0:
            raise ValueError(
                f"cost_per_turnover cannot be negative, got {self.cost_per_turnover}."
            )


@dataclass(frozen=True)
class Config:
    """Top-level config: the single object every module receives.

    Compose the sub-configs so a module takes `config.covariance`,
    `config.optimizer`, etc. — each module depends only on the slice it needs,
    which keeps the dependencies honest and the tests small.
    """

    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def default_config() -> Config:
    """Convenience constructor for the standard project configuration."""
    return Config()
