"""
run_backtest.py — Run the oil-optimizer backtest on real data and print results.

Loads prices.csv and fair_values.csv from the data/ folder, runs the weekly
walk-forward backtest, and prints the performance scorecard plus turnover and
hold-week diagnostics. This is the entry point for reproducing the strategy's
results once the data has been extracted.

Usage (from the repo root, after `pip install -e .`):
    python scripts/run_backtest.py
"""

from pathlib import Path

import pandas as pd

from oil_optimizer.config import Config
from oil_optimizer import backtest, metrics


def main():
    # Locate the data folder relative to the repo root.
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    prices = pd.read_csv(data_dir / "prices.csv", index_col=0, parse_dates=True)
    fair_values = pd.read_csv(data_dir / "fair_values.csv", index_col=0, parse_dates=True)

    print(f"Loaded prices:      {prices.shape[0]} weeks x {prices.shape[1]} tickers")
    print(f"Loaded fair values: {fair_values.shape[0]} weeks x {fair_values.shape[1]} tickers\n")

    config = Config()
    result = backtest.run_backtest(prices, fair_values, config)
    scorecard = metrics.score(result.weekly_returns)

    print("=" * 50)
    print("  SCORECARD")
    print("=" * 50)
    print(f"  weeks scored:          {scorecard.n_periods}")
    print(f"  hold weeks:            {result.n_holds}")
    print(f"  annualised return:     {scorecard.annualised_return:+.2%}")
    print(f"  annualised volatility: {scorecard.annualised_volatility:.2%}")
    print(f"  Sharpe ratio:          {scorecard.sharpe_ratio:.2f}")
    print(f"  max drawdown:          {scorecard.max_drawdown:.2%}")
    print(f"  avg weekly turnover:   {result.turnover_history.mean():.3f}")
    print(f"  total cost drag:       {result.cost_history.sum():.4f}")
    print("=" * 50)

    # Save the weekly returns and weights so the results page / graphs can use them.
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    result.weekly_returns.to_csv(out_dir / "weekly_returns.csv")
    result.weights_history.to_csv(out_dir / "weights_history.csv")
    print(f"\nSaved weekly_returns.csv and weights_history.csv to {out_dir}/")


if __name__ == "__main__":
    main()
