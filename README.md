# oil-optimizer

Constrained mean-variance portfolio optimization for an upstream oil-equity
strategy. A weekly DCF signal ranks ~12 upstream names by how far price sits below
fair value; this package turns that signal into expected returns, estimates risk
with a shrunk covariance matrix, and solves for weights that trade return against
risk under long-only, fully-invested, and per-name-cap constraints. It replaces an
earlier version that set weights with hand-picked caps.

The design goals are correctness and honesty: Ledoit–Wolf covariance shrinkage,
transaction costs charged on turnover, point-in-time inputs, and a test suite that
includes a proof the backtest cannot see the future.

## Install

```bash
git clone https://github.com/OwenPetropulos/oil-optimizer.git
cd oil-optimizer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest          # 24 tests, including test_no_lookahead
```

## Use

The package takes **prices** and **point-in-time fair values** as inputs (it does
not fetch data itself). Both are `DataFrame`s indexed by weekly date, one column per
ticker; `fair_values` is already lagged so each row contains only what was public
that week.

```python
import pandas as pd
from oil_optimizer.config import Config
from oil_optimizer import backtest, metrics

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
fair_values = pd.read_csv("data/fair_values.csv", index_col=0, parse_dates=True)

result = backtest.run_backtest(prices, fair_values, Config())
print(metrics.score(result.weekly_returns))
```

All assumptions (universe, covariance window, signal compression, correction
horizon, risk aversion, transaction cost, weight cap) live in `config.py` as
validated dataclasses.

## Example results

Weekly backtest over 2021–2026 on 11 upstream names, all three lines measured over
the same period:

| Strategy | Sharpe | Ann. return | Ann. vol | Max drawdown |
|---|---|---|---|---|
| Mean-variance optimizer | 1.15 | 26.7% | 23.2% | −19% |
| Equal-weight (same names) | 1.02 | 27.9% | 27.3% | −25% |
| Original arbitrary-cap strategy | 0.99 | 24.4% | 24.6% | −21% |

The optimizer's edge comes from lower volatility and drawdown rather than higher
return, and the Sharpe stays in a 1.08–1.21 band across a 100× sweep of the
risk-aversion parameter (λ was not tuned to the backtest).

Notes: the backtest starts in 2021 because the covariance estimate needs a 104-week
window, so it does not cover 2020. BP is excluded (no point-in-time fundamentals
available). The full analysis is written up separately.

## About

Part of [OPResearch](https://github.com/OwenPetropulos). Fair values come from a
separate point-in-time SEC EDGAR DCF pipeline.
