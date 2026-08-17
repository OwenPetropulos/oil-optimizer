# data/

Place the two extracted input files here:

- `prices.csv` — weekly prices, index = date, columns = tickers
- `fair_values.csv` — weekly point-in-time DCF fair values, same shape

Generate them by running `scripts/extract_data.py` in the folder containing
your `backtest_oil_dcf.py` and `edgar_fundamentals.py`, then copy the two CSVs
here. Both are point-in-time correct: each week's fair value uses only
fundamentals that were public as of that week.
