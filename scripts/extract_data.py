"""
extract_data.py — Dump prices and point-in-time fair values to CSV.

Run this ONCE on your machine, in the same folder as your existing
`backtest_oil_dcf.py` and `edgar_fundamentals.py`. It reuses your PROVEN
fair-value code untouched — same Monte Carlo DCF, same EDGAR point-in-time
fundamentals — so the valuations are identical to your validated backtest. It
does not build a portfolio; it just saves the two inputs the new oil-optimizer
package needs:

    prices.csv       weekly prices,      index = date, columns = tickers
    fair_values.csv  weekly fair values, index = date, columns = tickers

Both are point-in-time correct: each week's fair value uses only fundamentals
that were public as of that week (via edgar.as_of), exactly as your backtest does.

Usage:
    python extract_data.py

Then upload prices.csv and fair_values.csv to continue the new backtest.
"""

import numpy as np
import pandas as pd

# Import your existing, proven modules unchanged.
import backtest_oil_dcf as old
import edgar_fundamentals as edgar


def main():
    print("Extracting prices and point-in-time fair values...")
    print("(This reuses your existing DCF code, so it takes a few minutes — "
          "it runs the same Monte Carlo valuations your backtest does.)\n")

    # --- Same data assembly as your main() ---
    prices_df, wti = old.download_all_data(
        old.UNIVERSE, old.WTI_TICKER, old.START_DATE, old.END_DATE
    )
    if prices_df is None:
        print("Price download failed — aborting.")
        return

    pit_tables, static_fundamentals, source_summary = \
        old.build_fundamentals_sources(old.UNIVERSE)
    active_universe = [t for t in old.UNIVERSE if source_summary[t] != "none"]
    print(f"Active universe ({len(active_universe)}): {active_universe}\n")

    # --- Same weekly resampling as run_backtest ---
    weekly_prices = prices_df.resample("W-MON").first()
    weekly_wti = wti.resample("W-MON").first()
    common_idx = weekly_prices.index.intersection(weekly_wti.index)
    weekly_prices = weekly_prices.loc[common_idx]
    weekly_wti = weekly_wti.loc[common_idx]

    # --- Recompute fair values week by week, EXACTLY as run_backtest does,
    #     but store them into a table instead of building weights. ---
    fair_values_df = pd.DataFrame(index=common_idx, columns=active_universe, dtype=float)

    total = len(common_idx)
    for i, date in enumerate(common_idx):
        wti_price = weekly_wti.loc[date]
        if pd.isna(wti_price):
            continue

        for ticker in active_universe:
            fundamentals = old._resolve_fundamentals(
                ticker, date, pit_tables, static_fundamentals
            )
            if fundamentals is None:
                continue
            inputs = {**old.DEFAULT_ASSUMPTIONS, **old.MACRO_ASSUMPTIONS,
                      **fundamentals, **old.COMPANY_OVERRIDES.get(ticker, {})}
            try:
                fv = old.fair_value(
                    inputs, wti_price,
                    seed=hash((ticker, str(date))) % (2 ** 32)
                )
                fair_values_df.loc[date, ticker] = fv
            except Exception:
                continue

        if (i + 1) % 25 == 0:
            print(f"   ...week {i + 1}/{total} ({date.date()})")

    # --- Restrict prices to the same universe and index, then save ---
    prices_out = weekly_prices[active_universe].loc[common_idx]

    # Strip timezone from the index for clean CSVs (optional but tidy).
    prices_out.index = prices_out.index.tz_localize(None)
    fair_values_df.index = fair_values_df.index.tz_localize(None)

    prices_out.to_csv("prices.csv")
    fair_values_df.to_csv("fair_values.csv")

    print(f"\nSaved prices.csv       ({prices_out.shape[0]} weeks x {prices_out.shape[1]} tickers)")
    print(f"Saved fair_values.csv  ({fair_values_df.shape[0]} weeks x {fair_values_df.shape[1]} tickers)")
    print("\nDone. Upload these two files to continue.")


if __name__ == "__main__":
    main()
