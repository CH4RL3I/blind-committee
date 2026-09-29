"""Daily adjusted closes via yfinance, cached to CSV.

Stooq was the first choice but now sits behind a JavaScript proof-of-work challenge and
returns HTML instead of CSV to scripted clients; yfinance works.
"""

from __future__ import annotations

import pandas as pd

from ..config import BENCHMARK, UNIVERSE, cache_dir


def load_prices(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Adjusted close, columns = universe + benchmark, index = trading days."""
    path = cache_dir() / f"prices_{start}_{end}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    tickers = [*UNIVERSE, BENCHMARK]
    df = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )["Close"]
    df = df[tickers].dropna(how="all").ffill()
    df.to_csv(path)
    return df
