"""Free hourly bars from Yahoo Finance (via yfinance). Yahoo keeps about 730 days of 1h history."""
from __future__ import annotations

import time

import pandas as pd


def fetch_1h(ticker: str, period: str = "5d", tries: int = 3) -> tuple[pd.DataFrame, str | None]:
    """Returns (raw dataframe with tz-aware index, error text or None)."""
    import yfinance as yf

    err = None
    for i in range(tries):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1h", prepost=True,
                                           auto_adjust=False, actions=False, timeout=30)
            if df is not None and not df.empty:
                return df, None
            err = "no data returned"
        except Exception as e:  # network hiccups, Yahoo throttling
            err = f"{type(e).__name__}: {e}"
        time.sleep(3 * (i + 1))
    return pd.DataFrame(), err
