"""yfinance provider (working fallback). No API key required, so the whole
pipeline runs end-to-end out of the box. Quality is lower than Polygon and it
is survivorship-biased (delisted tickers return nothing), which is exactly the
kind of pitfall this project is about -- see README.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..bars import empty_bars, validate
from .base import DataProvider


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        # yfinance 'end' is exclusive; add a day so the range is inclusive.
        raw = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,   # keep raw close; we compute adj separately
            actions=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return empty_bars()

        # yfinance returns a MultiIndex column frame for single tickers in
        # recent versions; flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        out = pd.DataFrame(index=pd.to_datetime(raw.index, utc=True))
        out["open"] = raw["Open"].to_numpy()
        out["high"] = raw["High"].to_numpy()
        out["low"] = raw["Low"].to_numpy()
        out["close"] = raw["Close"].to_numpy()
        adj_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        out["adj_close"] = raw[adj_col].to_numpy()
        out["volume"] = raw["Volume"].to_numpy()
        return validate(out.dropna(subset=["open", "high", "low", "close"]))
