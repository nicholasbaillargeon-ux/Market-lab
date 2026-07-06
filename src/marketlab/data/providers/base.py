"""Provider abstraction. Everything downstream depends on this interface, not
on any specific vendor, so Polygon / yfinance / IEX are swappable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Fetches OHLCV bars for a symbol over [start, end] (inclusive dates)."""

    name: str = "abstract"

    @abstractmethod
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return a validated bar frame (see marketlab.data.bars.validate)."""
        raise NotImplementedError

    def get_many(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            out[sym] = self.get_daily_bars(sym, start, end)
        return out
