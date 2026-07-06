"""Storage abstraction for bar data."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BarStore(ABC):
    @abstractmethod
    def write_bars(self, symbol: str, bars: pd.DataFrame) -> int:
        """Upsert bars for a symbol. Returns the row count stored."""

    @abstractmethod
    def read_bars(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        """Read bars for a symbol, optionally sliced by [start, end]."""

    @abstractmethod
    def symbols(self) -> list[str]:
        """List stored symbols."""
