"""Partitioned Parquet data lake -- the primary bar store.

Layout: ``<root>/symbol=<SYM>/<SYM>.parquet`` (one file per symbol, columnar).
Parquet gives us fast columnar reads, compression, and predicate push-down on
the timestamp, with zero database to run. Writes are upserts: we merge on the
timestamp index so re-ingesting an overlapping range is idempotent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..bars import COLUMNS, empty_bars, validate
from .base import BarStore

# A ticker symbol used to build the on-disk partition path. Constrained to an
# allowlist so a caller-supplied symbol (the web UI forwards client-controlled
# Dash callback state straight to read_bars) cannot contain "/" or ".." and
# traverse out of the lake root when it is joined into a filesystem path.
_SAFE_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,23}$")


class ParquetStore(BarStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        if not _SAFE_SYMBOL.match(symbol or ""):
            raise ValueError(f"invalid symbol: {symbol!r}")
        d = self.root / f"symbol={symbol.upper()}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol.upper()}.parquet"

    def write_bars(self, symbol: str, bars: pd.DataFrame) -> int:
        bars = validate(bars)
        path = self._path(symbol)
        if path.exists():
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index, utc=True)
            existing.index.name = "ts"
            # New data wins on overlap (keep='last').
            merged = pd.concat([existing, bars])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = bars
        merged.to_parquet(path, engine="pyarrow", index=True)
        return len(merged)

    def read_bars(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            return empty_bars()
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "ts"
        df = df[COLUMNS].sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    def symbols(self) -> list[str]:
        return sorted(
            p.name.split("=", 1)[1]
            for p in self.root.glob("symbol=*")
            if p.is_dir()
        )
