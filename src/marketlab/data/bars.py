"""The canonical OHLCV bar schema and validation.

Every provider must return a DataFrame in *this* shape so the rest of the
system never has to care where the data came from.

Design decisions that matter for backtest correctness:

* We keep BOTH raw and adjusted prices. ``close`` is the raw traded price on
  that day; ``adj_close`` folds in splits and dividends. Using adjusted prices
  for signals is convenient but subtly dangerous (see corporate_actions.py),
  so we store both and let the caller choose explicitly.
* Timestamps are timezone-aware UTC and represent the *close* of the bar. A
  daily bar stamped 2023-01-03 is only knowable after that session closes.
"""
from __future__ import annotations

import pandas as pd

# Column contract for a bar frame. Index is a UTC DatetimeIndex named "ts".
COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class SchemaError(ValueError):
    pass


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}, index=idx)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + normalize a bar frame. Raises SchemaError on violations.

    Guarantees on return: sorted unique UTC index named 'ts', exactly COLUMNS,
    float dtypes, no NaNs in OHLC, and OHLC internal consistency
    (low <= open/close <= high).
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    df = df[COLUMNS].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError("index must be a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "ts"

    if df.index.has_duplicates:
        raise SchemaError("duplicate timestamps in bar frame")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    df = df.astype("float64")

    ohlc = ["open", "high", "low", "close"]
    if df[ohlc].isna().any().any():
        raise SchemaError("NaN in OHLC columns")
    # adj_close may be filled from close if a provider omits it.
    df["adj_close"] = df["adj_close"].fillna(df["close"])

    bad = (
        (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
    )
    if bad.any():
        n = int(bad.sum())
        raise SchemaError(f"{n} bars violate low<=open,close<=high")

    return df
