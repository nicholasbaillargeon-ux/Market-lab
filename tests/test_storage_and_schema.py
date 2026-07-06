"""Parquet round-trip + schema validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketlab.data.bars import SchemaError, validate
from marketlab.data.storage import ParquetStore


def test_parquet_roundtrip_and_upsert(tmp_path, synth_bars):
    store = ParquetStore(tmp_path)
    n = store.write_bars("TEST", synth_bars)
    assert n == len(synth_bars)

    back = store.read_bars("TEST")
    pd.testing.assert_frame_equal(back, validate(synth_bars))
    assert store.symbols() == ["TEST"]

    # Re-writing an overlapping range is idempotent (no dup rows).
    n2 = store.write_bars("TEST", synth_bars.iloc[100:300])
    assert n2 == len(synth_bars)


def test_read_slice(tmp_path, synth_bars):
    store = ParquetStore(tmp_path)
    store.write_bars("TEST", synth_bars)
    start = synth_bars.index[100].date()
    end = synth_bars.index[200].date()
    sliced = store.read_bars("TEST", start, end)
    assert sliced.index.min().date() >= start
    assert sliced.index.max().date() <= end


def test_validate_rejects_bad_ohlc(synth_bars):
    bad = synth_bars.copy()
    bad.iloc[5, bad.columns.get_loc("high")] = 0.0  # high below low
    with pytest.raises(SchemaError):
        validate(bad)


def test_validate_localizes_and_sorts():
    idx = pd.DatetimeIndex(["2021-01-05", "2021-01-04"])  # unsorted, naive
    df = pd.DataFrame(
        {"open": [1, 1], "high": [2, 2], "low": [0.5, 0.5],
         "close": [1.5, 1.5], "adj_close": [1.5, 1.5], "volume": [10, 10]},
        index=idx,
    )
    out = validate(df)
    assert out.index.tz is not None
    assert out.index.is_monotonic_increasing
