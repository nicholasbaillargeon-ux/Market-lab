"""Deterministic synthetic bars so tests never touch the network.

Builds a realistic-looking OHLCV frame with an embedded 2:1 split at day 200 so
corporate-action logic has something to bite on. Raw prices carry the split
(a downward jump); adj_close is the continuous back-adjusted series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SPLIT_IDX = 200
N = 400


def _make_bars(n: int = N, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n, tz="UTC", name="ts")

    # Continuous (adjusted) price path via GBM.
    rets = rng.normal(0.0004, 0.01, size=n)
    adj_close = 100.0 * np.cumprod(1.0 + rets)

    # Back-adjustment factor: pre-split prices traded at 2x, so factor=0.5 then.
    factor = np.where(np.arange(n) < SPLIT_IDX, 0.5, 1.0)
    raw_close = adj_close / factor  # raw jumps DOWN by half at the split

    # Build a consistent raw OHLC around raw_close.
    gap = rng.normal(0.0, 0.003, size=n)
    raw_open = np.empty(n)
    raw_open[0] = raw_close[0]
    raw_open[1:] = raw_close[:-1] * (1.0 + gap[1:])
    hi_noise = np.abs(rng.normal(0.0, 0.004, size=n))
    lo_noise = np.abs(rng.normal(0.0, 0.004, size=n))
    raw_high = np.maximum(raw_open, raw_close) * (1.0 + hi_noise)
    raw_low = np.minimum(raw_open, raw_close) * (1.0 - lo_noise)
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "open": raw_open,
            "high": raw_high,
            "low": raw_low,
            "close": raw_close,
            "adj_close": adj_close,
            "volume": volume,
        },
        index=idx,
    )


@pytest.fixture
def synth_bars() -> pd.DataFrame:
    return _make_bars()


@pytest.fixture
def split_idx() -> int:
    return SPLIT_IDX
