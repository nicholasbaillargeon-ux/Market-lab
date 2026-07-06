"""A small library of causal, single-instrument strategies.

The strategy is deliberately not the point of this project -- the point is that
the *same* strategy produces very different results once you add costs and honest
execution. These are just realistic vehicles for that demonstration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def target_weights(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index, name="weight")


class SMACrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise.

    Uses adjusted closes so the moving averages are split/dividend consistent.
    Both SMAs at bar t use only closes <= t, so the signal is knowable at that
    close; the engine handles the fill-next-open delay.
    """

    name = "sma_crossover"

    def __init__(self, fast: int = 20, slow: int = 100):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow")
        self.fast = fast
        self.slow = slow

    def target_weights(self, bars: pd.DataFrame) -> pd.Series:
        px = bars["adj_close"]
        sma_fast = px.rolling(self.fast).mean()
        sma_slow = px.rolling(self.slow).mean()
        w = (sma_fast > sma_slow).astype("float64")
        # Undefined until the slow window fills -> flat.
        w[sma_slow.isna()] = 0.0
        return w.rename("weight")


class MeanReversion(Strategy):
    """Long when price is `entry_z` std below its rolling mean; exit at the mean.

    A high-turnover strategy -- great for showing how transaction costs turn a
    beautiful gross equity curve into a losing net one.
    """

    name = "mean_reversion"

    def __init__(self, lookback: int = 20, entry_z: float = 1.0):
        self.lookback = lookback
        self.entry_z = entry_z

    def target_weights(self, bars: pd.DataFrame) -> pd.Series:
        px = bars["adj_close"]
        mean = px.rolling(self.lookback).mean()
        std = px.rolling(self.lookback).std()
        z = (px - mean) / std.replace(0.0, np.nan)
        # Long below -entry_z, flat once price reverts to/above the mean.
        w = pd.Series(0.0, index=bars.index)
        w[z <= -self.entry_z] = 1.0
        # Hold the long until z crosses back above 0 (stateful -> forward-fill).
        raw = pd.Series(np.nan, index=bars.index)
        raw[z <= -self.entry_z] = 1.0
        raw[z >= 0.0] = 0.0
        w = raw.ffill().fillna(0.0)
        return w.rename("weight")
