"""Multi-asset (cross-sectional) strategies.

A portfolio strategy maps a *dict* of per-symbol bar frames to a DataFrame of
target weights -- one column per symbol, one row per bar, each row the book you
want to hold going into the next bar, decided using only closes <= t. Same
causality rule as the single-asset ``Strategy``; the engine adds the
fill-next-open delay and rebalances to these weights every bar.

These allocate *across* names rather than timing one, so they're the natural
vehicles for showing that even a "just hold a diversified book" strategy pays a
continuous rebalancing cost the gross curve hides.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _closes(bars: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    """Adjusted closes for `symbols` on their common calendar (columns=symbols)."""
    cols = {s: bars[s]["adj_close"] for s in symbols}
    return pd.DataFrame(cols).dropna(how="any")


class PortfolioStrategy(ABC):
    name: str = "abstract_portfolio"

    @abstractmethod
    def target_weights(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Target weights in [-1, 1] per asset, indexed by the common calendar."""

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


class EqualWeight(PortfolioStrategy):
    """Hold every name at 1/N, rebalanced each bar. The textbook diversified
    book -- and a clean demonstration that rebalancing to fixed weights is not
    free once costs are on."""

    name = "equal_weight"

    def target_weights(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        px = _closes(bars, sorted(bars))
        n = px.shape[1]
        return pd.DataFrame(1.0 / n, index=px.index, columns=px.columns)


class InverseVolatility(PortfolioStrategy):
    """Weight each name by the inverse of its trailing return volatility, then
    normalize to a fully-invested book. Lower-vol names get more capital -- a
    causal, widely-used risk-weighting scheme. Flat until the vol window fills."""

    name = "inverse_vol"

    def __init__(self, lookback: int = 60):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback

    def target_weights(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        px = _closes(bars, sorted(bars))
        rets = px.pct_change()
        vol = rets.rolling(self.lookback).std()
        inv = 1.0 / vol.replace(0.0, np.nan)
        w = inv.div(inv.sum(axis=1), axis=0)      # normalize each row to sum 1
        return w.fillna(0.0)


PORTFOLIO_REGISTRY = {s.name: s for s in (EqualWeight, InverseVolatility)}
