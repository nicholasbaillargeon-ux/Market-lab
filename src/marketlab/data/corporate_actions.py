"""Corporate actions: the quiet source of two backtest lies.

1. **Adjustment look-ahead.** Vendors deliver *back-adjusted* prices: when a
   stock splits 2:1 today, every historical price is retroactively halved so the
   series is continuous. That is correct for charting and for computing returns,
   but a trader in 2015 never saw the 2020-adjusted price. If your signal keys
   off an absolute price level (e.g. "buy below $50") on an adjusted series, you
   are using information that did not exist at decision time.

2. **Adjustment inconsistency.** If you compute returns from raw ``close`` you
   get a fake -50% crash on the split day. If you compute *levels* from
   ``adj_close`` you smear future information backwards. The safe rule, used by
   the engines here: **drive P&L from an adjustment factor / total-return series,
   and drive absolute-level signals from raw prices.**

This module exposes the adjustment factor so both choices are explicit rather
than accidental.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adjustment_factor(bars: pd.DataFrame) -> pd.Series:
    """factor = adj_close / close in (0, 1]. Multiply a raw price by the factor
    for that date to get its back-adjusted equivalent. A change in the factor
    between consecutive days marks a split or dividend ex-date."""
    factor = bars["adj_close"] / bars["close"]
    return factor.rename("adj_factor")


def total_return_series(bars: pd.DataFrame) -> pd.Series:
    """Total-return (dividend-reinvested, split-consistent) close-to-close
    returns. This is the correct return stream for P&L: it is invariant to how
    the vendor anchored its back-adjustment."""
    return bars["adj_close"].pct_change().rename("ret").fillna(0.0)


def corporate_action_dates(bars: pd.DataFrame, *, tol: float = 1e-6) -> pd.DatetimeIndex:
    """Dates where the adjustment factor stepped -> a split or dividend."""
    f = adjustment_factor(bars)
    stepped = np.abs(f.diff()) > tol
    return bars.index[stepped.fillna(False)]
