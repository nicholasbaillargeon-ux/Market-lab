"""Splits/dividends: detect them, and prove the adjusted return stream doesn't
show the fake crash that raw prices would."""
from __future__ import annotations

import numpy as np

from marketlab.data.corporate_actions import (
    adjustment_factor,
    corporate_action_dates,
    total_return_series,
)


def test_split_is_detected(synth_bars, split_idx):
    dates = corporate_action_dates(synth_bars)
    assert synth_bars.index[split_idx] in dates


def test_adjustment_factor_shape(synth_bars, split_idx):
    f = adjustment_factor(synth_bars)
    assert np.isclose(f.iloc[0], 0.5)          # pre-split
    assert np.isclose(f.iloc[-1], 1.0)         # post-split (present)
    assert (f > 0).all() and (f <= 1.0 + 1e-9).all()


def test_raw_close_shows_fake_crash_but_adjusted_does_not(synth_bars, split_idx):
    raw_ret = synth_bars["close"].pct_change()
    # Raw price halves at the split -> a spurious ~ -50% day.
    assert raw_ret.iloc[split_idx] < -0.4

    tr = total_return_series(synth_bars)
    # The honest total-return series has no such crash on the split day.
    assert abs(tr.iloc[split_idx]) < 0.1
