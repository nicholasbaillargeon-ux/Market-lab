"""Sanity checks on the performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketlab.metrics import cagr, max_drawdown, sharpe


def test_sharpe_of_constant_positive_is_large():
    r = pd.Series([0.001] * 252)
    # Zero variance -> guarded to 0 by construction.
    assert sharpe(r) == 0.0
    noisy = pd.Series(np.linspace(0.0005, 0.0015, 252))
    assert sharpe(noisy) > 0


def test_max_drawdown_known():
    eq = pd.Series([100, 120, 90, 130])  # peak 120 -> trough 90 = -25%
    assert abs(max_drawdown(eq) - (90 / 120 - 1)) < 1e-12


def test_cagr_doubling_in_one_year():
    eq = pd.Series(np.linspace(100, 200, 252))
    assert abs(cagr(eq) - 1.0) < 0.05  # ~100% over ~1 year
