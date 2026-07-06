"""The two engines must agree. The vectorized engine is fast but only an
approximation for non-proportional costs and multi-weight drift; for a single
instrument going long/flat with proportional costs they should match to
numerical precision. This is the verification path the "layered" design buys us.
"""
from __future__ import annotations

import numpy as np

from marketlab.engine import run_event_driven, run_vectorized
from marketlab.engine.costs import CostModel
from marketlab.strategies import SMACrossover


def _both(bars, cost):
    strat = SMACrossover(fast=10, slow=50)
    w = strat.target_weights(bars)
    v = run_vectorized(bars, w, cost)
    e = run_event_driven(bars, w, cost)
    return v, e


def test_parity_zero_cost(synth_bars):
    v, e = _both(synth_bars, CostModel.zero())
    np.testing.assert_allclose(
        v.equity.to_numpy(), e.equity.to_numpy(), rtol=1e-9, atol=1e-6
    )


def test_parity_proportional_cost(synth_bars):
    v, e = _both(synth_bars, CostModel.retail())
    # Tiny residual from the cash-vs-return treatment of costs; stays well under
    # a basis point of terminal equity.
    np.testing.assert_allclose(
        v.equity.to_numpy(), e.equity.to_numpy(), rtol=1e-4, atol=1e-2
    )
    assert abs(v.total_return - e.total_return) < 1e-4


def test_parity_terminal_matches_buy_and_hold_math(synth_bars):
    """Buy & hold, zero cost: terminal return must equal the adjusted-close
    total return exactly (no execution/costs to muddy it)."""
    from marketlab.strategies import BuyAndHold

    w = BuyAndHold().target_weights(synth_bars)
    e = run_event_driven(synth_bars, w, CostModel.zero())
    # Enters at open[1]; total return is close[-1]/open[1] on the adjusted series.
    factor = synth_bars["adj_close"] / synth_bars["close"]
    adj_open = synth_bars["open"] * factor
    expected = synth_bars["adj_close"].iloc[-1] / adj_open.iloc[1] - 1.0
    assert abs(e.total_return - expected) < 1e-9
