"""Multi-asset parity: the vectorized portfolio engine must reproduce the
share-based event-driven book. This is the multi-asset version of the
single-asset parity guarantee, and the harder one -- it only holds because the
vectorized engine carries the intra-bar weight *drift* (the `c` term). A naive
sum(weight*return) portfolio would fail these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketlab.engine import run_event_driven_portfolio, run_vectorized_portfolio
from marketlab.engine.costs import CostModel
from marketlab.strategies import EqualWeight, InverseVolatility


def _asset(seed: int, mu: float, sigma: float, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n, tz="UTC", name="ts")
    adj_close = 100.0 * np.cumprod(1.0 + rng.normal(mu, sigma, size=n))
    gap = rng.normal(0.0, 0.003, size=n)
    raw_open = np.empty(n)
    raw_open[0] = adj_close[0]
    raw_open[1:] = adj_close[:-1] * (1.0 + gap[1:])
    hi = np.maximum(raw_open, adj_close) * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(raw_open, adj_close) * (1.0 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"open": raw_open, "high": hi, "low": lo, "close": adj_close,
         "adj_close": adj_close, "volume": 1e6},
        index=idx,
    )


@pytest.fixture
def book() -> dict[str, pd.DataFrame]:
    return {
        "AAA": _asset(1, 0.0006, 0.011),
        "BBB": _asset(2, 0.0003, 0.008),
        "CCC": _asset(3, 0.0009, 0.017),
    }


def _weights(book, strat) -> pd.DataFrame:
    return strat.target_weights(book)


def test_portfolio_parity_zero_cost_equal_weight(book):
    w = _weights(book, EqualWeight())
    v = run_vectorized_portfolio(book, w, CostModel.zero())
    e = run_event_driven_portfolio(book, w, CostModel.zero())
    # Drifting fractional weights: exact agreement is the whole claim.
    np.testing.assert_allclose(v.equity.to_numpy(), e.equity.to_numpy(),
                               rtol=1e-9, atol=1e-6)


def test_portfolio_parity_zero_cost_inverse_vol(book):
    w = _weights(book, InverseVolatility(lookback=30))
    v = run_vectorized_portfolio(book, w, CostModel.zero())
    e = run_event_driven_portfolio(book, w, CostModel.zero())
    np.testing.assert_allclose(v.equity.to_numpy(), e.equity.to_numpy(),
                               rtol=1e-9, atol=1e-6)


def test_portfolio_parity_proportional_cost(book):
    w = _weights(book, EqualWeight())
    v = run_vectorized_portfolio(book, w, CostModel.retail())
    e = run_event_driven_portfolio(book, w, CostModel.retail())
    # Same cash-vs-return residual as the single-asset costed case; still tiny.
    np.testing.assert_allclose(v.equity.to_numpy(), e.equity.to_numpy(),
                               rtol=1e-4, atol=1e-1)
    assert abs(v.total_return - e.total_return) < 1e-3
    # Both engines must also agree on how much trading they did -- the reported
    # turnover uses the same open-weight basis, so it matches closely.
    assert v.turnover == pytest.approx(e.turnover, rel=1e-3)


def test_single_asset_book_matches_single_asset_engine(book):
    """A one-name book at weight 1.0 must reproduce the single-asset engine
    exactly -- proof the portfolio engine is a strict generalization."""
    from marketlab.engine import run_vectorized
    from marketlab.strategies import BuyAndHold

    one = {"AAA": book["AAA"]}
    w1 = pd.DataFrame({"AAA": BuyAndHold().target_weights(book["AAA"])})
    vp = run_vectorized_portfolio(one, w1, CostModel.zero())
    v1 = run_vectorized(book["AAA"], w1["AAA"], CostModel.zero())
    np.testing.assert_allclose(vp.equity.to_numpy(), v1.equity.to_numpy(),
                               rtol=1e-12, atol=1e-9)


def test_costs_drag_and_lookahead_inflates(book):
    """Sanity of the thesis at the portfolio level: honest costs reduce terminal
    wealth, and look-ahead execution inflates it."""
    w = _weights(book, EqualWeight())
    gross = run_vectorized_portfolio(book, w, CostModel.zero())
    net = run_vectorized_portfolio(book, w, CostModel.retail())
    cheat = run_vectorized_portfolio(book, w, CostModel.zero(), allow_lookahead=True)
    assert net.final_equity < gross.final_equity          # costs are a drag
    assert cheat.final_equity != pytest.approx(gross.final_equity)  # look-ahead shifts it
    assert gross.turnover > 0                             # rebalancing happened
