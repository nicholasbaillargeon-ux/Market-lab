"""Transaction cost & slippage model, shared by both engines.

Two kinds of cost, because they behave differently in a backtest:

* **Proportional** (commission_bps, slippage_bps): scale with traded notional.
  These are what the fast vectorized engine models, and they are what actually
  kills high-turnover strategies -- a strategy that rebalances daily pays the
  spread every single day.
* **Non-proportional** (commission_per_share, fixed_per_trade): flat/per-share
  fees that only the event-driven engine models exactly (they depend on share
  count and portfolio size, which a pure weights-and-returns engine abstracts
  away). This asymmetry is *why* the two engines can only be expected to match
  when costs are proportional -- a fact the parity test pins down.

Slippage is modeled as a symmetric spread: you buy a touch above and sell a
touch below the reference (open) price. Economically that is identical to
charging ``slippage_bps`` on the traded notional, so we fold it into the
proportional rate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0        # proportional, per traded notional
    slippage_bps: float = 0.0          # proportional, models the half-spread
    commission_per_share: float = 0.0  # non-proportional (event engine only)
    fixed_per_trade: float = 0.0       # non-proportional (event engine only)

    @property
    def proportional_rate(self) -> float:
        """Fraction of traded notional lost to proportional costs."""
        return (self.commission_bps + self.slippage_bps) / 1e4

    @property
    def is_proportional(self) -> bool:
        return self.commission_per_share == 0.0 and self.fixed_per_trade == 0.0

    def trade_cost(self, qty: float, price: float) -> float:
        """Cost in currency for trading `qty` shares (sign-agnostic) at `price`."""
        q = abs(qty)
        if q == 0.0:
            return 0.0
        notional = q * price
        return (
            notional * self.proportional_rate
            + q * self.commission_per_share
            + self.fixed_per_trade
        )

    # --- presets ---
    @classmethod
    def zero(cls) -> "CostModel":
        """Frictionless -- the world naive backtests secretly assume."""
        return cls()

    @classmethod
    def retail(cls) -> "CostModel":
        """Commission-free broker, but you still pay ~5bps of spread/slippage."""
        return cls(commission_bps=0.0, slippage_bps=5.0)

    @classmethod
    def institutional(cls) -> "CostModel":
        """Low commission, tighter spread, but per-share exchange fees."""
        return cls(commission_bps=0.5, slippage_bps=2.0, commission_per_share=0.0005)
