from .base import Strategy
from .library import BuyAndHold, MeanReversion, SMACrossover
from .portfolio import (
    PORTFOLIO_REGISTRY,
    EqualWeight,
    InverseVolatility,
    PortfolioStrategy,
)

REGISTRY = {
    s.name: s
    for s in (BuyAndHold, SMACrossover, MeanReversion)
}

__all__ = [
    "Strategy", "BuyAndHold", "SMACrossover", "MeanReversion", "REGISTRY",
    "PortfolioStrategy", "EqualWeight", "InverseVolatility", "PORTFOLIO_REGISTRY",
]
