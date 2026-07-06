"""Provider registry / factory."""
from __future__ import annotations

from ...config import Settings
from .base import DataProvider
from .polygon import PolygonProvider
from .yfinance_provider import YFinanceProvider


def make_provider(name: str, settings: Settings) -> DataProvider:
    name = (name or settings.provider).lower()
    if name == "polygon":
        if not settings.polygon_api_key:
            raise RuntimeError(
                "provider=polygon but POLYGON_API_KEY is unset. "
                "Set it in .env, or use MARKETLAB_PROVIDER=yfinance."
            )
        return PolygonProvider(settings.polygon_api_key)
    if name in ("yfinance", "yf"):
        return YFinanceProvider()
    raise ValueError(f"unknown provider: {name!r}")


__all__ = ["DataProvider", "PolygonProvider", "YFinanceProvider", "make_provider"]
