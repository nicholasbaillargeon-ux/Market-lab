"""Polygon.io provider (primary). Uses the aggregates REST endpoint directly
via `requests` -- no vendor SDK, so the wire format is explicit and auditable.

Polygon's ``adjusted=true`` returns split/dividend-adjusted bars; we fetch BOTH
adjusted and raw so downstream code can reason about corporate actions instead
of silently inheriting a vendor's adjustment choices.

Free tier is 5 requests/minute, so we rate-limit politely and page through the
``next_url`` cursor.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from ..bars import validate
from .base import DataProvider

_BASE = "https://api.polygon.io"


class PolygonProvider(DataProvider):
    name = "polygon"

    def __init__(self, api_key: str, *, rps_delay: float = 12.0, session=None):
        if not api_key:
            raise ValueError("PolygonProvider requires an API key")
        self.api_key = api_key
        self.rps_delay = rps_delay  # seconds between calls (free tier = 5/min)
        self.session = session or requests.Session()

    def _agg_url(self, symbol: str, start: date, end: date, adjusted: bool) -> str:
        return (
            f"{_BASE}/v2/aggs/ticker/{symbol.upper()}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
            f"?adjusted={'true' if adjusted else 'false'}&sort=asc&limit=50000"
        )

    def _fetch_series(self, url: str) -> pd.DataFrame:
        rows: list[dict] = []
        while url:
            sep = "&" if "?" in url else "?"
            resp = self.session.get(f"{url}{sep}apiKey={self.api_key}", timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows.extend(payload.get("results", []) or [])
            url = payload.get("next_url", "")
            if url:
                time.sleep(self.rps_delay)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Polygon fields: t=epoch ms, o/h/l/c=OHLC, v=volume
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df.set_index("ts")[["o", "h", "l", "c", "v"]]

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        adj = self._fetch_series(self._agg_url(symbol, start, end, adjusted=True))
        time.sleep(self.rps_delay)
        raw = self._fetch_series(self._agg_url(symbol, start, end, adjusted=False))
        if raw.empty:
            from ..bars import empty_bars

            return empty_bars()

        out = pd.DataFrame(index=raw.index)
        out["open"] = raw["o"]
        out["high"] = raw["h"]
        out["low"] = raw["l"]
        out["close"] = raw["c"]
        out["volume"] = raw["v"]
        # Align adjusted close onto the raw index; fall back to raw close.
        out["adj_close"] = adj["c"].reindex(raw.index) if not adj.empty else raw["c"]
        return validate(out)
