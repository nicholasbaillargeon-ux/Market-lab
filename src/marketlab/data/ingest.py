"""Ingestion orchestration: provider -> validated bars -> store(s)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..config import Settings, get_settings
from .providers import make_provider
from .storage import BarStore, ParquetStore


@dataclass
class IngestResult:
    symbol: str
    rows: int
    start: date | None
    end: date | None


def ingest_symbols(
    symbols: list[str],
    start: date,
    end: date,
    *,
    provider: str | None = None,
    settings: Settings | None = None,
    stores: list[BarStore] | None = None,
) -> list[IngestResult]:
    settings = settings or get_settings()
    prov = make_provider(provider or settings.provider, settings)

    if stores is None:
        stores = [ParquetStore(settings.parquet_root)]
        if settings.postgres_enabled:
            try:
                from .storage import PostgresStore

                stores.append(PostgresStore(settings.pg_dsn))
            except Exception as e:  # DB not up -> degrade to Parquet only
                print(f"[ingest] Postgres unavailable, using Parquet only: {e}")

    results: list[IngestResult] = []
    for sym in symbols:
        bars = prov.get_daily_bars(sym, start, end)
        rows = 0
        for st in stores:
            rows = st.write_bars(sym, bars)
        results.append(
            IngestResult(
                symbol=sym.upper(),
                rows=rows,
                start=bars.index.min().date() if len(bars) else None,
                end=bars.index.max().date() if len(bars) else None,
            )
        )
        print(f"[ingest] {sym.upper():6} rows={rows} via {prov.name}")
    return results
