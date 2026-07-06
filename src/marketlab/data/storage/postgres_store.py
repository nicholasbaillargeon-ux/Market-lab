"""Postgres layer: symbol metadata, bars, and backtest results.

Parquet is the fast bar lake; Postgres is the system of record for *metadata*
and *reproducibility*:

* ``symbols`` carries an ``is_active`` / ``delisted_on`` flag. This is where you
  fight survivorship bias -- a point-in-time universe query filters on listing
  windows, not just "tickers that still trade today".
* ``backtest_runs`` stores the strategy, params, engine, and resulting metrics
  as JSON so every headline number is reproducible and auditable.

Uses SQLAlchemy Core (explicit tables + SQL), not the ORM.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..bars import COLUMNS, empty_bars, validate
from .base import BarStore

_meta = MetaData()

symbols_t = Table(
    "symbols", _meta,
    Column("symbol", String, primary_key=True),
    Column("name", String, nullable=True),
    Column("is_active", Boolean, default=True),
    Column("listed_on", Date, nullable=True),
    Column("delisted_on", Date, nullable=True),
)

bars_t = Table(
    "bars", _meta,
    Column("symbol", String, primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float),
    Column("close", Float), Column("adj_close", Float),
    Column("volume", Float),
)

runs_t = Table(
    "backtest_runs", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True)),
    Column("strategy", String),
    Column("engine", String),
    Column("params", JSON),
    Column("metrics", JSON),
)


class PostgresStore(BarStore):
    def __init__(self, dsn: str):
        # SQLAlchemy wants the psycopg3 driver spelled out.
        if dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        self.engine = create_engine(dsn, future=True)
        _meta.create_all(self.engine)

    # ---- symbol metadata (survivorship-aware universe) ----
    def upsert_symbol(self, symbol: str, *, name=None, is_active=True,
                      listed_on=None, delisted_on=None) -> None:
        stmt = pg_insert(symbols_t).values(
            symbol=symbol.upper(), name=name, is_active=is_active,
            listed_on=listed_on, delisted_on=delisted_on,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={"name": name, "is_active": is_active,
                  "listed_on": listed_on, "delisted_on": delisted_on},
        )
        with self.engine.begin() as cx:
            cx.execute(stmt)

    def universe_asof(self, as_of: date) -> list[str]:
        """Point-in-time universe: symbols listed on/before `as_of` and not yet
        delisted. This is the query that avoids survivorship bias."""
        q = select(symbols_t.c.symbol).where(
            (symbols_t.c.listed_on.is_(None)) | (symbols_t.c.listed_on <= as_of)
        ).where(
            (symbols_t.c.delisted_on.is_(None)) | (symbols_t.c.delisted_on >= as_of)
        )
        with self.engine.begin() as cx:
            return sorted(r[0] for r in cx.execute(q))

    # ---- BarStore ----
    def write_bars(self, symbol: str, bars: pd.DataFrame) -> int:
        bars = validate(bars)
        rows = [
            {"symbol": symbol.upper(), "ts": ts.to_pydatetime(),
             **{c: float(r[c]) for c in COLUMNS}}
            for ts, r in bars.iterrows()
        ]
        if not rows:
            return 0
        stmt = pg_insert(bars_t).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "ts"],
            set_={c: stmt.excluded[c] for c in COLUMNS},
        )
        with self.engine.begin() as cx:
            cx.execute(stmt)
        self.upsert_symbol(symbol)
        return len(rows)

    def read_bars(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        q = select(bars_t).where(bars_t.c.symbol == symbol.upper())
        if start is not None:
            q = q.where(bars_t.c.ts >= pd.Timestamp(start, tz="UTC").to_pydatetime())
        if end is not None:
            q = q.where(bars_t.c.ts <= pd.Timestamp(end, tz="UTC").to_pydatetime())
        q = q.order_by(bars_t.c.ts)
        with self.engine.begin() as cx:
            df = pd.DataFrame(cx.execute(q).mappings().all())
        if df.empty:
            return empty_bars()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.set_index("ts")[COLUMNS]

    def symbols(self) -> list[str]:
        with self.engine.begin() as cx:
            return sorted(r[0] for r in cx.execute(select(symbols_t.c.symbol)))

    # ---- backtest reproducibility ----
    def record_run(self, *, strategy: str, engine: str, params: dict,
                   metrics: dict) -> int:
        stmt = insert(runs_t).values(
            created_at=datetime.now(timezone.utc),
            strategy=strategy, engine=engine,
            params=json.loads(json.dumps(params, default=str)),
            metrics=json.loads(json.dumps(metrics, default=str)),
        ).returning(runs_t.c.id)
        with self.engine.begin() as cx:
            return int(cx.execute(stmt).scalar_one())
