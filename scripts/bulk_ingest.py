"""Bulk-ingest the entire US-listed universe (NASDAQ + NYSE + others) into the
Parquet lake, using yfinance batch downloads for speed.

Not every symbol resolves (delisted names, preferred/warrant tickers, test
issues) -- those are skipped and counted, not failed on. Run in the background;
it logs progress and a running success count.

Usage: python scripts/bulk_ingest.py [start_date] [chunk_size]
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marketlab.data.bars import validate  # noqa: E402
from marketlab.data.storage import ParquetStore  # noqa: E402

NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
LAKE = Path(__file__).resolve().parents[1] / "data" / "lake"


def universe() -> list[str]:
    """Full set of non-test US-listed symbols, normalized for yfinance."""
    syms: set[str] = set()

    nd = pd.read_csv(io.StringIO(requests.get(NASDAQ, timeout=30).text), sep="|")
    nd = nd[nd["Test Issue"] == "N"]
    syms.update(nd["Symbol"].dropna().astype(str))

    ot = pd.read_csv(io.StringIO(requests.get(OTHER, timeout=30).text), sep="|")
    ot = ot[ot["Test Issue"] == "N"]
    syms.update(ot["ACT Symbol"].dropna().astype(str))

    clean: set[str] = set()
    for s in syms:
        s = s.strip()
        if not s or "File Creation Time" in s or "$" in s:
            continue
        clean.add(s.replace(".", "-"))  # yfinance uses '-' for class/pref shares
    return sorted(clean)


def _one(df: pd.DataFrame) -> pd.DataFrame | None:
    """Turn a single-ticker OHLCV frame into a validated bar frame."""
    if df is None or df.empty or df["Close"].isna().all():
        return None
    out = pd.DataFrame(index=pd.to_datetime(df.index, utc=True))
    out["open"] = df["Open"].to_numpy()
    out["high"] = df["High"].to_numpy()
    out["low"] = df["Low"].to_numpy()
    out["close"] = df["Close"].to_numpy()
    adj = "Adj Close" if "Adj Close" in df.columns else "Close"
    out["adj_close"] = df[adj].to_numpy()
    out["volume"] = df["Volume"].to_numpy()
    out = out.dropna(subset=["open", "high", "low", "close"])
    if len(out) < 200:  # need enough history to be useful
        return None
    try:
        return validate(out)
    except Exception:
        return None


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    store = ParquetStore(LAKE)

    syms = universe()
    print(f"universe: {len(syms)} symbols | start={start} | chunk={chunk}", flush=True)

    stored, skipped, done = 0, 0, 0
    for i in range(0, len(syms), chunk):
        batch = syms[i:i + chunk]
        try:
            data = yf.download(batch, start=start, interval="1d", auto_adjust=False,
                               actions=False, progress=False, threads=True,
                               group_by="ticker")
        except Exception as e:
            print(f"  batch {i//chunk} download error: {e}", flush=True)
            skipped += len(batch); done += len(batch); continue

        for sym in batch:
            done += 1
            try:
                sub = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
                bars = _one(sub)
            except Exception:
                bars = None
            if bars is None:
                skipped += 1
                continue
            try:
                store.write_bars(sym, bars)
                stored += 1
            except Exception:
                skipped += 1

        print(f"[{done}/{len(syms)}] stored={stored} skipped={skipped}", flush=True)
        time.sleep(1.0)  # be polite

    print(f"DONE: stored={stored} skipped={skipped} of {len(syms)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
