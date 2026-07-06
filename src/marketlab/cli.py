"""Command-line interface: `marketlab ingest ...` and `marketlab backtest ...`."""
from __future__ import annotations

import argparse
from datetime import date, datetime

from .backtest import (
    COST_PRESETS,
    ENGINES,
    resolve_portfolio_strategy,
    resolve_strategy,
    run_backtest,
    run_portfolio_backtest,
)
from .config import get_settings
from .data.ingest import ingest_symbols
from .data.storage import ParquetStore
from .strategies import PORTFOLIO_REGISTRY, REGISTRY


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt_metrics(m: dict) -> str:
    order = ["engine", "total_return", "cagr", "ann_vol", "sharpe", "sortino",
             "max_drawdown", "turnover", "n_trades", "final_equity"]
    lines = []
    for k in order:
        v = m.get(k)
        if isinstance(v, float):
            if k in ("total_return", "cagr", "ann_vol", "max_drawdown"):
                lines.append(f"  {k:14} {v:+.2%}")
            else:
                lines.append(f"  {k:14} {v:,.4f}")
        else:
            lines.append(f"  {k:14} {v}")
    return "\n".join(lines)


def cmd_ingest(args) -> int:
    settings = get_settings()
    results = ingest_symbols(
        args.symbols, args.start, args.end,
        provider=args.provider, settings=settings,
    )
    total = sum(r.rows for r in results)
    print(f"\ningested {len(results)} symbols, {total} total rows -> {settings.parquet_root}")
    if settings.postgres_enabled:
        print(f"(also mirrored to Postgres: {settings.pg_dsn})")
    return 0


def cmd_backtest(args) -> int:
    settings = get_settings()
    store = ParquetStore(settings.parquet_root)
    bars = store.read_bars(args.symbol, args.start, args.end)
    if bars.empty:
        print(f"no bars for {args.symbol}; run `marketlab ingest` first.")
        return 1

    params = {}
    if args.fast:
        params["fast"] = args.fast
    if args.slow:
        params["slow"] = args.slow
    strat = resolve_strategy(args.strategy, **params)
    cost = COST_PRESETS[args.cost]()

    engines = list(ENGINES) if args.engine == "both" else [args.engine]
    print(f"\n{args.symbol}  {strat.name}  cost={args.cost}  "
          f"bars={len(bars)}  [{bars.index.min().date()} -> {bars.index.max().date()}]\n")
    for eng in engines:
        rep = run_backtest(bars, strat, engine=eng, cost_model=cost)
        print(f"[{eng}]")
        print(_fmt_metrics(rep.metrics))
        print()
    return 0


def cmd_portfolio(args) -> int:
    settings = get_settings()
    store = ParquetStore(settings.parquet_root)
    bars = {}
    for sym in args.symbols:
        b = store.read_bars(sym, args.start, args.end)
        if b.empty:
            print(f"no bars for {sym}; run `marketlab ingest` first.")
            return 1
        bars[sym] = b

    params = {"lookback": args.lookback} if args.lookback else {}
    strat = resolve_portfolio_strategy(args.strategy, **params)
    cost = COST_PRESETS[args.cost]()

    engines = list(ENGINES) if args.engine == "both" else [args.engine]
    print(f"\nportfolio [{' '.join(args.symbols)}]  {strat.name}  cost={args.cost}\n")
    for eng in engines:
        rep = run_portfolio_backtest(bars, strat, engine=eng, cost_model=cost)
        span = rep.result.equity.index
        print(f"[{eng}]  bars={len(span)}  "
              f"[{span.min().date()} -> {span.max().date()}]")
        print(_fmt_metrics(rep.metrics))
        print()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="marketlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="fetch + store OHLCV bars")
    pi.add_argument("--symbols", nargs="+", required=True)
    pi.add_argument("--start", type=_date, required=True)
    pi.add_argument("--end", type=_date, required=True)
    pi.add_argument("--provider", default=None, help="polygon|yfinance")
    pi.set_defaults(func=cmd_ingest)

    pb = sub.add_parser("backtest", help="run a strategy through the engine(s)")
    pb.add_argument("--symbol", required=True)
    pb.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    pb.add_argument("--engine", default="both", choices=[*ENGINES, "both"])
    pb.add_argument("--cost", default="retail", choices=sorted(COST_PRESETS))
    pb.add_argument("--start", type=_date, default=None)
    pb.add_argument("--end", type=_date, default=None)
    pb.add_argument("--fast", type=int, default=None)
    pb.add_argument("--slow", type=int, default=None)
    pb.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("portfolio", help="run a multi-asset strategy through the engine(s)")
    pp.add_argument("--symbols", nargs="+", required=True)
    pp.add_argument("--strategy", required=True, choices=sorted(PORTFOLIO_REGISTRY))
    pp.add_argument("--engine", default="both", choices=[*ENGINES, "both"])
    pp.add_argument("--cost", default="retail", choices=sorted(COST_PRESETS))
    pp.add_argument("--start", type=_date, default=None)
    pp.add_argument("--end", type=_date, default=None)
    pp.add_argument("--lookback", type=int, default=None, help="vol window for inverse_vol")
    pp.set_defaults(func=cmd_portfolio)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
