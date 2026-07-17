"""Interactive web demo for marketlab, served with Dash/Plotly.

The UI is organized around the project's thesis: pick a symbol, strategy, and
cost model, then *watch the backtest lie and get corrected* in real time --
engine parity, look-ahead inflation, and cost drag are all live comparisons.

Visual design follows a validated data-viz palette (dark surface): the three
equity lines are colored by meaning, not identity -- net-tradeable is the
primary blue (the truth), gross is muted (a reference ceiling), and the
look-ahead curve is warning-amber and dashed (the lie).

Run:  python -m marketlab.webapp   (serves on 0.0.0.0:8060)
"""
from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import send_from_directory

from .backtest import COST_PRESETS
from .config import get_settings
from .data.storage import ParquetStore
from .engine import run_event_driven, run_vectorized
from .engine.costs import CostModel
from .metrics import summary
from .strategies import REGISTRY

settings = get_settings()
STORE = ParquetStore(settings.parquet_root)

# ---- palette (validated dark surface) ----
PAGE = "#0d0d0d"        # page plane
SURFACE = "#1a1a19"     # chart / panel surface
INK = "#ffffff"         # primary text
INK2 = "#c3c2b7"        # secondary text
MUTED = "#898781"       # axis / labels
GRID = "#2c2c2a"        # hairline grid
HAIR = "rgba(255,255,255,0.10)"  # border ring
BLUE = "#3987e5"        # series 1 — net (the truth)
AMBER = "#fab219"       # status: warning — the lie
GOOD = "#0ca30c"        # status: good
CRIT = "#d03b3b"        # status: critical
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

POPULAR = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
           "TSLA", "TLT", "GLD"]


def _ret(x: float) -> str:
    """Format a total return: a wealth multiple past +1000%, else a percentage."""
    return f"{1 + x:,.0f}×" if abs(x) >= 10 else f"{x:+.1%}"

_cache: dict = {"syms": []}


def _all_symbols() -> list[str]:
    syms = [s for s in STORE.symbols() if s != "SYNTH"]
    _cache["syms"] = syms or STORE.symbols()
    return _cache["syms"]


def _default_options() -> list[str]:
    syms = _all_symbols()
    sset = set(syms)
    head = [s for s in POPULAR if s in sset]
    return head + [s for s in syms if s not in set(head)][:40]


# ---- reusable pieces ----
def _panel(children, **style):
    base = {"background": SURFACE, "borderRadius": "12px", "padding": "16px 18px",
            "border": f"1px solid {HAIR}"}
    base.update(style)
    return html.Div(children, style=base)


def _stat(label, value, sub, dot, status):
    return _panel(
        [
            html.Div(
                [html.Span(style={"display": "inline-block", "width": "8px",
                                  "height": "8px", "borderRadius": "50%",
                                  "background": dot, "marginRight": "7px"}),
                 html.Span(label, style={"color": MUTED, "fontSize": "11px",
                                         "textTransform": "uppercase",
                                         "letterSpacing": ".6px"})],
            ),
            html.Div(value, style={"color": INK, "fontSize": "27px",
                                   "fontWeight": 700, "marginTop": "6px",
                                   "fontVariantNumeric": "tabular-nums"}),
            html.Div([html.Span(status, style={"color": dot, "fontWeight": 600}),
                      html.Span(f"  {sub}", style={"color": MUTED})],
                     style={"fontSize": "12px", "marginTop": "2px"}),
        ],
        flex="1", minWidth="168px",
    )


def _help():
    """A hover/focus 'How to read this' pill that reveals a compact legend."""
    def _line(color, name, desc, dash=False):
        return html.Div(
            [html.Span(style={"display": "inline-block", "width": "18px",
                              "borderTop": f"2px {'dashed' if dash else 'solid'} {color}",
                              "marginRight": "9px", "verticalAlign": "middle"}),
             html.Span(name, style={"color": color, "fontWeight": 600}),
             html.Span(f" — {desc}", style={"color": INK2})],
            style={"marginBottom": "7px", "fontSize": "12.5px", "lineHeight": "1.4"})

    return html.Span(
        [
            html.Span(
                [html.Span("?", style={"display": "inline-block", "width": "15px",
                                       "height": "15px", "borderRadius": "50%",
                                       "border": f"1px solid {MUTED}", "color": MUTED,
                                       "fontSize": "10px", "fontWeight": 700,
                                       "textAlign": "center", "lineHeight": "15px",
                                       "marginRight": "6px"}),
                 "How to read this"],
                style={"color": MUTED, "fontSize": "12px", "fontWeight": 600,
                       "border": f"1px solid {HAIR}", "borderRadius": "999px",
                       "padding": "3px 10px 3px 8px", "whiteSpace": "nowrap"}),
            html.Div(
                [_line(BLUE, "Net (blue)",
                       "the only curve you could actually trade — real signals + costs"),
                 _line(MUTED, "Gross (grey)",
                       "honest signals, but zero trading costs — a ceiling"),
                 _line(AMBER, "Look-ahead (amber)",
                       "trades on prices it couldn't have known yet — the lie", dash=True),
                 html.Div(style={"borderTop": f"1px solid {HAIR}", "margin": "9px 0 8px"}),
                 html.Div(
                     [html.B("Cost drag", style={"color": INK2}),
                      html.Span(" how much costs erase · ", style={"color": MUTED}),
                      html.B("Look-ahead Δ", style={"color": INK2}),
                      html.Span(" the fantasy edge · ", style={"color": MUTED}),
                      html.B("Parity gap", style={"color": INK2}),
                      html.Span(" two engines agree ⇒ the fast number is trustworthy",
                                style={"color": MUTED})],
                     style={"fontSize": "12px", "lineHeight": "1.5"})],
                className="ml-help-pop"),
        ],
        className="ml-help", tabIndex=0, style={"marginLeft": "6px"},
    )


def _ctl(label, control):
    return html.Div([html.Label(label, style={"color": MUTED, "fontSize": "12px",
                                              "display": "block", "marginBottom": "6px"}),
                     control])


_DD = {"background": SURFACE, "color": INK, "border": f"1px solid {HAIR}"}
_NUM = {"background": SURFACE, "color": INK, "border": f"1px solid {HAIR}",
        "borderRadius": "7px", "padding": "5px 8px", "width": "58px",
        "fontVariantNumeric": "tabular-nums", "fontFamily": "inherit", "fontSize": "13px"}


def _numbox(label, ident, value):
    """A dark-themed number input, tied to the RangeSlider, with a small caption."""
    return html.Div(
        [dcc.Input(id=ident, type="number", min=5, max=250, step=1, value=value,
                   debounce=True, style=_NUM),
         html.Span(label, style={"color": MUTED, "fontSize": "11px",
                                 "textTransform": "uppercase", "letterSpacing": ".5px"})],
        style={"display": "flex", "alignItems": "center", "gap": "6px"},
    )

APP_BASE = "/app/"  # the dashboard; "/" is the landing page

app = Dash(__name__, title="marketlab — why backtests lie",
           update_title=None, suppress_callback_exceptions=True,
           url_base_pathname=APP_BASE)
server = app.server  # for gunicorn / systemd

_STATIC = Path(__file__).parent / "static"


@server.route("/")
def landing():
    """The marketing page. Its call-to-action links through to APP_BASE."""
    return send_from_directory(_STATIC, "landing.html")

# Dash 3 ships native Dropdown/Slider components (a <button> popover and a
# custom slider -- no more react-select / rc-slider), and their default theme is
# LIGHT: white control + menu backgrounds, dark option text. On our dark surface
# that reads as white-on-white boxes with invisible text. Override the (stable)
# `dash-*` class names here so every control renders dark-surface + legible.
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  /* --- Dropdown: closed trigger button --- */
  .dash-dropdown { background-color:#1a1a19 !important; color:#ffffff !important;
                   border:1px solid rgba(255,255,255,0.10) !important; }
  .dash-dropdown-trigger, .dash-dropdown-value,
  .dash-dropdown-value-item, .dash-dropdown-value-item * { color:#ffffff !important; }
  .dash-dropdown-trigger-icon { fill:#898781 !important; color:#898781 !important; }
  /* --- Dropdown: open popover + search + options --- */
  .dash-dropdown-content, .dash-dropdown-wrapper,
  .dash-dropdown-search-container { background-color:#1a1a19 !important; }
  .dash-dropdown-search { color:#ffffff !important; }
  .dash-dropdown-search::placeholder { color:#898781 !important; }
  .dash-dropdown-search-icon { fill:#898781 !important; color:#898781 !important; }
  .dash-options-list-option, .dash-options-list-option-wrapper,
  .dash-options-list-option-text { color:#c3c2b7 !important; }
  .dash-options-list-option:hover { background-color:#222b36 !important; }
  .dash-options-list-option:hover,
  .dash-options-list-option:hover .dash-options-list-option-text { color:#ffffff !important; }
  .dash-options-list-option.selected,
  .dash-options-list-option.selected .dash-options-list-option-text { color:#3987e5 !important; }
  /* --- 'How to read this' hover legend --- */
  .ml-help { position:relative; display:inline-block; cursor:help; vertical-align:middle; }
  .ml-help-pop { position:absolute; top:132%; left:0; z-index:50; width:340px;
    background:#1a1a19; border:1px solid rgba(255,255,255,0.10); border-radius:10px;
    padding:13px 15px; box-shadow:0 12px 34px rgba(0,0,0,0.55);
    opacity:0; visibility:hidden; transform:translateY(-4px);
    transition:opacity .12s ease, transform .12s ease; }
  .ml-help:hover .ml-help-pop, .ml-help:focus-within .ml-help-pop {
    opacity:1; visibility:visible; transform:translateY(0); }
  /* --- RangeSlider --- */
  .dash-slider-track { background-color:#2c2c2a !important; }
  .dash-slider-range { background-color:#3987e5 !important; }
  .dash-slider-thumb { border-color:#3987e5 !important; background-color:#1a1a19 !important; opacity:1 !important; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

app.layout = html.Div(
    style={"background": PAGE, "minHeight": "100vh", "fontFamily": FONT,
           "color": INK, "padding": "28px min(6vw, 60px)"},
    children=[
        dcc.Interval(id="tick", interval=12000, n_intervals=0),
        # header
        html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "flex-start", "flexWrap": "wrap", "gap": "12px"},
            children=[
                html.Div([
                    html.Div([
                        html.Span("marketlab", style={"fontSize": "30px",
                                                      "fontWeight": 800, "letterSpacing": "-.5px"}),
                        html.Span("why naive backtests lie", style={
                            "marginLeft": "12px", "color": BLUE, "fontSize": "13px",
                            "fontWeight": 600, "border": f"1px solid {BLUE}",
                            "borderRadius": "999px", "padding": "3px 10px"}),
                    ]),
                    html.Div(
                        [html.Span("Pick a symbol, strategy, and cost model. The chart shows "
                                   "three equity curves — only one is real."),
                         _help()],
                        style={"color": INK2, "marginTop": "6px", "fontSize": "14px"}),
                ]),
                html.Div(id="universe", style={"color": MUTED, "fontSize": "13px",
                                               "fontVariantNumeric": "tabular-nums",
                                               "textAlign": "right"}),
            ],
        ),
        # controls
        _panel(
            html.Div(
                style={"display": "flex", "gap": "22px", "flexWrap": "wrap",
                       "alignItems": "flex-start"},
                children=[
                    _ctl("Symbol", dcc.Dropdown(_default_options(), "SPY", id="symbol",
                                                clearable=False, className="ml-dd",
                                                style={**_DD, "width": "150px"},
                                                placeholder="search 12k tickers…")),
                    _ctl("Strategy", dcc.Dropdown(sorted(REGISTRY), "sma_crossover",
                                                  id="strategy", clearable=False, className="ml-dd",
                                                  style={**_DD, "width": "190px"})),
                    _ctl("Fast / slow window", html.Div(
                        # marks=None is load-bearing: the slider auto-generates
                        # marks when the prop is omitted. The number boxes below
                        # carry both values, so the track stays bare.
                        [dcc.RangeSlider(5, 250, value=[50, 200], id="windows",
                                         marks=None),
                         html.Div(
                             [_numbox("fast", "fast_in", 50),
                              _numbox("slow", "slow_in", 200)],
                             style={"display": "flex", "gap": "16px", "marginTop": "10px"}),
                         ],
                        style={"width": "280px", "paddingTop": "4px"})),
                    _ctl("Cost model", dcc.Dropdown(sorted(COST_PRESETS), "retail", id="cost",
                                                    clearable=False, className="ml-dd",
                                                    style={**_DD, "width": "170px"})),
                ],
            ),
            marginTop="20px",
        ),
        # KPI row
        html.Div(id="cards", style={"display": "flex", "gap": "14px",
                                    "flexWrap": "wrap", "margin": "18px 0"}),
        # charts
        _panel(dcc.Graph(id="equity", config={"displayModeBar": False}), padding="8px"),
        html.Div(style={"height": "14px"}),
        _panel(dcc.Graph(id="drawdown", config={"displayModeBar": False}), padding="8px"),
        # verdict
        html.Div(id="verdict", style={"marginTop": "16px"}),
        html.Div("marketlab · Parquet + Postgres · vectorized ∥ event-driven · "
                 "github.com/nicholasbaillargeon-ux/Market-lab",
                 style={"color": MUTED, "fontSize": "12px", "marginTop": "26px",
                        "borderTop": f"1px solid {HAIR}", "paddingTop": "14px"}),
    ],
)


def _layout_fig(title, height, ytitle, pct=False, legend=True):
    # The title and the horizontal legend both sit above the plot, left-aligned,
    # so they need separate bands: anchor each to an explicit edge (title to the
    # container top, legend to the plot top) and reserve enough top margin for
    # both. Mixing the two reference frames is what stacked them on one line.
    top = 78 if legend else 46
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK2), x=0.01, xref="paper",
                   y=1, yanchor="top", yref="container", pad=dict(t=14)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=MUTED, family=FONT),
        margin=dict(l=62, r=24, t=top, b=36), height=height,
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yref="paper", y=1, yanchor="bottom",
                    x=0, xref="paper", font=dict(color=INK2, size=12),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=12)), gridcolor=GRID,
                     zeroline=False, tickformat=".0%" if pct else "$,.0f",
                     tickfont=dict(color=MUTED))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


# ---- symbol search (server-side, keeps the 12k dropdown snappy) ----
@app.callback(Output("symbol", "options"), Input("symbol", "search_value"),
              State("symbol", "value"))
def search_symbols(search, current):
    syms = _cache["syms"] or _all_symbols()
    if not search:
        raise PreventUpdate
    s = search.upper()
    hits = [x for x in syms if x.startswith(s)][:60]
    if len(hits) < 60:
        hits += [x for x in syms if s in x and x not in hits][:60 - len(hits)]
    if current and current not in hits:
        hits = [current] + hits
    return hits


@app.callback(Output("universe", "children"), Input("tick", "n_intervals"))
def universe_count(_):
    return [html.Span(f"{len(_all_symbols()):,}", style={"color": INK, "fontWeight": 700}),
            html.Span(" symbols in the lake", style={"color": MUTED})]


# ---- keep the slider and the two number inputs in lock-step ----
@app.callback(
    Output("windows", "value"), Output("fast_in", "value"), Output("slow_in", "value"),
    Input("windows", "value"), Input("fast_in", "value"), Input("slow_in", "value"),
    prevent_initial_call=True,
)
def sync_windows(win, fast_in, slow_in):
    if ctx.triggered_id == "windows":
        # slider moved -> mirror its two handles into the number boxes
        return no_update, int(win[0]), int(win[1])
    # a number box was edited -> clamp, order (fast <= slow), drive the slider
    if fast_in is None or slow_in is None:
        raise PreventUpdate
    lo, hi = sorted((int(fast_in), int(slow_in)))
    lo = max(5, min(250, lo))
    hi = max(5, min(250, hi))
    return [lo, hi], lo, hi


@app.callback(
    Output("cards", "children"), Output("equity", "figure"),
    Output("drawdown", "figure"), Output("verdict", "children"),
    Input("symbol", "value"), Input("strategy", "value"),
    Input("windows", "value"), Input("cost", "value"),
)
def update(symbol, strategy, windows, cost):
    bars = STORE.read_bars(symbol)
    if bars.empty:
        empty = _layout_fig("no data", 300, "equity ($)")
        return [], empty, empty, ""

    fast, slow = int(windows[0]), int(windows[1])
    try:
        if strategy == "sma_crossover":
            strat = REGISTRY[strategy](fast=fast, slow=slow)
        elif strategy == "mean_reversion":
            strat = REGISTRY[strategy](lookback=max(2, fast), entry_z=1.0)
        else:
            strat = REGISTRY[strategy]()
    except ValueError as e:
        empty = _layout_fig("invalid parameters", 300, "equity ($)")
        return [], empty, empty, html.Div(f"⚠️ {e}", style={"color": AMBER})

    w = strat.target_weights(bars)
    cm = COST_PRESETS[cost]()

    ev = run_event_driven(bars, w, cm)                                    # honest + costed
    gross = run_event_driven(bars, w, CostModel.zero())                   # honest, no costs
    gross_vec = run_vectorized(bars, w, CostModel.zero())                 # fast engine, no costs
    cheat = run_vectorized(bars, w, CostModel.zero(), allow_lookahead=True)  # the lie

    m = summary(ev)
    # Engine equivalence is a ZERO-COST claim -- with costs the two engines
    # differ by design (cash vs return accounting), so compare them cost-free.
    parity_gap = abs(gross_vec.total_return - gross.total_return)
    # Drag / look-ahead as a fraction of gross terminal wealth: bounded and
    # honest even when the raw return is a 150x compounding monster.
    cost_drag = (gross.final_equity - ev.final_equity) / gross.final_equity
    lookahead = (cheat.final_equity - gross.final_equity) / gross.final_equity
    ret_pos = m["total_return"] >= 0

    cards = [
        _stat("Net total return", _ret(m["total_return"]), f"CAGR {m['cagr']:+.1%}",
              GOOD if ret_pos else CRIT, "tradeable"),
        _stat("Sharpe (net)", f"{m['sharpe']:.2f}", f"max drawdown {m['max_drawdown']:.1%}",
              BLUE, "risk-adj"),
        _stat("Transaction-cost drag", f"−{cost_drag:.1%}", f"{m['n_trades']} trades · of gross",
              AMBER, "friction"),
        _stat("Look-ahead Δ", f"{lookahead:+.1%}", "unrealizable edge",
              AMBER, "fantasy"),
        _stat("Engine parity gap", f"{parity_gap:.0e}", "vectorized ≡ event",
              GOOD, "verified"),
    ]

    # equity chart
    eq = _layout_fig(f"{symbol} · {strat.name} · {len(bars):,} bars", 430, "equity ($)")
    eq.add_trace(go.Scatter(x=gross.equity.index, y=gross.equity, name="gross (zero cost)",
                            line=dict(color=MUTED, width=1.6), hovertemplate="$%{y:,.0f}<extra>gross</extra>"))
    eq.add_trace(go.Scatter(x=cheat.equity.index, y=cheat.equity, name="look-ahead (fantasy)",
                            line=dict(color=AMBER, width=2, dash="dot"),
                            hovertemplate="$%{y:,.0f}<extra>look-ahead</extra>"))
    eq.add_trace(go.Scatter(x=ev.equity.index, y=ev.equity, name=f"net · {cost} (tradeable)",
                            line=dict(color=BLUE, width=2.6), fill="tozeroy",
                            fillcolor="rgba(57,135,229,0.07)",
                            hovertemplate="$%{y:,.0f}<extra>net</extra>"))

    # underwater / drawdown chart (net strategy)
    peak = ev.equity.cummax()
    dd = ev.equity / peak - 1.0
    uw = _layout_fig("drawdown — the pain you'd actually have felt (net)", 230,
                     "drawdown", pct=True, legend=False)
    uw.add_trace(go.Scatter(x=dd.index, y=dd, line=dict(color=CRIT, width=1.5),
                            fill="tozeroy", fillcolor="rgba(208,59,59,0.15)",
                            hovertemplate="%{y:.1%}<extra>drawdown</extra>"))

    la_verb = "fabricates a" if lookahead >= 0 else "distorts results by"
    verdict = _panel(
        html.Div([
            html.Span("The verdict.  ", style={"color": INK, "fontWeight": 700, "fontSize": "15px"}),
            html.Span("The ", style={"color": INK2}),
            html.Span("blue line", style={"color": BLUE, "fontWeight": 600}),
            html.Span(" is the only equity curve you could have actually traded. The ", style={"color": INK2}),
            html.Span("amber look-ahead curve", style={"color": AMBER, "fontWeight": 600}),
            html.Span(f" {la_verb} {lookahead:+.1%} of gross wealth by trading on prices it "
                      f"couldn't have known, and honest transaction costs erase {cost_drag:.1%} "
                      f"more. The vectorized and event-driven engines agree to {parity_gap:.0e} "
                      f"(zero-cost), so the fast research number is trustworthy.",
                      style={"color": INK2}),
        ], style={"lineHeight": "1.65", "fontSize": "14px"}),
        borderLeft=f"3px solid {BLUE}",
    )
    return cards, eq, uw, verdict


def main():
    port = int(os.environ.get("MARKETLAB_PORT", "8060"))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
