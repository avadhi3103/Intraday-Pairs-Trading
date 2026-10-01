"""
app.py — interactive web UI for the intraday pairs-trading project.

Run locally:   streamlit run app.py
It downloads the latest ~60 days of 5-minute bars from Yahoo Finance when opened
(nothing is redistributed) and runs the exact same engine as the notebooks (src/).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest_report as br
import cointegration as ci
import data_pipeline as dp
import garch_thresholds as gt
import strategy as strat
from costs import IndianEquityCostModel, round_trip_breakdown

REPO_URL = "https://github.com/avadhi3103/Intraday-Pairs-Trading"
BLUE, ORANGE, AQUA, YELLOW = br.SERIES_COLORS
GREY = "#8a8984"

st.set_page_config(page_title="Intraday Pairs Trading Simulator", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Data + computation (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading 5-minute bars from Yahoo Finance…")
def load_prices(ticker_a: str, ticker_b: str) -> pd.DataFrame:
    frames = []
    for t in (ticker_a, ticker_b):
        try:
            raw = dp.download_ticker(t)
        except Exception:
            if dp.raw_path(t).exists():           # offline fallback to a local cache, if any
                raw = pd.read_parquet(dp.raw_path(t))
            else:
                raise
        frames.append(dp.clean_bars(raw))
    return dp.align_pair(frames[0], frames[1], ticker_a.replace(".NS", ""), ticker_b.replace(".NS", ""))


@st.cache_data(show_spinner="Running the walk-forward backtest…")
def run_pipeline(prices, formation_days, z_window, hedge_days, entry, exit_, notional, slippage):
    formation, trading = dp.split_formation_trading(prices, formation_days)
    a, b = formation.iloc[:, 0], formation.iloc[:, 1]
    phase1 = {**ci.correlation_prescreen(a, b), **ci.engle_granger(a, b)}
    _, _, formation_resid = ci.estimate_hedge_ratio(a, b)

    sig = strat.build_signals(prices, trade_start=trading.index[0],
                              zscore_window=z_window, hedge_lookback_days=hedge_days)
    vol, params = gt.rolling_garch(prices, sig)
    sig = sig.loc[vol.index]                         # common window for every variant
    entry_t, exit_t = gt.adaptive_thresholds(vol, entry, exit_)

    cm = IndianEquityCostModel(slippage_bps=slippage)
    kw = dict(notional_per_leg=notional)
    res = {
        "fixed_gross": strat.run_backtest(sig, entry, exit_, label="Fixed threshold, before costs", **kw),
        "fixed_net": strat.run_backtest(sig, entry, exit_, cost_model=cm, label="Fixed threshold, after costs", **kw),
        "garch_gross": strat.run_backtest(sig, entry_t, exit_t, label="GARCH threshold, before costs", **kw),
        "garch_net": strat.run_backtest(sig, entry_t, exit_t, cost_model=cm, label="GARCH threshold, after costs", **kw),
        "bh": strat.buy_and_hold(sig[sig.columns[:2]], cost_model=cm, label="Buy & hold"),
    }
    sweep_grid = [0, 0.5, 1, 2, 3, 5, 10]
    sweep = pd.Series({s: br.compute_metrics(strat.run_backtest(
        sig, entry, exit_, cost_model=IndianEquityCostModel(slippage_bps=s), **kw))["Total return"]
        for s in sweep_grid})
    return dict(phase1=phase1, formation_resid=formation_resid, sig=sig, vol=vol, params=params,
                res=res, sweep=sweep, round_trip=round_trip_breakdown(notional, cm))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def naive(idx):
    """Plotly shows wall-clock time correctly only for tz-naive timestamps."""
    return idx.tz_localize(None)


def session_rangebreaks(index):
    """Hide nights, weekends and exchange holidays on multi-day time axes."""
    days = pd.DatetimeIndex(index.normalize().unique()).tz_localize(None)
    all_bdays = pd.bdate_range(days.min(), days.max())
    holidays = [d.strftime("%Y-%m-%d") for d in all_bdays.difference(days)]
    return [dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour"), dict(values=holidays)]


def inr(x):
    return f"₹{x:,.0f}"


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=90, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0), hovermode="x unified")
    for tr in fig.data:                          # plotly adds dots to short lines by default
        if tr.type == "scatter" and tr.mode is None:
            tr.mode = "lines"
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
presets = {f"{a.replace('.NS', '')} / {b.replace('.NS', '')}": (a, b) for a, b in dp.CANDIDATE_PAIRS}
choice = st.sidebar.selectbox("Stock pair", list(presets) + ["Custom pair…"], index=2)
if choice == "Custom pair…":
    c1, c2 = st.sidebar.columns(2)
    sym_a = c1.text_input("Stock A (NSE symbol)", "AXISBANK").strip().upper()
    sym_b = c2.text_input("Stock B (NSE symbol)", "KOTAKBANK").strip().upper()
    ticker_a = sym_a if sym_a.endswith(".NS") else f"{sym_a}.NS"
    ticker_b = sym_b if sym_b.endswith(".NS") else f"{sym_b}.NS"
else:
    ticker_a, ticker_b = presets[choice]

with st.sidebar.expander("Strategy parameters", expanded=True):
    entry = st.slider("Entry threshold |z|", 1.0, 3.5, strat.ENTRY_Z, 0.1,
                      help="Open a trade when the spread is this many standard deviations from its rolling mean.")
    exit_ = st.slider("Exit threshold |z|", 0.0, 1.5, strat.EXIT_Z, 0.1,
                      help="Close when z comes back inside this band (0 = full reversion to the mean).")
    z_window = st.slider("Z-score window (bars)", 20, 150, strat.ZSCORE_WINDOW, 5,
                         help="Rolling window for the spread mean and std. 75 bars ≈ one session.")
    hedge_days = st.slider("Hedge-ratio lookback (days)", 3, 15, strat.HEDGE_LOOKBACK_DAYS,
                           help="β is re-estimated every morning by OLS on this many previous days.")
    formation_days = st.slider("Formation period (days)", 15, 25, ci.FORMATION_DAYS,
                               help="Days used only for the cointegration test. Trading starts after them.")
with st.sidebar.expander("Sizing & costs"):
    notional = st.select_slider("Notional per leg (₹)", [100_000, 250_000, 500_000, 1_000_000],
                                strat.NOTIONAL_PER_LEG, format_func=inr)
    slippage = st.slider("Slippage (bps per fill)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.caption("Data: Yahoo Finance 5-minute bars (last ~60 days, may be delayed). "
                   "Educational project, not investment advice.")
st.sidebar.markdown(f"[Source code on GitHub]({REPO_URL})")


# ---------------------------------------------------------------------------
# Load + compute
# ---------------------------------------------------------------------------
try:
    prices = load_prices(ticker_a, ticker_b)
except Exception as e:
    st.error(f"Could not download data for {ticker_a} / {ticker_b}: {e}. "
             "Check the NSE symbols, or try again in a minute (Yahoo sometimes rate-limits).")
    st.stop()

A, B = prices.columns[:2]
if prices.index.normalize().nunique() < formation_days + 10:
    st.error("Not enough history for this pair and formation period.")
    st.stop()

out = run_pipeline(prices, formation_days, z_window, hedge_days, entry, exit_, notional, slippage)
p1, sig, vol, res = out["phase1"], out["sig"], out["vol"], out["res"]
eg_ok = p1["eg_p_value"] < ci.SIGNIFICANCE

st.title("Intraday Pairs Trading Simulator")
st.caption(f"**{A} / {B}** · {len(prices):,} five-minute bars · "
           f"{prices.index[0]:%d %b %Y} → {prices.index[-1]:%d %b %Y} · "
           f"trading period starts {sig.index[0]:%d %b}")
if not eg_ok:
    st.warning(f"**{A}/{B} is not cointegrated** in the formation period (Engle-Granger p = "
               f"{p1['eg_p_value']:.3f} ≥ 0.05). The simulation still runs so you can see how the engine works, "
               "but there's no statistical evidence that this spread mean-reverts.")

tab_how, tab_p1, tab_replay, tab_res, tab_garch = st.tabs(
    ["How it works", "1 · Pair test", "2 · Simulation replay", "3 · Results & costs", "4 · GARCH thresholds"])


# ---------------------------------------------------------------------------
# Tab: How it works
# ---------------------------------------------------------------------------
with tab_how:
    st.markdown(f"""
### The idea
Two stocks driven by the same economics (here **{A}** and **{B}**) usually move together.
When the gap between them becomes unusually wide, we **sell the expensive one, buy the cheap one**,
and close both when the gap returns to normal. The profit comes from the *relative* move, so a market-wide
rally or crash mostly cancels out.

### The pipeline
| Step | What happens | Tab |
|---|---|---|
| **1. Pair test** | On the first {formation_days} days only: OLS gives the hedge ratio β, then an ADF test checks whether the gap `A − β·B` is stationary (mean-reverting). | 1 |
| **2. Signals** | Each morning β is re-estimated on the previous {hedge_days} days. Every 5 minutes: `z = (spread − rolling mean) / rolling std` over {z_window} bars. | 2 |
| **3. Trading rules** | z < −{entry:.1f} → **long spread** (buy {A}, sell {B}). z > +{entry:.1f} → **short spread**. Close when z returns inside ±{exit_:.1f}. All positions closed by 15:10. | 2 |
| **4. Costs** | Every fill pays brokerage, STT (sell side), exchange & SEBI fees, stamp duty (buy side), GST and {slippage:g} bps slippage. | 3 |
| **5. GARCH** | A GARCH(1,1) model forecasts spread volatility. The entry threshold widens when vol is high and tightens when it's calm. | 4 |

### How one bar of the simulation works
Every 5-minute bar, the engine does these steps **in this order**, so it never uses information it wouldn't have had:
1. **Mark to market:** revalue the shares held over the last 5 minutes.
2. **Fill pending orders:** an order decided on the previous bar is executed at *this* bar's close (one-bar delay).
3. **Safety close:** if this is the day's last bar, close anything still open (no overnight positions).
4. **Decide:** compare z with the thresholds and the clock. Any new order fills on the *next* bar.

Open **2 · Simulation replay** to step through this bar by bar.
""")


# ---------------------------------------------------------------------------
# Tab: Phase 1
# ---------------------------------------------------------------------------
with tab_p1:
    st.subheader("Is the gap between these stocks mean-reverting?")
    c = st.columns(5)
    c[0].metric("Hedge ratio β", f"{p1['hedge_ratio']:.3f}", help="Shares of B that offset one share of A (OLS slope).")
    c[1].metric("ADF statistic", f"{p1['adf_stat']:.2f}", help="More negative = stronger evidence the spread is stationary.")
    c[2].metric("Plain ADF p-value", f"{p1['adf_p_value']:.3f}", help="Textbook adfuller p-value (too lenient on OLS residuals).")
    c[3].metric("Engle-Granger p-value", f"{p1['eg_p_value']:.3f}", help="Same statistic with correct MacKinnon critical values. This decides.")
    c[4].metric("Half-life", f"{p1['half_life_bars']:.0f} bars", help="Bars for a typical deviation to shrink by half (AR(1) fit).")

    if eg_ok:
        st.success(f"Cointegrated at 5%: the spread looks stationary, so trading its mean reversion has statistical support.")
    elif p1["adf_p_value"] < ci.SIGNIFICANCE:
        st.error("Not cointegrated. The plain ADF test passes but the correct Engle-Granger test fails. This is the "
                 "classic trap: OLS makes the residual look more stationary than it really is.")
    else:
        st.error("Not cointegrated: the data don't show evidence that this spread mean-reverts.")
    st.caption(f"Average rolling price correlation: {p1['avg_rolling_corr']:.2f} "
               f"({'passes' if p1['passes_prescreen'] else 'fails'} the {ci.MIN_AVG_CORRELATION} pre-screen). "
               "H0 of the ADF test: the spread has a unit root (random walk, no mean reversion). Reject if p < 0.05.")

    resid = out["formation_resid"]
    fig = go.Figure()
    fig.add_scatter(x=naive(resid.index), y=resid.values, name="Spread (OLS residual)", line=dict(color=BLUE, width=1.2))
    for k, dash in ((2, "dash"), (0, "solid"), (-2, "dash")):
        fig.add_hline(y=k * resid.std(), line=dict(color=GREY, width=1, dash=dash))
    fig.update_xaxes(rangebreaks=session_rangebreaks(resid.index))
    fig.update_layout(title=f"Formation-period spread {A} − {p1['hedge_ratio']:.2f}·{B} − α, with ±2σ bands")
    st.plotly_chart(style_fig(fig, 360), width="stretch")
    st.caption("A mean-reverting spread crosses its mean often. One that wanders away for days is not tradeable as mean-reverting.")

    daily_beta = sig["beta"].groupby(sig.index.normalize()).first()
    fig = go.Figure(go.Scatter(x=naive(daily_beta.index), y=daily_beta.values, mode="lines+markers",
                               line=dict(color=BLUE), name="β"))
    fig.update_layout(title="Walk-forward hedge ratio during the trading period (re-estimated each morning)")
    st.plotly_chart(style_fig(fig, 280), width="stretch")


# ---------------------------------------------------------------------------
# Tab: Simulation replay
# ---------------------------------------------------------------------------
def explain_bar(day: pd.DataFrame, k: int, result, trades: pd.DataFrame, use_costs: bool) -> list:
    """Plain-English account of the four engine steps at bar k of the chosen day."""
    row, ts = day.iloc[k], day.index[k]
    tm = ts.strftime("%H:%M")
    lines = []

    # 1) mark to market
    pos_before = day["position"].iloc[k - 1] if k > 0 else 0
    if k == 0:
        lines.append("**1 · Mark to market:** first bar of the day. The book is flat overnight, so there's nothing to revalue.")
    elif pos_before == 0:
        lines.append("**1 · Mark to market:** no position was held over the last 5 minutes.")
    else:
        mtm = row["sh_a_prev"] * (row[A] - day[A].iloc[k - 1]) + row["sh_b_prev"] * (row[B] - day[B].iloc[k - 1])
        lines.append(f"**1 · Mark to market:** {A} moved {row[A] - day[A].iloc[k-1]:+.2f}, "
                     f"{B} moved {row[B] - day[B].iloc[k-1]:+.2f} → open position P&L this bar **{inr(mtm)}**.")

    # 2/3) fills at this bar
    filled = False
    for _, t in trades[trades["exit_time"] == ts].iterrows():
        filled = True
        why = {"signal": "z reverted (exit signal from the previous bar)",
               "eod": "scheduled end-of-day flatten (signal at 15:05)",
               "session_end": "the day's data ended, safety close"}[t["exit_reason"]]
        cost_txt = f", costs {inr(t['costs'])}, **net {inr(t['net_pnl'])}**" if use_costs else ""
        lines.append(f"**2 · Fill:** position **closed** at {A} {t['exit_a']:.2f} / {B} {t['exit_b']:.2f}. "
                     f"Reason: {why}. Trade gross P&L {inr(t['gross_pnl'])}{cost_txt}, held {t['bars_held']} bars.")
    for _, t in trades[trades["entry_time"] == ts].iterrows():
        filled = True
        side_a = "Bought" if t["shares_a"] > 0 else "Sold"
        side_b = "bought" if t["shares_b"] > 0 else "sold"
        lines.append(f"**2 · Fill:** order from the previous bar executed. {side_a} **{abs(t['shares_a'])} {A}** @ "
                     f"{t['entry_a']:.2f} ({inr(abs(t['shares_a']) * t['entry_a'])}) and {side_b} "
                     f"**{abs(t['shares_b'])} {B}** @ {t['entry_b']:.2f} ({inr(abs(t['shares_b']) * t['entry_b'])}). "
                     "Equal rupees on each side = dollar-neutral.")
    if not filled:
        lines.append("**2 · Fill:** no pending order.")

    # 4) decision
    z, e, x, pos = row["zscore"], row["entry_threshold"], row["exit_threshold"], row["position"]
    is_last = k == len(day) - 1
    if is_last:
        d = "last bar of the day. Everything is flat, and no new trades until tomorrow."
    elif np.isnan(z):
        d = "not enough history for a z-score yet."
    elif tm >= strat.EOD_EXIT_TIME:
        d = ("it's 15:05 or later → **flatten signal**, fills at the next bar." if pos != 0
             else "it's after 15:05, so no new trades today.")
    elif pos == 0:
        if tm >= strat.NO_NEW_ENTRY_AFTER:
            d = f"flat, and it's after {strat.NO_NEW_ENTRY_AFTER}, so no new entries (not enough time left to revert)."
        elif z < -e:
            d = (f"flat and z = **{z:.2f} < −{e:.2f}**: the spread is unusually LOW → **signal: LONG spread** "
                 f"(buy {A}, sell {B}). Fills at the next bar's close.")
        elif z > e:
            d = (f"flat and z = **{z:.2f} > +{e:.2f}**: the spread is unusually HIGH → **signal: SHORT spread** "
                 f"(sell {A}, buy {B}). Fills at the next bar's close.")
        else:
            d = f"flat and z = {z:.2f} is inside ±{e:.2f} → no signal, stay flat."
    elif pos == 1:
        d = (f"long spread and z = **{z:.2f} ≥ −{x:.2f}** → the spread has reverted → **signal: CLOSE**." if z >= -x
             else f"long spread, z = {z:.2f} is still below −{x:.2f} → hold and wait for reversion.")
    else:
        d = (f"short spread and z = **{z:.2f} ≤ +{x:.2f}** → the spread has reverted → **signal: CLOSE**." if z <= x
             else f"short spread, z = {z:.2f} is still above +{x:.2f} → hold and wait for reversion.")
    lines.append(f"**3–4 · Decide:** {d}")
    return lines


def replay_figure(day: pd.DataFrame, k: int, trades: pd.DataFrame):
    x_all = naive(day.index)
    shown = day.iloc[: k + 1]
    x = naive(shown.index)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.32, 0.4, 0.28],
                        subplot_titles=("Prices, rebased to 100 at the open",
                                        "Z-score of the spread and trading thresholds",
                                        "Intraday P&L (₹, after costs if enabled)"))
    for col, colr in ((A, BLUE), (B, ORANGE)):
        fig.add_scatter(x=x, y=shown[col] / day[col].iloc[0] * 100, name=col, line=dict(color=colr, width=2), row=1, col=1)

    fig.add_scatter(x=x_all, y=day["entry_threshold"], name="entry band", line=dict(color=GREY, dash="dash", width=1),
                    row=2, col=1)
    fig.add_scatter(x=x_all, y=-day["entry_threshold"], showlegend=False, line=dict(color=GREY, dash="dash", width=1),
                    row=2, col=1)
    if (day["exit_threshold"] > 0).any():
        for sgn in (1, -1):
            fig.add_scatter(x=x_all, y=sgn * day["exit_threshold"], showlegend=sgn == 1, name="exit band",
                            line=dict(color=GREY, dash="dot", width=1), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=GREY, width=1), row=2, col=1)
    fig.add_scatter(x=x, y=shown["zscore"], name="z-score", line=dict(color=BLUE, width=2), row=2, col=1)

    # position shading
    pos = shown["position"].values
    start = None
    for i in range(len(pos) + 1):
        cur = pos[i] if i < len(pos) else 0
        if start is None and cur != 0:
            start, sgn = i, cur
        elif start is not None and cur != sgn:
            x1 = x[i] if i < len(x) else x[-1]
            fig.add_vrect(x0=x[start], x1=x1, fillcolor=AQUA if sgn > 0 else ORANGE, opacity=0.13, line_width=0)
            start = None
            if cur != 0:
                start, sgn = i, cur

    shown_ts = set(shown.index)
    ent = trades[trades["entry_time"].isin(shown_ts)]
    ext = trades[trades["exit_time"].isin(shown_ts)]
    if len(ent):
        fig.add_scatter(x=naive(pd.DatetimeIndex(ent["entry_time"])), y=day.loc[ent["entry_time"], "zscore"],
                        mode="markers", name="entry fill",
                        marker=dict(symbol="triangle-up", size=12, color=AQUA, line=dict(color="white", width=1.5)),
                        row=2, col=1)
    if len(ext):
        fig.add_scatter(x=naive(pd.DatetimeIndex(ext["exit_time"])), y=day.loc[ext["exit_time"], "zscore"],
                        mode="markers", name="exit fill",
                        marker=dict(symbol="square", size=10, color=ORANGE, line=dict(color="white", width=1.5)),
                        row=2, col=1)

    fig.add_scatter(x=x, y=shown["day_pnl"], name="P&L", line=dict(color=BLUE, width=2), showlegend=False,
                    row=3, col=1)
    fig.add_hline(y=0, line=dict(color=GREY, width=1), row=3, col=1)

    zmax = max(4.0, float(np.nanmax(np.abs(day["zscore"]))) + 0.5) if day["zscore"].notna().any() else 4.0
    fig.update_yaxes(range=[-zmax, zmax], row=2, col=1)
    lo, hi = float(day["day_pnl"].min()), float(day["day_pnl"].max())
    pad = max(500.0, (hi - lo) * 0.15)
    fig.update_yaxes(range=[min(lo, 0) - pad, max(hi, 0) + pad], row=3, col=1)
    pmin = min((day[c] / day[c].iloc[0] * 100).min() for c in (A, B))
    pmax = max((day[c] / day[c].iloc[0] * 100).max() for c in (A, B))
    fig.update_yaxes(range=[pmin - 0.1, pmax + 0.1], row=1, col=1)
    fig.update_xaxes(range=[x_all[0], x_all[-1]])
    return style_fig(fig, 640)


with tab_replay:
    st.subheader("Step through the backtest one 5-minute bar at a time")
    c1, c2 = st.columns(2)
    variant = c1.radio("Threshold type", ["Fixed", "GARCH-adjusted"], horizontal=True,
                       help="GARCH-adjusted: the entry band widens when forecast spread volatility is high.")
    use_costs = c2.toggle("Apply transaction costs", value=True)
    key = ("garch" if variant.startswith("GARCH") else "fixed") + ("_net" if use_costs else "_gross")
    result = res[key]
    trades = result.trades
    if trades.empty:
        trades = pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_z", "shares_a", "shares_b",
                                       "entry_a", "entry_b", "exit_a", "exit_b", "bars_held", "exit_reason",
                                       "gross_pnl", "costs", "net_pnl"])

    # per-bar frame for the replay
    bars = result.bars.join(sig[[A, B, "beta", "spread"]])
    bars["equity"] = result.equity
    shares = pd.DataFrame(0.0, index=bars.index, columns=["sh_a", "sh_b"])
    for _, t in trades.iterrows():
        held = (bars.index >= t["entry_time"]) & (bars.index < t["exit_time"])
        shares.loc[held, "sh_a"] = t["shares_a"]
        shares.loc[held, "sh_b"] = t["shares_b"]
    bars["sh_a_prev"] = shares["sh_a"].shift(1).fillna(0)
    bars["sh_b_prev"] = shares["sh_b"].shift(1).fillna(0)
    prev_close_eq = bars["equity"].groupby(bars.index.normalize()).last().shift(1).fillna(result.capital)
    bars["day_pnl"] = bars["equity"] - prev_close_eq.reindex(bars.index.normalize()).values

    days = bars.index.normalize().unique()
    trades_per_day = trades.groupby(pd.DatetimeIndex(trades["entry_time"]).normalize()).size() if len(trades) else pd.Series(dtype=int)
    day_labels = {d: f"{d:%a %d %b} · {trades_per_day.get(d, 0)} trade(s)" for d in days}
    c1, c2 = st.columns([2, 1])
    day = c1.selectbox("Trading day", days, format_func=lambda d: day_labels[d])
    speed = c2.select_slider("Playback speed", ["slow", "normal", "fast"], value="normal")
    dbars = bars[bars.index.normalize() == day]
    dtrades = trades[pd.DatetimeIndex(trades["entry_time"]).normalize() == day] if len(trades) else trades

    times = [t.strftime("%H:%M") for t in dbars.index]
    c1, c2 = st.columns([4, 1])
    k = times.index(c1.select_slider("Bar", times, value=times[0], key=f"bar_{day}"))
    play = c2.button("▶ Play day", width="stretch")

    metrics_ph, chart_ph, text_ph = st.empty(), st.empty(), st.empty()

    def draw(i):
        r = dbars.iloc[i]
        pos_txt = {1: "Long", -1: "Short", 0: "Flat"}[int(r["position"])]
        with metrics_ph.container():
            m = st.columns(5)
            m[0].metric("Time", times[i])
            m[1].metric("Z-score", "—" if np.isnan(r["zscore"]) else f"{r['zscore']:.2f}",
                        help=f"Entry when |z| > {r['entry_threshold']:.2f}")
            m[2].metric("Spread position", pos_txt)
            m[3].metric("Hedge ratio β (today)", f"{r['beta']:.3f}")
            m[4].metric("Day P&L", inr(r["day_pnl"]))
        chart_ph.plotly_chart(replay_figure(dbars, i, dtrades), width="stretch", key=f"chart_{day}_{i}_{time.time()}")
        text_ph.info("\n\n".join(explain_bar(dbars, i, result, dtrades, use_costs)))

    if play:
        delay = {"slow": 0.6, "normal": 0.25, "fast": 0.05}[speed]
        for i in range(k, len(dbars)):
            draw(i)
            time.sleep(delay)
    else:
        draw(k)

    with st.expander(f"Trades on this day ({len(dtrades)})"):
        if len(dtrades):
            show = dtrades[["entry_time", "exit_time", "direction", "entry_z", "shares_a", "shares_b",
                            "bars_held", "exit_reason", "gross_pnl", "costs", "net_pnl"]].copy()
            show["direction"] = show["direction"].map({1: "long spread", -1: "short spread"})
            for c in ("entry_time", "exit_time"):
                show[c] = pd.DatetimeIndex(show[c]).strftime("%H:%M")
            st.dataframe(show.round(2), width="stretch", hide_index=True)
        else:
            st.write("No trades opened on this day.")


# ---------------------------------------------------------------------------
# Tab: Results & costs
# ---------------------------------------------------------------------------
with tab_res:
    st.subheader("Whole trading period")
    variants = [res["fixed_gross"], res["fixed_net"], res["garch_net"], res["bh"]]
    table = br.format_table(br.comparison_table(variants)).drop(index=["Trading days"])
    st.table(table)
    st.caption(f"Sharpe uses daily excess returns over a {br.RISK_FREE_RATE_ANNUAL:.1%} risk-free rate. "
               "Annualised figures extrapolate a short sample and exaggerate noise.")

    rb = session_rangebreaks(sig.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08,
                        subplot_titles=("Equity (₹)", "Drawdown (%)"))
    for r, colr in zip(variants, [BLUE, ORANGE, AQUA, YELLOW]):
        x = naive(r.equity.index)
        fig.add_scatter(x=x, y=r.equity.values, name=r.label, line=dict(color=colr, width=1.8), row=1, col=1)
        dd = (r.equity / r.equity.cummax() - 1) * 100
        fig.add_scatter(x=x, y=dd.values, showlegend=False, line=dict(color=colr, width=1.2), row=2, col=1)
    fig.add_hline(y=res["fixed_gross"].capital, line=dict(color=GREY, width=1, dash="dot"), row=1, col=1)
    fig.update_xaxes(rangebreaks=rb)
    st.plotly_chart(style_fig(fig, 560), width="stretch")

    st.subheader("Transaction costs")
    c1, c2 = st.columns(2)
    with c1:
        rt = pd.Series(out["round_trip"])
        rt_tbl = pd.DataFrame({"₹": rt.round(1), "bps of one leg": (rt / notional * 1e4).round(2)})
        rt_tbl.loc["TOTAL"] = rt_tbl.sum()
        st.markdown(f"**One round trip** (4 fills, {inr(notional)} per leg)")
        st.table(rt_tbl.style.format("{:,.2f}"))
        g = res["fixed_gross"].trades
        if len(g):
            edge = g["gross_pnl"].mean() / notional * 1e4
            st.markdown(f"Average gross edge per trade: **{edge:.1f} bps** vs round-trip cost "
                        f"**{rt_tbl.loc['TOTAL', 'bps of one leg']:.1f} bps**. "
                        + ("Costs exceed the edge." if edge < rt_tbl.loc['TOTAL', 'bps of one leg'] else "The edge covers costs."))
    with c2:
        st.markdown("**Before vs after costs** (fixed threshold)")
        st.table(br.format_table(br.cost_comparison_table(res["fixed_gross"], res["fixed_net"])))

    sw = out["sweep"] * 100
    fig = go.Figure(go.Scatter(x=sw.index, y=sw.values, mode="lines+markers", line=dict(color=BLUE), name="net return"))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.add_vline(x=slippage, line=dict(color=ORANGE, dash="dash"), annotation_text="your setting")
    fig.update_layout(title="Net total return vs slippage assumption", xaxis_title="slippage per fill (bps)",
                      yaxis_title="total return (%)", hovermode="x")
    st.plotly_chart(style_fig(fig, 320), width="stretch")

    with st.expander("Full trade log (fixed threshold, after costs)"):
        t = res["fixed_net"].trades.copy()
        if len(t):
            t["direction"] = t["direction"].map({1: "long spread", -1: "short spread"})
            for c in ("entry_time", "exit_time"):
                t[c] = pd.DatetimeIndex(t[c]).strftime("%d %b %H:%M")
            st.dataframe(t.round(2), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab: GARCH
# ---------------------------------------------------------------------------
with tab_garch:
    st.subheader("GARCH(1,1) volatility forecasts and adaptive thresholds")
    st.markdown("`σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁`: **α** is the reaction to the latest shock, **β** is memory, "
                "and **α+β** is persistence. Refit every morning on the previous 15 days of 5-minute spread changes. "
                f"Entry threshold = {entry:.1f} × clip(forecast vol / average vol, 0.5, 2).")
    params = out["params"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("α: reaction to shocks", "GARCH β: memory", "Persistence α+β"))
    px = naive(params.index)
    fig.add_scatter(x=px, y=params["arch_alpha"], name="ARCH(1)", mode="lines+markers", line=dict(color=ORANGE), row=1, col=1)
    fig.add_scatter(x=px, y=params["garch_alpha"], name="GARCH(1,1)", mode="lines+markers", line=dict(color=BLUE), row=1, col=1)
    fig.add_scatter(x=px, y=params["garch_beta"], showlegend=False, mode="lines+markers", line=dict(color=BLUE), row=2, col=1)
    fig.add_scatter(x=px, y=params["persistence"], showlegend=False, mode="lines+markers", line=dict(color=BLUE), row=3, col=1)
    fig.add_scatter(x=px, y=params["arch_alpha"], showlegend=False, mode="lines+markers", line=dict(color=ORANGE), row=3, col=1)
    fig.add_hline(y=1, line=dict(color=GREY, dash="dot"), row=3, col=1)
    st.plotly_chart(style_fig(fig, 620), width="stretch")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Spread volatility forecast (bps per 5 min)", "Entry threshold |z|"))
    vx = naive(vol.index)
    fig.add_scatter(x=vx, y=vol["vol_forecast"], name="GARCH forecast", line=dict(color=BLUE, width=1), row=1, col=1)
    fig.add_scatter(x=vx, y=vol["vol_baseline"], name="baseline", line=dict(color=GREY, dash="dash"), row=1, col=1)
    fig.add_scatter(x=vx, y=entry * vol["vol_ratio"], name="adaptive", line=dict(color=BLUE, width=1), row=2, col=1)
    fig.add_hline(y=entry, line=dict(color=ORANGE, dash="dash"), annotation_text="fixed", row=2, col=1)
    fig.update_xaxes(rangebreaks=rb)
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    st.table(br.format_table(br.comparison_table([res["fixed_net"], res["garch_net"], res["bh"]]))
             .loc[["Total return", "Sharpe ratio", "Max drawdown", "Number of trades", "Win rate"]])
