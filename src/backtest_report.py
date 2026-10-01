"""
backtest_report.py — shared metrics, comparison tables and plots for every phase.

All phases report through these functions, so numbers are always computed the same
way and tables from Phases 2, 3 and 4 can be compared directly.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
BAR_MINUTES = 5

# Risk-free rate for Sharpe / excess return: roughly the 91-day Indian T-bill yield.
# An Indian strategy should be judged against the Indian risk-free rate, not 0% or a US rate.
RISK_FREE_RATE_ANNUAL = 0.065

# Fixed colour per series role, so "fixed threshold" is the same colour in every plot.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"

plt.rcParams.update({
    "figure.figsize": (11, 4),
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def daily_returns(equity: pd.Series, capital: float) -> pd.Series:
    """End-of-day equity -> daily % returns.

    Why daily and not per-bar for Sharpe: the book is flat overnight, so per-bar
    returns are a mix of intraday bars and zero-return gaps, and 5-minute returns are
    autocorrelated (bid-ask bounce). Annualising them with sqrt(75*252) overstates
    the Sharpe. Daily returns are the standard, conservative choice.
    """
    eod = equity.groupby(equity.index.normalize()).last()
    prev = eod.shift(1).fillna(capital)
    return eod / prev - 1


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall in equity, as a negative fraction."""
    return (equity / equity.cummax() - 1).min()


def compute_metrics(result, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> dict:
    eq, cap, trades = result.equity, result.capital, result.trades
    rets = daily_returns(eq, cap)
    n_days = len(rets)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = rets - rf_daily

    total_ret = eq.iloc[-1] / cap - 1
    ann_ret = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = excess.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if rets.std() > 0 else np.nan

    n_trades = len(trades)
    has_trades = n_trades > 0 and "net_pnl" in trades
    return {
        "Total return": total_ret,
        "Annualised return": ann_ret,
        "Annualised excess return": ann_ret - rf_annual,
        "Annualised volatility": ann_vol,
        "Sharpe ratio": sharpe,
        "Max drawdown": max_drawdown(eq),
        "Number of trades": n_trades,
        "Win rate": (trades["net_pnl"] > 0).mean() if has_trades else np.nan,
        "Avg holding (bars)": trades["bars_held"].mean() if has_trades else np.nan,
        "Avg holding (minutes)": trades["bars_held"].mean() * BAR_MINUTES if has_trades else np.nan,
        "Avg net P&L / trade (INR)": trades["net_pnl"].mean() if has_trades else np.nan,
        "Total costs (INR)": trades["costs"].sum() if has_trades else 0.0,
        "Trading days": n_days,
    }


PCT_ROWS = ["Total return", "Annualised return", "Annualised excess return",
            "Annualised volatility", "Max drawdown", "Win rate", "Excess-return degradation (ann.)"]


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format a metrics table (rows = metrics) for display."""
    out = df.astype(object).copy()
    for row in out.index:
        for col in out.columns:
            v = df.loc[row, col]
            if isinstance(v, str) or pd.isna(v):
                out.loc[row, col] = v if isinstance(v, str) else "-"
            elif row in PCT_ROWS or "%" in row:
                out.loc[row, col] = f"{v:.2%}"
            elif "INR" in row:
                out.loc[row, col] = f"{v:,.0f}"
            elif row in ("Number of trades", "Trading days"):
                out.loc[row, col] = f"{int(v)}"
            else:
                out.loc[row, col] = f"{v:.2f}"
    return out


def comparison_table(results: list, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> pd.DataFrame:
    """Side-by-side metrics, one column per strategy variant (raw numbers)."""
    return pd.DataFrame({r.label: compute_metrics(r, rf_annual) for r in results})


def cost_comparison_table(gross, net, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> pd.DataFrame:
    """Before/after-cost comparison, same structure as the single-asset version:
    total return before/after, cost impact, and excess-return degradation.
    """
    g, n = compute_metrics(gross, rf_annual), compute_metrics(net, rf_annual)
    gross_pnl = gross.equity.iloc[-1] - gross.capital
    total_costs = n["Total costs (INR)"]
    rows = ["Total return", "Annualised return", "Annualised excess return", "Sharpe ratio",
            "Max drawdown", "Win rate", "Number of trades", "Avg net P&L / trade (INR)"]
    df = pd.DataFrame({"Before costs": [g[r] for r in rows], "After costs": [n[r] for r in rows]}, index=rows)
    df["Change"] = df["After costs"] - df["Before costs"]
    extra = pd.DataFrame({
        "Before costs": [0.0, np.nan, np.nan],
        "After costs": [total_costs,
                        total_costs / abs(gross_pnl) if gross_pnl != 0 else np.nan,
                        n["Annualised excess return"] - g["Annualised excess return"]],
        "Change": [np.nan, np.nan, np.nan],
    }, index=["Total costs (INR)", "Cost impact (% of gross P&L)", "Excess-return degradation (ann.)"])
    return pd.concat([df, extra])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _session_axis(ax, index):
    """Intraday data has big overnight/weekend gaps. Plotting against bar number
    (not wall-clock time) removes the flat gaps; day boundaries are labelled instead."""
    days = index.normalize()
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    step = max(1, len(starts) // 8)
    ax.set_xticks(starts[::step])
    ax.set_xticklabels([index[i].strftime("%d-%b") for i in starts[::step]])
    ax.set_xlim(0, len(index) - 1)


def plot_equity(results: list, title: str = "Equity curve"):
    fig, ax = plt.subplots()
    for i, r in enumerate(results):
        ax.plot(np.arange(len(r.equity)), r.equity.values, color=SERIES_COLORS[i % 4], label=r.label)
    ax.axhline(results[0].capital, color=TEXT_COLOR, lw=0.8, ls=":")
    _session_axis(ax, results[0].equity.index)
    ax.set_ylabel("Equity (INR)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(title, loc="left")
    if len(results) > 1:
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(results))
    fig.tight_layout()
    return fig


def plot_drawdown(results: list, title: str = "Drawdown"):
    fig, ax = plt.subplots(figsize=(11, 3))
    for i, r in enumerate(results):
        dd = r.equity / r.equity.cummax() - 1
        c = SERIES_COLORS[i % 4]
        ax.plot(np.arange(len(dd)), dd.values * 100, color=c, label=r.label, lw=1.2)
        if len(results) == 1:
            ax.fill_between(np.arange(len(dd)), dd.values * 100, 0, color=c, alpha=0.15)
    _session_axis(ax, results[0].equity.index)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title, loc="left")
    if len(results) > 1:
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(results))
    fig.tight_layout()
    return fig


def plot_zscore_signals(result, title: str = "Z-score, thresholds and positions"):
    """Z-score with entry/exit thresholds; shading shows when a position is on."""
    b = result.bars
    x = np.arange(len(b))
    fig, ax = plt.subplots()
    ax.plot(x, b["zscore"].values, color=SERIES_COLORS[0], lw=0.8, label="z-score")
    ax.plot(x, b["entry_threshold"].values, color=TEXT_COLOR, lw=1, ls="--", label="entry band")
    ax.plot(x, -b["entry_threshold"].values, color=TEXT_COLOR, lw=1, ls="--")
    ax.axhline(0, color=TEXT_COLOR, lw=0.6)
    ax.fill_between(x, -6, 6, where=b["position"].values > 0, color=SERIES_COLORS[2], alpha=0.15,
                    label="long spread", step="mid")
    ax.fill_between(x, -6, 6, where=b["position"].values < 0, color=SERIES_COLORS[1], alpha=0.15,
                    label="short spread", step="mid")
    ax.set_ylim(-5, 5)
    _session_axis(ax, b.index)
    ax.set_ylabel("z")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    return fig


def plot_trade_pnl(result, title: str = "Net P&L per trade"):
    t = result.trades
    fig, ax = plt.subplots(figsize=(11, 3))
    colors = np.where(t["net_pnl"] > 0, SERIES_COLORS[2], SERIES_COLORS[1])
    ax.bar(np.arange(len(t)), t["net_pnl"].values, color=colors, width=0.8)
    ax.axhline(0, color=TEXT_COLOR, lw=0.6)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("INR")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig
