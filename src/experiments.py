"""
experiments.py — Wave 2 diagnostics and ablations on top of the Phase 2/3 engine.

  * window variants (Q5): how the z-score window treats the overnight gap
  * intraday seasonality of spread volatility (Q6)
  * Ornstein-Uhlenbeck time-to-revert, alternative exits (Q7)
  * half-life entry cutoff (Q8)
  * exit-reason split (Q9)

Everything here calls strategy.build_signals / strategy.run_backtest, so every number
comes from the same engine as the notebooks and the app.
"""

import numpy as np
import pandas as pd
from scipy import integrate, special

import backtest_report as br
import strategy as strat
from cointegration import FORMATION_DAYS, engle_granger
from costs import IndianEquityCostModel
from data_pipeline import split_formation_trading

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Q5 variants. Every variant uses the same data, betas, thresholds and costs; only the
# window treatment changes.
WINDOW_VARIANTS = {
    "(a) as-is": dict(window_mode="rolling"),
    "(b) reset at open": dict(window_mode="reset_daily"),
    "(c) wait for full window": dict(window_mode="rolling", min_bars_into_day=strat.ZSCORE_WINDOW - 1),
}

SESSION_OPEN_MIN = 9 * 60 + 15            # 09:15 in minutes after midnight
BAR_MINUTES = 5


def formation_half_life(prices: pd.DataFrame, formation_days: int = FORMATION_DAYS) -> float:
    """Phase 1 half-life (bars), estimated on the formation period only, so it's known
    before the trading period starts."""
    f, _ = split_formation_trading(prices, formation_days)
    return engle_granger(f.iloc[:, 0], f.iloc[:, 1])["half_life_bars"]


def run_pair(prices: pd.DataFrame, signal_kw: dict = None, backtest_kw: dict = None,
             cost_model=None, formation_days: int = FORMATION_DAYS):
    """Build signals for the trading period and run the backtest gross and net of costs."""
    signal_kw, backtest_kw = signal_kw or {}, backtest_kw or {}
    _, trading = split_formation_trading(prices, formation_days)
    sig = strat.build_signals(prices, trade_start=trading.index[0], **signal_kw)
    gross = strat.run_backtest(sig, **backtest_kw)
    net = strat.run_backtest(sig, cost_model=cost_model or IndianEquityCostModel(), **backtest_kw)
    return sig, gross, net


def split_kw(kw: dict):
    """Split one variant dict into build_signals and run_backtest keyword arguments."""
    sig_keys = {"window_mode", "min_bars_reset", "zscore_window", "hedge_lookback_days"}
    return ({k: v for k, v in kw.items() if k in sig_keys},
            {k: v for k, v in kw.items() if k not in sig_keys})


def minutes_after_open(ts) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return np.asarray(ts.hour * 60 + ts.minute - SESSION_OPEN_MIN)


def summarise(gross, net) -> dict:
    """One row of the comparison table: P&L, Sharpe, entry-time distribution, exit mix."""
    g, n = br.compute_metrics(gross), br.compute_metrics(net)
    t = gross.trades
    row = {"Trades": len(t), "Gross return": g["Total return"], "Net return": n["Total return"],
           "Gross Sharpe": g["Sharpe ratio"], "Net Sharpe": n["Sharpe ratio"]}
    if len(t):
        m = minutes_after_open(t["signal_time"])
        clock = pd.to_datetime(t["signal_time"]).dt.strftime("%H:%M").sort_values().reset_index(drop=True)
        pct = lambda q: clock.iloc[int(q * (len(clock) - 1))]
        row.update({
            "Entries in first 30 min": (m < 30).mean(),
            "Median entry signal": pct(0.5),
            "Entry window (10th-90th pct)": f"{pct(0.1)}-{pct(0.9)}",
            "Exits by reversion": (t["exit_reason"] == "signal").mean(),
            "Avg gross P&L / trade (INR)": t["gross_pnl"].mean(),
        })
    return row


def entry_minutes(trades: pd.DataFrame) -> np.ndarray:
    return minutes_after_open(trades["signal_time"]) if len(trades) else np.array([])


# ---------------------------------------------------------------------------
# Q6: intraday seasonality of spread volatility
# ---------------------------------------------------------------------------
def spread_change_by_bar(sig: pd.DataFrame) -> pd.DataFrame:
    """|change in spread| per bar, in bps of A's price, with each day's own beta.

    The first bar of each day (the overnight change) is kept but flagged, so it can be
    reported separately rather than mixed into the intraday profile.
    """
    a, b = sig.columns[:2]
    d_spread = (sig[a].diff() - sig["beta"] * sig[b].diff()) / sig[a].shift(1) * 1e4
    out = pd.DataFrame({"abs_change_bps": d_spread.abs(),
                        "slot": minutes_after_open(sig.index) // BAR_MINUTES,
                        "overnight": sig["bars_into_day"].values == 0})
    return out.dropna()


def z_exceedance_by_slot(sig: pd.DataFrame, threshold: float = strat.ENTRY_Z) -> pd.Series:
    """Share of bars with |z| > threshold, by bar-of-day. A well-standardised z should
    cross the threshold about equally often at every time of day."""
    s = pd.DataFrame({"hit": sig["zscore"].abs() > threshold,
                      "slot": minutes_after_open(sig.index) // BAR_MINUTES,
                      "valid": sig["zscore"].notna()})
    s = s[s["valid"]]
    return s.groupby("slot")["hit"].mean()


# ---------------------------------------------------------------------------
# Q7: Ornstein-Uhlenbeck time to revert
# ---------------------------------------------------------------------------
def ou_mean_first_passage(z0: float, z1: float = 0.0) -> float:
    """Expected time for a standardised OU process to first fall from z0 to z1 (z0 > z1),
    in units of 1/theta.

    Standardised OU: dY = -theta*Y dt + sqrt(2*theta) dW, stationary N(0,1).
    The mean first-passage time T(y) to a lower barrier z1 solves
        T'' - y T' = -1/theta,  T(z1) = 0,  T bounded as y -> infinity,
    whose solution is
        T(z0) = (1/theta) * integral_{z1}^{z0} e^{u^2/2} * integral_u^inf e^{-v^2/2} dv du
              = (1/theta) * sqrt(pi/2) * integral_{z1}^{z0} e^{u^2/2} erfc(u/sqrt2) du.
    """
    f = lambda u: np.exp(u * u / 2) * special.erfc(u / np.sqrt(2))
    val, _ = integrate.quad(f, z1, z0)
    return np.sqrt(np.pi / 2) * val


def ou_passage_in_half_lives(z0: float, z1: float = 0.0) -> float:
    """Same, in half-lives. theta = ln2 / half_life, so T / half_life = theta*T / ln2."""
    return ou_mean_first_passage(z0, z1) / np.log(2)


def deterministic_half_lives(z0: float, z1: float) -> float:
    """Without noise, E[z_t] = z0 * exp(-theta t): reaching z1 takes log2(z0/z1) half-lives
    (infinite for z1 = 0)."""
    return np.inf if z1 == 0 else np.log2(z0 / z1)


def simulate_ou_passage(z0: float, z1: float, half_life_bars: float, n_paths: int = 20_000,
                        max_bars: int = 2_000, seed: int = 0) -> np.ndarray:
    """Monte Carlo check of the formula on a discrete 5-minute grid (exact AR(1) step)."""
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1 / half_life_bars)                 # AR(1) coefficient per bar
    sd = np.sqrt(1 - phi ** 2)                        # keeps the stationary variance at 1
    y = np.full(n_paths, float(z0))                   # float, or integer z0 truncates every step
    hit = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, bool)
    for t in range(1, max_bars + 1):
        y[alive] = phi * y[alive] + sd * rng.standard_normal(alive.sum())
        done = alive & (y <= z1)
        hit[done] = t
        alive &= ~done
        if not alive.any():
            break
    return hit


# ---------------------------------------------------------------------------
# Q9: exit-reason split
# ---------------------------------------------------------------------------
def exit_reason_table(gross, net) -> pd.DataFrame:
    t = gross.trades.copy()
    t["net_pnl"] = net.trades["net_pnl"].values
    return (t.groupby("exit_reason")
             .agg(trades=("gross_pnl", "size"), mean_gross_pnl=("gross_pnl", "mean"),
                  mean_net_pnl=("net_pnl", "mean"), mean_bars_held=("bars_held", "mean")))
