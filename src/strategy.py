"""
strategy.py — Phase 2: walk-forward z-score pairs strategy and backtest engine.

How we avoid lookahead bias
---------------------------
1. Hedge ratio: re-estimated once per trading day by OLS on the
   HEDGE_LOOKBACK_DAYS days BEFORE that day. The beta used on day d never sees a
   price from day d or later. This is the rolling re-estimation idea from rolling
   ARIMA: refit on a trailing window, then forecast forward.
2. Z-score: the rolling mean/std at bar t use bars t-W+1 .. t only (the bar-t close
   is known at the moment we compute the signal).
3. Execution: a signal computed on the close of bar t is filled at the close of bar
   t+EXECUTION_LAG_BARS. We never trade at the same price that generated the signal.
4. Pair selection: pairs are chosen in a formation period, and the backtest runs only
   on the trading period after it (see data_pipeline.split_formation_trading).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 60                 # bars (60 x 5m = 5 hours, i.e. most of one session)

ENTRY_Z = 2.0                      # enter when |z| > 2: about a 2-sigma deviation
EXIT_Z = 0.0                       # exit when z crosses back through +/- EXIT_Z (0 = full reversion to mean)

HEDGE_LOOKBACK_DAYS = 10           # trailing days used to estimate beta each morning
# Beta is refit once per day, before the open. Refitting more often would make the
# spread definition move during the day, while trades opened under the old beta are
# still on.

CAPITAL = 1_000_000                # INR starting equity

# Position sizing: a fixed rupee notional on each leg. This is the "risk-per-trade" amount.
# Long leg = +NOTIONAL_PER_LEG, short leg = -NOTIONAL_PER_LEG, so gross exposure is
# 2 x NOTIONAL_PER_LEG = 1x capital (no leverage) and net exposure is ~0.
NOTIONAL_PER_LEG = 500_000

# "dollar_neutral": equal and opposite rupee notional in A and B. The book has no net
#     market exposure, so a broad market move (e.g. the whole banking sector rallying)
#     roughly cancels and the P&L comes mostly from the RELATIVE move we bet on. This is
#     the sizing the project spec asks for.
# "hedge_ratio": shares_B = beta * shares_A. This matches the spread definition
#     A - beta*B exactly, so P&L tracks the traded spread exactly, but net rupee
#     exposure is non-zero whenever beta != P_A / P_B.
# The two coincide only when the OLS intercept is ~0. That trade-off is worth knowing.
SIZING_MODE = "dollar_neutral"

EXECUTION_LAG_BARS = 1             # fill on the NEXT bar's close after the signal bar

# Intraday rules. NSE cash market: retail can't carry a short overnight, and brokers
# auto-square-off intraday (MIS) positions around 15:20. So we stop opening trades
# late in the session and flatten everything before the close.
# The flatten signal is at 15:05 (fill 15:10), earlier than strictly necessary, because
# yfinance often drops the last 2-3 bars of the session (on ~45% of days the data ends at
# 15:10 or 15:15). An exit scheduled later than that would often never fire.
NO_NEW_ENTRY_AFTER = "14:50"
EOD_EXIT_TIME = "15:05"            # signal to flatten at 15:05; with a 1-bar lag the fill is 15:10

BARS_PER_DAY = 75


# ---------------------------------------------------------------------------
# Walk-forward hedge ratio, spread and z-score
# ---------------------------------------------------------------------------
def ols_beta(price_a: pd.Series, price_b: pd.Series) -> float:
    """OLS slope of A on B (with intercept), same regression as Phase 1."""
    return sm.OLS(price_a.values, sm.add_constant(price_b.values)).fit().params[1]


def build_signals(prices: pd.DataFrame, trade_start=None,
                  zscore_window: int = ZSCORE_WINDOW,
                  hedge_lookback_days: int = HEDGE_LOOKBACK_DAYS) -> pd.DataFrame:
    """Compute beta_t, spread_t and z_t for every bar in the trading period.

    prices: two columns [A, B] covering formation + trading periods. The formation
            data is used only as warm-up history for the first betas and z-scores.
    trade_start: first timestamp of the trading period (default: first day that
                 has a full hedge lookback behind it).

    For each trading day d:
      beta_d   = OLS slope on the previous `hedge_lookback_days` days (strictly before d)
      spread_t = A_t - beta_d * B_t
      z_t      = (spread_t - mean(spread over last W bars)) / std(spread over last W bars)
    The rolling window for the first bars of day d reaches back into previous days.
    We recompute those earlier spreads with beta_d so that every bar in the window is
    measured in the same units.
    """
    a_col, b_col = prices.columns[:2]
    days = prices.index.normalize().unique()
    if trade_start is None:
        trade_start = days[hedge_lookback_days]
    trade_days = days[days >= pd.Timestamp(trade_start).normalize()]

    blocks = []
    for d in trade_days:
        day_pos = days.get_loc(d)
        if day_pos < hedge_lookback_days:
            continue
        hist = prices[(prices.index >= days[day_pos - hedge_lookback_days]) & (prices.index < d)]
        beta = ols_beta(hist[a_col], hist[b_col])

        today = prices[prices.index.normalize() == d]
        # warm-up: the last (W-1) bars before today, so the first bar of today has a full window
        warm = prices[prices.index < d].iloc[-(zscore_window - 1):]
        window_px = pd.concat([warm, today])
        spread = window_px[a_col] - beta * window_px[b_col]
        mean = spread.rolling(zscore_window).mean()
        std = spread.rolling(zscore_window).std()
        z = (spread - mean) / std

        block = today.copy()
        block["beta"] = beta
        block["spread"] = spread.loc[today.index]
        block["spread_mean"] = mean.loc[today.index]
        block["spread_std"] = std.loc[today.index]
        block["zscore"] = z.loc[today.index]
        blocks.append(block)

    return pd.concat(blocks)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    label: str
    equity: pd.Series                      # mark-to-market equity at every bar
    trades: pd.DataFrame                   # one row per round trip
    bars: pd.DataFrame                     # per-bar position, z-score, thresholds
    capital: float = CAPITAL
    params: dict = field(default_factory=dict)


def _as_series(x, index) -> pd.Series:
    """Allow thresholds to be a constant (Phase 2) or a time-varying series (Phase 4)."""
    if np.isscalar(x):
        return pd.Series(float(x), index=index)
    return x.reindex(index)


def run_backtest(sig: pd.DataFrame,
                 entry_threshold=ENTRY_Z,
                 exit_threshold=EXIT_Z,
                 cost_model=None,
                 capital: float = CAPITAL,
                 notional_per_leg: float = NOTIONAL_PER_LEG,
                 sizing_mode: str = SIZING_MODE,
                 execution_lag: int = EXECUTION_LAG_BARS,
                 label: str = "z-score") -> BacktestResult:
    """Bar-by-bar event loop. Written as an explicit loop (not vectorised) so every
    decision can be read in order: mark-to-market -> fill pending order -> new signal.

    sig: output of build_signals (columns A, B, beta, zscore).
    cost_model: None (gross) or a callable(notional, side) -> INR, e.g. costs.IndianEquityCostModel().
    Signal rules (pos = +1 is long spread = long A / short B):
      flat  and z < -entry -> go long spread   (spread unusually LOW, bet it rises)
      flat  and z > +entry -> go short spread  (spread unusually HIGH, bet it falls)
      long  and z >= -exit -> close            (spread has reverted)
      short and z <= +exit -> close
    """
    a_col, b_col = sig.columns[:2]
    idx = sig.index
    pa, pb = sig[a_col].values, sig[b_col].values
    z = sig["zscore"].values
    beta = sig["beta"].values
    entry = _as_series(entry_threshold, idx).values
    exit_ = _as_series(exit_threshold, idx).values

    times = idx.strftime("%H:%M")
    dates = idx.normalize()
    last_bar_of_day = np.r_[dates[1:] != dates[:-1], True]

    equity = np.empty(len(idx))
    position = np.zeros(len(idx), dtype=int)
    eq = capital
    pos, target = 0, 0
    sh_a = sh_b = 0                              # signed share holdings
    pending_lag = 0                              # bars until the pending order may fill
    pending_reason = ""
    trades, open_trade = [], None

    def fill(i, new_pos, reason):
        """Close any open position and/or open a new one at bar i's close prices."""
        nonlocal pos, sh_a, sh_b, eq, open_trade
        cost = 0.0
        if pos != 0:                             # close existing legs
            for shares, px in ((sh_a, pa[i]), (sh_b, pb[i])):
                if cost_model is not None:
                    cost += cost_model(abs(shares) * px, "sell" if shares > 0 else "buy")
            open_trade.update(exit_time=idx[i], exit_a=pa[i], exit_b=pb[i], exit_reason=reason,
                              bars_held=i - open_trade["entry_idx"])
            open_trade["costs"] += cost
            trades.append(open_trade)
            open_trade, sh_a, sh_b = None, 0, 0
        if new_pos != 0:                         # open new legs
            n_a = int(notional_per_leg // pa[i])                     # whole shares only
            if sizing_mode == "dollar_neutral":
                n_b = int(notional_per_leg // pb[i])                 # equal rupee notional
            else:
                n_b = int(round(beta[i] * n_a))                      # beta shares of B per share of A
            sh_a, sh_b = new_pos * n_a, -new_pos * n_b
            open_cost = 0.0
            if cost_model is not None:
                open_cost += cost_model(n_a * pa[i], "buy" if sh_a > 0 else "sell")
                open_cost += cost_model(n_b * pb[i], "buy" if sh_b > 0 else "sell")
            cost += open_cost
            open_trade = dict(entry_time=idx[i], entry_idx=i, direction=new_pos,
                              shares_a=sh_a, shares_b=sh_b, entry_a=pa[i], entry_b=pb[i],
                              entry_z=z[i - execution_lag] if i >= execution_lag else z[i],
                              costs=open_cost)
        eq -= cost
        pos = new_pos

    for i in range(len(idx)):
        # 1) mark to market: P&L from holding the shares over (t-1, t]
        if i > 0 and pos != 0:
            eq += sh_a * (pa[i] - pa[i - 1]) + sh_b * (pb[i] - pb[i - 1])

        # 2) fill the order decided `execution_lag` bars ago
        if target != pos:
            pending_lag -= 1
            if pending_lag <= 0:
                fill(i, target, pending_reason)

        # 3) safety net: if the day's data ends before the scheduled exit fills (missing
        #    bars), close at the last available bar. Never carry a position overnight.
        if last_bar_of_day[i]:
            if pos != 0:
                fill(i, 0, "session_end")
            target = 0
        else:
            # 4) decide the target position using information up to bar i only
            new_target, reason = pos, "signal"
            if np.isnan(z[i]):
                pass
            elif times[i] >= EOD_EXIT_TIME:
                new_target, reason = 0, "eod"
            elif pos == 0 and times[i] < NO_NEW_ENTRY_AFTER:
                if z[i] < -entry[i]:
                    new_target = 1
                elif z[i] > entry[i]:
                    new_target = -1
            elif pos == 1 and z[i] >= -exit_[i]:
                new_target = 0
            elif pos == -1 and z[i] <= exit_[i]:
                new_target = 0
            if new_target != target:
                target = new_target
                pending_reason = reason
                pending_lag = execution_lag
                if execution_lag == 0:
                    fill(i, target, pending_reason)

        equity[i] = eq
        position[i] = pos

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["gross_pnl"] = (trades_df["shares_a"] * (trades_df["exit_a"] - trades_df["entry_a"])
                                  + trades_df["shares_b"] * (trades_df["exit_b"] - trades_df["entry_b"]))
        trades_df["net_pnl"] = trades_df["gross_pnl"] - trades_df["costs"]
        trades_df = trades_df.drop(columns="entry_idx")

    bars = pd.DataFrame({"position": position, "zscore": z,
                         "entry_threshold": entry, "exit_threshold": exit_}, index=idx)
    return BacktestResult(label=label, equity=pd.Series(equity, index=idx, name="equity"),
                          trades=trades_df, bars=bars, capital=capital,
                          params=dict(sizing_mode=sizing_mode, notional_per_leg=notional_per_leg,
                                      execution_lag=execution_lag, costs=cost_model is not None))


# ---------------------------------------------------------------------------
# Benchmark: buy-and-hold the pair
# ---------------------------------------------------------------------------
def buy_and_hold(prices: pd.DataFrame, capital: float = CAPITAL, cost_model=None,
                 label: str = "buy & hold") -> BacktestResult:
    """Put half the capital into each stock at the first bar and hold to the end.

    This is the "just own the stocks" alternative. It carries full market risk, which
    the dollar-neutral pairs book is designed to remove, so the comparison shows what
    we gave up (or avoided) by hedging. If a cost model is passed, entry and exit are
    charged at DELIVERY rates, since the position is held overnight.
    """
    a_col, b_col = prices.columns[:2]
    sh_a = (capital / 2) // prices[a_col].iloc[0]
    sh_b = (capital / 2) // prices[b_col].iloc[0]
    cash = capital - sh_a * prices[a_col].iloc[0] - sh_b * prices[b_col].iloc[0]
    equity = cash + sh_a * prices[a_col] + sh_b * prices[b_col]

    costs = 0.0
    if cost_model is not None:
        from costs import IndianEquityCostModel
        delivery = IndianEquityCostModel(slippage_bps=getattr(cost_model, "slippage_bps", 0.0),
                                         product="delivery")
        for sh, col in ((sh_a, a_col), (sh_b, b_col)):
            costs += delivery(sh * prices[col].iloc[0], "buy") + delivery(sh * prices[col].iloc[-1], "sell")
        equity = equity - costs                  # charged up front for simplicity

    trades = pd.DataFrame([dict(entry_time=prices.index[0], exit_time=prices.index[-1], direction=1,
                                bars_held=len(prices) - 1, costs=costs,
                                gross_pnl=equity.iloc[-1] + costs - capital,
                                net_pnl=equity.iloc[-1] - capital)])
    bars = pd.DataFrame({"position": 1}, index=prices.index)
    return BacktestResult(label=label, equity=equity.rename("equity"), trades=trades,
                          bars=bars, capital=capital, params=dict(costs=cost_model is not None))
