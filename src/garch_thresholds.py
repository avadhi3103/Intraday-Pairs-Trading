"""
garch_thresholds.py — Phase 4: GARCH(1,1) volatility forecasts for the spread and
volatility-scaled z-score thresholds.

Idea
----
The z-score already divides by a rolling 60-bar std. But that std is a lagging,
equally weighted estimate: when volatility jumps, it takes many bars to catch up.
In that window a perfectly ordinary move looks like a 2-sigma "opportunity", and we
enter just as the regime turns noisy. GARCH reacts faster, because it puts heavy
weight on the most recent squared shock. So we use the GARCH forecast as a regime
indicator:

    entry_threshold_t = ENTRY_Z * (sigma_forecast_t / sigma_baseline)

  * high-vol regime (ratio > 1): demand a bigger deviation before entering
  * calm regime     (ratio < 1): accept a smaller deviation

GARCH(1,1):  r_t = mu + e_t,   e_t = sigma_t * z_t
             sigma^2_t = omega + alpha * e^2_{t-1} + beta * sigma^2_{t-1}
  alpha        = reaction to the latest shock (ARCH effect)
  beta         = memory of past variance
  alpha + beta = persistence; close to 1 means vol shocks die out slowly
ARCH(1) is the special case beta = 0 (vol depends only on the last shock). We fit
both at every refit and plot how the parameters evolve over time.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

from backtest_report import SERIES_COLORS, TEXT_COLOR, _session_axis
from strategy import ENTRY_Z, EXIT_Z

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BARS_PER_DAY = 75

# Rolling fit: before each trading day, refit on the trailing GARCH_WINDOW_DAYS days
# (strictly before that day), the same walk-forward logic as the hedge ratio.
# 15 days x ~74 returns = ~1,100 observations, plenty for 4 GARCH parameters.
GARCH_WINDOW_DAYS = 15

# Spread "returns" are scaled to basis points of stock A's price (see spread_returns).
# arch's optimiser behaves best when the data have std roughly between 1 and 1000.
# 5-minute moves are ~0.05%, i.e. ~5 bps, which fits that range.
RETURN_SCALE = 10_000

# The overnight move (15:25 close -> 09:15 next day) covers ~18 hours of news, not 5
# minutes. Left in, it shows up as a huge "shock" every morning and GARCH reads it as
# a vol spike. Since we never hold overnight, we drop it.
DROP_OVERNIGHT_RETURNS = True

MEAN_MODEL = "Constant"          # spread changes have ~zero mean; estimating mu costs little
ERROR_DIST = "normal"            # simplest choice; 't' would capture fat tails (an extension)

# Threshold scaling: ratio = forecast vol / baseline vol, where baseline = average GARCH
# conditional vol over the fit window (so it's also lookahead-free). The ratio is clipped
# so one wild bar can't push the threshold to 0.2 or 10 sigma.
SCALE_MIN, SCALE_MAX = 0.5, 2.0


# ---------------------------------------------------------------------------
# Spread returns
# ---------------------------------------------------------------------------
def spread_returns(prices: pd.DataFrame, beta: float) -> pd.Series:
    """5-minute change in the spread A - beta*B, in bps of A's previous price.

    Why first-differences and not % returns of the spread: the spread is a
    difference of prices and can sit near zero or go negative, so a % change of it
    is meaningless. The first difference dSpread_t = dA_t - beta*dB_t is exactly the
    rupee P&L of holding 1 share of A against beta shares of B. Dividing by A's price
    makes it unit-free, comparable across pairs, and well scaled for the optimiser.

    beta is held FIXED inside the function. Mixing betas from different days would add
    artificial jumps where beta changes.
    """
    a, b = prices.iloc[:, 0], prices.iloc[:, 1]
    r = (a.diff() - beta * b.diff()) / a.shift(1) * RETURN_SCALE
    if DROP_OVERNIGHT_RETURNS:
        new_day = prices.index.normalize() != pd.Series(prices.index, index=prices.index).shift(1).dt.normalize()
        r[new_day] = np.nan
    return r.dropna().rename("spread_ret_bps")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_garch(r: pd.Series, p: int = 1, q: int = 1):
    """Fit GARCH(p,q) (q=0 -> ARCH(p)) with arch. Returns the fitted result."""
    model = arch_model(r, mean=MEAN_MODEL, vol="GARCH", p=p, q=q, dist=ERROR_DIST, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # convergence chatter; we check the flag ourselves
        return model.fit(disp="off")


def garch_filter(r: np.ndarray, mu: float, omega: float, alpha: float, beta: float,
                 sigma2_init: float) -> np.ndarray:
    """Run the GARCH(1,1) variance recursion by hand with FIXED parameters.

    Returns sigma2_next[t] = forecast of the variance of r_{t+1}, made at time t
    using r_0..r_t only. That is the one-step-ahead forecast we can actually trade on.
    Writing the recursion out (instead of calling forecast() 3,000 times) keeps it fast
    and shows that the forecast is nothing more than the GARCH equation.
    """
    sigma2_next = np.empty(len(r))
    s2 = sigma2_init
    for t in range(len(r)):
        eps = r[t] - mu
        s2 = omega + alpha * eps ** 2 + beta * s2
        sigma2_next[t] = s2
    return sigma2_next


def rolling_garch(prices: pd.DataFrame, sig: pd.DataFrame,
                  window_days: int = GARCH_WINDOW_DAYS):
    """Walk-forward GARCH volatility forecasts for every bar in the trading period.

    prices: full price history (formation + trading), used for fit windows.
    sig:    output of strategy.build_signals (gives the trading days and each day's beta).

    For each trading day d:
      1. take the day's beta (estimated before d) and build spread returns
      2. fit ARCH(1) and GARCH(1,1) on the `window_days` days before d
      3. run the GARCH recursion with those fixed parameters through day d,
         giving a forecast at every bar that uses only data up to that bar
    Returns (per-bar DataFrame of forecasts, per-day DataFrame of fitted parameters).
    """
    days = prices.index.normalize().unique()
    trade_days = sig.index.normalize().unique()
    bar_rows, param_rows = [], []
    prev = None

    for d in trade_days:
        pos = days.get_loc(d)
        if pos < window_days:
            continue
        beta_d = sig.loc[sig.index.normalize() == d, "beta"].iloc[0]
        block = prices[(prices.index >= days[pos - window_days]) & (prices.index < d + pd.Timedelta(days=1))]
        r = spread_returns(block, beta_d)
        r_fit = r[r.index < d]

        g = fit_garch(r_fit, p=1, q=1)
        a1 = fit_garch(r_fit, p=1, q=0)
        converged = g.convergence_flag == 0
        if not converged and prev is not None:
            params = prev                         # keep yesterday's params if today's fit fails
        else:
            params = g.params
        prev = params

        mu, om, al, be = params["mu"], params["omega"], params["alpha[1]"], params["beta[1]"]
        s2_next = garch_filter(r.values, mu, om, al, be, sigma2_init=r_fit.var())
        fc = pd.Series(np.sqrt(s2_next), index=r.index)

        # baseline = average forecast vol over the fit window (known before day d)
        baseline = fc[fc.index < d].mean()

        today_idx = sig.index[sig.index.normalize() == d]
        # the 09:15 bar has no return (overnight dropped) -> carry the last forecast forward
        vol_today = fc.reindex(fc.index.union(today_idx)).ffill().loc[today_idx]
        bar_rows.append(pd.DataFrame({"vol_forecast": vol_today, "vol_baseline": baseline}))

        param_rows.append({
            "date": d,
            "arch_omega": a1.params["omega"], "arch_alpha": a1.params["alpha[1]"],
            "garch_omega": om, "garch_alpha": al, "garch_beta": be,
            "persistence": al + be,
            "arch_aic": a1.aic, "garch_aic": g.aic,
            "converged": converged,
        })

    vol = pd.concat(bar_rows)
    vol["vol_ratio"] = (vol["vol_forecast"] / vol["vol_baseline"]).clip(SCALE_MIN, SCALE_MAX)
    return vol, pd.DataFrame(param_rows).set_index("date")


def adaptive_thresholds(vol: pd.DataFrame, base_entry: float = ENTRY_Z, base_exit: float = EXIT_Z):
    """entry_t = base_entry * ratio_t, exit_t = base_exit * ratio_t.

    With base_exit = 0 the exit stays at the mean. Scaling only matters for a
    non-zero exit band.
    """
    return base_entry * vol["vol_ratio"], base_exit * vol["vol_ratio"]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_parameter_evolution(params: pd.DataFrame, title: str = "ARCH(1) vs GARCH(1,1) parameters, spread volatility"):
    """Parameter evolution across daily refits: ARCH vs GARCH alpha, GARCH beta, and
    persistence. Three panels, one y-scale each."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    x = params.index

    axes[0].plot(x, params["arch_alpha"], color=SERIES_COLORS[1], marker="o", ms=4, label="ARCH(1) alpha")
    axes[0].plot(x, params["garch_alpha"], color=SERIES_COLORS[0], marker="o", ms=4, label="GARCH(1,1) alpha")
    axes[0].set_ylabel("alpha (shock reaction)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)

    axes[1].plot(x, params["garch_beta"], color=SERIES_COLORS[0], marker="o", ms=4)
    axes[1].set_ylabel("GARCH beta (memory)")

    axes[2].plot(x, params["persistence"], color=SERIES_COLORS[0], marker="o", ms=4, label="GARCH alpha + beta")
    axes[2].plot(x, params["arch_alpha"], color=SERIES_COLORS[1], marker="o", ms=4, label="ARCH alpha (its persistence)")
    axes[2].axhline(1.0, color=TEXT_COLOR, lw=0.8, ls=":")
    axes[2].set_ylabel("persistence")
    axes[2].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)

    fig.suptitle(title, x=0.01, ha="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_vol_and_thresholds(vol: pd.DataFrame, base_entry: float = ENTRY_Z,
                            title: str = "GARCH vol forecast and adaptive entry threshold"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x = np.arange(len(vol))
    axes[0].plot(x, vol["vol_forecast"].values, color=SERIES_COLORS[0], lw=0.8, label="GARCH 1-step forecast")
    axes[0].plot(x, vol["vol_baseline"].values, color=TEXT_COLOR, lw=1.2, ls="--", label="baseline (fit-window avg)")
    axes[0].set_ylabel("spread vol (bps / 5m)")
    axes[0].legend(loc="upper left")
    axes[0].set_title(title, loc="left")

    axes[1].plot(x, base_entry * vol["vol_ratio"].values, color=SERIES_COLORS[0], lw=0.9, label="adaptive entry |z|")
    axes[1].axhline(base_entry, color=SERIES_COLORS[1], lw=1.2, ls="--", label="fixed entry |z|")
    axes[1].set_ylabel("entry threshold")
    axes[1].legend(loc="upper left")
    _session_axis(axes[1], vol.index)
    fig.tight_layout()
    return fig
