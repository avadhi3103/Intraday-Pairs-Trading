"""
cointegration.py — Phase 1: pair pre-screen and Engle-Granger cointegration test.

Why cointegration and not just correlation?
  Correlation measures whether two series move together. Cointegration asks
  whether a linear combination of them, A - beta*B, is stationary, meaning it
  keeps returning to a fixed mean. Pairs trading only makes money in the second
  case: we bet that a wide spread will narrow again. Two trending stocks can have
  0.95 price correlation and still drift apart permanently (a spurious
  regression). So correlation is a cheap first filter, and cointegration is
  the actual test.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BARS_PER_DAY = 75                 # 5-minute bars in an NSE session (09:15-15:30)

# Pre-screen: rolling correlation of PRICES over ~5 trading days. We demand a high
# average correlation. This is only a loose filter (see module docstring), so the
# threshold is deliberately generous.
CORR_WINDOW = 5 * BARS_PER_DAY
MIN_AVG_CORRELATION = 0.60

# Formation period used for pair selection. Backtests run on the data AFTER it
# (see data_pipeline.split_formation_trading for why).
FORMATION_DAYS = 15

SIGNIFICANCE = 0.05               # 5% significance level for all hypothesis tests

# ADF settings.
#  regression='c': the test regression includes a constant, because a regression
#                  residual has mean ~0 but we don't want to force that. No trend
#                  term: a spread with a deterministic trend isn't tradeable as
#                  mean-reverting anyway.
#  autolag='AIC':  choose the number of lagged differences by AIC. The lags soak up
#                  autocorrelation in the residual so the test statistic is valid.
ADF_REGRESSION = "c"
ADF_AUTOLAG = "AIC"

# Which p-value decides "cointegrated"?
#   A plain ADF on OLS residuals is too lenient. OLS picks beta to make the residual
#   as small (and so as stationary-looking) as possible, which pushes the ADF
#   statistic towards rejection. Engle-Granger therefore needs stricter
#   (MacKinnon) critical values. statsmodels' `coint` applies them.
#   True  -> decide using the Engle-Granger-corrected p-value (correct choice)
#   False -> decide using the plain adfuller p-value (textbook simplification)
# Both p-values are always reported so the difference is visible.
USE_ENGLE_GRANGER_PVALUE = True


# ---------------------------------------------------------------------------
# STEP 0: correlation pre-screen
# ---------------------------------------------------------------------------
def rolling_correlation(price_a: pd.Series, price_b: pd.Series, window: int = CORR_WINDOW) -> pd.Series:
    """Rolling Pearson correlation of the two price series."""
    return price_a.rolling(window).corr(price_b)


def correlation_prescreen(price_a: pd.Series, price_b: pd.Series,
                          window: int = CORR_WINDOW,
                          min_avg: float = MIN_AVG_CORRELATION) -> dict:
    rc = rolling_correlation(price_a, price_b, window).dropna()
    return {
        "avg_rolling_corr": rc.mean(),
        "min_rolling_corr": rc.min(),
        "passes_prescreen": rc.mean() >= min_avg,
    }


# ---------------------------------------------------------------------------
# STEP 1: OLS hedge ratio
# ---------------------------------------------------------------------------
def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series):
    """OLS of price_A = alpha + beta * price_B + residual.

    beta is the hedge ratio: how many shares of B offset one share of A.
    The residual is the spread we will test for stationarity.
    Returns (alpha, beta, residual series).
    """
    X = sm.add_constant(price_b.values)
    model = sm.OLS(price_a.values, X).fit()
    alpha, beta = model.params
    residual = pd.Series(model.resid, index=price_a.index, name="residual")
    return alpha, beta, residual


# ---------------------------------------------------------------------------
# STEP 2: ADF test on the residual
# ---------------------------------------------------------------------------
def adf_test(series: pd.Series, regression: str = ADF_REGRESSION,
             autolag: str = ADF_AUTOLAG, significance: float = SIGNIFICANCE) -> dict:
    """Augmented Dickey-Fuller test, same pattern as the earlier returns work.

    H0 (null):        the series has a unit root, i.e. it is NON-stationary
                      (a random walk; shocks never die out).
    H1 (alternative): the series is stationary (shocks decay; it mean-reverts).

    A more negative test statistic is stronger evidence against H0.
    If p-value < significance -> reject H0 -> treat the series as stationary.
    If p-value >= significance -> fail to reject H0. That is NOT proof of a unit
    root; it means the data don't give enough evidence of mean reversion to trade it.
    """
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(series.dropna(), regression=regression, autolag=autolag,
                                             result_object=False)
    return {
        "adf_stat": stat,
        "p_value": pvalue,
        "used_lag": usedlag,
        "n_obs": nobs,
        "crit_1%": crit["1%"],
        "crit_5%": crit["5%"],
        "reject_null": pvalue < significance,
    }


# ---------------------------------------------------------------------------
# Engle-Granger two-step test for one pair
# ---------------------------------------------------------------------------
def engle_granger(price_a: pd.Series, price_b: pd.Series, significance: float = SIGNIFICANCE) -> dict:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS gives the hedge ratio and the residual (spread).
    Step 2: ADF on the residual. If the residual is stationary, A and B are cointegrated.
    """
    alpha, beta, resid = estimate_hedge_ratio(price_a, price_b)
    adf = adf_test(resid, significance=significance)

    # Same test with Engle-Granger (MacKinnon) critical values. `coint` reruns the OLS
    # internally, so its statistic matches ours; only the p-value distribution differs.
    eg_stat, eg_pvalue, _ = coint(price_a, price_b, trend=ADF_REGRESSION, autolag=ADF_AUTOLAG)

    decision_p = eg_pvalue if USE_ENGLE_GRANGER_PVALUE else adf["p_value"]

    # Half-life of mean reversion from an AR(1) fit on the spread:
    #   d(spread_t) = lambda * spread_{t-1} + e  ->  half-life = -ln(2) / lambda.
    # It tells us roughly how many bars a typical deviation takes to halve, which
    # sanity-checks the z-score window chosen in Phase 2.
    lagged = resid.shift(1).dropna()
    delta = resid.diff().dropna()
    lam = sm.OLS(delta.values, sm.add_constant(lagged.values)).fit().params[1]
    half_life = -np.log(2) / lam if lam < 0 else np.inf

    return {
        "alpha": alpha,
        "hedge_ratio": beta,
        "adf_stat": adf["adf_stat"],
        "adf_p_value": adf["p_value"],
        "adf_crit_5%": adf["crit_5%"],
        "eg_p_value": eg_pvalue,
        "half_life_bars": half_life,
        "cointegrated": decision_p < significance,
    }


def screen_pairs(pair_frames: dict, significance: float = SIGNIFICANCE) -> pd.DataFrame:
    """Run the pre-screen and the Engle-Granger test on every pair.

    pair_frames: {"A/B": DataFrame with two price columns [A, B]}
    Returns one row per pair. A pair proceeds to Phase 2 only if it passes BOTH the
    correlation pre-screen and the cointegration test (column `proceed`).
    """
    rows = []
    for label, df in pair_frames.items():
        a, b = df.iloc[:, 0], df.iloc[:, 1]
        pre = correlation_prescreen(a, b)
        eg = engle_granger(a, b, significance)
        rows.append({"pair": label, "n_bars": len(df), **pre, **eg})

    table = pd.DataFrame(rows).set_index("pair")
    table["cointegrated"] = table["cointegrated"].map({True: "Y", False: "N"})
    table["proceed"] = table["passes_prescreen"] & (table["cointegrated"] == "Y")
    return table


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """The headline table requested for Phase 1."""
    cols = ["hedge_ratio", "adf_stat", "adf_p_value", "eg_p_value", "cointegrated",
            "avg_rolling_corr", "half_life_bars", "proceed"]
    return table[cols].round(4)


# If NO pair passes, should later phases still run on the closest candidate?
#   False -> strict: nothing proceeds (the statistically correct answer).
#   True  -> run Phases 2-4 on the pair with the lowest decision p-value, loudly
#            flagged as NOT cointegrated. Use this only to exercise the pipeline.
#            A pair that failed the test has no statistical case for mean reversion,
#            so any profit it shows should be treated as luck until proven otherwise.
ALLOW_FALLBACK_TO_BEST_PAIR = True


def select_pairs_for_trading(table: pd.DataFrame, allow_fallback: bool = ALLOW_FALLBACK_TO_BEST_PAIR):
    """Return (list of pair labels to trade, was_fallback_used)."""
    passed = table.index[table["proceed"]].tolist()
    if passed or not allow_fallback:
        return passed, False
    p_col = "eg_p_value" if USE_ENGLE_GRANGER_PVALUE else "adf_p_value"
    best = table[p_col].idxmin()
    print(f"WARNING: no pair passed cointegration at {SIGNIFICANCE:.0%}. Falling back to "
          f"'{best}' ({p_col}={table.loc[best, p_col]:.3f}) FOR DEMONSTRATION ONLY.")
    return [best], True
