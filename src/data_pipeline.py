"""
data_pipeline.py — fetch, clean and align intraday bars for a pair of NSE stocks.

DATA LIMITATION (read this before trusting any result in this project)
----------------------------------------------------------------------
This project uses yfinance intraday data as a free stand-in for proper
exchange tick/bar data. Consequences:
  * History is capped: 5-minute bars go back only ~60 calendar days,
    1-minute bars only ~30 days (and only 7 days per request). That is
    roughly 40 trading days, far too short for strong statistical claims.
  * Bars are built by Yahoo, not NSE. There can be missing bars, the
    09:15 bar often has volume 0, and prices are last-trade closes, not
    bid/ask. We can't see the spread we'd actually pay.
  * The 60-day window rolls forward every day, so re-downloading gives a
    different sample. We cache raw data to parquet and reuse it, so a set
    of results can be reproduced exactly from /data.
A production version would swap `fetch_ticker` for a paid NSE data feed;
nothing downstream would need to change.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Candidate pairs, matched by sector so that they share a common economic driver
# (that shared driver is what could make the spread mean-revert).
#   * HDFCBANK / ICICIBANK: the two largest private-sector banks
#   * TCS / INFY: the two largest IT services exporters (both exposed to USD/INR and US tech spend)
#   * SBIN / BANKBARODA: the two largest public-sector banks.
#     PLACEHOLDER: swap BANKBARODA for the SBIN-adjacent name you choose.
CANDIDATE_PAIRS = [
    ("HDFCBANK.NS", "ICICIBANK.NS"),
    ("TCS.NS", "INFY.NS"),
    ("SBIN.NS", "BANKBARODA.NS"),
]

# 5-minute bars give ~60 days of history from yfinance. 1-minute bars give only ~30 days,
# so 5m is the better trade-off between sample length and resolution.
BAR_INTERVAL = "5m"
HISTORY_PERIOD = "60d"          # the maximum yfinance allows for 5m bars

# NSE continuous session is 09:15 to 15:30 IST. A 5m bar is stamped with its start time,
# so the last regular bar starts at 15:25.
SESSION_START = "09:15"
SESSION_END = "15:25"
MARKET_TZ = "Asia/Kolkata"

PRICE_COLUMN = "Close"          # we trade on bar closes (see strategy.py for the execution lag)


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker.replace('.', '_')}_{BAR_INTERVAL}.parquet"


def fetch_ticker(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Download OHLCV bars for one ticker and cache them as parquet.

    If a cached file exists and refresh=False, the cache is used, so results are
    reproducible even though Yahoo's 60-day window keeps rolling forward.
    """
    path = raw_path(ticker)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        interval=BAR_INTERVAL,
        auto_adjust=False,          # no splits/dividends to adjust within a 60-day intraday window
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# CLEAN
# ---------------------------------------------------------------------------
def clean_bars(df: pd.DataFrame) -> pd.Series:
    """Return a clean close-price series for one ticker.

    Steps:
      1. Make sure timestamps are in IST, so session filters line up with NSE hours.
      2. Keep only regular-session bars (drops any pre-open or stray post-close prints).
      3. Drop duplicate timestamps and non-positive or missing prices.
    """
    s = df[PRICE_COLUMN].copy()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s.index = s.index.tz_convert(MARKET_TZ)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.between_time(SESSION_START, SESSION_END)
    s = s[s > 0].dropna()
    return s


# ---------------------------------------------------------------------------
# ALIGN
# ---------------------------------------------------------------------------
def align_pair(price_a: pd.Series, price_b: pd.Series, name_a: str, name_b: str) -> pd.DataFrame:
    """Inner-join two price series on timestamp.

    We use an INNER join and do NOT forward-fill. Forward-filling a missing bar would
    show a stale price for one leg, which creates a fake spread move and a fake
    trading signal. Dropping the bar is the honest choice; it costs us only a few
    bars out of thousands.
    """
    df = pd.concat({name_a: price_a, name_b: price_b}, axis=1, join="inner").dropna()
    df.index.name = "timestamp"
    return df


def load_pair(ticker_a: str, ticker_b: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch, clean and align a pair. Saves the aligned frame to data/processed.

    Returned columns are the ticker names without the '.NS' suffix, e.g. ['HDFCBANK', 'ICICIBANK'].
    """
    name_a, name_b = ticker_a.replace(".NS", ""), ticker_b.replace(".NS", "")
    a = clean_bars(fetch_ticker(ticker_a, refresh))
    b = clean_bars(fetch_ticker(ticker_b, refresh))
    pair = align_pair(a, b, name_a, name_b)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pair.to_parquet(PROCESSED_DIR / f"{name_a}_{name_b}_{BAR_INTERVAL}.parquet")
    return pair


def split_formation_trading(pair: pd.DataFrame, formation_days: int):
    """Split the sample into a formation period (first `formation_days` trading days)
    and a trading period (everything after).

    Why: if we chose pairs by testing cointegration on the WHOLE sample and then
    backtested on that same sample, the backtest would benefit from knowing in
    advance which pairs were going to mean-revert. That is selection lookahead bias.
    Phase 1 therefore uses only the formation period, and every backtest number
    is reported on the trading period that comes after it.
    """
    days = pair.index.normalize().unique()
    if formation_days >= len(days):
        raise ValueError(f"formation_days={formation_days} but only {len(days)} days of data")
    cutoff = days[formation_days]
    return pair[pair.index < cutoff], pair[pair.index >= cutoff]


def load_universe(pairs=CANDIDATE_PAIRS, refresh: bool = False) -> dict:
    """Load every candidate pair. Returns {"A/B": aligned price DataFrame}."""
    out = {}
    for ticker_a, ticker_b in pairs:
        df = load_pair(ticker_a, ticker_b, refresh)
        out[f"{df.columns[0]}/{df.columns[1]}"] = df
    return out


if __name__ == "__main__":
    for a, b in CANDIDATE_PAIRS:
        df = load_pair(a, b)
        n_days = df.index.normalize().nunique()
        print(f"{a}/{b}: {len(df)} aligned bars over {n_days} days "
              f"({df.index[0]} -> {df.index[-1]})")
