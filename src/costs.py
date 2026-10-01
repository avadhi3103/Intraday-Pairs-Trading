"""
costs.py — Phase 3: transaction cost model for NSE cash equities.

Every rupee of cost is charged per FILL (one order on one leg). A pairs round trip
has 4 fills: open A, open B, close A, close B. Half are buys and half are sells,
which matters in India because some charges apply to only one side.

The statutory rates below are as published by NSE/SEBI and discount brokers as of
2024-25. They change from time to time (NSE revised its transaction charge in
Oct 2024), so check them against the current NSE circular and your broker's
charge sheet before relying on them.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CONFIG (all rates are fractions of traded notional unless stated)
# ---------------------------------------------------------------------------

# Brokerage: discount-broker style "0.03% or Rs 20 per executed order, whichever is lower".
# With Rs 5 lakh per leg, 0.03% = Rs 150, so the Rs 20 cap applies and brokerage is effectively flat.
BROKERAGE_PCT = 0.0003
BROKERAGE_CAP_PER_ORDER = 20.0          # INR

# Securities Transaction Tax (STT).
#   Intraday (MIS):   0.025% on the SELL side only.
#   Delivery (CNC):   0.1% on BOTH buy and sell.
# Our pairs trades are flattened before close, so they are intraday. That is also
# forced: retail traders can't hold a short cash-equity position overnight in India.
STT_INTRADAY_SELL = 0.00025
STT_DELIVERY_BOTH = 0.001

# NSE exchange transaction charge (cash segment), both sides.
EXCHANGE_TXN_CHARGE = 0.0000297         # 0.00297%

# SEBI turnover fee: Rs 10 per crore = 0.0001%, both sides.
SEBI_FEE = 0.000001

# Stamp duty, BUY side only.
STAMP_DUTY_INTRADAY_BUY = 0.00003       # 0.003%
STAMP_DUTY_DELIVERY_BUY = 0.00015       # 0.015%

# GST at 18%, levied on brokerage + exchange charges + SEBI fee (not on STT or stamp duty).
GST_RATE = 0.18

# Slippage, in basis points of notional, charged on EVERY fill (each leg, each side).
# Default 2 bps. Why not the 3 bps (0.03%) often used for US equities:
#   * The pairs here are NIFTY-50 large caps, among the most liquid stocks on NSE.
#     The quoted spread is usually 1 tick, and on a Rs 1,000-3,000 stock that is
#     about 0.3-1 bp. Crossing it with a market order costs half the spread, under 0.5 bp.
#   * Our order (about Rs 5 lakh per leg) is tiny next to the depth on these books,
#     so market impact is close to zero.
#   * We add a margin for adverse moves between the bar-close signal and the actual
#     fill, and for the less liquid leg of a pair (e.g. BANKBARODA vs SBIN).
# Together that gives about 2 bps per fill. This is a judgement call, not a
# measurement (yfinance has no bid/ask data), so notebook 03 shows how results
# change across a range of slippage values.
SLIPPAGE_BPS = 2.0


@dataclass
class IndianEquityCostModel:
    """Callable cost model: cost_model(notional, side) -> cost in INR for one fill.

    product: "intraday" (MIS) or "delivery" (CNC). Changes STT and stamp duty.
    """
    slippage_bps: float = SLIPPAGE_BPS
    product: str = "intraday"
    brokerage_pct: float = BROKERAGE_PCT
    brokerage_cap: float = BROKERAGE_CAP_PER_ORDER

    def breakdown(self, notional: float, side: str) -> dict:
        """Itemised cost of one fill. side is 'buy' or 'sell'."""
        notional = abs(notional)
        is_buy = side == "buy"
        intraday = self.product == "intraday"

        brokerage = min(notional * self.brokerage_pct, self.brokerage_cap)
        if intraday:
            stt = 0.0 if is_buy else notional * STT_INTRADAY_SELL
            stamp = notional * STAMP_DUTY_INTRADAY_BUY if is_buy else 0.0
        else:
            stt = notional * STT_DELIVERY_BOTH
            stamp = notional * STAMP_DUTY_DELIVERY_BUY if is_buy else 0.0
        exchange = notional * EXCHANGE_TXN_CHARGE
        sebi = notional * SEBI_FEE
        gst = GST_RATE * (brokerage + exchange + sebi)
        slippage = notional * self.slippage_bps / 10_000

        return {"brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi,
                "stamp_duty": stamp, "gst": gst, "slippage": slippage}

    def __call__(self, notional: float, side: str) -> float:
        return sum(self.breakdown(notional, side).values())

    def round_trip_bps(self, notional_per_leg: float) -> float:
        """Total cost of a full pairs round trip (4 fills), in bps of ONE leg's notional.

        Handy for a back-of-envelope check: the spread must revert by more than this,
        on average, for the strategy to make money after costs.
        """
        total = 2 * self(notional_per_leg, "buy") + 2 * self(notional_per_leg, "sell")
        return total / notional_per_leg * 10_000


def round_trip_breakdown(notional_per_leg: float, model: IndianEquityCostModel = None) -> dict:
    """Itemised cost of one pairs round trip (buy+sell on each of the two legs)."""
    model = model or IndianEquityCostModel()
    buy, sell = model.breakdown(notional_per_leg, "buy"), model.breakdown(notional_per_leg, "sell")
    return {k: 2 * (buy[k] + sell[k]) for k in buy}
