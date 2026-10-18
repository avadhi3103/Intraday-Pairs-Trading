# Q5 results vs the pre-registered predictions

Predictions: [`q5_predictions.md`](q5_predictions.md), committed and pushed (commit `0536eb0`)
before the experiment ran. Full analysis: [`notebooks/05_diagnostics.ipynb`](../notebooks/05_diagnostics.ipynb), section 5.

Same data, betas, thresholds and NSE cost model for every variant; only the treatment of the
overnight gap in the z-score window changes.

## Results (SBIN/BANKBARODA)

| | (a) as-is | (b) reset at open | (c) wait for full window |
|---|---|---|---|
| Trades | 59 | 34 | 13 |
| Gross return | +1.29% | −0.93% | +1.68% |
| Net return | −3.70% | −3.80% | +0.58% |
| Gross Sharpe | 0.26 | −3.33 | 1.35 |
| Net Sharpe | −6.17 | −7.56 | −1.28 |
| Entries in first 30 min | 51% | 0% | 0% |
| Median entry signal | 09:40 | 12:45 | 14:20 |
| Entry window (10th–90th pct) | 09:15–14:10 | 10:50–14:20 | 14:10–14:35 |
| Exits by reversion | 56% | 26% | 8% |

All three pairs: [`notebooks/05_diagnostics.ipynb`](../notebooks/05_diagnostics.ipynb). Each Sharpe has a standard error of ~2.4 (44 days).

## Scorecard

| Variant | Prediction | Result | Right? |
|---|---|---|---|
| (b) trades | fewer, 30–45 | 34 | ✓ |
| (b) entry times | mostly 10:55–12:30 | median 12:45, spread across 10:50–14:20 | ✗ later and wider |
| (b) gross return | lower, near zero or slightly negative | −0.93% | ✓ |
| (b) net return | about the same, −3% to −5% | −3.80% | ✓ |
| (b) gross Sharpe | lower, not meaningful | −3.33, about one standard error of the difference below (a) | ✓ |
| (b) mechanism | more end-of-day exits | reversion exits 56% → 26% | ✓ |
| (c) trades | far fewer, 8–15 | 13 | ✓ |
| (c) entry times | all 14:10–14:50 | 14:10–14:40 | ✓ |
| (c) gross return | lower, close to zero | **+1.68%, higher than (a)** | ✗ |
| (c) net return | higher, −1% to −1.5% | +0.58% | direction ✓, size ✗ |
| (c) Sharpe | gross lower, net less negative | gross higher, net less negative | half |
| (c) mechanism | ~all clock exits; gain purely from trading less | 92% clock exits ✓; gross also rose ✗ | half |

**Score: 7 of 12 right, 3 half right, 2 wrong.** Trade counts, timing and mechanisms were predicted
well; returns were not, which is what the stated confidence levels expected.

## The two "would surprise me" cases

1. **(b) warm-up cluster: happened, moderately.** Pooled across the three pairs, the first three
   allowed bars (10:50–11:00) hold 17 of 111 trades, against ~7 if entries were spread evenly over
   the 48 eligible bars. They are the worst group: −₹634 gross and −₹1,480 net per trade, vs −₹120 to
   −₹201 gross later in the day. That supports the small-sample explanation: a 20-bar std
   underestimates morning spread volatility, which is still elevated at 10:50 (notebook 05, section 6).
2. **(c) clearly positive gross: happened on SBIN/BANKBARODA, but it's noise.** +₹1,296 per trade,
   but one trade (07-Aug, +₹8,589) is 51% of the total, t = 1.72, and the other pairs show −₹408
   (t = −0.62) and +₹119 (t = 0.12). Pooled over 40 trades, t = 0.67. No evidence of a real
   late-afternoon reversal effect; worth re-testing on more data, not worth trading.

## Conclusion

Neither variant fixes the strategy. (b) removes the overnight-gap artifact and replaces it with
worse-timed trades. (c) trades so little that costs shrink, which flatters net return without
showing a better signal. Resetting the window also leaves the intraday volatility seasonality in
place (notebook 05, section 6).
