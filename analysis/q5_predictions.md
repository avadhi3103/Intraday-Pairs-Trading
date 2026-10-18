# Q5 predictions, written before running (date: 18 Oct 2026)
# Author: Help with Claude (AI assistant)

Baseline (a), already known, on SBIN/BANKBARODA: 59 trades, gross +1.29%, net −3.70%,
gross Sharpe 0.26, about half of all entries in the first 30 minutes.

## (b) z-score window resets at the open
- Trades vs (a):        fewer, roughly 30–45
- Entry times:          most entries late morning (about 10:55–12:30), because the z-score
                        doesn't exist until 20 bars in, and the first values come from a short,
                        noisy window that crosses ±2 easily
- Gross return vs (a):  lower, probably near zero or slightly negative
- Net return vs (a):    about the same (somewhere around −3% to −5%)
- Sharpe vs (a):        lower (gross), but the difference won't be meaningful given a
                        standard error of ~2.4
- Reasoning:            The reset removes the 09:15 gap trades, which were the only group on
                        SBIN that made money after costs (+₹1,033 gross on average). The trades
                        that replace them start around 11:00, with about 49 bars to the 15:05
                        flatten, which is roughly one 46-bar half-life. So I expect many more
                        end-of-day exits. Fewer trades cuts costs, which roughly offsets the
                        lost gross edge on net return.
- Confidence:           medium on trade count and timing, low on returns

## (c) window unchanged, but no entries until all 60 bars are from today
- Trades vs (a):        far fewer, roughly 8–15
- Entry times:          all between about 14:10 and 14:50, because 60 of today's bars only
                        exist from about 14:10 and new entries stop at 14:50
- Gross return vs (a):  lower, close to zero
- Net return vs (a):    higher (less negative), roughly −1% to −1.5%
- Sharpe vs (a):        gross lower, net less negative, and the difference won't be
                        meaningful given a standard error of ~2.4
- Reasoning:            Every trade has at most ~11 bars before the 15:05 flatten, against a
                        46-bar half-life, so nearly all should exit on the clock rather than by
                        reversion. The better net number will come purely from trading less
                        (~₹847 cost per round trip), not from a better signal.
- Confidence:           high on trade count, timing and exit reason; low on returns

## What would surprise me
- If (b) puts a big cluster of entries right at its first allowed bar (~10:55) and those
  trades lose money. That would mean a 20-bar warm-up is too short: the early standard
  deviation is underestimated, and the reset just swaps the overnight-gap artifact for a
  small-sample artifact.
- If (c) shows a clearly positive gross return despite almost every trade being closed by
  the clock. That would mean late-afternoon deviations reverse quickly for some other
  reason, not the mean reversion I've been assuming.
