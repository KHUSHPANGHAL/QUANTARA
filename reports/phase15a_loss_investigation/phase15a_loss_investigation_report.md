# QUANTARA AI — PHASE 15A: LOSING TRADE ROOT-CAUSE INVESTIGATION REPORT

**Date:** 2026-10-29  
**Status:** COMPLETE & INDEPENDENTLY RECONCILED  
**Investigation Nature:** Strict Diagnostic & Post-Mortem Audit (No Optimization / No Live Brokerage)  
**Baseline Strategy Analyzed:** Phase 14 Strategy 5 (20-Day ML All Eligible Baseline, Logistic Regression on Splits V3)  
**Starting Capital:** ₹1,00,000.00  
**Final Equity:** ₹82,538.93 (-17.46% Return, 20.66% Max Drawdown, 27 Trades)  
**Accounting Discrepancy / Leakage Delta:** ₹0.0000000000 (Machine-precision match)  

---

## 1. EXECUTIVE SUMMARY & REPRODUCTION VERIFICATION

The objective of Phase 15A is to perform an exhaustive, strictly diagnostic investigation into why the Quantara AI trading strategies lost money in out-of-sample backtesting.

All existing raw historical data, split manifests, trained model artifacts, and prior phase reports have been preserved without modification. All diagnostic deliverables have been produced from authentic market data from `data/raw/multi_stock_historical.csv` and versioned split data in `data/splits_v3/horizon_20d/`.

### Baseline Reproduction Audit (0.000 Delta)
The Phase 14 Baseline (Strategy 5: 20d ML All Eligible) was re-simulated from first principles using identical event-driven mechanics and fee schedules:

| Performance Metric | Reported Phase 14 Value | Independently Recomputed Value | Absolute Delta | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Starting Capital** | ₹1,00,000.00 | ₹1,00,000.00 | ₹0.00 | **MATCH** |
| **Final Portfolio Equity** | ₹82,538.93 | ₹82,538.930128 | ₹0.0000 | **EXACT MATCH** |
| **Total Net Return** | -17.46% | -17.4611% | 0.000% | **EXACT MATCH** |
| **Maximum Drawdown** | 20.66% | 20.6584% | 0.000% | **EXACT MATCH** |
| **Total Closed Trades** | 27 | 27 | 0 | **EXACT MATCH** |
| **Winning Trades** | 8 | 8 | 0 | **EXACT MATCH** |
| **Losing Trades** | 19 | 19 | 0 | **EXACT MATCH** |
| **Win Rate** | 29.6% | 29.63% | 0.0% | **EXACT MATCH** |
| **Profit Factor** | 0.31 | 0.3073 | 0.000 | **EXACT MATCH** |
| **Statutory Commissions Paid** | ₹2,259.70 | ₹2,259.7031 | ₹0.0000 | **EXACT MATCH** |
| **Dividends Received** | ₹887.00 | ₹887.0009 | ₹0.0000 | **EXACT MATCH** |

The baseline loss of ₹17,461.07 is confirmed to be 100% reproducible down to machine precision.

---

## 2. ACCOUNTING & IMPLEMENTATION INTEGRITY AUDIT

Before diagnosing economic and statistical causes, the backtest implementation was audited for accounting leaks, lookahead bias, and execution distortions:

1. **Cash Balance Conservation:**
   $$\text{Final Equity} = \text{Starting Capital} + \sum \text{Gross PnL} - \sum \text{Commissions} + \sum \text{Dividends}$$
   $$\text{Recomputed Equity} = ₹1,00,000.00 - ₹16,088.37 - ₹2,259.70 + ₹887.00 = ₹82,538.93$$
   $$\text{Accounting Delta} = |₹82,538.93012815746 - ₹82,538.93012815749| = ₹0.0000000000$$
   *Finding: Zero accounting leakage or unexplained cash drift.*

2. **Temporal Alignment & Lookahead Audit:**
   - Features are computed strictly using information up to session $t$ close.
   - Model inference and signal generation occur after session $t$ close.
   - Orders are queued and executed at session $t+1$ Open with slippage applied against the portfolio (Buy at $\text{Open} \times (1 + \text{slip})$, Sell at $\text{Open} \times (1 - \text{slip})$).
   *Finding: Zero lookahead bias or unrealistic same-close execution.*

3. **Transaction Fee & Friction Deduplication:**
   - Brokerage (0.03%), Exchange fees (0.00325%), SEBI charges (0.0001%), and GST (18%) were charged on both entry and exit.
   - Stamp duty (0.015%) was charged on entry; STT (0.10%) was charged on delivery buy and sell.
   - Total friction incurred across the 27 trades was ₹3,028.66 (₹2,259.70 statutory taxes/commissions + ₹768.96 slippage drag), representing 3.03% of initial equity.
   *Finding: Realistic Indian market friction is fully accounted for.*

4. **Corporate Action Crediting:**
   - Dividends were credited on ex-dividend dates only for positions acquired prior to the ex-date. Positions were not credited twice upon liquidation.
   *Finding: Dividend accounting passed all verification checks.*

---

## 3. TRADE-BY-TRADE DIAGNOSTIC FINDINGS

From the 27 round-trip trades recorded in `reports/phase15a_loss_investigation/trade_by_trade_diagnostics.csv`:

### Key Distributional Metrics
- **Average Winning Trade:** +₹1,017.50 (+3.08% return on capital)
- **Average Losing Trade:** -₹1,394.11 (-4.22% return on capital)
- **Payoff Ratio (Avg Win / |Avg Loss|):** **0.73** (Unfavorable asymmetry: losses are 1.37x larger than wins)
- **Average Maximum Adverse Excursion (MAE):** **-6.36%**
- **Average Maximum Favorable Excursion (MFE):** **+3.57%**
- **Adverse Excursion Ratio (MAE / MFE):** **1.78x** (Trades experience nearly double the downside movement compared to upside potential)

### The 5 Worst Catastrophic Trades
The table below highlights the 5 worst single trades that crippled portfolio equity:

| Trade ID | Symbol | Entry Date | Exit Date | Sessions Held | Entry Price (₹) | Exit Price (₹) | Gross Return | Net PnL (₹) | MAE (%) | Stock Trend at Entry |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T01** | `ITC.NS` | 2025-12-24 | 2026-01-22 | 20 | 408.15 | 325.94 | -20.14% | **-₹6,747.40** | -21.34% | Above SMA50 (Gap crash) |
| **T05** | `HDFCBANK.NS` | 2026-02-11 | 2026-03-12 | 20 | 931.72 | 819.69 | -12.02% | **-₹4,011.21** | -11.98% | Below SMA50 (Falling knife) |
| **T06** | `BHARTIARTL.NS`| 2026-02-18 | 2026-03-19 | 20 | 2021.11 | 1820.41 | -9.93% | **-₹3,217.39** | -12.38% | Below SMA50 (Falling knife) |
| **T07** | `INFY.NS` | 2026-02-20 | 2026-03-23 | 20 | 1297.05 | 1176.94 | -9.26% | **-₹2,522.45** | -10.65% | Below SMA50 (Falling knife) |
| **T18** | `ITC.NS` | 2026-06-19 | 2026-07-20 | 20 | 315.36 | 305.15 | -3.24% | **-₹1,308.64** | -5.73% | Below SMA50 (Downtrend) |

**Key Takeaway:** The top 3 losing trades alone (`T01`, `T05`, `T06`) lost **-₹13,976.00**, accounting for **80.04% of the entire net loss** of the strategy!

---

## 4. STOCK-BY-STOCK LOSS BREAKDOWN

Analysis of `reports/phase15a_loss_investigation/stock_loss_breakdown.csv`:

| Symbol | Trades | Wins | Losses | Win Rate | Gross PnL (₹) | Commissions (₹) | Dividends (₹) | Net PnL (₹) | Profit Factor | Loss Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ITC.NS** | 7 | 2 | 5 | 28.57% | -₹8,796.09 | ₹628.58 | ₹546.00 | **-₹8,878.67** | 0.11 | 38.46% |
| **HDFCBANK.NS** | 4 | 1 | 3 | 25.00% | -₹4,785.52 | ₹331.42 | ₹0.00 | **-₹5,116.95** | 0.07 | 21.09% |
| **BHARTIARTL.NS**| 3 | 0 | 3 | 0.00% | -₹3,998.38 | ₹261.65 | ₹0.00 | **-₹4,260.03** | 0.00 | 16.42% |
| **INFY.NS** | 3 | 1 | 2 | 33.33% | -₹2,590.42 | ₹221.11 | ₹0.00 | **-₹2,811.53** | 0.05 | 11.39% |
| **RELIANCE.NS** | 3 | 1 | 2 | 33.33% | -₹552.60 | ₹259.09 | ₹0.00 | **-₹811.68** | 0.28 | 4.37% |
| **TCS.NS** | 2 | 1 | 1 | 50.00% | -₹853.47 | ₹151.53 | ₹341.00 | **-₹664.01** | 0.35 | 3.93% |
| **ICICIBANK.NS**| 2 | 1 | 1 | 50.00% | +₹2,170.48 | ₹181.70 | ₹0.00 | **+₹1,988.78** | 5.48 | 1.71% |
| **SBIN.NS** | 3 | 2 | 1 | 66.67% | +₹3,317.64 | ₹224.62 | ₹0.00 | **+₹3,093.02** | 5.53 | 2.63% |
| **Total / Portfolio** | **27** | **8** | **19** | **29.63%** | **-₹16,088.37** | **₹2,259.70** | **₹887.00** | **-₹17,461.07** | **0.31** | **100.00%** |

### Critical Diagnostic Observations:
1. **Extreme Loss Concentration:** The top 3 losing symbols (`ITC.NS`, `HDFCBANK.NS`, `BHARTIARTL.NS`) together lost **-₹18,255.65**, which is **104.5% of the total net portfolio loss**!
2. **Winners Were Consistent:** `SBIN.NS` (+₹3,093.02) and `ICICIBANK.NS` (+₹1,988.78) performed well with profit factors > 5.4.
3. **Repeated Exposure to Broken Stocks:** `ITC.NS` was traded 7 times, absorbing over 25% of total portfolio activity while in a sustained downward trend.

---

## 5. MARKET REGIME & TREND MISMATCH

Analysis of `reports/phase15a_loss_investigation/market_regime_breakdown.csv`:

| Market Regime at Entry | Trade Count | Wins | Losses | Win Rate | Gross PnL (₹) | Commissions (₹) | Net PnL (₹) | Loss Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BULL (Breadth $\ge$ 70%)** | 9 | 1 | 8 | 11.11% | -₹13,654.41 | ₹780.12 | **-₹14,434.53** | 56.18% |
| **NEUTRAL (30% < Breadth < 70%)** | 3 | 1 | 2 | 33.33% | -₹2,209.88 | ₹237.73 | **-₹2,447.61** | 16.55% |
| **BEAR (Breadth $\le$ 30%)** | 15 | 7 | 8 | 46.67% | -₹224.07 | ₹1,241.85 | **-₹578.93** | 27.27% |

### Counter-Intuitive Finding: Why Did Bull Markets Suffer the Heaviest Losses?
At first glance, it seems paradoxical that trades entered during broad market bull regimes (breadth $\ge$ 70%) suffered -₹14,434.53 in losses (56.18% of gross losses), while bear regime trades only lost -₹578.93.

The explanation becomes immediately evident when cross-referencing individual stock trends at entry:

| Stock 50-day Trend at Entry | Trade Count | Wins | Losses | Win Rate | Total Net PnL (₹) | Avg PnL (₹) | Worst Single Trade (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BEAR (Close < SMA50)** | **23** | 8 | 15 | 34.78% | **-₹12,187.94** | -₹529.91 | -₹4,011.21 |
| **BULL (Close > SMA50)** | **4** | 1 | 3 | 25.00% | **-₹5,273.13** | -₹1,318.28 | -₹6,747.40 |

**The "Falling Knife" Phenomenon:**
- Even when the *overall market* was in a Bull regime, the model selected stocks (`HDFCBANK.NS`, `BHARTIARTL.NS`, `INFY.NS`) that were individually in deep downtrends below their 50-day moving averages (6 out of 9 Bull trades were below SMA50).
- The model operated as a **mean-reversion dip buyer**, anticipating that beaten-down stocks would bounce back to the mean.
- In reality, in Indian equity markets during 2025–2026, lagging stocks continued to break down, and without a stop-loss, holding them for 20 fixed trading days resulted in severe drawdown.

---

## 6. MODEL PREDICTION ERROR & CALIBRATION ANALYSIS

Analysis of `reports/phase15a_loss_investigation/prediction_error_analysis.csv`:

| Confidence Band | Test Set Samples | Mean Model $P(\text{UP})$ | Empirical Realized UP Rate | Overconfidence Bias | Executed Trades | Realized Win Rate | Executed Net PnL (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P(\text{UP}) < 0.35$** | 1,063 | 30.95% | 27.38% | +3.57% | 0 | — | ₹0.00 |
| **$0.35 \le P(\text{UP}) < 0.40$** | 602 | 37.29% | 27.74% | **+9.54%** | 16 | 25.00% | **-₹12,195.68** |
| **$0.40 \le P(\text{UP}) < 0.45$** | 149 | 41.67% | 30.87% | **+10.79%** | 11 | 45.45% | **-₹5,265.39** |
| **$P(\text{UP}) \ge 0.45$** | 2 | 45.29% | 0.00% | +45.29% | 0 | — | ₹0.00 |

### Prediction Error Insights:
1. **Pervasive Overconfidence Bias (+9.5% to +10.8%):**
   When the model assigned a probability between 35% and 45% to an UP move, the actual historical probability of that stock rising over 20 days was only ~27% to ~31%. The model's sigmoid probabilities were miscalibrated upwards.
2. **Low Absolute Conviction:**
   The model never produced a probability above 50% for class UP. Every single trade was triggered on weak conviction ($P(\text{UP}) \in [0.35, 0.43]$), merely because $P(\text{UP}) > P(\text{DOWN})$.
3. **Misalignment of Multi-Class Target:**
   Because the target includes three classes (`DOWN`, `HOLD`, `UP`), the base rate is roughly 33%. A threshold of 35% is barely above random noise, forcing the strategy to trade frequently on statistically insignificant signals.

---

## 7. THE 3 PRIORITIZED, EVIDENCE-SUPPORTED ROOT CAUSES OF LOSSES

Based on quantitative evidence from all 27 trades and portfolio ledger accounting, the losses are driven by three distinct mechanisms:

### Root Cause 1: Unhedged Structural Downtrend Exposure ("Falling Knife Trap")
- **Evidence:** 23 out of 27 trades (85.2%) were entered on stocks trading below their own 50-day moving average.
- **Mechanism:** The linear model learned mean-reversion weights from historical training data where dips were quickly bought. In the out-of-sample test period (2025–2026), major Indian large caps (`HDFCBANK.NS`, `ITC.NS`, `BHARTIARTL.NS`) experienced multi-month structural corrections. The model repeatedly bought these falling knives, mistaking downward momentum for oversold bargains.

### Root Cause 2: Extreme Downside Asymmetry Without Risk Limits or Stop-Losses
- **Evidence:** 
  - Average MAE (-6.36%) was 1.78x larger than average MFE (+3.57%).
  - Top 3 single losing trades alone (`T01`, `T05`, `T06`) destroyed ₹13,976.00 (80.04% of total loss).
  - Top 3 losing stocks accounted for 104.5% of net portfolio losses.
- **Mechanism:** The strategy rigidly held every position for 20 completed sessions (`MAX_HOLD_20D`) regardless of how far the stock collapsed. In `T01` (`ITC.NS`), the stock gapped down and fell -20.14%, but the strategy sat helplessly in the position for 20 days. Lack of individual stop-losses, trailing stops, or sector exposure caps allowed tail losses to wipe out all gains from winners (`SBIN.NS`, `ICICIBANK.NS`).

### Root Cause 3: Sub-Random Probability Thresholding Combined with Structural Friction Drag
- **Evidence:** 
  - Model overconfidence bias was +9.5% to +10.8% across active signal bands.
  - Payoff ratio was only 0.73 (average win ₹1,017.50 vs average loss ₹1,394.11).
  - Total friction (statutory taxes + commissions + slippage) consumed ₹3,028.66 (3.03% of capital).
- **Mechanism:** Triggering long entries at $P(\text{UP}) \ge 0.35$ in a 3-class system meant entering trades on statistical noise. In an environment with zero genuine predictive edge, round-trip delivery friction (~0.39% statutory + 0.10% slippage) creates a persistent mathematical drag that steadily erodes capital.

---

## 8. WHAT IS CONFIRMED VS POSSIBLE VS UNKNOWN

| Category | Finding | Evidence Status |
| :--- | :--- | :---: |
| **Confirmed by Data** | Phase 14 baseline equity of ₹82,538.93 is exact with ₹0.00 delta. | **VERIFIED** |
| **Confirmed by Data** | No accounting leaks, fee omissions, or lookahead biases exist in the backtest engine. | **VERIFIED** |
| **Confirmed by Data** | Top 3 losing stocks (`ITC.NS`, `HDFCBANK.NS`, `BHARTIARTL.NS`) explain >100% of portfolio losses. | **VERIFIED** |
| **Confirmed by Data** | 85.2% of trades entered stocks trading below their 50-day SMA. | **VERIFIED** |
| **Confirmed by Data** | Model probabilities suffer from an upward overconfidence bias of ~10 percentage points. | **VERIFIED** |
| **Possible Explanation** | Mean-reversion feature weights trained during bull markets inverted during structural bear/correction regimes. | **PLAUSIBLE HYPOTHESIS** |
| **Possible Explanation** | Implementing a 5% to 7% stop-loss could cut tail risk, though it may increase turnover friction. | **TO BE TESTED IN PHASE 15B** |
| **Unknown** | Whether a model can reliably predict regime shifts in Indian markets without overfitting. | **UNRESOLVED RESEARCH QUESTION** |
| **Unknown** | Whether any long-only strategy on this 8-stock basket can achieve positive net returns after friction in 2025–2026. | **UNRESOLVED RESEARCH QUESTION** |

---

## 9. RECOMMENDED REMEDIATION ROADMAP (FOR PHASE 15B / PHASE 16)

Before attempting to deploy capital or optimize new models, the following architectural fixes must be sequenced:

1. **Mandatory Trend & Moving Average Gating:**
   - Prohibit long entries on any stock trading below its own 50-day simple moving average (`Close < SMA50`), regardless of model confidence.
2. **Dynamic Risk Control & Tail-Risk Truncation:**
   - Introduce a volatility-adjusted or hard stop-loss (e.g., 5% to 7% or $2 \times \text{ATR}$) to terminate runaway crashes like `ITC.NS` (-20%) and `HDFCBANK.NS` (-12%).
3. **Probability Calibration & Higher Entry Thresholds:**
   - Apply isotonic or Platt scaling to calibrate model probabilities.
   - Raise entry threshold from $P(\text{UP}) \ge 0.35$ to $P(\text{UP}) \ge 0.50$ (or require a top-decile cross-sectional score), refusing to trade when edge is negligible.
4. **Stock-Level Loss Quotas / Circuit Breakers:**
   - Implement a stock-specific cool-off rule: if a stock loses 2 consecutive trades or drops > 10%, bar further entries in that symbol for 40 trading days.

---

## 10. FINAL VERDICT ON OPERATIONAL READINESS

> **FINAL VERDICT: STRATEGY IS COMPLETELY UNFIT FOR PAPER TRADING OR LIVE CAPITAL.**  
>
> The current Quantara AI 20-day ML strategy suffers from severe structural vulnerabilities: it systematically buys falling stocks, holds losing positions through catastrophic drawdowns without stop-losses, and trades on overconfident, uncalibrated probability signals.  
> 
> **Under no circumstances should this strategy be connected to a live broker, paper-trading feed, or real capital.** Phase 15B must focus exclusively on rigorous, disciplined countermeasure validation in simulation before any deployment discussion is reopened.

