# QUANTARA AI — RESEARCH PHASE HISTORY

This document provides a comprehensive chronological record of the major research phases, engineering milestones, mathematical discoveries, and audit findings in the Quantara AI project.

---

## Part 1: Foundations, Data Hygiene & Initial Baselines (Phases 1–12)

### Phase 1 & 2: Forensic Data Hygiene & Baseline Construction
- **Objective:** Establish an authentic, zero-synthetic historical database for eight NSE stocks (`BHARTIARTL.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `ITC.NS`, `RELIANCE.NS`, `SBIN.NS`, `TCS.NS`) spanning 2016–2026.
- **Key Actions:** Audited raw OHLCV price series, verified date alignment, engineered 20 technical indicators, and established initial chronological train/val/test splits.
- **Finding:** Initial naive models on short horizons (1-day and 5-day) produced directional accuracy near 50%, highlighting that raw daily noise dominates short-term returns.

### Phases 3 & 4: Leakage Prevention & Audit Gate 1
- **Objective:** Verify temporal isolation between training, validation, and testing partitions.
- **Key Actions:** Implemented strict embargo purging (removing boundary-crossing samples whose forward target window overlaps future evaluation boundaries) and ensured feature scalers are fitted strictly on historical training rows.

---

## Part 2: 20-Day Horizon & The Phase 13 ML Baseline (Phases 13–15B)

### Phase 13: Advanced Multi-Horizon & Feature Group Research
- **Objective:** Extend predictive horizon to 20 trading sessions (~1 month) where macroeconomic and cyclical trends have greater signal-to-noise ratio.
- **The Core Baseline Established:** Using `group_1_trend` (15 features: returns, SMA ratios, EMA ratios, RSI, MACD, and Bollinger Bands) and `Target_Binary_20d` (UP: $>+2.0\%$, DOWN: $<-2.0\%$, HOLD excluded):
  * **Test Balanced Accuracy:** **52.63%**
  * **Test Standard Accuracy:** **47.28%**
  * **Test Macro F1-Score:** **46.32%**
  * **Confusion Matrix:** `[[218, 564], [114, 390]]` on 1,286 test observations.
- **Significance:** This 52.63% balanced accuracy became the authoritative Machine Learning baseline against which all subsequent models were compared.

### Phase 14 & 15: Profitability Research & Strategy Pitfalls
- **Objective:** Test whether directional ML predictions can generate positive portfolio returns after costs.
- **Finding:** A baseline long-only strategy entering every UP signal suffered net losses (-17.46% on the test period), driven by entering during market-wide downtrends and paying cumulative transaction costs.

---

## Part 3: Institutional Accounting, Dividends & Dynamic Allocation (Phases 16–23)

### Phase 16 & 17: Portfolio Accounting Reconciliation & Dividend Recovery
- **Objective:** Reconcile portfolio accounting, cash ledger, and corporate actions.
- **Key Actions:** Formally implemented the full Indian statutory fee schedule (0.3937% round-trip delivery friction covering STT, exchange fees, SEBI charges, stamp duty, GST, and brokerage) and extracted 124 authentic corporate dividend events to ensure accurate cash credits.

### Phase 18: Model Failure Diagnosis
- **Objective:** Understand why ML models struggle in certain regimes.
- **Finding:** Diagnosed that market-wide regime shifts (macro liquidity vs tightening) dramatically override individual stock technical patterns. When the broad market corrects, individual bullish patterns fail at high rates.

### Phase 19: High-Conviction Research
- **Objective:** Test if filtering for high model confidence (>65% probability) creates positive expectancy.
- **Finding:** While win rates improved on small subsets, trade frequency dropped drastically, and out-of-sample returns remained vulnerable to regime shifts.

### Phase 20: The Breakthrough — Dynamic Cash Allocation & Regime Strategies (Rules D, E, F)
- **Objective:** Investigate dynamic cash allocation to protect capital during hostile market regimes.
- **Key Strategies Defined:**
  * **Rule D:** Pure ML Signal Allocation.
  * **Rule E:** ML + Market Breadth Filter (20-day SMA breadth).
  * **Rule F (Authoritative Benchmark):** Dynamic Regime Allocation combining 20-day market breadth, trend filters, and volatility controls.
- **Crucial Finding:** Rule F preserved capital and generated positive risk-adjusted returns by dynamically holding up to 100% cash during unfavorable market downturns. Capital preservation proved to be the true driver of performance, not raw directional prediction accuracy.

### Phases 21–23: Independent Reproduction, Strategy Freezing & Cash Validation
- **Objective:** Independently audit and mathematically freeze Rules D, E, and F.
- **Key Actions:** Audited cash interest calculations, compared idle cash against liquid mutual funds (5.20% net annual yield), verified ledger consistency, and permanently froze Rules D, E, and F against further modification.

---

## Part 4: Deep Learning Implementation, Flaw Discovery & Correction (Phases 24–26)

### Phase 24: Deep Learning Sequential Models (LSTM & GRU)
- **Objective:** Implement 2-layer LSTM (56,162 parameters) and 2-layer GRU (42,658 parameters) using 60-day historical sequences ($L=60$) to test if recurrent memory beats linear models.
- **Initial Reported Metrics:**
  * Reused Test Set: GRU achieved **50.27%** balanced accuracy, LSTM achieved **48.64%**, while Logistic Regression scored **44.56%**.
  * Walk-Forward Folds: Reported 54%–60% accuracy in earlier years (2022–2024).
  * Anomaly Observed: The Naive Majority Baseline reported an astronomical log loss of **21.9177**.

### Phase 25: Independent Reproduction & Audit of Phase 24
- **Objective:** Verify whether Phase 24 results were reproducible and methodologically sound.
- **The Two Major Defects Discovered:**
  1. **Defect 1 (Majority Baseline Bug):** In Phase 24, hard deterministic labels `[0.0, 1.0]` were passed into `log_loss()` instead of valid probabilities. This triggered scikit-learn's extreme probability clipping penalty ($-\ln(10^{-15}) \approx 34.54$) on 782 DOWN samples, artificially inflating log loss to 21.9177. When recalculated using authentic training class priors ($P=0.6238$), the true baseline log loss was **0.7794**.
  2. **Defect 2 (Static Model Walk-Forward Flaw):** In Phase 24, the LSTM and GRU models were **NOT retrained** for each fold. They were evaluated using static weights trained through September 2024. Consequently, Folds 1–3 were **in-sample evaluations** on training data, explaining their inflated 55%–60% accuracy.
  3. **Bootstrap Flaw:** Standard i.i.d. row resampling underestimated confidence intervals by 3x by ignoring cross-stock market correlation and 20-day serial return overlap.

### Phase 26: Final Corrective Run & True Walk-Forward Retraining
- **Objective:** Fix both defects in executable code and execute true walk-forward retraining across four chronological folds (4,647 out-of-sample observations).
- **Corrective Actions Taken:**
  * Fixed baseline probabilities using empirical training class priors per fold.
  * Added automated regression tests (`REG_T1`, `REG_T2`, `REG_T3`) rejecting degenerate hard probabilities.
  * Executed true chronological walk-forward retraining: for every fold, fresh LSTM and GRU models were initialized from scratch, trained on preceding data only, validated with 20-day embargoes, and tested on unseen future periods.
- **The Definitive Scientific Verdict:**
  Across 4,647 pooled out-of-sample observations:
  * **Logistic Regression:** Balanced Accuracy = **50.07%** [46.19%, 54.25%], Macro F1 = **49.91%**, Log Loss = **0.6942**.
  * **LSTM Classifier:** Balanced Accuracy = **49.92%** [44.96%, 55.11%], Macro F1 = **46.89%**, Log Loss = **0.6983**.
  * **GRU Classifier:** Balanced Accuracy = **48.68%** [45.73%, 51.44%], Macro F1 = **39.37%**, Log Loss = **0.6992**.
  * **Paired Hypothesis Tests:** $\Delta \text{BalAcc} (\text{LSTM} - \text{LR}) = -0.15\%$ (95% CI: `[-5.10%, +5.22%]`). Both deep learning models showed **zero statistically significant advantage** over simple Logistic Regression.

