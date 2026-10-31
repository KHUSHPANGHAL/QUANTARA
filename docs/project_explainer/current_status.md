# QUANTARA AI — CURRENT SCIENTIFIC & OPERATIONAL STATUS

This document provides an objective, transparent assessment of what has been built, what has been independently verified, what statistical uncertainties remain, and why live trading is strictly disabled.

---

## 1. What is Completed and Verified

### Data Pipeline & Preprocessing Integrity
- **Universe:** 8 liquid Indian blue chips (`BHARTIARTL.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `ITC.NS`, `RELIANCE.NS`, `SBIN.NS`, `TCS.NS`) spanning 2016–2026.
- **Zero Synthetic Data:** 100% of price rows originate from authentic historical market data; zero imputation or synthetic price generation.
- **Leakage Controls Verified:** All scalers are fitted strictly on historical training dates. Chronological embargo periods (20 to 22 trading sessions) are strictly enforced between train, validation, and test splits to prevent forward-return overlap.
- **60-Day Lookback Sequences:** Input sequences for sequential models ($L=60$, 15 features) are constructed strictly backward-looking per stock with zero future visibility.

### Baseline Machine Learning Models
- **Phase 13 Baseline Reproduced Bit-for-Bit:**
  * Test Balanced Accuracy: **52.63%**
  * Test Macro F1: **46.32%**
  * Confusion Matrix: `[[218, 564], [114, 390]]` on 1,286 test observations.
- **Institutional Delivery Cost Model:** Validated 0.3937% statutory round-trip friction covering STT, exchange turnover fees, SEBI charges, stamp duty, GST, and brokerage, plus 0.05% slippage per leg.
- **Dividend Accounting:** 124 authentic corporate dividend actions extracted and credited to cash on ex-dates.

### Dynamic Cash Allocation & Frozen Strategies (Rules D, E, F)
- **Phase 20–23 Strategies:** Evaluated dynamic cash preservation rules based on 20-day market breadth and volatility.
- **Rule F Frozen:** Rule F demonstrated that holding up to 100% cash during unfavorable market regimes protects capital and avoids the drawdowns suffered by continuous long-only exposure. Rules D, E, and F are permanently frozen with verified file hashes.

### Deep Learning Walk-Forward Corrections (Phases 24–26)
- **Resolution of Problem 1 (Majority Baseline Bug):** Replaced hard labels with training empirical priors, reducing test log loss from an invalid **21.9177** down to **0.7794**. Automated regression tests prevent regression.
- **Resolution of Problem 2 (True Walk-Forward Retraining):** Successfully executed fresh walk-forward retraining across 4 distinct expanding folds (4,647 out-of-sample observations).
- **Authoritative DL Performance:** Proved that across 4,647 pooled out-of-sample observations, Logistic Regression (**50.07%**) slightly outperformed LSTM (**49.92%**) and GRU (**48.68%**), with paired difference confidence intervals spanning zero.

---

## 2. What Remains Uncertain & Unresolved

1. **Persistent Bullish Mode Collapse in Deep Models:**
   In multiple folds (especially Folds 2, 3, and 4), fresh LSTM and GRU networks collapsed toward predicting the historical majority class (`UP`), yielding UP recall of 79%–93% while DOWN recall collapsed to 6%–21%. When market conditions turned sharply negative, the recurrent models failed to adapt.
2. **Signal-to-Noise Ratio at 20-Day Horizons:**
   Daily technical indicators (RSI, MACD, Bollinger Bands, return lags) have limited predictive power for 20-day forward equity returns. Macroeconomic regime shifts, earnings surprises, and geopolitical events dwarf the geometric patterns of technical indicators.
3. **Probability Calibration:**
   While softmax probabilities sum to 1.0, they remain uncalibrated across shifting volatility environments, tending toward overconfidence during structural transitions.

---

## 3. Status of Master PDF Document

- **Target File:** `Quantara_AI_Full_Project_Explained.pdf`
- **Location Status:** An exhaustive scan of the repository and local drives confirmed that `Quantara_AI_Full_Project_Explained.pdf` is **not currently present on the local Windows machine**.
- **User Action Required:** If this PDF was downloaded from an external conversation or interface, it must be placed manually into:
  `C:\Project Data\Quantara_AI_Audited_Data\docs\project_explainer\Quantara_AI_Full_Project_Explained.pdf`

---

## 4. Why Live Trading Remains Strictly Disabled

> [!WARNING]
> **Live trading is completely disabled. No broker APIs are connected, and no live orders can be placed.**

The scientific rationale for maintaining strict research isolation:
1. **No Directional Alpha:** Out-of-sample directional prediction accuracy hovers near 50% (equivalent to a random coin flip).
2. **Transaction Cost Drag:** In a market with ~0.40%–0.50% round-trip delivery friction, trading without a statistically significant directional edge guarantees long-term capital erosion.
3. **Institutional Prudence:** High-risk live capital deployment requires verifiable positive expectancy across varying market regimes under conservative execution assumptions.

