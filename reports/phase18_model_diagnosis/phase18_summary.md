# Quantara AI — Phase 18 Executive Summary
## Model Failure Diagnosis and Predictive Signal Investigation

**Date:** October 29, 2026  
**Auditor / Engineering Lead:** Antigravity AI Team  
**Evaluation Scope:** 8 NSE Large-Cap Equities (2017 to 2026)  
**Primary Research Horizon:** 20-Day Forward Trend Classification and Backtesting  
**Final Diagnostic Verdict:** **WEAK UNDERLYING SIGNAL COMPOUNDED BY SEVERE REGIME DRIFT AND TRANSACTION COST DRAG**

---

### 1. Executive Summary & Core Mission

Phase 18 was initiated to diagnose the fundamental root causes behind Quantara AI's inability to achieve profitable long-only equity performance. Rather than adding model complexity, tuning hyperparameters, or searching for curve-fitted thresholds, Phase 18 conducted a systematic scientific autopsy of:
1. The mathematical integrity of Phase 17 accounting.
2. The prediction target formulation and potential lookahead leakage.
3. The presence (or absence) of useful predictive information against naive and heuristic baselines.
4. The marginal utility of 27 features organized into predefined economic groups.
5. Probability calibration and confidence fidelity under shifting market distributions.
6. Multi-year walk-forward generalization across 5 distinct macroeconomic cycles (2022 to 2026).
7. The mathematical link connecting directional prediction quality to trading profitability and cost sensitivity.

---

### 2. Key Findings & Diagnostic Evidence

#### Finding 1: Phase 17 Independent Verification (Zero Discrepancies)
All trade calculations, cash ledgers, commissions, dividend credits, and final equity figures reported in Phase 17 were independently reproduced from raw trades with zero errors:
- **Total Trades:** 11 completed round trips (7 wins, 4 losses, Win Rate: 63.64%).
- **Gross PnL:** $-₹233.02$ | **Commissions:** $₹1,042.74$ | **Dividends Credited:** $+₹1,053.00$.
- **Net Portfolio Change:** $-₹222.75$ | **Final Equity:** $₹99,777.25$ vs reported $₹99,777.26$ ($\Delta = 0.008\text{ INR}$).
- **Status:** **100% Mathematically Verified**.

#### Finding 2: Severe Non-Stationary Regime Shift in Prediction Target
The prediction target exhibits catastrophic distribution drift across chronological splits:
- **Train (2017–2024, N=15,256):** 47.73% UP, 23.48% HOLD, 28.79% DOWN (Mean return: $+1.84\%$). Bull regime.
- **Validation (2024–2025, N=1,808):** 32.52% UP, 36.50% HOLD, 30.97% DOWN (Mean return: $+0.12\%$). Sideways regime.
- **Test (2025–2026, N=1,816):** 27.75% UP, 29.19% HOLD, 43.06% DOWN (Mean return: $-1.58\%$). Bear regime.
Models fitted on a 7-year bull regime inevitably develop a structural bias toward upward continuation that fails when market regimes invert.

#### Finding 3: ML Models Fail to Beat Naive Baselines
In a 3-class classification problem (DOWN, HOLD, UP), pure random guessing yields a Balanced Accuracy of **33.33%**:
- **Majority-Class Baseline:** 33.33% Balanced Accuracy.
- **Simple Momentum Baseline (20d Trend):** 32.62% (Val) / 29.91% (Test).
- **Random Forest (150 Trees):** 31.71% (Val) / 34.93% (Test) — **Worse than random on validation!**
- **LightGBM (Gradient Boosted Trees):** 31.08% (Val) / 33.00% (Test) — **Worse than random on validation!**
- **Logistic Regression:** 36.54% (Val) / 36.55% (Test) — Marginal $+3.2\%$ over random guessing.
Sophisticated non-linear tree models overfit historical market noise and perform worse than simple linear or naive heuristics out-of-sample.

#### Finding 4: Feature Groups Add Virtually Zero Predictive Value
Testing feature subsets fitted strictly on training data revealed no genuine out-of-sample edge:
- **Price & Momentum (15 features):** 33.63% Val Balanced Accuracy (Random noise).
- **Volume & Liquidity (3 features):** 32.38% Val Balanced Accuracy (Sub-random).
- **Market Breadth & Macro (3 features):** 33.72% Val Balanced Accuracy (Random noise).
- **Volatility & Range (6 features):** 36.61% Val Balanced Accuracy (Slightly informative).
- **All 27 Features Combined:** 36.54% Val Balanced Accuracy for LR, 31.08% for LightGBM.
Adding 21 technical and momentum features to volatility indicators actually degraded tree-based model performance due to the curse of dimensionality.

#### Finding 5: Catastrophic Calibration Breakdown in Bear Markets
When models were calibrated on training data via Platt Scaling (Sigmoid) or Isotonic Regression:
- On Validation data (2024–2025), calibration appeared reasonable (mean predicted $P(\text{UP}) = 51.3\%$ vs. observed $52.6\%$).
- On Test data (2025–2026), Platt Scaling severely over-confidently predicted a mean $P(\text{UP})$ of **$51.92\%$** on 62 high-confidence instances, but the actual realized frequency of UP was only **$22.58\%$**!
- The calibration gap was an astonishing **$29.34\%$**! A calibration model fitted in a bull regime cannot calibrate probabilities for an unhedged bear regime.

#### Finding 6: Chronic Walk-Forward Failure Across 5 Consecutive Years
Evaluating expanding-window models across 5 distinct chronological annual folds from 2022 to 2026 proved that failure is not an isolated bad year:
- **Fold 1 (2022 Inflation Shock):** 34.71% Balanced Accuracy.
- **Fold 2 (2023 Post-Shock Recovery):** 31.41% Balanced Accuracy (Worse than majority baseline!).
- **Fold 3 (2024 Pre-Election Bull):** 34.07% Balanced Accuracy.
- **Fold 4 (2024–2025 Sideways):** 36.45% Balanced Accuracy.
- **Fold 5 (2025–2026 Bear Market):** 37.77% Balanced Accuracy.
Across half a decade of data, the model's balanced accuracy never once exceeded 38%.

#### Finding 7: Prediction Quality vs. Trading Outcomes & The Friction Trap
Connecting classification predictions to trading simulations revealed the structural flaw of thresholding:
- In the Validation window (2024–2025), threshold $0.38$ achieved an attractive win rate of $77.78\%$ and a net return of **$+14.35\%$** ($18$ trades).
- In the Reused Test window (2025–2026), the exact same threshold $0.38$ suffered a win rate collapse to **$36.36\%$** ($22$ trades) and lost **$-11.75\%$**!
- At threshold $0.42$ in 2025–2026: Gross return without friction was $+1.27\%$, but transaction fees of $₹1,042.74$ dragged net return to $-0.22\%$, and conservative friction of $0.50\%$ dragged it to $-0.84\%$.

---

### 3. Summary Performance Matrix (from Generated Deliverables)

| Evaluation Window | Model / Baseline | Balanced Acc (%) | Macro F1 (%) | Brier Score | Log Loss | Status vs Random |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation (2024–25)** | Random Guess (Theoretical) | 33.33% | 33.33% | 0.6667 | 1.0986 | Baseline Reference |
| **Validation (2024–25)** | Majority Class Baseline | 33.33% | 16.36% | 0.7056 | 1.1552 | Predicts UP always |
| **Validation (2024–25)** | Naive 1-Day Direction | 32.85% | 31.82% | 0.7820 | 1.2552 | Sub-random noise |
| **Validation (2024–25)** | 20-Day Momentum | 32.62% | 32.60% | 0.7785 | 1.2503 | Sub-random noise |
| **Validation (2024–25)** | Random Forest | 31.71% | 30.93% | 0.6701 | 1.1034 | Overfitted / Negative Edge |
| **Validation (2024–25)** | LightGBM | 31.08% | 29.30% | 0.6807 | 1.1189 | Overfitted / Negative Edge |
| **Validation (2024–25)** | **Logistic Regression** | **36.54%** | **35.74%** | **0.6642** | **1.0946** | Slight edge (+3.2% over random) |
| **Reused Test (2025–26)** | Majority Class Baseline | 33.33% | 14.48% | 0.7159 | 1.1644 | Predicts UP always |
| **Reused Test (2025–26)** | Random Forest | 34.93% | 34.76% | 0.6686 | 1.1023 | Noise level |
| **Reused Test (2025–26)** | LightGBM | 33.00% | 32.72% | 0.6714 | 1.1069 | Exactly random |
| **Reused Test (2025–26)** | **Logistic Regression** | **36.55%** | **34.80%** | **0.6664** | **1.0984** | Slight edge (+3.2% over random) |

---

### 4. Root-Cause Synthesis & Conclusion

Why does Quantara AI fail to make money?
1. **Lack of Directional Predictive Signal:** The primary bottleneck is **not** model capacity, hyperparameters, or threshold tuning. The 27 engineered technical, momentum, and volume features simply contain near-zero statistical information about 20-day forward equity returns out-of-sample (Balanced Accuracy $\approx 31\%$ to $36\%$).
2. **Unhedged Long-Only Structural Vulnerability:** The strategy is strictly long-only. When the broader market undergoes an extended correction (such as 2025–2026 where Buy & Hold lost $-15.60\%$), long-only models cannot generate positive returns without directional hedging or shorting capabilities.
3. **The Friction Barrier:** At realistic Indian delivery trading costs ($0.39\%$ to $0.50\%$ round-trip), any faint theoretical edge is completely consumed by statutory fees, exchange turnover charges, and bid-ask slippage.

**Definitive Conclusion:** The primary failure mechanism is **weak underlying signal and non-stationary regime instability**, not implementation bugs or threshold miscalibration. Real-world capital must not be deployed.

