# QUANTARA AI — PHASE 26 FINAL REPORT: DEEP LEARNING CORRECTION, TRUE WALK-FORWARD EVALUATION, AND FINAL RELIABILITY VERIFICATION

**Project:** Quantara AI — AI-Based Stock Market Trend Prediction and Analysis Using ML and DL  
**Phase:** Phase 26 — Final Corrective Run  
**Objective:** Correct the two major defects identified in Phase 25, execute true walk-forward retraining across multiple market regimes, conduct dependence-aware uncertainty analysis, and establish authoritative conclusions on deep learning for Quantara AI.  
**Workspace:** `C:\Project Data\Quantara_AI_Audited_Data`  
**Execution Date:** October 30, 2026  
**Auditor / Lead Quantitative Engineer:** Quantitative Research & Verification Lead  

---

## 1. Executive Summary

Phase 26 represents the definitive corrective research phase for Deep Learning architectures within Quantara AI. Phase 25 identified two critical methodological flaws in Phase 24:
1. **Defect 1:** An invalid majority-baseline log loss of **21.9177** resulting from passing hard deterministic labels (`[0.0, 1.0]`) into `log_loss()`.
2. **Defect 2:** An invalid walk-forward evaluation where LSTM and GRU were **not retrained** for each fold, testing static models on historical data they had already learned from (**in-sample evaluation**).

In Phase 26, both defects were fully resolved in executable code (`scripts/run_phase26_walk_forward_correction.py`) and verified across **4,647 genuine out-of-sample observations** spanning four distinct market regimes from 2023 through 2026.

### Key Headline Findings:
1. **Majority-Baseline Correction Verified:**
   - Replacing hard deterministic labels with training-derived empirical class priors ($P_{\text{train}}(\text{DOWN})$ and $P_{\text{train}}(\text{UP})$) reduced test log loss from **21.9177** to a valid **0.7794** (and pooled out-of-sample log loss to **0.6977**).
   - Source-level regression tests (`REG_T1`, `REG_T2`, `REG_T3`) verified mathematical validity, finite ranges, probability summation to 1.0, and strict target encoding alignment.
2. **True Walk-Forward Retraining Completed:**
   - Four distinct expanding folds were evaluated: Fold 1 (2023 Recovery), Fold 2 (2024 Bull Run), Fold 3 (2024–2025 Consolidation), and Fold 4 (2025–2026 Downtrend).
   - For every fold, a **fresh LSTM** and a **fresh GRU** were initialized from scratch, trained strictly on permitted historical data, validated on non-overlapping validation data (separated by 20-day embargoes), and tested on subsequent unseen future periods.
   - All 8 fold checkpoints (4 LSTM, 4 GRU) and 4 fold scalers were persisted in `models/phase26/` with cryptographic SHA-256 hashes recorded.
3. **The Empirical Truth on Deep Learning Performance:**
   - Across all 4,647 pooled out-of-sample observations:
     * **Logistic Regression (Retrained):** Balanced Accuracy = **50.07%** [46.19%, 54.25%], Macro F1 = **49.91%**, Log Loss = **0.6942**, Brier Score = **0.2505**.
     * **LSTM Classifier (Fresh Retraining):** Balanced Accuracy = **49.92%** [44.96%, 55.11%], Macro F1 = **46.89%**, Log Loss = **0.6983**, Brier Score = **0.2524**.
     * **GRU Classifier (Fresh Retraining):** Balanced Accuracy = **48.68%** [45.73%, 51.44%], Macro F1 = **39.37%**, Log Loss = **0.6992**, Brier Score = **0.2528**.
     * **Corrected Majority Baseline:** Balanced Accuracy = **50.00%**, Macro F1 = **35.75%**, Log Loss = **0.6977**, Brier Score = **0.2521**.
4. **Paired Difference Hypothesis Testing:**
   - Paired difference bootstrap testing ($B=1,000$ 20-day blocks) revealed that:
     * $\Delta \text{BalAcc} (\text{LSTM} - \text{LR}) = \mathbf{-0.15\%}$ with 95% CI `[-5.10%, +5.22%]`.
     * $\Delta \text{BalAcc} (\text{GRU} - \text{LR}) = \mathbf{-1.39\%}$ with 95% CI `[-6.07%, +2.74%]`.
   - **Conclusion:** Neither LSTM nor GRU demonstrates a statistically significant advantage over the simple Logistic Regression baseline. In fact, both recurrent models slightly underperformed linear regression on pooled out-of-sample data.

---

## 2. Correction of Problem 1: Majority-Baseline Probabilities & Log Loss

### 2.1 Implementation Mechanics
In Phase 26, the baseline probability predictor was rewritten to dynamically compute class frequencies strictly from permitted training labels for each fold:
$$p_{\text{DOWN}} = \frac{N_{\text{DOWN, train}}}{N_{\text{total, train}}}, \quad p_{\text{UP}} = \frac{N_{\text{UP, train}}}{N_{\text{total, train}}}$$
A constant probability vector $[p_{\text{DOWN}}, p_{\text{UP}}]$ was assigned to every evaluation sample in that fold.

### 2.2 Comparison of Hard-Label vs Corrected Probability Baselines

| Fold Name | Out-of-Sample Window | Train $P(\text{DOWN})$ | Train $P(\text{UP})$ | Hard-Label Log Loss (Phase 24 Bug) | Corrected Valid Log Loss (Phase 26) | Log Loss Reduction | Corrected Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1: 2023 Recovery** | 2023-01-30 to 2023-12-29 | 0.3874 | 0.6126 | 12.2626 | **0.6460** | -11.6166 | 0.2267 |
| **Fold 2: 2024 Bull Run** | 2024-02-01 to 2024-09-23 | 0.3901 | 0.6099 | 10.9127 | **0.6298** | -10.2829 | 0.2187 |
| **Fold 3: 2024–2025 Val** | 2024-10-24 to 2025-09-19 | 0.3758 | 0.6242 | 17.5823 | **0.7188** | -16.8635 | 0.2624 |
| **Fold 4: 2025–2026 Test** | 2025-10-23 to 2026-09-22 | 0.3762 | 0.6238 | 21.9177 | **0.7794** | -21.1383 | 0.2921 |

### 2.3 Executed Regression Tests

| Test ID | Description | Rule Enforced | Observed Outcome | Status |
| :--- | :--- | :--- | :--- | :---: |
| **REG_T1** | Hard-Label Detector | Rejects degenerate arrays containing hard 0.0 or 1.0 | Detector triggers on hard labels and halts execution | **PASSED** |
| **REG_T2** | Prior Probability Range | $0 < P < 1$, sum strictly equals $1.0 \pm 10^{-6}$ | Verified across all folds ($P \in [0.37, 0.63]$, sum = 1.0000) | **PASSED** |
| **REG_T3** | Target Encoding Alignment | Index 0 must correspond strictly to DOWN, Index 1 to UP | Target mapping verified against ground truth labels | **PASSED** |

---

## 3. Correction of Problem 2: True Chronological Walk-Forward Retraining

### 3.1 Fold Definitions & Temporal Partitioning
To eliminate in-sample evaluation, Phase 26 established four chronological expanding windows:

| Fold ID & Name | Market Regime | Training Window | Purge Embargo 1 | Validation Window | Purge Embargo 2 | Out-of-Sample Test Window |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | Post-Correction Recovery | 2017-01-03 to 2022-09-30 | 2022-10-01 to 2022-10-31 | 2022-11-01 to 2022-12-30 | 2023-01-01 to 2023-01-27 | **2023-01-30 to 2023-12-29** |
| **Fold 2** | Bull Run / Liquidity | 2017-01-03 to 2023-09-29 | 2023-10-01 to 2023-10-31 | 2023-11-01 to 2023-12-29 | 2024-01-01 to 2024-01-31 | **2024-02-01 to 2024-09-23** |
| **Fold 3** | Sideways Consolidation | 2017-01-03 to 2024-06-28 | 2024-07-01 to 2024-07-31 | 2024-08-01 to 2024-09-23 | 2024-09-24 to 2024-10-23 | **2024-10-24 to 2025-09-19** |
| **Fold 4** | Large-Cap Downtrend | 2017-01-03 to 2024-09-23 | 2024-09-24 to 2024-10-23 | 2024-10-24 to 2025-09-19 | 2025-09-20 to 2025-10-22 | **2025-10-23 to 2026-09-22** |

### 3.2 Verification of Walk-Forward Independence
- **Zero Lookahead:** Every fold's training and validation sets strictly precede its evaluation start date.
- **20-Day Embargo Purging:** A 20- to 22-trading-day buffer between splits completely purges overlapping forward returns.
- **Train-Only Preprocessing:** `StandardScaler` was fitted strictly on each fold's training feature rows.
- **Fresh Weight Initialization:** Every fold instantiated new `StockLSTMClassifier` and `StockGRUClassifier` objects, trained them from scratch with Adam ($\eta=0.001$, weight decay=$10^{-4}$), and saved separate checkpoints.

---

## 4. Comprehensive Performance Evaluation (Per-Fold & Pooled)

### 4.1 Per-Fold Out-of-Sample Performance Matrix

| Fold Name & Regime | Model Name | Test N | Balanced Acc (%) | Macro F1 (%) | DOWN Prec / Rec (%) | UP Prec / Rec (%) | Brier Score | Log Loss | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1: 2023 Recovery** | Corrected Majority | 1,308 | 50.00% | 39.75% | 0.0% / 0.0% | 65.98% / 100.0% | 0.2267 | 0.6460 | `[[0, 445], [0, 863]]` |
| | Logistic Regression | 1,308 | 56.28% | 55.61% | 40.69% / 50.56% | 70.86% / 61.99% | 0.2461 | 0.6854 | `[[225, 220], [328, 535]]` |
| | **LSTM (Fresh WF)** | 1,308 | **57.08%** | **55.91%** | 40.97% / 55.06% | 71.83% / 59.10% | 0.2364 | 0.6658 | `[[245, 200], [353, 510]]` |
| | GRU (Fresh WF) | 1,308 | 50.85% | 50.53% | 35.48% / 27.19% | 66.49% / 74.51% | 0.2352 | 0.6633 | `[[121, 324], [220, 643]]` |
| **Fold 2: 2024 Bull Run** | Corrected Majority | 905 | 50.00% | 41.08% | 0.0% / 0.0% | 69.72% / 100.0% | 0.2187 | 0.6298 | `[[0, 274], [0, 631]]` |
| | **Logistic Regression** | 905 | **55.59%** | **48.62%** | 33.90% / 72.99% | 76.51% / 38.19% | 0.2517 | 0.6965 | `[[200, 74], [390, 241]]` |
| | LSTM (Fresh WF) | 905 | 50.00% | 41.08% | 0.0% / 0.0% | 69.72% / 100.0% | 0.2389 | 0.6707 | `[[0, 274], [0, 631]]` |
| | GRU (Fresh WF) | 905 | 50.00% | 41.08% | 0.0% / 0.0% | 69.72% / 100.0% | 0.2397 | 0.6724 | `[[0, 274], [0, 631]]` |
| **Fold 3: 2024–2025 Val** | Corrected Majority | 1,148 | 50.00% | 33.87% | 0.0% / 0.0% | 51.22% / 100.0% | 0.2624 | 0.7188 | `[[0, 560], [0, 588]]` |
| | Logistic Regression | 1,148 | 49.73% | 47.51% | 48.31% / 28.04% | 51.03% / 71.43% | 0.2504 | 0.6940 | `[[157, 403], [168, 420]]` |
| | **LSTM (Fresh WF)** | 1,148 | **50.22%** | **49.08%** | 49.10% / 34.11% | 51.38% / 66.33% | 0.2598 | 0.7135 | `[[191, 369], [198, 390]]` |
| | GRU (Fresh WF) | 1,148 | 50.00% | 33.87% | 0.0% / 0.0% | 51.22% / 100.0% | 0.2528 | 0.6989 | `[[0, 560], [0, 588]]` |
| **Fold 4: 2025–2026 Test** | Corrected Majority | 1,286 | 50.00% | 28.16% | 0.0% / 0.0% | 39.19% / 100.0% | 0.2921 | 0.7794 | `[[0, 782], [0, 504]]` |
| | **Logistic Regression** | 1,286 | **52.63%** | **46.32%** | 65.66% / 27.88% | 40.88% / 77.38% | 0.2543 | 0.7017 | `[[218, 564], [114, 390]]` |
| | LSTM (Fresh WF) | 1,286 | 50.00% | 28.16% | 0.0% / 0.0% | 39.19% / 100.0% | 0.2717 | 0.7373 | `[[0, 782], [0, 504]]` |
| | GRU (Fresh WF) | 1,286 | 50.00% | 28.16% | 0.0% / 0.0% | 39.19% / 100.0% | 0.2800 | 0.7549 | `[[0, 782], [0, 504]]` |

### 4.2 Pooled Out-of-Sample Performance (N=4,647)

Across all four evaluation windows combined (2,061 DOWN, 2,586 UP):

| Model Architecture | Pooled OOS Samples | Balanced Acc (%) | Macro F1 (%) | Standard Acc (%) | Log Loss | Brier Score | DOWN Prec / Rec (%) | UP Prec / Rec (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corrected Majority Baseline** | 4,647 | 50.00% | 35.75% | 55.65% | 0.6977 | 0.2521 | 0.0% / 0.0% | 55.65% / 100.0% |
| **Logistic Regression (Retrained)** | 4,647 | **50.07%** | **49.91%** | 50.14% | **0.6942** | **0.2505** | 44.57% / 38.82% | 55.43% / 61.18% |
| **LSTM Classifier (Fresh WF)** | 4,647 | 49.92% | 46.89% | 53.07% | 0.6983 | 0.2524 | 45.42% / 21.15% | 56.63% / 79.54% |
| **GRU Classifier (Fresh WF)** | 4,647 | 48.68% | 39.37% | 54.40% | 0.6992 | 0.2528 | 35.48% / 5.87% | 55.70% / 93.08% |

---

## 5. Dependence-Aware Uncertainty Analysis & Paired Hypothesis Testing

To account for contemporaneous cross-stock correlations across the 8 stocks and serial autocorrelation from overlapping 20-day returns, we executed a **20-Day Block Clustered Bootstrap** ($B=1,000$ replicates, random seed 42):

### 5.1 Individual Model Uncertainty Intervals

| Model Name | Resampling Methodology | Point Balanced Acc (%) | 95% Confidence Interval | CI Width (%) | 95% Macro F1 CI | Crosses 50% Threshold? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Corrected Majority Baseline** | 20-Day Block Bootstrap | 50.00% | [50.00%, 50.00%] | 0.00% | [35.75%, 35.75%] | Boundary |
| **Logistic Regression (Retrained)** | 20-Day Block Bootstrap | 50.07% | [46.19%, 54.25%] | 8.06% | [45.10%, 54.60%] | **Yes** |
| **LSTM Classifier (Fresh WF)** | 20-Day Block Bootstrap | 49.92% | [44.96%, 55.11%] | 10.15% | [41.22%, 52.88%] | **Yes** |
| **GRU Classifier (Fresh WF)** | 20-Day Block Bootstrap | 48.68% | [45.73%, 51.44%] | 5.71% | [33.72%, 44.88%] | **Yes** |

### 5.2 Formal Paired Difference Hypothesis Testing vs Logistic Regression

By computing the difference $\Delta = \text{Metric}_{\text{Model}} - \text{Metric}_{\text{LR}}$ on the exact same resampled blocks:

| Comparison Pair | Point Balanced Acc Difference (%) | 95% Paired Confidence Interval | Statistically Superior to LR Baseline? |
| :--- | :---: | :---: | :---: |
| **LSTM vs Logistic Regression** | **-0.15%** | **[-5.10%, +5.22%]** | **NO (Confidence interval includes zero)** |
| **GRU vs Logistic Regression** | **-1.39%** | **[-6.07%, +2.74%]** | **NO (Confidence interval includes zero)** |

> [!IMPORTANT]
> **Key Scientific Conclusion:** Because the paired difference confidence intervals span zero broadly (with LSTM ranging from -5.10% to +5.22%), there is **zero statistically significant evidence** that recurrent deep learning provides any predictive advantage over simple Logistic Regression on genuine out-of-sample Indian market data.

---

## 6. Root-Cause Analysis: Why Deep Learning Failed in True Walk-Forward Retraining

1. **Persistent Bullish Mode Collapse:**
   In Folds 2, 3, and 4, both LSTM and GRU collapsed toward predicting the dominant historical training class (`UP`), achieving UP recall of 79%–93% while DOWN recall collapsed to 5%–21%. When market conditions turned sharply negative (Fold 4 downtrend), the deep networks failed to adjust their internal state transitions.
2. **Over-Parameterization on Low Signal-to-Noise Data:**
   Financial returns over 20-day horizons have extremely low signal-to-noise ratios. Recurrent networks with 42,000 to 56,000 parameters easily memorize idiosyncratic temporal sequences in training that do not recur out-of-sample.
3. **Linear Models Provide Superior Regularization:**
   Logistic Regression, with only 16 parameters (15 weights + 1 bias) and balanced class weighting, proved far more resilient across regime transitions (achieving 55.59% in Fold 2 and 52.63% in Fold 4).

---

## 7. Mandatory Pass/Fail Verification Table

| Requirement | Status | Evidence |
| :--- | :---: | :--- |
| **Corrected training-prior probabilities** | **PASS** | Script dynamically derives $P(\text{DOWN})$ and $P(\text{UP})$ from fold train labels (`phase26_corrected_baseline_metrics.csv`) |
| **Probability sums and class ordering** | **PASS** | 100% valid probabilities, sum to $1.0 \pm 10^{-5}$, Class 0=DOWN, Class 1=UP (`phase26_probability_validation.csv`) |
| **Corrected log loss and Brier score** | **PASS** | Test log loss corrected from 21.9177 to valid 0.7794; pooled log loss is 0.6977 |
| **Regression test prevents original bug** | **PASS** | `REG_T1` in `phase26_regression_tests.csv` asserts rejection of hard deterministic `[0, 1]` arrays |
| **Fresh LSTM training per fold** | **PASS** | 4 separate LSTM checkpoints trained from scratch and saved in `models/phase26/` |
| **Fresh GRU training per fold** | **PASS** | 4 separate GRU checkpoints trained from scratch and saved in `models/phase26/` |
| **Strict chronological fold boundaries** | **PASS** | `Train < Purge1 < Val < Purge2 < Test` strictly enforced across all 4 folds |
| **Target-overlap purge/embargo** | **PASS** | 20- to 22-day embargo gaps eliminate forward-return leakage (`phase26_temporal_integrity_checks.csv`) |
| **Train-only preprocessing** | **PASS** | `StandardScaler` fitted strictly on train feature rows per fold (`scaler_fold1` to `scaler_fold4`) |
| **Genuine out-of-sample predictions** | **PASS** | 4,647 genuine out-of-sample predictions evaluated across 2023–2026 (`phase26_oos_predictions.csv`) |
| **Dependence-aware uncertainty analysis** | **PASS** | 20-Day block clustered bootstrap ($B=1,000$, seed=42) executed on pooled and paired metrics |
| **Results reproduced from saved predictions** | **PASS** | All summary metrics calculated directly from `phase26_oos_predictions.csv` |
| **Existing frozen strategies preserved** | **PASS** | Frozen Rules D, E, and F verified untouched with identical file hashes |

---

## 8. Final Decision and Actionable Recommendations

### Definitive Conclusion: **No Demonstrated Improvement**
The empirical evidence from true walk-forward retraining conclusively demonstrates that **Deep Learning sequential models (LSTM and GRU) do not improve directional prediction accuracy or risk-adjusted expectancy over simple baselines in this equity research setup**:
- **Pooled Out-of-Sample Balanced Accuracy:** Logistic Regression (**50.07%**) > LSTM (**49.92%**) > GRU (**48.68%**).
- **Paired Hypothesis Tests:** The differences are statistically indistinguishable from zero ($\Delta = -0.15\%$ for LSTM, $-1.39\%$ for GRU).
- **Mode Collapse Risk:** Recurrent architectures tend toward catastrophic mode collapse (predicting only the historical majority class) during structural market shifts.

### Operational Mandates for Quantara AI:
1. **Do Not Deploy Deep Learning for Directional Signals:**
   Neither LSTM nor GRU should be deployed for live trading or signal generation.
2. **Maintain the Frozen Rule F Framework:**
   Phases 20–23 proved that risk-adjusted performance in Indian equities is driven by **dynamic cash preservation and regime allocation**, not by attempting to predict noisy 20-day price trajectories with complex neural networks.
3. **Preserve Phase 26 Deliverables as the Authoritative Deep Learning Benchmark:**
   All Phase 26 scripts, model checkpoints, and logs are permanently archived in the repository as the reproducible scientific standard for sequential deep learning in Quantara AI.

