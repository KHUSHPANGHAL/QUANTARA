# QUANTARA AI — PHASE 20 RESEARCH SUMMARY
## Dynamic Cash Allocation and Market-Regime Strategy Research

**Date:** October 2026  
**Auditor / Researcher:** Quantara AI Lead Research Agent  
**Status:** Completed & Empirically Audited  
**Working Directory:** `reports/phase20_regime_allocation/`  

---

### 1. Executive Summary & Core Verdict

Phase 20 investigates whether Quantara AI can improve risk-adjusted returns and prevent catastrophic drawdowns by dynamically allocating capital between equities and interest-earning cash based on strictly causal market-regime signals.

In previous phases, directional equity strategies suffered severe capital degradation during the 2025–2026 market downturn (e.g., Phase 14 baseline lost -17.46%, and Buy & Hold lost -15.60% with a 23.92% peak-to-trough drawdown). In Phase 19, the Combined Alpha strategy showed +0.58% in validation ($N=4$) but collapsed to -9.02% in the reused test set ($N=8$).

#### Core Empirical Verdict
1. **Dynamic Cash Allocation Drastically Reduces Downside Risk:**
   - Multi-Regime Tiered Allocation (**Rule D**) slashed portfolio maximum drawdown from **23.92%** (Buy & Hold) and **16.82%** (Unfiltered Baseline ML) down to **5.74%**, while ML-Plus-Regime Dynamic Alpha (**Rule F**) reduced drawdown to **6.60%**.
   - Net return in the 2025–2026 bear correction improved from **-15.60%** (Buy & Hold) and **-12.21%** (Rule A) to **-1.39%** (Rule D) and **+1.19%** (Rule E Heuristic Trend-Follower).
2. **The "Cash Hurdle" Reality:**
   - While dynamic cash allocation successfully insulated the portfolio against equity market losses, **100% Safe Cash** earning a realistic net yield of **5.2%** (liquid mutual fund equivalent after 20% capital gains tax and expense ratio) delivered **+5.09%** net return with **0.0% drawdown**.
   - **No active equity strategy beat the 5.2% net cash benchmark in the 2025–2026 correction.** In a bear market regime, holding cash is an active, superior investment decision rather than an idle drag.
3. **Regime Conditioning Protects Strong Up-Markets:**
   - During the 2024–2025 Validation period, dynamic allocation allowed the ML model to participate in equity upside: Rule A delivered **+13.01%**, Rule B/C delivered **+12.66%**, Rule D delivered **+10.42%**, and Rule F delivered **+7.10%**, all easily outperforming Buy & Hold (**+1.57%**) and Net Cash (**+5.07%**).

---

### 2. Comprehensive Strategy Comparison Matrix

The table below summarizes performance across the 4 benchmarks and 6 dynamic allocation strategies for both the **Reused Test (2025–2026)** and **Validation (2024–2025)** windows, starting with ₹1,00,000 capital.

| Strategy / Benchmark | Window | Trades | Win Rate (95% CI) | Realized Payoff | Net Expectancy | Final Equity (₹) | Net Return | Max Drawdown | Mean Eq. Exposure | Time in 100% Cash |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Buy & Hold Benchmark** | Reused Test | 0 | N/A | N/A | ₹0 | ₹84,401.47 | **-15.60%** | 23.92% | 100.0% | 0.0% |
| **1. Cash (0% Idle)** | Reused Test | 0 | N/A | N/A | ₹0 | ₹1,00,000.00 | **0.00%** | 0.00% | 0.0% | 100.0% |
| **2. Net Cash (5.2% Yield)** | Reused Test | 0 | N/A | N/A | ₹0 | ₹1,05,115.38 | **+5.09%** | 0.00% | 0.0% | 100.0% |
| **3. Gross Cash (6.5% Repo)** | Reused Test | 0 | N/A | N/A | ₹0 | ₹1,06,393.60 | **+6.37%** | 0.00% | 0.0% | 100.0% |
| **4. Rule A: Baseline LR (No Filter)** | Reused Test | 22 | 36.36% [19.7%, 57.0%] | 0.75 | -₹554.86 | ₹87,793.09 | **-12.21%** | 16.82% | 58.8% | 29.4% |
| **5. Rule B: Dynamic Breadth Switch** | Reused Test | 14 | 7.14% [1.3%, 31.5%] | 1.04 | -₹1,204.15 | ₹86,292.73 | **-13.72%** | 16.59% | 33.2% | 56.1% |
| **6. Rule C: Dual Unfavorable Switch** | Reused Test | 15 | 13.33% [3.7%, 37.9%] | 0.63 | -₹1,103.01 | ₹86,472.06 | **-13.55%** | 17.02% | 36.3% | 48.4% |
| **7. Rule D: Multi-Regime Tiered** | Reused Test | 8 | 12.50% [2.2%, 47.1%] | 0.19 | -₹676.32 | ₹98,627.40 | **-1.39%** | **5.74%** | 19.5% | 57.7% |
| **8. Rule E: Heuristic Trend (No ML)**| Reused Test | 16 | 50.00% [28.0%, 72.0%] | 0.81 | -₹123.75 | ₹1,01,212.27 | **+1.19%** | **5.53%** | 36.9% | 59.7% |
| **9. Rule F: ML-Plus-Regime Alpha** | Reused Test | 6 | 16.67% [3.0%, 56.4%] | 0.74 | -₹1,239.26 | ₹97,163.38 | **-2.86%** | **6.60%** | 7.7% | 79.4% |
| | | | | | | | | | | |
| **0. Buy & Hold Benchmark** | Validation | 0 | N/A | N/A | ₹0 | ₹1,01,570.45 | **+1.57%** | 12.59% | 100.0% | 0.0% |
| **2. Net Cash (5.2% Yield)** | Validation | 0 | N/A | N/A | ₹0 | ₹1,05,094.24 | **+5.07%** | 0.00% | 0.0% | 100.0% |
| **4. Rule A: Baseline LR (No Filter)** | Validation | 18 | 77.78% [54.8%, 91.0%] | 1.34 | +₹722.84 | ₹1,13,011.13 | **+13.01%** | 6.41% | 45.0% | 35.2% |
| **5. Rule B: Dynamic Breadth Switch** | Validation | 12 | 75.00% [46.8%, 91.1%] | 0.93 | +₹742.39 | ₹1,12,682.73 | **+12.66%** | 5.85% | 29.7% | 59.9% |
| **6. Rule C: Dual Unfavorable Switch** | Validation | 12 | 75.00% [46.8%, 91.1%] | 0.93 | +₹742.39 | ₹1,12,682.73 | **+12.66%** | 5.85% | 29.7% | 59.9% |
| **7. Rule D: Multi-Regime Tiered** | Validation | 8 | 87.50% [52.9%, 97.8%] | 0.58 | +₹776.69 | ₹1,10,443.19 | **+10.42%** | **3.21%** | 19.5% | 57.5% |
| **8. Rule E: Heuristic Trend (No ML)**| Validation | 16 | 31.25% [14.2%, 55.6%] | 0.56 | -₹634.33 | ₹92,950.90 | **-7.07%** | 11.31% | 36.1% | 58.3% |
| **9. Rule F: ML-Plus-Regime Alpha** | Validation | 3 | 66.67% [20.8%, 93.9%] | 4.27 | +₹785.43 | ₹1,07,119.54 | **+7.10%** | **1.21%** | 7.7% | 76.5% |

---

### 3. Key Findings and Insights

#### A. Drawdown Suppression is Remarkable
In an unhedged long-only portfolio, taking full equity exposure during a bear market leads to severe double-digit drawdowns.
- Buy & Hold suffered a **23.92%** peak drawdown during the 2025–2026 test period.
- Baseline ML without regime rules suffered a **16.82%** drawdown.
- By enforcing **Rule D (Multi-Regime Tiered)**, capital was throttled to 0% equity whenever breadth dropped below 40% or market broke its 50-day moving average. As a result, Rule D's max drawdown was compressed to only **5.74%** — a 76% reduction in drawdown risk.
- **Rule F (ML-Plus-Regime Dynamic Alpha)** achieved a tiny **1.21%** max drawdown in Validation and **6.60%** in Test, maintaining over 76%–79% of its portfolio in cash.

#### B. The True Cost of Bear Market Rallies (False Breakouts)
Why did Rule B and Rule C lose -13.72% and -13.55% in Test?
- Binary switching rules (allocating 100% equity when a condition is met and 0% otherwise) suffered from "whipsaw" risk.
- In 2025–2026, the market experienced frequent bear-market counter-trend rallies. Breadth temporarily ticked above 50%, triggering long entries. Almost immediately, the broader downtrend resumed, hitting trailing stops or generating losses.
- In contrast, **Rule D** succeeded because it graded exposure continuously (100% in Favorable, 50% in Uncertain, 0% in Unfavorable) and capped individual positions at 10% when uncertain, preventing whipsaws from inflicting heavy portfolio damage.

#### C. Heuristic Trend-Follower (Rule E) vs. ML Models
- In the 2025–2026 bear period, **Rule E (Heuristic Trend-Follower)** was the only active strategy to achieve a positive net nominal return (**+1.19%**, ₹1,01,212.27), winning 8 out of 16 trades (50.0% win rate).
- However, in the 2024–2025 validation period, Rule E lost **-7.07%** because sideways choppy conditions resulted in repeated failed momentum breakouts.
- This highlights the regime-dependence of classic momentum/trend-following: it thrives in sustained single-stock trend persistence, but suffers in choppy rotational markets.

#### D. Multi-Year Walk-Forward Generalization (2022–2026)
Across 5 annual walk-forward folds ($N=8,168$ total out-of-sample observations):
- **Fold 1 (2022 Inflation/Rate Shock):** Market return +1.13%, Model Balanced Acc = 34.71%, Macro F1 = 31.09%.
- **Fold 2 (2023 Post-Correction Recovery):** Market return +1.35%, Model Balanced Acc = 31.41%, Macro F1 = 24.54%.
- **Fold 3 (2024 Pre-Election Bull Run):** Market return +2.65%, Model Balanced Acc = 34.07%, Macro F1 = 31.32%.
- **Fold 4 (2024–2025 Validation Consolidation):** Market return +0.15%, Model Balanced Acc = 36.45%, Macro F1 = 35.67%.
- **Fold 5 (2025–2026 Reused Test Bear Correction):** Market return -1.46%, Model Balanced Acc = 37.77%, Macro F1 = 35.59%.

Across all 5 folds, 3-class Balanced Accuracy hovers tightly between 31.4% and 37.8% (against a random baseline of 33.33%). This proves that out-of-sample directional prediction using statistical classifiers remains modest. **The primary driver of portfolio capital preservation is not higher directional accuracy, but strict regime filtering and cash management.**

---

### 4. Methodological Compliance & Integrity Safeguards

1. **Zero Lookahead in Regime Signals:**
   Market Proxy close, SMA50, and breadth indicators were computed strictly as of session $t$ close and applied only to session $t+1$ entries.
2. **Realistic Round-Trip Transaction Costs:**
   All trades incorporated Zerodha-style Indian delivery fees (STT/CTT 0.1% buy/sell, NSE exchange turnover fees, SEBI charges, stamp duty, and ₹15.93 flat DP charges on sales), yielding an average round-trip frictional cost of ~0.3937% plus 0.05% slippage.
3. **Cash Interest Realism:**
   Idle cash was credited on an exact 365-day compounding basis. A clear distinction was maintained between:
   - Idle trading cash yielding 0.0%.
   - Net Liquid Mutual Fund equivalent yielding 5.2% annually (reflecting gross 6.5% repo minus 20% short-term capital gains tax and 25 bps TER).
   - Gross Repo Rate yielding 6.5% annually.
4. **Transparent Reused-Holdout Reporting:**
   The 2025–2026 period is openly documented as a repeatedly evaluated test dataset.

