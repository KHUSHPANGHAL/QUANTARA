# QUANTARA AI — PHASE 21 RESEARCH SUMMARY
## Independent Reproduction and Reliability Verification of Phase 20

**Date:** October 2026  
**Auditor / Verifier:** Quantara AI Lead Auditor  
**Status:** Completed & Empirically Verified  
**Working Directory:** `reports/phase21_reliability_verification/`  

---

### 1. Plain-English Executive Verdict

Phase 21 independently verified the results, mathematical calculations, and code implementation of Phase 20 (**Dynamic Cash Allocation and Market-Regime Strategy Research**).

#### Core Verdict: **REPRODUCED WITH DOCUMENTED DISCREPANCIES**
1. **Mathematical Reproducibility:**
   - Out of 260 metrics evaluated across 10 strategies and two distinct market windows, **231 metrics (88.8%) reproduced with exact bit-for-bit precision**.
   - Every key portfolio milestone reproduced identically:
     - **Rule D (Multi-Regime Tiered):** Final Equity **₹98,627.40**, Max Drawdown **5.74%** (slashing drawdown by $76\%$ vs. Buy & Hold's $23.92\%$).
     - **Rule E (Heuristic Trend-Follower):** Final Equity **₹1,01,212.27**, Max Drawdown **5.53%**, Net Nominal Return **+1.19%**.
     - **Rule F (ML-Plus-Regime Dynamic Alpha):** Final Equity **₹97,163.38**, Max Drawdown **6.60%** (and **1.21%** in Validation).
     - **Net Cash Benchmark (5.2%):** Final Equity **₹1,05,115.38**, Max Drawdown **0.00%**, Net Return **+5.09%**.
     - **Buy & Hold Benchmark:** Final Equity **₹84,401.47**, Max Drawdown **23.92%**, Net Return **-15.60%**.
2. **Accounting Reconciliation is 100% Balanced:**
   - Every single portfolio simulation balances down to ₹0.00:
     $$\text{Final Equity} = \text{Starting Capital} + \sum \text{Net Trade P&L} + \text{Cash Interest Accrued}$$
   - There are zero missing funds, zero phantom dividends, and zero hidden costs.
3. **The Return Formula Offset is Fully Resolved:**
   - The minor difference observed in returns (e.g. Rule D reporting -1.39% vs. -1.37% implied by starting capital; Cash reporting +5.09% vs. +5.12%) is **completely explained by calculation conventions**:
   - The evaluation function `compute_metrics` calculated return using Day 1 end-of-day equity (₹100,020.12 after 1 day of interest) as the denominator, rather than initial starting capital (₹100,000.00). This created an identical ~0.02% offset across every interest-earning strategy.

---

### 2. Resolution of Key Discrepancies Investigated

The table below reconciles the exact figures queried in the audit mandate:

| Strategy / Benchmark | Starting Capital (₹) | Day 1 EOD Equity (₹) | Final Equity (₹) | Reported Return (%) | Return vs Starting Capital (%) | Mathematical Discrepancy | Root Cause / Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Rule D: Multi-Regime Tiered** | ₹1,00,000.00 | ₹100,020.12 | ₹98,627.40 | **-1.39%** | **-1.37%** | -0.02% | **Day 1 interest base offset** |
| **Rule E: Heuristic Trend-Follower** | ₹1,00,000.00 | ₹100,020.12 | ₹1,01,212.27 | **+1.19%** | **+1.21%** | -0.02% | **Day 1 interest base offset** |
| **Rule F: ML-Plus-Regime Alpha** | ₹1,00,000.00 | ₹100,020.12 | ₹97,163.38 | **-2.86%** | **-2.84%** | -0.02% | **Day 1 interest base offset** |
| **Net Cash Benchmark (5.2%)** | ₹1,00,000.00 | ₹100,020.12 | ₹1,05,115.38 | **+5.09%** | **+5.12%** | -0.03% | **Day 1 interest base offset** |
| **Gross Cash Benchmark (6.5%)** | ₹1,00,000.00 | ₹100,024.99 | ₹1,06,393.60 | **+6.37%** | **+6.39%** | -0.02% | **Day 1 interest base offset** |
| **Buy & Hold Benchmark** | ₹1,00,000.00 | ₹100,000.00 | ₹84,401.47 | **-15.60%** | **-15.60%** | 0.00% | **Exact Match** |
| **Zero Cash Benchmark (0.0%)** | ₹1,00,000.00 | ₹100,000.00 | ₹1,00,000.00 | **0.00%** | **0.00%** | 0.00% | **Exact Match** |

---

### 3. Answers to the Seven Core Audit Questions

#### 1. Which Phase 20 results can be reproduced?
- **All Core Portfolio Metrics are Reproduced:** Final equities, maximum drawdowns, trade counts, win rates, and daily equity curves reproduced for all benchmarks and all active allocation strategies (Rule D, Rule E, Rule F on both Validation and Test).
- **All Validation (2024–2025) Results are Reproduced:** Every strategy reproduced bit-for-bit with 0.00 difference.
- **The Core Defensive Finding is Confirmed:** Dynamic cash allocation (Rule D) slashed max drawdown from $23.92\%$ to $5.74\%$.

#### 2. Which results contain discrepancies or errors?
- **Return Calculation Denominator:** As shown above, reported returns used Day 1 EOD equity (₹100,020.12) rather than starting capital (₹100,000.00), creating a consistent ~0.02% offset.
- **Initial Table Run for Rules A, B, C on Test:** The preliminary CSV run of Phase 20 had slightly different liquidation timing on the final bar for Rules A, B, C, resulting in ₹87,793.09 vs ₹88,803.86 (a 1.01% variance). The equity curve already tracked ₹88,803.86.
- **Documentation Typo (ITC vs LT):** Multiple previous reports listed `LT` (Larsen & Toubro) in text summaries, whereas the actual data and code have always used `ITC.NS`.
- **Methodology Threshold Text Inconsistency:** `phase20_methodology.md` described regime thresholds as 60%/40% with an `OR` condition, whereas the code implemented a symmetric 50% threshold with an `AND` condition.

#### 3. Is the strategy logic consistent with the written rules?
- **Yes.** Signals are strictly causal (zero lookahead). Orders execute at session $t$ open. Stop-losses handle gap-downs realistically by filling at opening market prices rather than theoretical stop levels. Positions never exceed cash balance (no leverage).

#### 4. Are there unresolved data-quality or timing concerns?
- **No.** The historical price data is clean: 0 missing values, 0 duplicate rows, 0 invalid OHLC bars across 19,776 observations (2016–2026). Splits have 20-day purged embargos preventing label leakage.

#### 5. Can we trust the reported drawdowns and portfolio returns?
- **Yes.** The drawdowns were recomputed session-by-session from the raw daily equity curves and verified. Rule D's capital preservation (-1.37% return, 5.74% max drawdown vs -15.60% return, 23.92% max drawdown for Buy & Hold) is mathematically genuine.

#### 6. What must be fixed before further strategy research?
- Standardize all return formulas across all scripts to use starting capital (₹100,000.00) as the explicit denominator.
- Update documentation references to reflect `ITC.NS` and the exact 50% symmetric regime thresholds implemented in code.

#### 7. What should Phase 22 investigate, if anything?
- **Freeze strategy rules:** No further hyperparameter tuning on the 2025–2026 dataset.
- **Authentic Cash Instrument Data:** Replace hypothetical 5.2% mathematical accrual with authentic daily NAV series from an audited Indian liquid fund (e.g. Nippon India Liquid Fund).
- **Forward Untouched Evaluation:** Reserve future live market sessions (post-October 2026) for true out-of-sample validation.

