# QUANTARA AI — PHASE 21 LIMITATIONS & DISCLOSURES
## Methodological Vulnerabilities, Repeated Holdout Bias, and Protocol Safeguards

**Project:** Quantara AI — Independent Reliability Verification  
**Auditor:** Quantara AI Lead Auditor  
**Date:** October 2026  
**Target Target:** Phase 20 Verification and Research Integrity  
**Output Path:** `reports/phase21_reliability_verification/phase21_limitations.md`  

---

### 1. Repeated Holdout Bias (Reused 2025–2026 Test Window)

The most significant methodological limitation across the Quantara AI research trajectory is the repeated evaluation of the **2025–2026 test window (2025-10-23 to 2026-09-22)**:
- This period has been tested repeatedly across **Phases 14, 15A, 15B, 16, 17, 18, 19, 20, and now Phase 21**.
- **The Mechanism of Overfitting:** Even when each individual model is fitted strictly on data prior to October 2025 without future lookahead, the researcher knows in hindsight that 2025–2026 was a bear market. Strategies that allocate heavily to cash or enforce defensive moving-average stops naturally perform best.
- **Walk-Forward Limitation:** While 5-fold expanding walk-forward testing (2022–2026) demonstrates how the classifier behaved in earlier regimes, it does not erase the fact that strategy hyperparameters (such as 50% breadth thresholds and 5% stops) were selected after observing the test period.
- **Auditor Verdict:** The 2025–2026 period must be formally classified as a **known historical stress-test period**, not an untouched out-of-sample forward test.

---

### 2. Disclosed Typo in Stock Universe Documentation

During the Phase 21 data audit, a persistent documentation discrepancy was uncovered:
- **Textual Reports (Phases 15B, 17, 18, 19, 20):** Repeatedly listed the eight audited stocks as `RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, LT`.
- **Actual Code & Data Files (`multi_stock_historical.csv`, `splits_v3`):** Contain `ITC.NS` (ITC Limited) and have never contained `LT.NS` (Larsen & Toubro).
- **Audit Confirmation:** This was an unintentional textual documentation carry-over from early project design notes. All feature matrices, price pivots, labels, and order blotters executed strictly on `ITC.NS`. All calculations are internally consistent with the actual data on disk.

---

### 3. Small Sample Size in High-Conviction Regimes

1. **Trade Count Compression:**
   - In Rule F (ML-Plus-Regime Dynamic Alpha), high selectivity restricted execution to only **3 trades in Validation** and **6 trades in Test**.
   - In Rule D (Multi-Regime Tiered), execution was restricted to **8 trades in Validation** and **8 trades in Test**.
2. **Statistical Uncertainty:**
   - With $N \le 8$ trades, 95% Wilson confidence intervals around the win rate remain extraordinarily wide (e.g. $[3.0\%, 56.4\%]$ for Rule F on Test).
   - A single lucky or unlucky trade alters the overall win rate and profit factor drastically. True statistical edge cannot be proven on sample sizes of this magnitude.

---

### 4. Operational Sweep & Tax Friction Assumptions

1. **Tax Slab Variability:**
   - The hypothetical net cash yield ($5.20\%$) assumes an effective tax rate of $20\%$.
   - Under Finance Act 2023, debt mutual funds in India are taxed at the investor's marginal income tax slab. For high-earning individuals in the $30\%$ slab (plus surcharge and cess = $39\%$), the net yield drops to $\approx 3.72\%$.
2. **Operational Friction:**
   - The simulation assumes instantaneous overnight cash interest accrual. In real-world brokerage setups, capital must be manually or automatically swept into liquid ETFs or overnight mutual funds, which introduces settlement cycle constraints (T+1) and cut-off timing restrictions.

---

### 5. Recommended Future Evaluation Protocol (Phase 22+)

To eliminate repeated holdout bias, the following **Frozen Rule Protocol** is strictly recommended before any further strategy development:
1. **Rule Freezing:** Formally freeze the exact parameters of Rule D (Tiered Allocation), Rule E (Heuristic Trend), and Rule F (Dynamic Alpha) without further threshold tuning.
2. **Genuinely Untouched Out-of-Sample Window:** Reserve subsequent live market data (sessions after October 2026) as an uninspected, virgin holdout.
3. **Pre-Registration:** Pre-register entry rules, stop-loss formulas, and position limits prior to generating or inspecting future market observations.

