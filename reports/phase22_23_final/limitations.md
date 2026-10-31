# QUANTARA AI — PHASE 22–23 RESEARCH LIMITATIONS & DISCLOSURES
## Sample Sufficiency, Repeated Holdout Bias, and Evidentiary Boundaries

**Auditor / Researcher:** Quantara AI Lead Auditor  
**Date:** October 2026  
**Path:** `reports/phase22_23_final/limitations.md`  

---

### 1. Chronology and Sample Size Limitations

1. **Preliminary Sample Size ($N = 21$ Sessions):**
   The evaluated forward window (2026-09-23 to 2026-10-23) contains only **21 trading sessions** (30 calendar days) and yielded only **1 closed trade in Rule E** and **0 closed trades in Rules D and F** prior to batch cut-off liquidation.
   - **Audit Verdict:** This sample size is **preliminary and statistically insufficient** to establish commercial viability, calculate meaningful Sharpe ratios, or assert a durable market edge.
2. **Pseudo-Out-of-Sample Classification:**
   Because strategy code was frozen on October 29, 2026 while historical price records through October 23, 2026 were already resident on disk, this window is formally classified as a **pseudo-out-of-sample holdout replay**, not a real-time contemporaneous prospective forward test.

---

### 2. Repeated Holdout Bias (2025–2026 Window)

1. **Reused Dataset Disclosure:**
   The historical period spanning **October 23, 2025 to September 22, 2026** has been repeatedly examined across **Phases 14, 15A, 15B, 16, 17, 18, 19, 20, 21, and 22**.
   - Even though individual machine learning models maintain temporal causality, the human researcher has observed this period's bear market dynamics repeatedly.
   - Designing strategies that allocate heavily to cash or enforce tight moving average filters naturally succeeds on a known bear market.
   - **Audit Verdict:** The 2025–2026 period must never be described as an untouched holdout or virgin forward test.

---

### 3. Empirical Evidentiary Prerequisites for Paper Trading

To prevent premature commercial deployment or false confidence, Quantara AI formally defines the following **proposed research criteria** before paper trading or forward deployment may be considered:

1. **Sample Size & Horizon:**
   - Minimum of **125 to 250 prospective trading sessions** (6 to 12 continuous calendar months).
   - Minimum of **50 to 100 closed prospective trades** executed under frozen rules.
2. **Net Trade Expectancy:**
   - Realized trade expectancy must remain positive ($> +₹200$ per trade) after realistic statutory delivery fees, slippage, and STT.
3. **Regime Diversity:**
   - The strategy must be tested across at least three distinct market regimes: a sustained bull market, a volatile sideways market, and a severe correction.
4. **Benchmark-Relative Alpha:**
   - Net performance must demonstrate positive risk-adjusted excess returns over both Buy & Hold and the authentic Liquid Fund NAV benchmark.

---

### 4. Regulatory & Scope Disclosures

- **Research Environment Only:** Quantara AI is an experimental AI coding and financial quantitative research project.
- **No Financial Advice:** Nothing in this audit, codebase, or report constitutes an investment recommendation, financial advice, or a guarantee of returns.
- **Zero Real-Money Execution:** The codebase contains zero broker API keys, zero paper order hooks, and zero automated trading capabilities. Real-money trading is strictly out of scope.

