# QUANTARA AI — PHASE 20 RESEARCH LIMITATIONS & DISCLOSURES

**Project:** Quantara AI — Phase 20: Dynamic Cash Allocation and Market-Regime Strategy Research  
**Date:** October 2026  
**Auditor:** Quantara AI Lead Research Agent  
**Path:** `reports/phase20_regime_allocation/phase20_limitations.md`  

---

### 1. Repeated Holdout Evaluation Bias (Reused Test Period)

A primary methodological vulnerability of the ongoing Quantara AI research pipeline is the repeated reuse of the **2025–2026 historical period (2025-10-23 to 2026-09-22)** across Phases 14 through 20.
- While each individual model and regime filter was trained strictly on data prior to October 2025 (strictly zero lookahead within the algorithmic execution), **the human researcher and agent have observed the test period characteristics repeatedly**.
- Knowing in advance that 2025–2026 was a bear-market correction naturally incentivizes the design of defensive regime switches and high cash allocations.
- **Scientific Implication:** The 2025–2026 test window must be characterized as an **audited historical stress test**, not an untouched virgin holdout. True statistical validation requires subsequent forward evaluation on newly generated live market sessions.

---

### 2. Cash Yield and Operational Assumptions

1. **Tax Drag and Tax Slab Variability:**
   - In Phase 20, the net cash yield is modeled at **5.20% per annum**, calculated from a gross 6.50% repo rate minus 25 bps TER and an assumed 20% effective tax rate on short-term liquid fund capital gains.
   - In practical Indian taxation under Finance Act 2023, debt mutual funds are taxed at the investor's marginal income tax slab (which can reach 30% plus surcharge and cess, effectively ~39% for high-net-worth individuals, or 25% for domestic corporations).
   - An investor in the 39% slab would realize a net cash yield closer to **3.8%–4.0%**, whereas a tax-exempt entity would realize ~6.25%.
2. **Operational Friction of Overnight Sweep:**
   - The simulation assumes unallocated cash is automatically swept into overnight/liquid instruments daily without cut-off timing friction (e.g., 1:30 PM liquid fund cut-off) or settlement cycle delays (T+1 equity settlement).
   - In live trading, idle funds in a broker ledger typically earn 0.0% unless an automated cash-management / sweep account product is active.

---

### 3. Small Sample Sizes in Selective Subsets

1. **Wide Confidence Intervals:**
   - High-conviction filters drastically reduce trading frequency:
     - Rule F executed only **3 trades** in Validation (Win Rate 66.7%, 95% CI $[20.8\%, 93.9\%]$) and **6 trades** in Test (Win Rate 16.7%, 95% CI $[3.0\%, 56.4\%]$).
     - Rule D executed **8 trades** in Validation and **8 trades** in Test.
   - Because the sample size $N$ is small, individual trade outcomes exert an outsized impact on the aggregate performance metrics.
2. **Statistical Significance Limitations:**
   - Although Rule D achieved strong capital preservation (-1.39% vs. -15.60% Buy & Hold), an $N=8$ sample size cannot establish statistically significant outperformance at the $p < 0.05$ level against random variation.

---

### 4. Vulnerability to Choppy Counter-Trend Whipsaws

1. **The Breadth Whipsaw Phenomenon:**
   - Binary threshold switches (such as Rule B and Rule C) suffered substantial drawdowns (-13.72% and -13.55%) because bear markets feature sharp, deceptive counter-trend rallies.
   - When market breadth briefly ticked above 50%, the binary switch allocated capital into long equity positions right before the primary downtrend resumed, resulting in consecutive stopped-out trades.
2. **Superiority of Graded Multi-Tiered Rules:**
   - The failure of binary switches underscores that regime allocation must be **graded and tiered** (e.g., Rule D), scaling exposure down gradually rather than toggling between 0% and 100% equity.

---

### 5. Scope and Regulatory Boundary

- **Research Only:** Quantara AI is an experimental Machine Learning and Quantitative Finance research environment.
- **No Financial Advice:** Nothing in this audit, codebase, or report constitutes investment advice, trading recommendations, or a guarantee of future returns.
- **No Broker Integration:** The codebase contains zero live order execution hooks or real-money brokerage connections.

