# QUANTARA AI — PHASE 21 CASH BENCHMARK AUDIT & REVIEW
## Rigorous Separation of Idle Cash, Hypothetical Yields, and Actual Market Returns

**Project:** Quantara AI — Independent Reliability Verification  
**Auditor:** Quantara AI Lead Auditor  
**Date:** October 2026  
**Output Path:** `reports/phase21_reliability_verification/cash_benchmark_review.md`  

---

### 1. Conceptual Separation of Cash Benchmarks

To maintain strict scientific and regulatory integrity, cash returns must never be conflated or presented as uniform. Phase 21 enforces strict separation across three distinct categories:

| Cash Benchmark Tier | Mathematical Assumption | Real-World Vehicle | Status in Research |
| :--- | :--- | :--- | :--- |
| **1. Zero-Yield Cash (0.0%)** | $r_{\text{annual}} = 0.00\%$ | Uninvested idle cash sitting in an Indian broker trading ledger | **Authentic Baseline** (No return, zero risk) |
| **2. Hypothetical Net Cash (5.2%)** | $r_{\text{annual}} = 5.20\%$ | Overnight sweep into Indian Liquid / Overnight Debt Mutual Funds | **Hypothetical Model** (Assumes 6.5% gross minus 25 bps TER and 20% effective tax) |
| **3. Hypothetical Gross Cash (6.5%)** | $r_{\text{annual}} = 6.50\%$ | Theoretical gross RBI Repo Rate with zero expenses and zero taxation | **Hypothetical Upper Bound** (Unachievable by retail investors) |
| **4. Actual Historical Investment** | Verified daily NAV series | Specific audited mutual fund historical units (e.g. Nippon India Liquid Fund NAV) | **Not Implemented in Phase 20** (Synthetic mathematical accrual used) |

> [!WARNING] Scientific Disclosure
> The returns reported for Net Cash (+5.09%) and Gross Cash (+6.37%) are **hypothetical mathematical models** based on static annualized rate assumptions, not historical fund NAV records. They must not be described as guaranteed, risk-free, or independently verified actual investment returns.

---

### 2. Audit of Mathematical Accrual Formula

In `scripts/run_phase20_regime_allocation.py` (line 216):
$$\text{daily\_cash\_rate} = (1.0 + r_{\text{annual}})^{1/252.0} - 1.0$$
Interest was compounded across 252 trading sessions rather than 365 calendar days.

#### Numerical Verification across 248 Trading Sessions (Test Period)
For an initial principal of $P_0 = \text{₹1,00,000.00}$:
$$P_{248} = 100,000 \times \left(1 + (1.052)^{1/252} - 1\right)^{248} = 100,000 \times (1.052)^{248/252} = \text{₹1,05,115.38}$$
- Implied Return vs Starting Capital:
  $$\frac{105,115.38 - 100,000.00}{100,000.00} \times 100\% = \mathbf{+5.1154\%} \approx \mathbf{+5.12\%}$$
- Reported Return in Phase 20 Table:
  $$\mathbf{+5.09\%}$$

#### Root Cause of the 0.02% Offset
On Session 0 (`2025-10-23`), the simulation credited interest at the end of the first day:
$$\text{Day 1 EOD Equity} = 100,000 \times (1.052)^{1/252} = \text{₹100,020.12}$$
When the evaluation function (`compute_metrics` in `run_phase17_reconciliation_research.py`, line 133) computed:
$$\text{total\_ret} = \frac{\text{equity\_series.iloc}[-1]}{\text{equity\_series.iloc}[0]} - 1.0 = \frac{105,115.38}{100,020.12} - 1.0 = \mathbf{+5.0942\%} \approx \mathbf{+5.09\%}$$
**Conclusion:** The reported +5.09% return was calculated relative to the Day 1 end-of-day equity balance (₹100,020.12) rather than the starting capital (₹100,000.00). Relative to starting capital, the true return is **+5.12%**.

---

### 3. Practical Taxation Realities in India (Finance Act 2023)

Phase 20 modeled the net yield at $5.20\%$ based on an assumed flat $20\%$ effective tax rate ($6.50\% \times 0.80 = 5.20\%$). In live Indian financial markets:

1. **Marginal Slab Rate Taxation:**
   Under the Finance Act 2023 amendment to Section 50AA of the Income Tax Act, gains from specified mutual funds (debt funds investing $<35\%$ in equity) acquired on or after April 1, 2023, are treated as **short-term capital gains** and taxed at the investor's applicable slab rate, regardless of the holding period.
2. **Impact across Investor Slabs:**
   - **High-Net-Worth Individual (30% Slab + Surcharge/Cess $\approx 39\%$):**
     $$\text{Net Yield} = 6.50\% \times (1 - 0.39) - 0.25\% \text{ TER} \approx \mathbf{3.72\%}$$
     *Final Equity on ₹1,00,000:* $\approx \text{₹1,03,660.00}$ (vs ₹1,05,115.38 reported).
   - **Domestic Corporate Tax Rate (22% + Surcharge/Cess $\approx 25.17\%$):**
     $$\text{Net Yield} = 6.50\% \times (1 - 0.2517) - 0.25\% \text{ TER} \approx \mathbf{4.61\%}$$
     *Final Equity on ₹1,00,000:* $\approx \text{₹1,04,540.00}$.
   - **Tax-Exempt Trust / Lower Bracket (0% to 10% Slab):**
     $$\text{Net Yield} \approx \mathbf{5.60\% - 6.25\%}$$.

---

### 4. Operational Sweep Limitations

In live institutional and retail brokerage accounts:
- Cash parked in equity trading accounts does not earn interest automatically unless linked to an active **automated sweep facility** or daily liquid ETF investment (e.g. Liquid BeES).
- Liquid BeES units distribute returns as fractional dividend units, which require regular reconciliation and broker DP charges upon redemption.
- **Auditor Verdict:** Zero-yield cash (0.0%) represents the true default state of uninvested capital in the absence of explicit treasury operations.

