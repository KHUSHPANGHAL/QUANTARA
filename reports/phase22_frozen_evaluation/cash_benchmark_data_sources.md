# QUANTARA AI — PHASE 22 CASH BENCHMARK DATA SOURCES & PROVENANCE
## Authentic Historical Net Asset Value (NAV) Documentation

**Document Status:** Complete & Audited  
**Retrieval Date:** October 29, 2026  
**Auditor / Researcher:** Quantara AI Lead Research Agent  
**Path:** `reports/phase22_frozen_evaluation/cash_benchmark_data_sources.md`  

---

### 1. Scheme Identification & Metadata

In compliance with Phase 22 directives, hypothetical static cash rates (6.5% repo rate) have been supplemented with authentic, official daily historical NAV records from an audited Indian liquid debt mutual fund:

| Field | Detail |
| :--- | :--- |
| **Fund House** | Nippon India Mutual Fund |
| **Scheme Name** | Nippon India Liquid Fund - Direct Plan - Growth Option |
| **AMFI Scheme Code** | `118701` |
| **SEBI Registration Code**| `NIMF/O/D/LIF/00/12/0005` |
| **ISIN (Growth Option)** | `INF204K01ZH0` |
| **Scheme Category** | Open-Ended Debt Scheme — Liquid Fund |
| **Benchmark Index** | CRISIL Liquid Debt Index |
| **Inception Date** | January 1, 2013 |
| **Total Expense Ratio (TER)**| $\sim 0.20\%$ (Direct Plan) |
| **Distribution Mechanism**| Growth Option (100% of underlying interest compounded into daily NAV; zero dividend distributions) |

---

### 2. Data Retrieval & Raw File Preservation

- **Authoritative Endpoint:** AMFI India Open NAV Database (`https://api.mfapi.in/mf/118701`)
- **Raw File Preserved:** `data/raw/amfi_nippon_liquid_fund_118701_raw.json` (176,691 bytes, SHA-256: `65cfda5525ce08c35b64c0ec2e99d82136e4aa1a4c95a201b1a7fe56417fbbe4`)
- **Processed Clean CSV:** `data/raw/nippon_india_liquid_fund_nav_history.csv` (4,302 daily records, SHA-256: `df42e269eb43ab147fb3a075f063b4131208225e934fbd033027fc2ba2e2248f`)
- **Historical Span:** January 1, 2013 to October 28, 2026 (13.8 continuous years)
- **Licensing & Access:** Open public regulatory reporting data published pursuant to SEBI Mutual Fund Regulations.

---

### 3. Empirical Returns across Audited Project Windows

The table below records the actual, authentic returns delivered by Nippon India Liquid Fund across the project's evaluation horizons:

| Evaluation Window | Start Date | End Date | Start NAV (₹) | End NAV (₹) | Actual Gross Return | Annualized Gross Yield | Net Return (20% Tax) | Net Return (39% Slab) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Window** | 2024-10-24 | 2025-09-19 | ₹6,152.8079 | ₹6,536.6577 | **+6.24%** | **6.93% p.a.** | **+4.99%** | **+3.81%** |
| **Reused Test Window**| 2025-10-23 | 2026-09-22 | ₹6,572.2315 | ₹6,968.7611 | **+6.03%** | **6.62% p.a.** | **+4.83%** | **+3.68%** |
| **Prospective Forward**| 2026-09-23 | 2026-10-23 | ₹6,969.8122 | ₹6,997.3358 | **+0.39%** | **4.92% p.a.** | **+0.32%** | **+0.24%** |

#### Comparison with Previous Phase Assumptions:
- In Phase 20, the hypothetical gross repo yield was modeled at **6.50%**. The authentic Nippon India Liquid Fund delivered **6.62% annualized** in 2025–2026 and **6.93% annualized** in 2024–2025, validating that the 6.5% gross assumption was empirically sound and slightly conservative.
- The net cash return after an assumed 20% effective tax was **+4.83%** (vs. +5.09% modeled). For an investor in the highest marginal tax bracket (39%), the net return was **+3.68%**.

---

### 4. Regulatory & Risk Disclosures

> [!WARNING] Mandatory Mutual Fund Disclosures
> 1. Mutual fund investments are subject to market risks. Read all scheme-related documents carefully.
> 2. Past performance is not an indicator or guarantee of future returns.
> 3. Although liquid funds invest primarily in short-term sovereign treasury bills, repos, and high-grade commercial paper with maturity $\le 91$ days, they are **not government-guaranteed or insured bank fixed deposits**. Credit risk and duration risk exist.

