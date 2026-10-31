# QUANTARA AI — PHASE 22 FROZEN STRATEGY SPECIFICATION
## Formal Rules, Mathematical Parameters, and Cryptographic Version Identifiers

**Status:** FROZEN & IMMUTABLE  
**Effective Date:** October 2026  
**Auditor / Researcher:** Quantara AI Lead Research Agent  
**Path:** `reports/phase22_frozen_evaluation/frozen_strategy_specification.md`  

---

### 1. Cryptographic Version Identifiers (SHA-256 Hashes)

To guarantee that strategy rules, training matrices, and evaluation code are permanently locked without retroactive parameter tuning, the following SHA-256 hashes are recorded:

| Component / Artifact | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Phase 20 Simulation Engine** | `scripts/run_phase20_regime_allocation.py` | `7321cc36fbf28b3df9b76c790f388c163221b0b2d3791768cabb75b917b16f75` |
| **Phase 21 Verification Engine** | `scripts/run_phase21_reliability_verification.py` | `79adefa2fd2c3d2b64a9280adad6067756ee13709ed96b8fe32d81cbfc519511` |
| **Phase 22 Frozen Execution Engine**| `scripts/run_phase22_frozen_evaluation.py` | `6b6900298aef828230559eb482a47781b2da1e7195764b85c324eb294efbcda8` |
| **Authentic Historical Stock Prices** | `data/raw/multi_stock_historical.csv` | `a30f9ad861cba8e44b1e1aa317d6e0216a0a519963b5c9d254482e043d52ff91` |
| **Authentic Liquid Fund NAV History**| `data/raw/nippon_india_liquid_fund_nav_history.csv` | `df42e269eb43ab147fb3a075f063b4131208225e934fbd033027fc2ba2e2248f` |
| **Feature Engineering Matrix v3** | `data/processed_v3/multi_stock_features_v3_validated.csv` | `37841c195123d5a230756784d09252c892837bc4ee5bc138bb67d98ee6717a10` |

---

### 2. Specification of Frozen Rule D (Multi-Regime Tiered Allocation)

- **Strategy Identifier:** `FROZEN_RULE_D`
- **Core Philosophy:** Graded capital allocation where exposure throttles down dynamically as market internal breadth and trend deteriorate, moving uninvested capital into liquid cash equivalents.
- **Market Regime Determination (Session $t-1$ Close):**
  1. *Market Proxy:* $P_{M, t-1} = \frac{1}{K} \sum_{k=1}^K Close_{k, t-1}$
  2. *Market Trend:* $P_{M, t-1} > SMA50_{M, t-1}$
  3. *Market Breadth:* $Breadth_{20, t-1} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(Close_{k, t-1} > SMA20_{k, t-1})$
  4. *Regime State:*
     - **FAVORABLE:** $Breadth_{20, t-1} \ge 0.50 \quad \land \quad P_{M, t-1} > SMA50_{M, t-1}$
     - **UNFAVORABLE:** $Breadth_{20, t-1} < 0.50 \quad \land \quad P_{M, t-1} \le SMA50_{M, t-1}$
     - **UNCERTAIN:** Mixed conditions
- **Exposure & Position Limits:**
  - *Favorable:* Max 3 open positions, 33.3% capital per position (100% max equity).
  - *Uncertain:* Max 1 open position, 33.3% capital per position (33.3% max equity, 66.7% cash).
  - *Unfavorable:* 0 open positions (0% equity, 100% cash).
- **ML Signal Logic:** 20-day Logistic Regression model trained strictly on training partition. Entry triggered when $P(\text{Up}) \ge 0.38 \land P(\text{Up}) > P(\text{Down})$.
- **Risk Management & Exits:**
  - Hard Stop-Loss: $-6.0\%$ (executed at opening price minus slippage on overnight gap-downs).
  - Profit Target: $+12.0\%$ (intraday execution).
  - Time-based Exit: Scheduled exit at open on session $t = \text{entry} + 20$.
  - Post-Stop Cooldown: Stock blocked from re-entry for remaining holding window.

---

### 3. Specification of Frozen Rule E (Heuristic Trend-Follower)

- **Strategy Identifier:** `FROZEN_RULE_E`
- **Core Philosophy:** Pure price action momentum without machine learning predictions, capturing sustained relative strength.
- **Market Filter:** $P_{M, t-1} > SMA50_{M, t-1} \quad \land \quad Breadth_{20, t-1} \ge 0.50$
- **Stock Selection:** Ranks stocks by 20-day momentum return $r_{20d}$. Eligible if $r_{20d} > 0.02$ and individual stock $Close_{t-1} > SMA50_{t-1}$.
- **Allocation:** Up to 3 positions equal-weighted (33.3% per slot).
- **Risk Management & Exits:**
  - Hard Stop-Loss: $-6.0\%$.
  - Profit Target: None (allows winners to run until time exit).
  - Time-based Exit: 20 sessions maximum hold.

---

### 4. Specification of Frozen Rule F (ML-Plus-Regime Dynamic Alpha)

- **Strategy Identifier:** `FROZEN_RULE_F`
- **Core Philosophy:** High-conviction statistical alpha requiring positive model probability, individual stock trend alignment, and broad market regime safety.
- **Market Filter:** Regime $\ne \text{UNFAVORABLE}$.
- **Stock Trend Filter:** Individual stock $Close_{t-1} > SMA50_{t-1}$.
- **ML Signal Logic:** $P(\text{Up}) \ge 0.38 \land P(\text{Up}) > P(\text{Down})$.
- **Allocation:** Max 3 positions in Favorable, 1 position in Uncertain, 0 in Unfavorable.
- **Risk Management & Asymmetric Exits:**
  - Hard Stop-Loss: $-5.0\%$.
  - Profit Target: $+12.0\%$.
  - Realized Target Ratio: $2.4 : 1$ reward-to-risk asymmetry.
  - Time-based Exit: 20 sessions maximum hold.

---

### 5. Standardized Transaction Friction Schedule

All frozen strategies operate under standard Indian delivery fee regulations:
- Brokerage: 0.03% (₹30 per crore delivery)
- Securities Transaction Tax (STT): 0.10% buy, 0.10% sell
- Exchange Turnover Fee: 0.00297%
- SEBI Turnover Fee: 0.0001%
- Stamp Duty: 0.015% buy
- GST: 18% on (Brokerage + Exchange + SEBI fees)
- Flat DP Charge: ₹15.93 per executed sell order
- Execution Slippage: 0.05% applied to all open/stop executions

