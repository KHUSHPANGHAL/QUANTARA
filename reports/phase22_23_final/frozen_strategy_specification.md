# QUANTARA AI — PHASE 22–23 FROZEN STRATEGY SPECIFICATION
## Immutable Algorithmic Rules, Mathematical Parameters, and Cryptographic Checksums

**Document Status:** FROZEN & VERSIONED  
**Effective Audit Date:** October 2026  
**Auditor / Researcher:** Quantara AI Lead Auditor  
**Output Path:** `reports/phase22_23_final/frozen_strategy_specification.md`  

---

### 1. Cryptographic Version Checksums (SHA-256)

To establish an immutable boundary preventing post-hoc parameter adjustments, the exact SHA-256 hashes of all authoritative code, feature matrices, and market data files are recorded:

| Component | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Phase 20 Simulation Engine** | `scripts/run_phase20_regime_allocation.py` | `7321cc36fbf28b3df9b76c790f388c163221b0b2d3791768cabb75b917b16f75` |
| **Phase 21 Verification Engine** | `scripts/run_phase21_reliability_verification.py` | `79adefa2fd2c3d2b64a9280adad6067756ee13709ed96b8fe32d81cbfc519511` |
| **Phase 22 Frozen Engine** | `scripts/run_phase22_frozen_evaluation.py` | `6b6900298aef828230559eb482a47781b2da1e7195764b85c324eb294efbcda8` |
| **Phase 22–23 Unified Engine**| `scripts/run_phase22_23_unified_verification.py` | `68c3a9d592a4e9b97779fdfd7cfd197ea94819777085c13b28ee8d8d38e21a20` |
| **Authentic Stock Prices** | `data/raw/multi_stock_historical.csv` | `a30f9ad861cba8e44b1e1aa317d6e0216a0a519963b5c9d254482e043d52ff91` |
| **Authentic Liquid Fund NAV** | `data/raw/nippon_india_liquid_fund_nav_history.csv` | `df42e269eb43ab147fb3a075f063b4131208225e934fbd033027fc2ba2e2248f` |
| **Feature Engineering v3** | `data/processed_v3/multi_stock_features_v3_validated.csv` | `37841c195123d5a230756784d09252c892837bc4ee5bc138bb67d98ee6717a10` |

---

### 2. Strategy Specifications

#### A. Frozen Rule D: Multi-Regime Tiered Allocation (`FROZEN_RULE_D`)
- **Philosophy:** Graded exposure scaling where capital automatically rotates to interest-bearing cash during market breakdown.
- **Regime Classification (Session $t-1$ Close):**
  - *Equal-Weight Market Index:* $P_{M, t-1} = \frac{1}{8} \sum_{k=1}^8 Close_{k, t-1}$
  - *Market 50d SMA:* $SMA50_{M, t-1} = \frac{1}{50} \sum_{i=0}^{49} P_{M, t-1-i}$
  - *Short-Term Breadth:* $Breadth_{20, t-1} = \frac{1}{8} \sum_{k=1}^8 \mathbb{I}(Close_{k, t-1} > SMA20_{k, t-1})$
  - *State Definitions:*
    - **FAVORABLE:** $Breadth_{20, t-1} \ge 0.50 \quad \land \quad P_{M, t-1} > SMA50_{M, t-1}$
    - **UNFAVORABLE:** $Breadth_{20, t-1} < 0.50 \quad \land \quad P_{M, t-1} \le SMA50_{M, t-1}$
    - **UNCERTAIN:** Mixed conditions
- **Exposure Limits:**
  - *Favorable:* Max 3 open positions, 33.3% capital per position (100% max equity exposure).
  - *Uncertain:* Max 1 open position, 33.3% capital per position (33.3% max equity exposure, 66.7% cash).
  - *Unfavorable:* 0 open positions (0% equity exposure, 100% cash).
- **Signal Logic:** 20-day Logistic Regression model trained strictly on training data. Entry condition: $P(\text{Up}) \ge 0.38 \land P(\text{Up}) > P(\text{Down})$.
- **Risk Management & Exits:**
  - Hard Stop-Loss: $-6.0\%$ (fills at open minus slippage if opening bar gaps down).
  - Profit Target: $+12.0\%$ (intraday fill minus slippage).
  - Time-based Exit: Open execution on session $t = \text{entry} + 20$.
  - Post-Stop Cooldown: Stock barred from re-entry for remaining holding window.

#### B. Frozen Rule E: Heuristic Trend-Follower (`FROZEN_RULE_E`)
- **Philosophy:** Classic price action momentum without machine learning predictions.
- **Regime Filter:** Permitted only when $P_{M, t-1} > SMA50_{M, t-1} \land Breadth_{20, t-1} \ge 0.50$.
- **Stock Selection:** Highest 20-day return $r_{20d} > 0.02$ among stocks trading above their individual 50-day SMA ($Close_{t-1} > SMA50_{t-1}$).
- **Allocation:** Up to 3 positions equal-weighted (33.3% capital per slot).
- **Risk Management & Exits:**
  - Hard Stop-Loss: $-6.0\%$.
  - Profit Target: None (runs to maximum holding period).
  - Time Exit: 20 sessions maximum hold.

#### C. Frozen Rule F: ML-Plus-Regime Dynamic Alpha (`FROZEN_RULE_F`)
- **Philosophy:** High-conviction statistical alpha requiring positive ML probability, individual stock trend alignment, and broad market regime safety.
- **Regime Filter:** Permitted in Favorable and Uncertain states; strictly blocked in Unfavorable state.
- **Stock Trend Filter:** Individual stock $Close_{t-1} > SMA50_{t-1}$.
- **ML Signal Logic:** $P(\text{Up}) \ge 0.38 \land P(\text{Up}) > P(\text{Down})$.
- **Allocation:** Max 3 positions in Favorable, 1 position in Uncertain, 0 in Unfavorable.
- **Asymmetric Risk Management:**
  - Hard Stop-Loss: $-5.0\%$.
  - Profit Target: $+12.0\%$.
  - Reward-to-Risk Target Asymmetry: $2.4 : 1$.
  - Time Exit: 20 sessions maximum hold.

---

### 3. Execution Friction & Statutory Costs

All simulated fills strictly incorporate Indian cash delivery fee structures:
- Brokerage: 0.03%
- Securities Transaction Tax (STT): 0.10% buy, 0.10% sell
- Exchange Turnover Fee: 0.00297%
- SEBI Charges: 0.0001%
- Stamp Duty: 0.015% buy
- GST: 18% on (Brokerage + Exchange + SEBI fees)
- Depository Participant (DP) Charge: Flat ₹15.93 per sell order
- Execution Slippage: 0.05% applied to all open/stop executions

