# QUANTARA AI — PROJECT DOCUMENTATION & EXPLAINER GUIDE

Welcome to the institutional documentation repository for **Quantara AI**, an applied machine learning and deep learning quantitative research project evaluating trend prediction and systematic trading strategies in the Indian equity market.

---

## What is Quantara AI?

Quantara AI is an experimental research platform designed to investigate whether modern computational methods—spanning traditional regularized machine learning (Logistic Regression), gradient boosting, and deep recurrent neural networks (Long Short-Term Memory [LSTM] and Gated Recurrent Units [GRU])—can generate statistically reliable directional predictions and profitable trading strategies when applied to liquid Indian equities listed on the National Stock Exchange (NSE).

### Core Universe & Technical Scope
- **Universe:** 8 liquid Indian blue-chip stocks:
  1. `BHARTIARTL.NS` (Bharti Airtel)
  2. `HDFCBANK.NS` (HDFC Bank)
  3. `ICICIBANK.NS` (ICICI Bank)
  4. `INFY.NS` (Infosys)
  5. `ITC.NS` (ITC Limited)
  6. `RELIANCE.NS` (Reliance Industries)
  7. `SBIN.NS` (State Bank of India)
  8. `TCS.NS` (Tata Consultancy Services)
- **Historical Horizon:** 2017 to 2026 (over 2,420 trading sessions per stock; 19,380+ observations).
- **Core Signal Horizon:** 20 completed trading sessions (approximately 1 calendar month), with strict chronological signal-to-execution mechanics (Signal at Close $t$, Entry at Open $t+1$, Exit at Open $t+21$).
- **Statutory Friction Model:** Realistic round-trip transaction costs incorporating Securities Transaction Tax (STT), exchange turnover fees, SEBI charges, stamp duty, Goods & Services Tax (GST), brokerage, and execution slippage (~0.3937% to 0.4937% round-trip).

---

## Documentation Navigation Guide

This folder contains a complete, transparent, and audited record of the project's evolution, empirical findings, error corrections, and current status:

| Document | Primary Focus & Content |
| :--- | :--- |
| [`project_history.md`](./project_history.md) | **Chronological Phase Summary (Phases 1–26):** A comprehensive summary of every major research milestone, from the initial data hygiene audits and baseline models through dynamic cash allocation research and deep learning walk-forward corrections. |
| [`current_status.md`](./current_status.md) | **Current Scientific & Operational Status:** An objective assessment of what has been verified, what defects were identified and resolved, what statistical uncertainties remain, and why live trading is strictly disabled. |
| [`next_steps.md`](./next_steps.md) | **Roadmap & Pending Milestones:** Detailed implementation plan for Phase 27 (hypothetical ₹1,00,000 backtest under tradable 3-class targets) and future algorithmic enhancements. |
| `Quantara_AI_Full_Project_Explained.pdf` | **Institutional Master PDF:** Master project explanation document. *(Note: Must be placed into this folder manually if downloaded externally).* |

---

## Key Takeaways for Beginners

1. **Machine Learning is Not Magic:** Stock market returns have an extremely low signal-to-noise ratio. Across nearly a decade of data, standard technical indicators provide weak directional edge (hovering near a 50% coin toss).
2. **Complexity Does Not Equal Alpha:** Adding 42,000 to 56,000 neural network parameters (LSTM and GRU) did not outperform a simple 16-parameter Logistic Regression model. In genuine out-of-sample walk-forward testing, deep learning recurrent networks suffered from mode collapse and slightly underperformed simple linear models.
3. **Risk Management Trumps Prediction:** As proven in Phases 20–23, capital preservation in equities is driven by **dynamic cash allocation and market regime filters** (holding cash during unfavorable market downturns), rather than attempting to aggressively trade low-conviction directional predictions.
4. **Research Only:** **No live trading is enabled, and no broker connections exist.** All performance figures represent historical research simulations.

