# QUANTARA AI — RESEARCH ROADMAP & NEXT STEPS

This document outlines the planned future milestones for the Quantara AI research project, with primary focus on **Phase 27: Final Out-of-Sample Profit and Loss Backtest**.

---

## 1. Immediate Planned Milestone: Phase 27 Profit & Loss Simulation

### The Core Research Mission
Calculate the actual hypothetical rupee profit or loss from investing **₹1,00,000** starting capital across Logistic Regression, LSTM, and GRU under realistic Indian equity execution rules.

### Critical Target Correction: Genuinely Tradable 3-Class Target
In previous research phases (Phases 13–26), classification performance was evaluated on a binary target (`Target_Binary_20d`) that excluded `HOLD` observations (forward returns between -2.0% and +2.0%). 
- In live market operations, future returns cannot be filtered out because they are unknown when a trading decision is executed.
- Phase 27 implements an unfiltered **3-class target** across all 19,608 eligible historical observations:
  * **Class 0 (DOWN):** Forward 20-session return $< -2.0\%$
  * **Class 1 (HOLD):** Forward 20-session return between $-2.0\%$ and $+2.0\%$
  * **Class 2 (UP):** Forward 20-session return $> +2.0\%$

### Strategy & Portfolio Allocation Rules
- **Starting Capital:** ₹1,00,000 per strategy.
- **Long-Only Sizing:** Equal maximum allocation of $1/8 = 12.5\%$ per stock across the 8 stocks. Maximum portfolio equity exposure cannot exceed 100%.
- **Decision Rule:**
  * Predict `UP` (Class 2) at session $t$ close $\to$ Target $12.5\%$ equity allocation. Enter at Open $t+1$.
  * Predict `HOLD` (Class 1) or `DOWN` (Class 0) at session $t$ close $\to$ Target $0\%$ equity (maintain cash).
- **Holding Period:** 20 completed trading sessions (Exit at Open $t+21$).
- **Friction & Cash Accounting:** Full statutory delivery charges (0.3937%) plus execution slippage (0.05% per leg), cash balances, and corporate dividend credits on ex-dates.
- **Benchmarks:**
  1. **Buy-and-Hold:** Equal-weight ₹12,500 invested across the 8 stocks on day 1, holding through the period and crediting dividends.
  2. **Idle Cash:** ₹1,00,000 held in risk-free cash at 0% nominal return.

### Key Deliverables for Phase 27
- `phase27_final_profit_report.md`
- `phase27_strategy_comparison.csv`
- `phase27_fold_profitability.csv`
- `phase27_trade_ledger.csv`
- `phase27_daily_equity_curves.csv`
- `phase27_transaction_costs.csv`
- `phase27_benchmark_comparison.csv`
- `phase27_sensitivity_analysis.csv`
- `phase27_accounting_reconciliation.csv`

---

## 2. Longer-Term Architectural Opportunities

### 1. Alternative Feature Modalities Beyond Price Technicals
- Technical chart indicators (RSI, MACD, Bollinger Bands) contain limited forward-looking information over 20-day horizons.
- Future phases should investigate incorporating:
  * Macroeconomic series: Reserve Bank of India (RBI) policy repo rate changes, 10-year Indian government bond yields.
  * Intermarket signals: Crude oil prices, USD/INR exchange rate volatility, US 10-year Treasury yields.
  * Fundamental metrics: Trailing Price-to-Earnings (P/E), quarterly earnings surprise momentum, and return on equity (ROE).

### 2. Post-Hoc Probability Calibration
- Recurrent neural networks (LSTM and GRU) exhibit uncalibrated, overconfident softmax probabilities during market sell-offs.
- Introducing temperature scaling, Platt scaling, or Conformal Prediction could restrain aggressive false-positive UP predictions in bear regimes.

### 3. Asymmetric Loss Functions & Dynamic Thresholds
- Rather than standard cross-entropy loss, investigate cost-sensitive loss functions that penalize false-positive bullish predictions more heavily than false-negative missed rallies, directly reflecting the asymmetric financial penalty of bear-market drawdowns.

