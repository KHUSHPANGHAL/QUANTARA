# Quantara AI — Phase 17 Methodology & Accounting Specifications

**Auditor / Engineering Lead:** Antigravity AI Team  
**Scope:** Reconciled Portfolio Accounting, Trade Ledgers, Risk Metrics, Bootstrap Analysis, and Robustness Experiments

---

### 1. Unified Accounting Identity & Capital Equations

To prevent discrepancies, omissions, or double-counting, all portfolio simulations strictly enforce the following closed-form mathematical identities:

$$\text{Final Equity} = \text{Initial Equity} + \sum_{i=1}^N \text{Gross PnL}_i - \sum_{i=1}^N \text{Commissions}_i + \sum_{k=1}^M \text{Dividends}_k$$

Where:
- $\text{Initial Equity} = ₹1,00,000.00$
- $N$ is the total count of completed round-trip trades.
- $M$ is the total count of dividend distributions received while holding stock over ex-dates.
- $\text{Gross PnL}_i = (\text{Exit Price}_i - \text{Entry Price}_i) \times \text{Shares}_i$
- $\text{Commissions}_i = \text{Commission}_{\text{entry}, i} + \text{Commission}_{\text{exit}, i}$
- $\text{Net PnL}_i = \text{Gross PnL}_i - \text{Commissions}_i$

At any trading session $t$, the total portfolio equity $E_t$ is defined as:
$$E_t = \text{Cash}_t + \sum_{j \in \text{Open Positions}} (\text{Shares}_j \times \text{Close Price}_{j, t})$$

When no positions are open, $E_t = \text{Cash}_t$. At the conclusion of the test period, all open positions are marked-to-market or closed at the final session open, ensuring $E_{\text{final}} = \text{Cash}_{\text{final}}$.

---

### 2. Signal Generation, Order Execution, and Gap Risk Modeling

#### A. Chronological Timeline
1. **Signal Generation (Close $t$):** At the end of session $t$, all closing prices, volume, technical indicators, and machine learning probabilities are calculated using strictly past and current information (features up to and including session $t$).
2. **Order Submission:** If $P(\text{UP}) \ge \text{Threshold}$ (and any optional filters like SMA50 are satisfied), an entry order is queued.
3. **Execution (Open $t+1$):** Orders are filled at session $t+1$ Open price plus execution slippage.
4. **Holding Duration:** The baseline strategy holds for 20 completed sessions, exiting at Open $t+21$ plus slippage.

#### B. Intraday Stop-Loss and Take-Profit Execution
When intraday stop-loss or take-profit rules are active:
- **Intraday Stop-Loss:** If session low $L_{\tau} \le \text{Stop Price}$, an exit is triggered.
  - *Standard Fill:* If Open $O_{\tau} > \text{Stop Price}$, execution occurs at the exact Stop Price minus slippage.
  - *Gap Down Fill:* If Open $O_{\tau} \le \text{Stop Price}$, execution occurs at Open $O_{\tau}$ minus slippage. (Zero unrealistic protection across overnight gaps).
- **Intraday Take-Profit:** If session high $H_{\tau} \ge \text{Target Price}$, execution occurs at Target Price plus/minus slippage.
  - *Gap Up Fill:* If Open $O_{\tau} \ge \text{Target Price}$, execution occurs at Open $O_{\tau}$.

---

### 3. Realistic Fee Schedule and Friction Models

Quantara AI evaluates two distinct cost structures representing realistic Indian equity delivery trading:

#### A. Base Cost Structure (0.3937% Round-Trip)
- **Brokerage:** Flat ₹20 per executed order (discount broker model) or 0.03% (whichever is lower).
- **Securities Transaction Tax (STT):** 0.10% on both Buy and Sell turnover (delivery equity).
- **Exchange Turnover Fees (NSE):** 0.00297% on turnover.
- **GST:** 18% on (Brokerage + Exchange Fees + SEBI charges).
- **SEBI Turnover Charges:** ₹10 per crore (0.0001%).
- **Stamp Duty:** 0.015% on Buy turnover.
- **Slippage:** 0.05% on entry and 0.05% on exit to account for bid-ask spreads and impact.
- **Total Round-Trip Drag:** Approximately ~0.39% to 0.40% of trade turnover.

#### B. Conservative Cost Structure (0.5000% Round-Trip)
- Scaled up to model higher market impact, illiquidity, or full-service brokerage charges.
- A fixed 0.25% drag is applied at Entry and 0.25% at Exit (0.50% total round-trip turnover).

---

### 4. Corporate Action & Dividend Accounting

To prevent corporate actions from distorting trading results:
- Stock prices used for entry and exit execution are **unadjusted historical raw prices**, accurately reflecting cash spent and cash received.
- When an active position is held across an ex-dividend date (i.e. acquired strictly prior to the ex-date and held through market open on the ex-date), cash dividend is calculated:
$$\text{Dividend Received} = \text{Shares Held} \times \text{Dividend Amount Per Share}$$
- The cash is credited to the portfolio cash balance on the ex-date.
- Dividends are tracked explicitly in `dividends_inr` and are never double-counted into the stock price return.

---

### 5. Statistical Rigor: Expectancy & Bootstrap Methodology

#### A. Net Trade Expectancy
Net expectancy measures the average expected profit or loss per executed trade:
$$\text{Expectancy} = (P_{\text{win}} \times \overline{\text{Win}}) - (P_{\text{loss}} \times |\overline{\text{Loss}}|)$$
Where:
- $P_{\text{win}} = \frac{N_{\text{wins}}}{N_{\text{trades}}}$
- $P_{\text{loss}} = \frac{N_{\text{losses}}}{N_{\text{trades}}} = 1 - P_{\text{win}}$
- $\overline{\text{Win}}$ is the average net rupee gain of winning trades.
- $\overline{\text{Loss}}$ is the average net rupee loss of losing trades.

#### B. Realized Payoff Ratio
$$\text{Payoff Ratio} = \frac{\overline{\text{Win}}}{|\overline{\text{Loss}}|}$$
A strategy with a win rate of $63.6\%$ requires a minimum payoff ratio of:
$$\text{Breakeven Payoff Ratio} = \frac{1 - 0.6364}{0.6364} = 0.5714$$
If the realized payoff ratio is below 0.5714, the strategy will inevitably lose money despite winning nearly two-thirds of its trades.

#### C. Non-Parametric Bootstrap Resampling
To quantify sample-size uncertainty without assuming a normal distribution:
1. For a strategy trade sample of size $N$, draw $N$ trades with replacement.
2. Recompute Win Rate, Mean Win, Mean Loss, Payoff Ratio, and Net Expectancy.
3. Repeat for $B = 1,000$ independent resamples.
4. Sort bootstrap estimates and determine the 95% Confidence Interval using the 2.5th and 97.5th percentiles:
$$\text{CI}_{95\%} = [\hat{\theta}_{0.025}^*, \hat{\theta}_{0.975}^*]$$

---

### 6. Chronological Windows & Walk-Forward Protocol

All model training and evaluations strictly observe time boundaries:
1. **Training Period (2018-01-01 to 2024-03-31):** Used exclusively for model fitting and hyperparameter searches.
2. **Validation Period (2024-04-01 to 2025-03-31):** Used for threshold selection, feature calibration, and walk-forward verification.
3. **Reused Test Period (2025-04-01 to 2026-03-31):** Used across Phases 14–17 to test out-of-sample behavior. In Phase 17, this is explicitly treated as a **reused research window** rather than an untouched holdout to avoid confirmation bias.

