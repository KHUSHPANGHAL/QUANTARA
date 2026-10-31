# Quantara AI — Phase 18 Issues, Limitations, and Research Risks

**Auditor / Engineering Lead:** Antigravity AI Team  
**Audit Scope:** Model Diagnostics, Feature Information Limits, and Statistical Integrity  

---

### 1. Fundamental Information Limits (Low Signal-to-Noise Ratio)

#### Issue Description
Across all tested models (Logistic Regression, Random Forest, LightGBM), Balanced Accuracy on out-of-sample data hovers between **$31.08\%$ and $36.55\%$**. For a 3-class target, pure random guessing is $33.33\%$.

#### Implications
- Technical indicators derived from standard price and volume series (RSI, Moving Averages, MACD, Bollinger Bands, Volume vs SMA) contain virtually **zero predictive information** regarding 20-day forward returns in the Indian large-cap equity universe.
- Complex tree models (Random Forest and LightGBM) perform **worse than random guessing** on validation data ($31.71\%$ and $31.08\%$), demonstrating that they fit idiosyncratic training noise rather than persistent market dynamics.
- Trying to engineer profitability by adding more layers, deep neural networks, or complex hyperparameter sweeps will simply exacerbate overfitting without fundamentally improving predictive signal.

---

### 2. Severe Non-Stationary Regime Instability

#### Issue Description
Market dynamics vary radically across market cycles:
- **Training Period (2017–2024):** $47.7\%$ UP moves (sustained bull market).
- **Validation Period (2024–2025):** $36.5\%$ HOLD moves, $32.5\%$ UP (sideways churn).
- **Test Period (2025–2026):** $43.1\%$ DOWN moves, $27.8\%$ UP (bear market correction).

#### Implications
- Models trained on historical data learn conditional probabilities that assume an upward-drifting baseline.
- Probability calibration procedures (such as Platt scaling) fit in a bull regime become dangerously overconfident in bear regimes: on the 2025–2026 test set, Platt scaling assigned a $51.9\%$ probability of going UP to a group of stocks where only $22.6\%$ actually moved upward!
- In an unhedged long-only equity model, regime drift acts as a direct headwind that overwhelms any micro-level technical signal.

---

### 3. Repeated Use of the 2025–2026 Test Period (Data Snooping Risk)

#### Issue Description
The 2025–2026 chronological window (248 trading sessions) has been repeatedly evaluated across Phases 14, 15A, 15B, 16, 17, and 18.

#### Implications
- Even though models were originally trained on pre-2024 data, testing multiple hypotheses, threshold levels, stop-loss percentages, and filter combinations on the exact same test dataset introduces **implicit data snooping** and selection bias.
- The 2025–2026 dataset must be formally designated as a **reused research period**, never an untouched virgin holdout.
- Any future strategy validation must wait for genuinely unseen post-2026 market data.

---

### 4. Long-Only Structural Constraint in a Down Market

#### Issue Description
Quantara AI is currently implemented strictly as a long-only cash delivery system. It does not support short selling, index futures hedging, or inverse exchange-traded funds.

#### Implications
- During the 2025–2026 period, the benchmark equity universe fell by $-15.60\%$ (Buy & Hold).
- In a market where 43% of stocks drop and fewer than 28% rise, a long-only cash model cannot generate substantial positive nominal returns without either sitting entirely in cash (earning 0% from equities) or taking highly concentrated, risky bets that fail when costs are deducted.

---

### 5. Transaction Costs Exceed Model Edge (The Friction Barrier)

#### Issue Description
Under zero trading friction, the LR Threshold 0.42 strategy produces a positive gross return of $+1.27\%$ on the test period and $+17.06\%$ at threshold 0.38 on validation. However, realistic delivery trading costs (brokerage, STT, exchange fees, GST, SEBI charges, stamp duty, and $0.05\%$ bid-ask slippage) drag test return to $-0.22\%$ under Base costs and $-0.84\%$ under Conservative costs.

#### Implications
- The gross statistical edge of the model is narrower than the bid-ask spread and statutory taxation of Indian markets.
- Higher turnover strategies generate cumulative commissions (over $₹2,000$ to $₹3,000$ per year on a $₹1,00,000$ account) that inevitably drain investor equity.

---

### 6. Summary Risk Disclosure
Deploying current Quantara AI models in live trading would guarantee capital erosion due to a combination of weak predictive signal, severe regime vulnerability, and transaction cost drag. Models must remain purely exploratory research tools.

