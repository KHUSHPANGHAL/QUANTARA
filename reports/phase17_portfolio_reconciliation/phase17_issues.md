# Quantara AI — Phase 17 Issues, Limitations, and Risk Disclosures

**Auditor / Engineering Lead:** Antigravity AI Team  
**Audit Scope:** Data Integrity, Statistical Constraints, and Operational Risk Analysis  

---

### 1. Reused Test Period & Multiple Hypothesis Testing (Data Snooping)

#### Issue Description
The 2025–2026 chronological evaluation window (248 trading sessions) has been repeatedly evaluated across Phase 14, Phase 15A, Phase 15B, Phase 16, and Phase 17. 

#### Implications
- In quantitative finance, evaluating multiple hypothesis variants (thresholds, stop-losses, filters) on the exact same historical test set introduces **selection bias** and data snooping.
- Even though models were originally trained on pre-2024 data, iterating strategy rules and thresholds on the 2025–2026 window turns this period into a **reused research window**, not an untouched virgin holdout.
- Any future deployment would require testing on an entirely fresh, unseen time horizon (e.g. late 2026+ data).

---

### 2. Extremely Small Sample Sizes and Wide Confidence Intervals

#### Issue Description
By increasing the decision threshold to $P(\text{UP}) \ge 0.42$, the strategy became highly selective. Over the 248 trading sessions of 2025–2026, the model generated only **11 completed trades**.

#### Implications
- A sample size of $N = 11$ lacks statistical power to establish positive expectancy.
- As demonstrated by the 1,000-iteration bootstrap analysis:
  - **Win Rate 95% Confidence Interval:** $[36.4\%, 90.9\%]$
  - **Net Expectancy 95% Confidence Interval:** $[−₹913.43, +₹823.86]$
- A 95% confidence interval that ranges from losing over ₹900 per trade to making ₹800 per trade indicates massive parameter uncertainty. The observed win rate of 63.64% cannot be statistically distinguished from pure random chance or market noise.

---

### 3. Adverse Payoff Asymmetry (The Payoff Ratio Trap)

#### Issue Description
The strategy suffers from a structural payoff deficit:
- **Average Winning Trade:** $+₹945.10$
- **Average Losing Trade:** $−₹1,709.62$
- **Realized Payoff Ratio:** $0.55$

#### Implications
- Winning trades make less than ₹1,000 on average, while losing trades give back over ₹1,700.
- For a strategy with a 0.55 payoff ratio to break even, it requires a win rate greater than:
$$\text{Breakeven Win Rate} = \frac{1}{1 + 0.55} = 64.52\%$$
- Because the realized win rate was 63.64% (slightly below breakeven), the strategy ended with a net negative return (−₹222.75). Any mild decrease in win rate (which is highly likely over longer samples) will result in steep portfolio losses.

---

### 4. Long-Only Structural Constraint in a Bearish Market Regime

#### Issue Description
During the 2025–2026 test window, the Indian stock universe underwent a substantial market correction:
- **Buy & Hold Benchmark Return:** $−15.60\%$
- **Buy & Hold Maximum Drawdown:** $23.92\%$

#### Implications
- The current Quantara AI engine is strictly a **long-only equity cash delivery** system. It possesses no ability to take short positions or hedge downside risk via derivatives or inverse index instruments.
- While the model demonstrated strong capital preservation (losing only −0.22% compared to the market's −15.60% crash), long-only strategies will inevitably struggle to produce positive absolute profits during sustained sectoral or macroeconomic downturns.

---

### 5. Failure to Beat the Risk-Free Cash Benchmark

#### Issue Description
Across both the bullish validation window (2024–2025) and the bearish test window (2025–2026), holding 100% liquid cash earning a standard Indian repo/overnight yield (6.5% annualized) completely outperformed every active ML equity strategy after trading costs.

| Evaluation Window | Strategy | Return | Sharpe | Max Drawdown |
| :--- | :--- | :---: | :---: | :---: |
| **Validation (2024–2025)** | Cash Benchmark (6.5% Yield) | **+6.34%** | N/A | **0.00%** |
| Validation (2024–2025) | Reconciled Baseline (Th 0.42) | +5.14% | −0.18 | 3.76% |
| **Reused Test (2025–2026)** | Cash Benchmark (6.5% Yield) | **+6.37%** | N/A | **0.00%** |
| Reused Test (2025–2026) | Reconciled Baseline (Th 0.42) | −0.22% | −0.63 | 7.97% |
| Reused Test (2025–2026) | Buy & Hold Equities | −15.60% | −1.63 | 23.92% |

#### Conclusion
When an active trading system incurs turnover, transaction fees, bid-ask slippage, and market risk without exceeding the risk-free return of safe cash, deploying it in live trading is financially irrational.

---

### 6. Operational & Execution Constraints
- **Execution Latency:** Models assume exact Open execution at session $t+1$. In live markets, pre-open session volatility and opening auctions can introduce unexpected execution slippage beyond 0.05%.
- **Concentration:** With only 8 large-cap stocks in the candidate universe, trading opportunities are sparse, leading to prolonged periods of idle cash (54.4% of time in cash for Th 0.42).

**Mandatory Recommendation:** Maintain the strategy in the research and diagnostic phase. Under no circumstances should live broker accounts or real capital be connected.

