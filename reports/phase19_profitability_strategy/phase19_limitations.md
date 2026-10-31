# Quantara AI — Phase 19 Limitations, Constraints, and Risk Disclosures

**Auditor / Engineering Lead:** Antigravity AI Engineering Team  
**Evaluation Scope:** Strategy Feasibility, Feature Coverage, and Statistical Significance  

---

### 1. Complete Non-Availability of Fundamental Data

#### Disclosure
- **Repository Audit:** Inspection of the data directory and configuration manifests confirmed that no fundamental data files (quarterly balance sheets, P/E ratios, earnings surprises, or dividend declaration timestamps) exist in the authenticated dataset.
- **Compliance Action:** In strict compliance with guidelines (*"Fundamental information only when reliable historical values and publication timestamps are available. Verify source availability before implementation. Do not fabricate missing data"*), no synthetic fundamentals were created. Group E features are formally designated as **UNAVAILABLE**.
- **Impact:** Models rely entirely on technical OHLCV price action, volume proxies, and cross-stock breadth.

---

### 2. The Short-Horizon Turnover Friction Trap (1-Day Horizon)

#### Disclosure
- In the 1-Day forward return horizon, the model generated **181 executed trades** over 248 trading sessions on the 2025–2026 test period.
- Total transaction friction (brokerage, STT, exchange fees, GST, SEBI charges, stamp duty, and bid-ask slippage) amounted to **$₹14,951.96$**, eroding approximately **$15.0\%$ of the initial account capital**.
- The average 1-day gross price move in large-cap equities ($0.3\%$ to $0.4\%$, or $\approx ₹350$ to $₹420$ on a $₹33,000$ position) is completely consumed by the $0.39\%$ round-trip friction ($\approx ₹90$ to $₹100$).
- **Conclusion:** Daily active long-only cash trading in Indian delivery equities is economically non-viable for retail account sizes.

---

### 3. Fragility of Asymmetric Exits Under Market Corrections

#### Disclosure
- Achieving positive expectancy at 40%–50% accuracy requires an average win-to-loss ratio (payoff ratio) of at least **$1.25$ to $1.50$**.
- While asymmetric targets (e.g., 5% stop-loss vs. 10%–12% take-profit) achieved a 1.28 payoff ratio on the Validation period, they broke down during the 2025–2026 bear correction:
  1. Overnight gap-down openings routinely breached stop-loss triggers, causing execution at prices significantly worse than the nominal 5% threshold.
  2. Sustained downward momentum prevented positions from reaching the 10%–12% profit targets, forcing premature stops or time-decay exits.
  3. Win rates collapsed below the break-even threshold ($12.5\%$ vs. $54.6\%$ required), generating net portfolio losses.

---

### 4. Extreme Statistical Uncertainty in Highly Selective Subsets

#### Disclosure
- Applying strict filters (e.g. Stock > SMA50 AND Market Breadth $\ge 50\%$) dramatically reduced trade frequency:
  - On Validation: Only **4 completed trades** were executed over 248 sessions.
  - On Reused Test: Only **8 completed trades** were executed over 248 sessions.
- With sample sizes of $N = 4$ and $N = 8$, the $95\%$ Wilson score confidence intervals for trade win rate are extremely wide:
  - Validation: $[15.0\%, 85.0\%]$
  - Reused Test: $[2.2\%, 47.1\%]$
- A sample of 4 to 8 trades provides insufficient statistical power to establish an enduring quantitative edge.

---

### 5. Reused Test Period Status

#### Disclosure
- The 2025–2026 evaluation period has been repeatedly examined across Phases 14 through 19.
- This period must strictly be treated as a **reused exploratory research benchmark**, not an untouched holdout.
- Any future strategy claiming live-readiness must be validated on an untouched, future out-of-sample dataset (e.g., late 2026+ data).

---

### 6. Absence of Directional Hedging / Short Selling
- The system remains strictly long-only delivery equity. When broad indices decline (e.g., $-15.60\%$ Buy & Hold), long-only strategies cannot produce positive nominal returns without sitting entirely in cash. Future research must evaluate market-neutral or dynamic cash-hedging rules.

