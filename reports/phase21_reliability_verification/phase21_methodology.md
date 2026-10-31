# QUANTARA AI — PHASE 21 VERIFICATION METHODOLOGY
## Technical Protocol, Audit Formulations, and Execution Commands

**Project:** Quantara AI — Independent Reliability Verification  
**Auditor:** Quantara AI Lead Auditor  
**Date:** October 2026  
**Audited Target:** Phase 20 Dynamic Cash Allocation Experiments  
**Output Path:** `reports/phase21_reliability_verification/phase21_methodology.md`  

---

### 1. Audit Commands & Execution Environment

The entire Phase 21 verification suite runs deterministically via standard Python.

#### Exact Command to Execute Verification:
```powershell
python scripts/run_phase21_reliability_verification.py
```

#### Execution Environment:
- **Operating System:** Windows 11 Enterprise
- **Python Interpreter:** Python 3.14.0 (x64)
- **Core Dependencies:** `numpy`, `pandas`, `scikit-learn`
- **Data Source:** `data/raw/multi_stock_historical.csv` and `data/splits_v3/horizon_20d/`
- **Zero Live Connection:** 100% offline, isolated execution with no network or live broker hooks.

---

### 2. Accounting Reconciliation Formulations

Every portfolio simulation was evaluated for complete conservation of capital down to ₹0.01 across all closed trades, open equity, fees, dividends, and interest accrual.

#### A. Equity Balance Equation
At the conclusion of trading session $T$, the final portfolio equity $E_T$ must reconcile exactly with the starting capital $E_0$, total net trade profits/losses, and total cash interest accrued:
$$E_T = E_0 + \sum_{j=1}^{N_{\text{trades}}} \text{NetPnL}_j + \text{CashInterest}_{\text{total}}$$
where:
$$\text{NetPnL}_j = \text{GrossProceeds}_j - \text{SellFee}_j - \text{CostBasis}_j + \text{Dividends}_j$$
$$\text{CostBasis}_j = \text{Shares}_j \times P_{\text{entry}, j} + \text{BuyFee}_j$$
$$\text{GrossProceeds}_j = \text{Shares}_j \times P_{\text{exit}, j}$$

#### B. Accounting Residual Check
For every evaluated strategy, the accounting residual $\epsilon$ was computed:
$$\epsilon = \left|E_0 + \sum_{j=1}^{N_{\text{trades}}} \text{NetPnL}_j + \text{CashInterest}_{\text{total}} - E_T\right|$$
The verification test is marked **PASSED** if and only if $\epsilon < 0.001$.

#### C. Return Basis Reconciliation
The return was analyzed under two distinct calculation conventions:
1. **Reported Convention (Day 1 End-of-Day Base):**
   $$R_{\text{reported}} = \left(\frac{E_T}{E_{\text{Day 1 EOD}}} - 1.0\right) \times 100\%$$
2. **Standard Financial Convention (Starting Capital Base):**
   $$R_{\text{standard}} = \left(\frac{E_T}{E_0} - 1.0\right) \times 100\%$$
3. **Mathematical Offset:**
   $$\Delta R = R_{\text{reported}} - R_{\text{standard}} = E_T \times \left(\frac{1}{E_{\text{Day 1 EOD}}} - \frac{1}{E_0}\right) \times 100\%$$

---

### 3. Trade Ledger Verification Formulations

Individual trade blotters were verified by re-parsing every execution event:
1. **Execution Prices:**
   - Standard Buy: $P_{\text{open}} \times (1 + \text{slippage})$
   - Gap-Down Exit: $P_{\text{open}} \times (1 - \text{slippage})$ if $P_{\text{open}} \le P_{\text{stop}}$
   - Intraday Stop Exit: $P_{\text{stop}} \times (1 - \text{slippage})$ if $P_{\text{low}} \le P_{\text{stop}}$
   - Intraday Take-Profit Exit: $P_{\text{target}} \times (1 - \text{slippage})$ if $P_{\text{high}} \ge P_{\text{target}}$
   - Scheduled Exit: $P_{\text{open}} \times (1 - \text{slippage})$ at session $t = \text{entry} + 20$
2. **Total Friction Check:**
   $$\text{Commissions Paid} = \text{Buy Commission} + \text{Sell Commission}$$
   where commissions incorporate STT, NSE turnover fees, SEBI charges, stamp duty, GST, and ₹15.93 flat DP charges on sell orders.

---

### 4. Stock Universe and Data Verification

The universe verification checked:
1. Set identity: $\mathcal{U}_{\text{raw}} = \mathcal{U}_{\text{train}} = \mathcal{U}_{\text{val}} = \mathcal{U}_{\text{test}}$.
2. Duplicate row counts: $\sum \mathbb{I}(\text{Date}_i = \text{Date}_j \land \text{Symbol}_i = \text{Symbol}_j) = 0$.
3. Missing price fields: $\sum \mathbb{I}(\text{Field} = \text{NaN}) = 0$.
4. Price envelope constraint: $\text{Low}_t \le \min(\text{Open}_t, \text{Close}_t) \le \max(\text{Open}_t, \text{Close}_t) \le \text{High}_t$.

