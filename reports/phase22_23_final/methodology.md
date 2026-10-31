# QUANTARA AI — PHASE 22–23 METHODOLOGY & REPRODUCIBILITY
## Audit Commands, Mathematical Formulations, and Execution Architecture

**Auditor / Researcher:** Quantara AI Lead Auditor  
**Date:** October 2026  
**Path:** `reports/phase22_23_final/methodology.md`  

---

### 1. Execution Commands & Environment

The entire unified Phase 22–23 audit is deterministic and fully reproducible with a single script execution:

```powershell
# Execute complete unified verification and generate all deliverables
python scripts/run_phase22_23_unified_verification.py
```

#### System Environment:
- **Operating System:** Windows 11 Enterprise (x64)
- **Interpreter:** Python 3.14.0
- **Primary Libraries:** `numpy`, `pandas`, `scikit-learn`
- **Isolation:** 100% offline, local execution with no live broker connectivity.

---

### 2. Standardized Mathematical Formulations

#### A. Standard Portfolio Return
$$\text{Return}_{\text{standard}} = \left(\frac{E_{\text{Final}} - \text{₹1,00,000.00}}{\text{₹1,00,000.00}}\right) \times 100\%$$

#### B. Complete Accounting Balance Identity
$$E_{\text{Final}} = E_{\text{Starting}} + \sum_{j=1}^{N_{\text{trades}}} \text{NetPnL}_j + \text{Interest}_{\text{accrued}}$$
where:
$$\text{NetPnL}_j = \text{Gross Proceeds}_j - \text{Sell Fee}_j - \text{Cost Basis}_j + \text{Dividends}_j$$
$$\text{Cost Basis}_j = \text{Shares}_j \times P_{\text{entry}, j} + \text{Buy Fee}_j$$
$$\text{Gross Proceeds}_j = \text{Shares}_j \times P_{\text{exit}, j}$$

#### C. Maximum Drawdown Calculation
$$\text{Peak}_t = \max_{1 \le \tau \le t} E_\tau$$
$$\text{Drawdown}_t = \frac{E_t - \text{Peak}_t}{\text{Peak}_t}$$
$$\text{Max Drawdown} = \max_{1 \le t \le T} \left|\text{Drawdown}_t\right| \times 100\%$$

#### D. Authentic Mutual Fund NAV Returns
For evaluation window $[t_0, t_1]$ using authentic NAV observations:
$$R_{\text{gross}} = \left(\frac{NAV_{t_1}}{NAV_{t_0}} - 1.0\right) \times 100\%$$
$$\text{Annualized Yield} = \left[\left(\frac{NAV_{t_1}}{NAV_{t_0}}\right)^{\frac{365.25}{\Delta \text{days}}} - 1.0\right] \times 100\%$$
$$R_{\text{net, 20\%}} = R_{\text{gross}} \times (1.0 - 0.20) \quad \text{(Standard Domestic Rate)}$$
$$R_{\text{net, 39\%}} = R_{\text{gross}} \times (1.0 - 0.39) \quad \text{(Highest Marginal Tax Slab)}$$

#### E. Wilson Score Confidence Interval for Win Rates
$$\hat{p} = \frac{w}{n}, \quad z = 1.959964$$
$$CI_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

