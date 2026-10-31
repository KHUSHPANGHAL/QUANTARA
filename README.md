# Quantara AI — audited data preparation package

## Project Overview & Institutional Documentation
Quantara AI is an experimental quantitative research project developing and evaluating machine-learning and deep-learning models (Logistic Regression, LSTM, and GRU) for trend prediction and systematic trading in Indian equities (8 liquid NSE blue-chip stocks).

Comprehensive, plain-English documentation explaining the research phases, methodology, audit discoveries, and current status is available in the dedicated project explainer suite:
- **Project Documentation Suite:** [`docs/project_explainer/README.md`](docs/project_explainer/README.md)
  - [`project_history.md`](docs/project_explainer/project_history.md): Full narrative of Phases 1 through 26.
  - [`current_status.md`](docs/project_explainer/current_status.md): Verified findings, known limitations, and unresolved challenges.
  - [`next_steps.md`](docs/project_explainer/next_steps.md): Roadmap for Phase 27 (hypothetical ₹1,00,000 backtest).
- **Master Explainer PDF:** `docs/project_explainer/Quantara_AI_Full_Project_Explained.pdf` *(Note: Must be placed into this folder manually if downloaded externally).*
- **Current Research Status:** Phase 26 true walk-forward retraining completed across 4,647 out-of-sample observations (confirming that deep recurrent networks do not outperform simple linear models). Phase 27 portfolio P&L backtesting is currently scheduled.
- **Risk & Operational Warning:** All results represent historical research simulations. Past performance does not guarantee future results. **Live trading is strictly disabled, and no broker connections or automated order routing exist.**

---

## What this package is
This package rebuilds model inputs and future-return labels from the historical CSV supplied in `AI-Stock-Market-Project-main(2).zip`. It does **not** download, invent, interpolate, or randomly generate market observations. Original supplied files are preserved under `source_snapshot/`; the raw multi-stock CSV is also kept at `data/raw/multi_stock_historical.csv`.

## Stocks included
- BHARTIARTL.NS
- HDFCBANK.NS
- ICICIBANK.NS
- INFY.NS
- ITC.NS
- RELIANCE.NS
- SBIN.NS
- TCS.NS

The source file has 19,776 rows (2,472 rows per symbol), with dates from 2016-10-24 through 2026-10-23. These facts are measured from the uploaded file, not independently confirmed against an exchange.

## Folder guide
- `source_snapshot/`: unmodified copies of key source CSVs/reports from the uploaded repository.
- `data/raw/multi_stock_historical.csv`: source input used by the build script.
- `data/processed/multi_stock_features_validated.csv`: feature table with Date/Symbol/Company metadata.
- `data/processed/features_by_stock/`: feature tables separated by symbol.
- `data/processed/labels_by_horizon/`: forward-return and classification labels for 1, 5, and 20 observed trading sessions.
- `data/processed/master_dataset_*_audit_only.csv`: convenient joined tables for inspection only. **Do not pass every column in these files directly to a model.**
- `data/splits/horizon_1d/`, `horizon_5d/`, `horizon_20d/`: model-ready `X_*` and `y_*` files, plus metadata.
- `data/reports/`: data audit, feature dictionary, and automated validation report.
- `scripts/build_datasets.py`: reproducible build script. It checks the source CSV SHA-256 before processing.
- `scripts/validate_outputs.py`: automated checks for schema, finite inputs, label correctness, row alignment, and split boundaries.
- `config/data_config.json`: source hash, feature list, split dates, and label thresholds.

## Correct training usage
For each horizon, use:
- `X_train.csv` + `y_train.csv` for model fitting.
- `X_validation.csv` + `y_validation.csv` for model selection and threshold/hyperparameter tuning.
- `X_test.csv` + `y_test.csv` once for final evaluation.

The X files contain **only 20 numeric input features**. They do not contain `Target`, future returns, dates, symbols, company names, or sample IDs. X and y rows align by row order; the corresponding `metadata_*.csv` has the same row order and supplies Date/Symbol/Company for audit. Never shuffle rows independently across X, y, and metadata.

## Target definitions
For horizon h, the forward return is:
`Adj Close[t+h] / Adj Close[t] - 1`, where t+h means h subsequent rows for the same stock (observed sessions, not calendar days).

Initial UP/DOWN/HOLD thresholds are configurable assumptions:
- 1 session: +0.5% / -0.5%
- 5 sessions: +1.0% / -1.0%
- 20 sessions: +2.0% / -2.0%

A return above the positive threshold is UP; below the negative threshold is DOWN; otherwise HOLD. These are **not externally authenticated financial standards**. Choose/tune them using training/validation data only; do not tune against the test set.

## Split policy
Global chronological cutoffs are shared by all symbols:
- Train cutoff inclusive: 2024-10-23
- Validation cutoff inclusive: 2025-10-21
- Latest source date: 2026-10-23

A row is included in a split only if its future target endpoint remains within that split's date window. Boundary-crossing rows are written to `purged_boundary_rows.csv`, not assigned to a split. The last h observations per stock without an observed future target are excluded for that horizon.

## Rebuild and verify
Use Python with pandas and numpy installed:
```bash
python scripts/build_datasets.py
python scripts/validate_outputs.py
```
The build script deliberately stops if the source file hash differs from the one audited. If you intentionally replace the source file, re-audit it and update the config hash explicitly; do not bypass the check silently.

## Validation results
The included automated report records 75 checks passed, 0 failed at package creation time. Run the validation script again after rebuilding or editing any files.

## Important limitations
1. **No synthetic market observations were added.** All raw market rows originate in the user-supplied CSV.
2. **Authenticity is not the same as structural validity.** This environment did not independently compare historical prices against NSE/BSE or another trusted data provider. Therefore this package cannot certify the source prices as externally authenticated.
3. The source contains `Adj Close` and unadjusted OHLC columns. Returns and technical indicators are based on supplied `Adj Close`; intraday range/location and open-gap features use supplied OHLC. Confirm the provider's corporate-action methodology before making financial claims.
4. Thresholds for 5- and 20-session classification are explicitly configurable modelling assumptions, not facts retrieved from an authoritative source.
5. This package prepares data only. It does not claim that a model has been trained or that any predictive performance has been achieved.
6. Market data can have provider errors or retrospective adjustments. Passing these checks does not guarantee future market predictability or eliminate all possible leakage. Review feature definitions before adding new features.
