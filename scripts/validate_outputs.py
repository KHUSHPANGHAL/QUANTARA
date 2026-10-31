from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "data_config.json").read_text(encoding="utf-8"))
FEATURES = CONFIG["model_input_features"]
THRESHOLDS = {int(k.replace("d", "")): v for k, v in CONFIG["classification_thresholds"].items()}
SOURCE = pd.read_csv(ROOT / "data" / "raw" / "multi_stock_historical.csv", parse_dates=["Date"])
failures = []
checks = []

def check(name, condition, detail=""):
    checks.append((name, bool(condition), detail))
    if not condition:
        failures.append(name)

check("source_has_no_duplicate_symbol_dates", not SOURCE.duplicated(["Symbol", "Date"]).any())
check("source_required_values_present", not SOURCE[["Date","Adj Close","Close","High","Low","Open","Volume","Symbol","Company"]].isna().any().any())
check("source_rows_not_synthetic_claimed", CONFIG.get("no_synthetic_data") is True, "Build uses supplied source rows only.")
for h in CONFIG["horizons_trading_sessions"]:
    d = ROOT / "data" / "splits" / f"horizon_{h}d"
    for s in ["train", "validation", "test"]:
        X = pd.read_csv(d / f"X_{s}.csv")
        y = pd.read_csv(d / f"y_{s}.csv")
        m = pd.read_csv(d / f"metadata_{s}.csv", parse_dates=["Date"])
        check(f"{h}d_{s}_X_y_metadata_row_counts_match", len(X) == len(y) == len(m))
        check(f"{h}d_{s}_X_has_only_declared_features", list(X.columns) == FEATURES)
        check(f"{h}d_{s}_X_all_numeric_finite", np.isfinite(X.to_numpy(dtype=float)).all())
        target_col = f"Target_{h}d"
        return_col = f"forward_return_{h}d"
        check(f"{h}d_{s}_labels_have_expected_columns", list(y.columns) == [return_col, target_col])
        check(f"{h}d_{s}_labels_are_not_in_X", not any("target" in c.lower() or "forward_return" in c.lower() for c in X.columns))
        check(f"{h}d_{s}_target_classes_valid", set(y[target_col].dropna().unique()).issubset({"UP","DOWN","HOLD"}))
        threshold = THRESHOLDS[h]
        expected = np.select([y[return_col] > threshold, y[return_col] < -threshold], ["UP","DOWN"], default="HOLD")
        check(f"{h}d_{s}_class_labels_match_configured_thresholds", np.array_equal(expected, y[target_col].to_numpy()))
    # Verify no split target-endpoint crosses its split boundary.
    raw_dates = pd.Index(sorted(SOURCE["Date"].unique()))
    n = len(raw_dates)
    train_end = pd.Timestamp(raw_dates[int(np.floor(n * 0.80)) - 1])
    val_end = pd.Timestamp(raw_dates[int(np.floor(n * 0.90)) - 1])
    for s in ["train", "validation", "test"]:
        m = pd.read_csv(d / f"metadata_{s}.csv", parse_dates=["Date"])
        y = pd.read_csv(d / f"y_{s}.csv")
        joined = m.copy()
        joined["ret"] = y[f"forward_return_{h}d"].values
        # compute endpoint date per symbol from source's observed session order
        end_map = {}
        for sym, g in SOURCE.groupby("Symbol", sort=True):
            g = g.sort_values("Date").reset_index(drop=True)
            pos = {dt:i for i,dt in enumerate(g["Date"])}
            end_map[sym] = (g["Date"].tolist(), pos)
        enddates = []
        for _, row in joined.iterrows():
            dates, pos = end_map[row["Symbol"]]
            idx = pos[row["Date"]]
            enddates.append(dates[idx+h])
        joined["end_date"] = enddates
        if s == "train":
            ok = (joined["Date"] <= train_end).all() and (joined["end_date"] <= train_end).all()
        elif s == "validation":
            ok = (joined["Date"] > train_end).all() and (joined["Date"] <= val_end).all() and (joined["end_date"] <= val_end).all()
        else:
            ok = (joined["Date"] > val_end).all() and (joined["end_date"] <= raw_dates[-1]).all()
        check(f"{h}d_{s}_chronological_and_target_boundary_safe", ok, f"train cutoff {train_end.date()}, validation cutoff {val_end.date()}")
report = ["QUANTARA AI OUTPUT VALIDATION", "="*32]
for name, ok, detail in checks:
    report.append(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
report.append("")
report.append(f"Checks passed: {sum(ok for _,ok,_ in checks)} / {len(checks)}")
report.append(f"Failures: {len(failures)}")
(ROOT / "data" / "reports" / "output_validation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
if failures:
    raise SystemExit("Validation failed: " + ", ".join(failures))
