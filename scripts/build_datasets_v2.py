"""
Quantara AI — Phase 3.4: Versioned Dataset Generation (v2 Pipeline)
Execution-Aligned Target & Audited Feature Pipeline

Key Updates in v2:
1. Target: Execution-aligned 5-day open-to-open forward return:
   Adj_Open(t+6) / Adj_Open(t+1) - 1
   where Adj_Open(k) = Open(k) * (Adj Close(k) / Close(k)).
   Signal date: Session t close (features strictly use <= t).
   Entry: Session t+1 open.
   Exit: Session t+6 open (after exactly 5 completed trading sessions).
   Classification threshold: +/- 1.0% (UP: > +0.010, DOWN: < -0.010, HOLD: [-0.010, +0.010]).
2. Feature: Corrected open_gap_pct:
   Adj_Open(t) / Adj_Close(t-1) - 1
   Eliminates artificial ex-dividend gap spikes without leaking future information.
3. Purging: Target spans 6 sessions forward (t to t+6). Any sample whose target end date
   crosses a split boundary (train_cutoff=2024-10-23, val_cutoff=2025-10-21) is purged.
4. Absolute Isolation: Outputs saved strictly to data/processed_v2/ and data/splits_v2/horizon_5d/.
   Never modifies original data/splits/ or data/processed/.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "multi_stock_historical.csv"
OUT = ROOT / "data"
CONFIG = json.loads((ROOT / "config" / "data_config.json").read_text(encoding="utf-8"))
FEATURES = CONFIG["model_input_features"]
REQUIRED = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "Symbol", "Company"]
HORIZON_SESSIONS = 5
TARGET_THRESHOLD = 0.010  # 1.0%


def build_features_and_target_v2(g):
    g = g.sort_values("Date").copy().reset_index(drop=True)
    p = g["Adj Close"].astype(float)
    close, high, low, op, volume = (g[c].astype(float) for c in ["Close", "High", "Low", "Open", "Volume"])
    
    # Technical Indicators (identical stationary definitions from v1)
    daily = p.pct_change()
    sma20, sma50 = p.rolling(20, min_periods=20).mean(), p.rolling(50, min_periods=50).mean()
    ema12 = p.ewm(span=12, adjust=False, min_periods=12).mean()
    ema20 = p.ewm(span=20, adjust=False, min_periods=20).mean()
    ema26 = p.ewm(span=26, adjust=False, min_periods=26).mean()
    delta = p.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi.loc[avg_loss == 0] = 100
    rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    std20 = p.rolling(20, min_periods=20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    band_range = (upper - lower).replace(0, np.nan)
    vol_sma = volume.rolling(20, min_periods=20).mean()
    day_range = high - low

    # Adjusted Open calculation:
    # Adj_Open(k) = Open(k) * (Adj Close(k) / Close(k))
    adj_factor = p / close
    adj_open = op * adj_factor

    # Feature: Corrected open_gap_pct (Session t Adj_Open vs Session t-1 Adj_Close)
    # Both are fully known at Session t close without future leakage.
    open_gap_pct_corrected = adj_open / p.shift(1) - 1

    f = pd.DataFrame({
        "Date": g["Date"],
        "Symbol": g["Symbol"],
        "Company": g["Company"],
        "daily_return_1d": daily,
        "return_5d": p.pct_change(5),
        "return_20d": p.pct_change(20),
        "adj_close_vs_sma20": p / sma20 - 1,
        "adj_close_vs_sma50": p / sma50 - 1,
        "adj_close_vs_ema12": p / ema12 - 1,
        "adj_close_vs_ema20": p / ema20 - 1,
        "adj_close_vs_ema26": p / ema26 - 1,
        "rsi_14": rsi,
        "macd_pct": macd / p,
        "macd_signal_pct": signal / p,
        "macd_histogram_pct": (macd - signal) / p,
        "bollinger_percent_b": (p - lower) / band_range,
        "bollinger_width_pct": band_range / sma20,
        "volatility_20d_annualized": daily.rolling(20, min_periods=20).std() * np.sqrt(252),
        "log_volume_change": np.log1p(volume).diff(),
        "volume_vs_sma20": volume / vol_sma - 1,
        "intraday_range_pct": (high - low) / close,
        "open_gap_pct": open_gap_pct_corrected,
        "close_location_in_range": np.where(day_range > 0, (close - low) / day_range, 0.5),
    })

    # Execution-Aligned Target:
    # Entry at session t+1 Open, exit at session t+6 Open (5 trading sessions held)
    adj_open_next = adj_open.shift(-1)
    adj_open_exit = adj_open.shift(-6)
    f["forward_return_5d"] = adj_open_exit / adj_open_next - 1
    f["target_end_date_5d"] = g["Date"].shift(-6)

    return f


def main():
    print("=" * 70)
    print("QUANTARA AI — PHASE 3.4: VERSIONED DATASET GENERATION (v2)")
    print("=" * 70)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Source CSV missing: {SOURCE}")
    actual_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    expected_hash = CONFIG["source_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"Source hash mismatch. Expected {expected_hash}; got {actual_hash}. Refusing to process.")
    print(f"Verified source CSV SHA-256: {actual_hash}")

    raw = pd.read_csv(SOURCE)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for c in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Process all symbols
    all_features = pd.concat([build_features_and_target_v2(g) for _, g in raw.groupby("Symbol", sort=True)], ignore_index=True)
    all_features = all_features.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # Feature validation
    finite = np.isfinite(all_features[FEATURES].to_numpy(dtype=float)).all(axis=1)
    features_valid = all_features.loc[finite, ["Date", "Symbol", "Company"] + FEATURES].copy()

    # Save processed_v2 directory
    processed_dir = OUT / "processed_v2"
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_valid.to_csv(processed_dir / "multi_stock_features_validated.csv", index=False, float_format="%.10g")

    stock_dir = processed_dir / "features_by_stock"
    stock_dir.mkdir(parents=True, exist_ok=True)
    for symbol, part in features_valid.groupby("Symbol", sort=True):
        slug = symbol.replace(".NS", "").replace(".", "_")
        part.to_csv(stock_dir / f"{slug}_features.csv", index=False, float_format="%.10g")

    # Split boundaries
    unique_dates = pd.Index(sorted(raw["Date"].unique()))
    n = len(unique_dates)
    train_end = pd.Timestamp(unique_dates[int(np.floor(n * 0.80)) - 1])
    validation_end = pd.Timestamp(unique_dates[int(np.floor(n * 0.90)) - 1])
    latest = pd.Timestamp(unique_dates[-1])

    print(f"Split boundaries:")
    print(f"  Train:      <= {train_end.date()}")
    print(f"  Validation: > {train_end.date()} and <= {validation_end.date()}")
    print(f"  Test:       > {validation_end.date()} and <= {latest.date()}")

    # Master audit dataset and splits for 5d horizon
    work = all_features.copy()
    work["target_end_date"] = pd.to_datetime(work["target_end_date_5d"])
    work["forward_return"] = work["forward_return_5d"]
    work["Target"] = np.select(
        [work["forward_return"] > TARGET_THRESHOLD, work["forward_return"] < -TARGET_THRESHOLD],
        ["UP", "DOWN"],
        default="HOLD"
    )

    valid_features = np.isfinite(work[FEATURES].to_numpy(dtype=float)).all(axis=1)
    valid_target = work["forward_return"].notna() & work["target_end_date"].notna()
    work = work.loc[valid_features & valid_target].copy()

    date, end = work["Date"], work["target_end_date"]
    work["split"] = np.select([
        (date <= train_end) & (end <= train_end),
        (date > train_end) & (date <= validation_end) & (end <= validation_end),
        (date > validation_end) & (date <= latest) & (end <= latest)
    ], ["train", "validation", "test"], default="purged")

    # Master audit table
    master_cols = ["Date", "Symbol", "Company"] + FEATURES + ["forward_return", "Target", "target_end_date", "split"]
    work[master_cols].rename(columns={
        "forward_return": "forward_return_5d",
        "Target": "Target_5d",
        "target_end_date": "target_end_date_5d",
    }).to_csv(processed_dir / "master_dataset_5d_audit_only.csv", index=False, float_format="%.10g")

    label_dir = processed_dir / "labels_by_horizon"
    label_dir.mkdir(parents=True, exist_ok=True)
    work[["Date", "Symbol", "Company", "forward_return", "Target"]].rename(
        columns={"forward_return": "forward_return_5d", "Target": "Target_5d"}
    ).to_csv(label_dir / "labels_5d.csv", index=False, float_format="%.10g")

    # Save to data/splits_v2/horizon_5d/
    splits_dir = OUT / "splits_v2" / "horizon_5d"
    splits_dir.mkdir(parents=True, exist_ok=True)

    summary_counts = {}
    class_distributions = {}

    for split_name in ["train", "validation", "test"]:
        part = work.loc[work["split"] == split_name].sort_values(["Date", "Symbol"]).reset_index(drop=True)
        part[FEATURES].to_csv(splits_dir / f"X_{split_name}.csv", index=False, float_format="%.10g")
        part[["forward_return", "Target"]].rename(
            columns={"forward_return": "forward_return_5d", "Target": "Target_5d"}
        ).to_csv(splits_dir / f"y_{split_name}.csv", index=False, float_format="%.10g")
        part[["Date", "Symbol", "Company"]].to_csv(splits_dir / f"metadata_{split_name}.csv", index=False)
        
        assert len(part[FEATURES]) == len(part[["forward_return", "Target"]]) == len(part[["Date", "Symbol", "Company"]]), \
            f"Row count mismatch in {split_name}"

        summary_counts[split_name] = len(part)
        class_counts = part["Target"].value_counts().to_dict()
        class_distributions[split_name] = {
            cls: {"count": int(class_counts.get(cls, 0)), "pct": float(class_counts.get(cls, 0) / len(part) * 100.0)}
            for cls in ["DOWN", "HOLD", "UP"]
        }

    purged_part = work.loc[work["split"] == "purged", ["Date", "Symbol", "Company", "forward_return", "Target", "target_end_date"]].copy()
    purged_part.to_csv(splits_dir / "purged_boundary_rows.csv", index=False, float_format="%.10g")
    summary_counts["purged"] = len(purged_part)

    print("\nDataset split summary (v2):")
    for s, cnt in summary_counts.items():
        print(f"  {s:12s}: {cnt:6d} samples")

    print("\nTarget class distributions (v2):")
    for s, dist in class_distributions.items():
        print(f"  {s.upper()}:")
        for cls, d in dist.items():
            print(f"    {cls:4s}: {d['count']:5d} ({d['pct']:5.2f}%)")

    dataset_summary = {
        "pipeline_version": "v2_execution_aligned",
        "source_sha256": actual_hash,
        "features": FEATURES,
        "sample_counts": summary_counts,
        "class_distributions": class_distributions,
        "cutoffs": {
            "train_end": str(train_end.date()),
            "validation_end": str(validation_end.date()),
            "latest": str(latest.date()),
        }
    }
    with open(processed_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    print(f"\nAll v2 datasets successfully generated in {splits_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

