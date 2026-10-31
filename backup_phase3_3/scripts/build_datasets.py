from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "multi_stock_historical.csv"
OUT = ROOT / "data"
CONFIG = json.loads((ROOT / "config" / "data_config.json").read_text(encoding="utf-8"))
HORIZONS = CONFIG["horizons_trading_sessions"]
THRESHOLDS = {int(k.replace("d", "")): v for k, v in CONFIG["classification_thresholds"].items()}
FEATURES = CONFIG["model_input_features"]
REQUIRED = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "Symbol", "Company"]


def build_features(g):
    g = g.sort_values("Date").copy().reset_index(drop=True)
    p = g["Adj Close"].astype(float)
    close, high, low, op, volume = (g[c].astype(float) for c in ["Close", "High", "Low", "Open", "Volume"])
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
    prev_close = close.shift(1)
    day_range = high - low

    f = pd.DataFrame({
        "Date": g["Date"], "Symbol": g["Symbol"], "Company": g["Company"],
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
        "open_gap_pct": op / prev_close - 1,
        "close_location_in_range": np.where(day_range > 0, (close - low) / day_range, 0.5),
    })
    for h in HORIZONS:
        f[f"forward_return_{h}d"] = p.shift(-h) / p - 1
        f[f"target_end_date_{h}d"] = g["Date"].shift(-h)
    return f


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source CSV missing: {SOURCE}")
    actual_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    expected_hash = CONFIG["source_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"Source hash mismatch. Expected {expected_hash}; got {actual_hash}. Refusing to silently process changed data.")
    raw = pd.read_csv(SOURCE)
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for c in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if raw[REQUIRED].isna().any().any():
        raise ValueError("Missing/invalid values in required source columns. Review source; this script will not invent replacements.")
    if raw.duplicated(["Symbol", "Date"]).any():
        raise ValueError("Duplicate Symbol-Date rows; resolve explicitly before processing.")
    if (raw[["Adj Close", "Close", "High", "Low", "Open"]] <= 0).any().any():
        raise ValueError("Non-positive price values found.")
    if (raw["Volume"] < 0).any():
        raise ValueError("Negative volume values found.")
    bad_ohlc = (raw["High"] < raw[["Open", "Low", "Close"]].max(axis=1)) | (raw["Low"] > raw[["Open", "High", "Close"]].min(axis=1))
    if bad_ohlc.any():
        raise ValueError(f"{int(bad_ohlc.sum())} rows violate OHLC consistency.")
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    if not all(g["Date"].is_monotonic_increasing for _, g in raw.groupby("Symbol")):
        raise ValueError("Dates not monotonic within one or more symbols.")

    all_features = pd.concat([build_features(g) for _, g in raw.groupby("Symbol", sort=True)], ignore_index=True)
    all_features = all_features.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    finite = np.isfinite(all_features[FEATURES].to_numpy(dtype=float)).all(axis=1)
    # Keep the warm-up rule explicit; labels are kept separately from X.
    features = all_features.loc[finite, ["Date", "Symbol", "Company"] + FEATURES].copy()
    features.to_csv(OUT / "processed" / "multi_stock_features_validated.csv", index=False, float_format="%.10g")
    stock_dir = OUT / "processed" / "features_by_stock"
    stock_dir.mkdir(parents=True, exist_ok=True)
    for symbol, part in features.groupby("Symbol", sort=True):
        slug = symbol.replace(".NS", "").replace(".", "_")
        part.to_csv(stock_dir / f"{slug}_features.csv", index=False, float_format="%.10g")

    unique_dates = pd.Index(sorted(raw["Date"].unique()))
    n = len(unique_dates)
    train_end = pd.Timestamp(unique_dates[int(np.floor(n * 0.80)) - 1])
    validation_end = pd.Timestamp(unique_dates[int(np.floor(n * 0.90)) - 1])
    latest = pd.Timestamp(unique_dates[-1])
    summaries = []
    for h in HORIZONS:
        return_col = f"forward_return_{h}d"
        end_col = f"target_end_date_{h}d"
        target_col = f"Target_{h}d"
        work = all_features.copy()
        work["target_end_date"] = pd.to_datetime(work[end_col])
        work["forward_return"] = work[return_col]
        work["Target"] = np.select(
            [work["forward_return"] > THRESHOLDS[h], work["forward_return"] < -THRESHOLDS[h]],
            ["UP", "DOWN"], default="HOLD")
        valid_features = np.isfinite(work[FEATURES].to_numpy(dtype=float)).all(axis=1)
        valid_target = work["forward_return"].notna() & work["target_end_date"].notna()
        work = work.loc[valid_features & valid_target].copy()
        date, end = work["Date"], work["target_end_date"]
        work["split"] = np.select([
            (date <= train_end) & (end <= train_end),
            (date > train_end) & (date <= validation_end) & (end <= validation_end),
            (date > validation_end) & (date <= latest) & (end <= latest)
        ], ["train", "validation", "test"], default="purged")
        # Joined audit table is for inspection only; never use all its columns as model inputs.
        master_cols = ["Date", "Symbol", "Company"] + FEATURES + ["forward_return", "Target", "target_end_date", "split"]
        work[master_cols].rename(columns={
            "forward_return": return_col,
            "Target": target_col,
            "target_end_date": end_col,
        }).to_csv(OUT / "processed" / f"master_dataset_{h}d_audit_only.csv", index=False, float_format="%.10g")
        label_dir = OUT / "processed" / "labels_by_horizon"
        label_dir.mkdir(parents=True, exist_ok=True)
        work[["Date", "Symbol", "Company", "forward_return", "Target"]].rename(
            columns={"forward_return": return_col, "Target": target_col}
        ).to_csv(label_dir / f"labels_{h}d.csv", index=False, float_format="%.10g")
        split_dir = OUT / "splits" / f"horizon_{h}d"
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ["train", "validation", "test"]:
            part = work.loc[work["split"] == split_name].sort_values(["Date", "Symbol"]).reset_index(drop=True)
            part[FEATURES].to_csv(split_dir / f"X_{split_name}.csv", index=False, float_format="%.10g")
            part[["forward_return", "Target"]].rename(
                columns={"forward_return": return_col, "Target": target_col}
            ).to_csv(split_dir / f"y_{split_name}.csv", index=False, float_format="%.10g")
            part[["Date", "Symbol", "Company"]].to_csv(split_dir / f"metadata_{split_name}.csv", index=False)
            if not (len(part[FEATURES]) == len(part[["forward_return", "Target"]]) == len(part[["Date", "Symbol", "Company"]])):
                raise AssertionError("X/y/metadata row count mismatch")
        work.loc[work["split"] == "purged", ["Date", "Symbol", "Company", "forward_return", "Target", "target_end_date"]].to_csv(
            split_dir / "purged_boundary_rows.csv", index=False, float_format="%.10g")
        summaries.append({"horizon": h, "train": int((work.split == "train").sum()),
                          "validation": int((work.split == "validation").sum()),
                          "test": int((work.split == "test").sum()),
                          "purged": int((work.split == "purged").sum())})
    print("Dataset build completed from supplied historical rows only.")
    print(f"Training cutoff: {train_end.date()}, validation cutoff: {validation_end.date()}, last date: {latest.date()}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
