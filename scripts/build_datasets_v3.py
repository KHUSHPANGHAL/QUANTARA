"""
Quantara AI — Phase 3 & 4: Multi-Horizon & Feature-Engineered Research Pipeline (v3)
Horizons: 1-Day, 5-Day, and 20-Day Forward Open-to-Open Returns
Target Formulations: 3-Class Categorical & Binary (Excluding Noise Threshold)

Key Methodological Guarantees:
1. Exact Execution Alignment:
   - Signal: Session t close (3:30 PM IST)
   - Entry: Session t+1 morning open
   - Exit: Session t+1+H morning open (where H = 1, 5, or 20 completed sessions)
   - Realized Return: Adj_Open(t+1+H) / Adj_Open(t+1) - 1
2. Boundary Purging:
   - Samples near train/validation cutoffs where t+1+H crosses into the next partition are purged.
3. Feature Engineering (4 Structured Groups):
   - Group 1: Core Trend & Multi-Window Momentum (1d, 5d, 10d, 20d returns, SMA/EMA ratios, RSI, MACD)
   - Group 2: Volatility, Intraday Range & Rolling Drawdown (10d/20d vol, 20d max drawdown, range)
   - Group 3: Volume & Liquidity (volume/SMA, log volume diff, 5d volume-price momentum)
   - Group 4: Market Breadth & Macro Regime (market breadth % > SMA20, avg market return, market vol)
   - ALL strictly derived from prices <= session t close.
4. Storage:
   - data/processed_v3/
   - data/splits_v3/horizon_1d/, horizon_5d/, horizon_20d/
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

HORIZONS = [1, 5, 20]
THRESHOLDS = {1: 0.005, 5: 0.010, 20: 0.020}  # +/-0.5% for 1d, +/-1.0% for 5d, +/-2.0% for 20d


def build_stock_features_v3(g):
    """Compute individual stock technical, momentum, volatility, and volume indicators."""
    g = g.sort_values("Date").copy().reset_index(drop=True)
    p = g["Adj Close"].astype(float)
    close, high, low, op, volume = (g[c].astype(float) for c in ["Close", "High", "Low", "Open", "Volume"])
    
    # Core Trend & Moving Averages
    daily = p.pct_change()
    ret_5d = p.pct_change(5)
    ret_10d = p.pct_change(10)
    ret_20d = p.pct_change(20)
    
    sma20 = p.rolling(20, min_periods=20).mean()
    sma50 = p.rolling(50, min_periods=50).mean()
    ema12 = p.ewm(span=12, adjust=False, min_periods=12).mean()
    ema20 = p.ewm(span=20, adjust=False, min_periods=20).mean()
    ema26 = p.ewm(span=26, adjust=False, min_periods=26).mean()
    
    # RSI 14
    delta = p.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi.loc[avg_loss == 0] = 100
    rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50
    
    # MACD
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    
    # Bollinger Bands
    std20 = p.rolling(20, min_periods=20).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    band_range = (upper - lower).replace(0, np.nan)
    
    # Volatilities & Drawdown
    vol10 = daily.rolling(10, min_periods=10).std() * np.sqrt(252)
    vol20 = daily.rolling(20, min_periods=20).std() * np.sqrt(252)
    rolling_peak_20 = p.rolling(20, min_periods=20).max()
    rolling_dd_20 = (p - rolling_peak_20) / rolling_peak_20
    
    # Volume Dynamics
    vol_sma = volume.rolling(20, min_periods=20).mean()
    vpt_5d = ((p - p.shift(5)) / p.shift(5)) * (volume / vol_sma)
    
    # Execution Open & Gap
    adj_factor = p / close
    adj_open = op * adj_factor
    open_gap_pct_corrected = adj_open / p.shift(1) - 1
    day_range = high - low
    
    f = pd.DataFrame({
        "Date": g["Date"],
        "Symbol": g["Symbol"],
        "Company": g["Company"],
        "Close": close,
        "Adj_Close": p,
        "Adj_Open": adj_open,
        # Feature Group 1: Trend & Momentum
        "daily_return_1d": daily,
        "return_5d": ret_5d,
        "return_10d": ret_10d,
        "return_20d": ret_20d,
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
        # Feature Group 2: Volatility & Range
        "volatility_10d_annualized": vol10,
        "volatility_20d_annualized": vol20,
        "rolling_drawdown_20d": rolling_dd_20,
        "intraday_range_pct": (high - low) / close,
        "open_gap_pct": open_gap_pct_corrected,
        "close_location_in_range": np.where(day_range > 0, (close - low) / day_range, 0.5),
        # Feature Group 3: Volume & Liquidity
        "log_volume_change": np.log1p(volume).diff(),
        "volume_vs_sma20": volume / vol_sma - 1,
        "volume_price_trend_5d": vpt_5d,
    })
    
    # Compute Forward Execution Targets for each horizon H
    adj_open_next = adj_open.shift(-1)
    for h in HORIZONS:
        adj_open_exit = adj_open.shift(-(h + 1))
        f[f"forward_return_{h}d"] = adj_open_exit / adj_open_next - 1
        f[f"target_end_date_{h}d"] = g["Date"].shift(-(h + 1))
        
    return f


def compute_market_regime_features(df):
    """
    Compute market-wide aggregate indicators strictly across session t close:
    - market_avg_return_1d: average 1d return across 8 large-cap stocks
    - market_breadth_above_sma20: proportion of stocks above their 20d SMA
    - market_avg_volatility_20d: average annualized 20d volatility
    """
    df["above_sma20"] = (df["adj_close_vs_sma20"] > 0).astype(float)
    
    daily_stats = df.groupby("Date").agg({
        "daily_return_1d": "mean",
        "above_sma20": "mean",
        "volatility_20d_annualized": "mean"
    }).rename(columns={
        "daily_return_1d": "market_avg_return_1d",
        "above_sma20": "market_breadth_above_sma20",
        "volatility_20d_annualized": "market_avg_volatility_20d"
    }).reset_index()
    
    # Market Regime Label (derived strictly from past information):
    # UPTREND if breadth > 0.50 and market_avg_return_1d > -0.002
    # DOWNTREND if breadth < 0.35
    # SIDEWAYS otherwise
    daily_stats["market_regime"] = np.select(
        [
            (daily_stats["market_breadth_above_sma20"] >= 0.50),
            (daily_stats["market_breadth_above_sma20"] <= 0.30)
        ],
        ["BULL_REGIME", "BEAR_REGIME"],
        default="NEUTRAL_REGIME"
    )
    
    df = df.merge(daily_stats, on="Date", how="left")
    return df


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 3 & 4: MULTI-HORIZON & REGIME DATASET PIPELINE (v3)")
    print("=" * 80)

    if not SOURCE.exists():
        raise FileNotFoundError(f"Source CSV missing: {SOURCE}")
    actual_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    expected_hash = CONFIG["source_sha256"]
    if actual_hash != expected_hash:
        raise ValueError("Source hash mismatch!")
    print(f"Verified source CSV SHA-256: {actual_hash}")

    raw = pd.read_csv(SOURCE)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    for c in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # 1. Build Stock Features
    print("\n[Step 1/4] Computing technical and volume indicators across 8 stocks...")
    all_features = pd.concat([build_stock_features_v3(g) for _, g in raw.groupby("Symbol", sort=True)], ignore_index=True)
    all_features = all_features.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # 2. Add Market Regime & Cross-Stock Breadth Features
    print("[Step 2/4] Computing market breadth and regime features...")
    all_features = compute_market_regime_features(all_features)

    # Feature List
    FEATURE_COLS = [
        # Group 1: Momentum & Trend (15)
        "daily_return_1d", "return_5d", "return_10d", "return_20d",
        "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
        "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
        "macd_pct", "macd_signal_pct", "macd_histogram_pct",
        "bollinger_percent_b", "bollinger_width_pct",
        # Group 2: Volatility & Range (6)
        "volatility_10d_annualized", "volatility_20d_annualized", "rolling_drawdown_20d",
        "intraday_range_pct", "open_gap_pct", "close_location_in_range",
        # Group 3: Volume & Liquidity (3)
        "log_volume_change", "volume_vs_sma20", "volume_price_trend_5d",
        # Group 4: Market Breadth & Macro Dynamics (3)
        "market_avg_return_1d", "market_breadth_above_sma20", "market_avg_volatility_20d"
    ]
    print(f"Total stationary features generated: {len(FEATURE_COLS)}")

    # Save to data/processed_v3/
    processed_dir = OUT / "processed_v3"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    finite_mask = np.isfinite(all_features[FEATURE_COLS].to_numpy(dtype=float)).all(axis=1)
    all_features_clean = all_features.loc[finite_mask].copy().reset_index(drop=True)
    all_features_clean[["Date", "Symbol", "Company", "market_regime"] + FEATURE_COLS].to_csv(
        processed_dir / "multi_stock_features_v3_validated.csv", index=False, float_format="%.10g"
    )

    # Split boundaries
    unique_dates = pd.Index(sorted(raw["Date"].unique()))
    n = len(unique_dates)
    train_end = pd.Timestamp(unique_dates[int(np.floor(n * 0.80)) - 1])      # 2024-10-23
    validation_end = pd.Timestamp(unique_dates[int(np.floor(n * 0.90)) - 1]) # 2025-10-21
    latest = pd.Timestamp(unique_dates[-1])                                 # 2026-10-23

    print(f"\nSplit cutoffs:")
    print(f"  Train:      <= {train_end.date()}")
    print(f"  Validation: > {train_end.date()} and <= {validation_end.date()}")
    print(f"  Test:       > {validation_end.date()} and <= {latest.date()}")

    # 3. Build Horizon Splits (1d, 5d, 20d) for both 3-Class and Binary
    pipeline_summary = {}

    for h in HORIZONS:
        print(f"\n[Step 3/4] Processing Horizon {h}d (Threshold: +/-{THRESHOLDS[h]*100:.1f}%)...")
        splits_dir = OUT / "splits_v3" / f"horizon_{h}d"
        splits_dir.mkdir(parents=True, exist_ok=True)

        ret_col = f"forward_return_{h}d"
        end_col = f"target_end_date_{h}d"
        th = THRESHOLDS[h]

        work = all_features_clean.copy()
        work["forward_return"] = work[ret_col]
        work["target_end_date"] = pd.to_datetime(work[end_col])

        # 3-Class Target: DOWN, HOLD, UP
        work[f"Target_{h}d"] = np.select(
            [work["forward_return"] > th, work["forward_return"] < -th],
            ["UP", "DOWN"],
            default="HOLD"
        )

        # Binary Target: UP vs DOWN (with 'EXCLUDE_HOLD' for noise instances)
        work[f"Target_Binary_{h}d"] = np.select(
            [work["forward_return"] > th, work["forward_return"] < -th],
            ["UP", "DOWN"],
            default="EXCLUDE_HOLD"
        )

        valid_sample = work["forward_return"].notna() & work["target_end_date"].notna()
        work = work.loc[valid_sample].copy()

        date, end = work["Date"], work["target_end_date"]
        work["split"] = np.select([
            (date <= train_end) & (end <= train_end),
            (date > train_end) & (date <= validation_end) & (end <= validation_end),
            (date > validation_end) & (date <= latest) & (end <= latest)
        ], ["train", "validation", "test"], default="purged")

        # Save 3-Class Splits
        horizon_counts = {}
        for s in ["train", "validation", "test"]:
            part = work.loc[work["split"] == s].sort_values(["Date", "Symbol"]).reset_index(drop=True)
            part[FEATURE_COLS].to_csv(splits_dir / f"X_{s}.csv", index=False, float_format="%.10g")
            
            y_df = part[["forward_return", f"Target_{h}d", f"Target_Binary_{h}d"]].rename(
                columns={"forward_return": ret_col}
            )
            y_df.to_csv(splits_dir / f"y_{s}.csv", index=False, float_format="%.10g")
            
            meta_cols = ["Date", "Symbol", "Company", "market_regime"]
            part[meta_cols].to_csv(splits_dir / f"metadata_{s}.csv", index=False)
            
            horizon_counts[s] = len(part)

        purged_df = work.loc[work["split"] == "purged", ["Date", "Symbol", "Company", "forward_return", f"Target_{h}d", "target_end_date"]].copy()
        purged_df.to_csv(splits_dir / "purged_boundary_rows.csv", index=False, float_format="%.10g")
        horizon_counts["purged"] = len(purged_df)

        print(f"  Horizon {h}d sample counts: Train={horizon_counts['train']}, Val={horizon_counts['validation']}, Test={horizon_counts['test']}, Purged={horizon_counts['purged']}")
        pipeline_summary[f"horizon_{h}d"] = horizon_counts

    # Master summary
    with open(processed_dir / "dataset_v3_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v3_multi_horizon_regime",
            "source_sha256": actual_hash,
            "feature_count": len(FEATURE_COLS),
            "features": FEATURE_COLS,
            "horizons": HORIZONS,
            "thresholds": THRESHOLDS,
            "pipeline_summary": pipeline_summary
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3 & 4 DATASET REGENERATION COMPLETED SUCCESSFULLY")
    print(f"Datasets written to: {OUT / 'splits_v3'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
