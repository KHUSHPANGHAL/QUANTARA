"""
Quantara AI — Interactive Prediction Tool (Beginner Friendly)
Loads the trained Phase 2 models and generates clear, actionable 5-day forecasts.

Usage:
  python scripts/predict_sample.py
  python scripts/predict_sample.py --symbol TCS.NS --model logistic_regression
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits" / "horizon_5d"
MODELS_DIR = ROOT / "models" / "horizon_5d"

CLASS_NAMES = ["DOWN (Drop > -1.0%)", "HOLD (Flat between -1% and +1%)", "UP (Gain > +1.0%)"]
SIGNALS = {
    0: "SELL / HEDGE (High downside risk)",
    1: "CASH / WAIT (Market expected to chop sideways)",
    2: "BUY / LONG (Bullish 5-day swing signal)",
}


def main():
    parser = argparse.ArgumentParser(description="Generate 5-day stock market forecasts using Phase 2 models.")
    parser.add_argument("--symbol", type=str, default="TCS.NS", help="Stock symbol to inspect (e.g. TCS.NS, RELIANCE.NS, INFY.NS)")
    parser.add_argument("--model", type=str, choices=["lightgbm", "logistic_regression", "random_forest"], default="logistic_regression", help="Model to use")
    parser.add_argument("--rows", type=int, default=5, help="Number of recent test observations to display")
    args = parser.parse_args()

    # Load Model
    model_file = MODELS_DIR / f"{args.model}_5d_classifier.joblib" if args.model == "lightgbm" else MODELS_DIR / f"{args.model}_baseline.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    print("=" * 75)
    print(f"QUANTARA AI — 5-DAY SWING PREDICTOR: {args.symbol}")
    print(f"Model Engine: {args.model.upper()} | Horizon: 5 Trading Days (~1 Week)")
    print("=" * 75)

    model = joblib.load(model_file)

    # Load Test Data & Metadata
    X_test = pd.read_csv(SPLITS_DIR / "X_test.csv")
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv")
    meta_test = pd.read_csv(SPLITS_DIR / "metadata_test.csv")

    mask = (meta_test["Symbol"] == args.symbol).values
    if not mask.any():
        print(f"Available symbols: {sorted(meta_test['Symbol'].unique())}")
        return

    sub_X = X_test.loc[mask].tail(args.rows)
    sub_y = y_test.loc[mask].tail(args.rows)
    sub_meta = meta_test.loc[mask].tail(args.rows)

    preds = model.predict(sub_X)
    probs = model.predict_proba(sub_X)

    for i in range(len(sub_X)):
        row_date = sub_meta.iloc[i]["Date"]
        p_class = int(preds[i])
        p_prob = probs[i]
        actual_ret = sub_y.iloc[i]["forward_return_5d"] * 100
        actual_target = sub_y.iloc[i]["Target_5d"]

        print(f"\n[Session Date: {row_date}]")
        print(f"  Forecast:        {SIGNALS[p_class]}")
        print(f"  Confidence:      DOWN: {p_prob[0]*100:.1f}% | HOLD: {p_prob[1]*100:.1f}% | UP: {p_prob[2]*100:.1f}%")
        print(f"  Realized Return: {actual_ret:+.2f}% (Actual Outcome: {actual_target})")

    print("\n" + "=" * 75)
    print("Interpretation Note:")
    print("• Forecasts represent probabilistic edges over 5 trading sessions.")
    print("• Always adhere to strict risk management (stop-losses, position sizing).")
    print("=" * 75)


if __name__ == "__main__":
    main()
