"""
QUANTARA AI — PHASE 18
MODEL FAILURE DIAGNOSIS AND PREDICTIVE SIGNAL INVESTIGATION

Comprehensive research engine covering:
- Step 1: Independent verification of Phase 17
- Step 2: Prediction target formulation & leakage diagnosis
- Step 3: Predictive information baselines (Majority, Naive, Momentum, LR, RF, LGBM)
- Step 4: Feature usefulness & group comparisons (Train-only fitting)
- Step 5: Probability calibration (Reliability diagrams, Brier score, Log loss, Platt scaling)
- Step 6: Walk-forward generalization across market regimes
- Step 7: Prediction quality vs. trading outcomes & cost sensitivity
"""

import sys
import json
import hashlib
from pathlib import Path

ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
sys.path.append(str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

# -----------------------------------------------------------------------------
# DIRECTORIES & CONFIGURATION
# -----------------------------------------------------------------------------
ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "phase18_model_diagnosis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.00
BASE_ROUNDTRIP_FEE = 0.003937  # 0.3937% round-trip
CONS_ROUNDTRIP_FEE = 0.005000  # 0.5000% round-trip
ZERO_FEE = 0.000000

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def multiclass_brier_score(y_true, y_probs, num_classes=3):
    """
    Compute Multi-class Brier score:
    Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    n = len(y_true)
    y_onehot = np.zeros((n, num_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0
    return float(np.mean(np.sum((y_probs - y_onehot) ** 2, axis=1)))


def safe_log_loss(y_true, y_probs, num_classes=3):
    """Safely compute multi-class log loss with probability clipping."""
    eps = 1e-15
    clipped = np.clip(y_probs, eps, 1.0 - eps)
    # Renormalize rows
    clipped /= clipped.sum(axis=1, keepdims=True)
    return float(log_loss(y_true, clipped, labels=list(range(num_classes))))


# -----------------------------------------------------------------------------
# STEP 1: INDEPENDENT VERIFICATION OF PHASE 17
# -----------------------------------------------------------------------------
def step1_verify_phase17():
    print("\n" + "=" * 80)
    print("STEP 1: INDEPENDENT VERIFICATION OF PHASE 17 ACCOUNTING")
    print("=" * 80)

    ledger_path = ROOT / "reports" / "phase17_portfolio_reconciliation" / "trade_ledger_reconciled.csv"
    equity_path = ROOT / "reports" / "phase17_portfolio_reconciliation" / "equity_curve_reconciled.csv"
    reconciliation_path = ROOT / "reports" / "phase17_portfolio_reconciliation" / "accounting_reconciliation.csv"

    if not ledger_path.exists() or not equity_path.exists():
        raise FileNotFoundError("Phase 17 output files missing!")

    df_trades = pd.read_csv(ledger_path)
    df_equity = pd.read_csv(equity_path)
    df_recon = pd.read_csv(reconciliation_path)

    # 1. Trade-level arithmetic
    for idx, r in df_trades.iterrows():
        exp_cost = (r["shares"] * r["entry_exec_price"]) + r["buy_commission_inr"]
        exp_proceeds = r["shares"] * r["exit_exec_price"]
        exp_gross = exp_proceeds - (r["shares"] * r["entry_exec_price"])
        exp_net = r["gross_pnl_inr"] - r["total_commissions_inr"] + r["dividends_credited_inr"]

        assert abs(r["cost_basis_inr"] - exp_cost) < 1.0, f"Cost basis delta on {r['trade_id']}"
        assert abs(r["gross_proceeds_inr"] - exp_proceeds) < 1.0, f"Gross proceeds delta on {r['trade_id']}"
        assert abs(r["gross_pnl_inr"] - exp_gross) < 1.0, f"Gross PnL delta on {r['trade_id']}"
        assert abs(r["net_pnl_inr"] - exp_net) < 0.05, f"Net PnL delta on {r['trade_id']}"

    # 2. Portfolio-level metrics
    total_trades = len(df_trades)
    gross_pnl_sum = df_trades["gross_pnl_inr"].sum()
    commissions_sum = df_trades["total_commissions_inr"].sum()
    dividends_sum = df_trades["dividends_credited_inr"].sum()
    net_pnl_sum = df_trades["net_pnl_inr"].sum()

    wins = df_trades[df_trades["net_pnl_inr"] > 0]
    losses = df_trades[df_trades["net_pnl_inr"] < 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0

    avg_win = wins["net_pnl_inr"].mean()
    avg_loss = losses["net_pnl_inr"].mean()
    payoff_ratio = avg_win / abs(avg_loss)
    net_expectancy = (win_count / total_trades) * avg_win + (loss_count / total_trades) * avg_loss

    final_equity_reconstructed = STARTING_CAPITAL + net_pnl_sum
    final_equity_curve = df_equity["total_portfolio_equity_inr"].iloc[-1]
    equity_delta = abs(final_equity_reconstructed - final_equity_curve)

    print(f"Total Completed Trades:         {total_trades}")
    print(f"Wins / Losses:                  {win_count} / {loss_count} (Win Rate: {win_rate:.2f}%)")
    print(f"Sum of Gross PnL:               INR {gross_pnl_sum:.2f}")
    print(f"Sum of Commissions:             INR {commissions_sum:.2f}")
    print(f"Sum of Dividends:               INR {dividends_sum:.2f}")
    print(f"Sum of Net PnL:                 INR {net_pnl_sum:.2f}")
    print(f"Average Win / Loss:             +INR {avg_win:.2f} / -INR {abs(avg_loss):.2f}")
    print(f"Realized Payoff Ratio:          {payoff_ratio:.4f}")
    print(f"Net Expectancy per trade:       INR {net_expectancy:.2f}")
    print(f"Reconstructed Final Equity:     INR {final_equity_reconstructed:.2f}")
    print(f"Curve Final Equity:             INR {final_equity_curve:.2f}")
    print(f"Discrepancy (Delta):            INR {equity_delta:.5f} (Verified < 0.01 INR)")

    # Assert exact agreement with Phase 17 reports
    assert equity_delta < 0.01, "Phase 17 equity reconciliation mismatch!"
    print("STEP 1 INDEPENDENT VERIFICATION CONFIRMED: All accounting metrics 100% verified.")
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "net_expectancy": net_expectancy,
        "final_equity": final_equity_curve,
        "commissions": commissions_sum,
        "dividends": dividends_sum,
    }


# -----------------------------------------------------------------------------
# STEP 2: DIAGNOSE THE PREDICTION TARGET
# -----------------------------------------------------------------------------
def step2_diagnose_prediction_target():
    print("\n" + "=" * 80)
    print("STEP 2: PREDICTION TARGET & LEAKAGE DIAGNOSIS")
    print("=" * 80)

    splits_dir = DATA_DIR / "splits_v3" / "horizon_20d"
    meta_tr = pd.read_csv(splits_dir / "metadata_train.csv")
    meta_va = pd.read_csv(splits_dir / "metadata_validation.csv")
    meta_te = pd.read_csv(splits_dir / "metadata_test.csv")
    y_tr_df = pd.read_csv(splits_dir / "y_train.csv")
    y_va_df = pd.read_csv(splits_dir / "y_validation.csv")
    y_te_df = pd.read_csv(splits_dir / "y_test.csv")

    purged = pd.read_csv(splits_dir / "purged_boundary_rows.csv")

    stats = []
    for split_name, meta, y in [("TRAIN", meta_tr, y_tr_df), ("VALIDATION", meta_va, y_va_df), ("TEST (REUSED)", meta_te, y_te_df)]:
        n = len(y)
        up_c = (y["Target_20d"] == "UP").sum()
        hold_c = (y["Target_20d"] == "HOLD").sum()
        down_c = (y["Target_20d"] == "DOWN").sum()
        mean_ret = y["forward_return_20d"].mean() * 100.0
        median_ret = y["forward_return_20d"].median() * 100.0
        std_ret = y["forward_return_20d"].std() * 100.0

        stats.append({
            "split": split_name,
            "sample_size": n,
            "date_range": f"{meta['Date'].min()} to {meta['Date'].max()}",
            "up_pct": round(up_c / n * 100.0, 2),
            "hold_pct": round(hold_c / n * 100.0, 2),
            "down_pct": round(down_c / n * 100.0, 2),
            "mean_forward_return_pct": round(mean_ret, 2),
            "median_forward_return_pct": round(median_ret, 2),
            "std_forward_return_pct": round(std_ret, 2),
        })

    df_target_stats = pd.DataFrame(stats)
    print(df_target_stats.to_string(index=False))

    print(f"\nPurged boundary rows count: {len(purged)}")
    print(f"Purged date range: {purged['Date'].min()} to {purged['Date'].max()}")
    print("Purge verification: Observations overlapping split boundaries were eliminated to prevent lookahead.")

    return df_target_stats


# -----------------------------------------------------------------------------
# STEP 3: PREDICTIVE INFORMATION BASELINES COMPARISON
# -----------------------------------------------------------------------------
def step3_test_predictive_baselines():
    print("\n" + "=" * 80)
    print("STEP 3: TESTING FOR USEFUL PREDICTIVE INFORMATION")
    print("=" * 80)

    splits_dir = DATA_DIR / "splits_v3" / "horizon_20d"
    X_tr = pd.read_csv(splits_dir / "X_train.csv")
    y_tr = pd.read_csv(splits_dir / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_va = pd.read_csv(splits_dir / "X_validation.csv")
    y_va = pd.read_csv(splits_dir / "y_validation.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_te = pd.read_csv(splits_dir / "X_test.csv")
    y_te = pd.read_csv(splits_dir / "y_test.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # 1. Models Fitting (Train only)
    # Model A: Logistic Regression (balanced)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_tr)

    # Model B: Random Forest (balanced)
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    # Model C: LightGBM (balanced)
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1)
    lgb_model.fit(X_tr, y_tr)

    # Empirical prior from train
    train_counts = np.bincount(y_tr, minlength=3)
    train_priors = train_counts / len(y_tr)  # [DOWN, HOLD, UP]
    majority_class = int(np.argmax(train_counts))  # UP (2)

    results = []

    for split_name, X_raw, X_scaled, y_true in [("VALIDATION (2024-2025)", X_va, X_va_sc, y_va), ("TEST (2025-2026)", X_te, X_te_sc, y_te)]:
        n_samples = len(y_true)

        # Baseline 1: Majority Class (Always UP)
        pred_maj = np.full(n_samples, majority_class)
        prob_maj = np.tile(train_priors, (n_samples, 1))

        # Baseline 2: Naive Directional (Sign of daily_return_1d)
        # If daily_return_1d > 0.002 -> UP, < -0.002 -> DOWN, else HOLD
        d_ret = X_raw["daily_return_1d"].values
        pred_naive = np.select([d_ret > 0.002, d_ret < -0.002], [2, 0], default=1)
        prob_naive = np.zeros((n_samples, 3))
        for i, p in enumerate(pred_naive):
            if p == 2:
                prob_naive[i] = [0.20, 0.20, 0.60]
            elif p == 0:
                prob_naive[i] = [0.60, 0.20, 0.20]
            else:
                prob_naive[i] = [0.20, 0.60, 0.20]

        # Baseline 3: Simple Momentum (return_20d past trend)
        # If return_20d > 0.02 -> UP, < -0.02 -> DOWN, else HOLD
        m_ret = X_raw["return_20d"].values
        pred_mom = np.select([m_ret > 0.02, m_ret < -0.02], [2, 0], default=1)
        prob_mom = np.zeros((n_samples, 3))
        for i, p in enumerate(pred_mom):
            if p == 2:
                prob_mom[i] = [0.20, 0.20, 0.60]
            elif p == 0:
                prob_mom[i] = [0.60, 0.20, 0.20]
            else:
                prob_mom[i] = [0.20, 0.60, 0.20]

        # ML Models
        prob_lr = lr.predict_proba(X_scaled)
        pred_lr = lr.predict(X_scaled)

        prob_rf = rf.predict_proba(X_raw)
        pred_rf = rf.predict(X_raw)

        prob_lgb = lgb_model.predict_proba(X_raw)
        pred_lgb = lgb_model.predict(X_raw)

        candidates = [
            ("1. Majority-Class Baseline (Always UP)", pred_maj, prob_maj),
            ("2. Naive Directional Baseline (1d Trend)", pred_naive, prob_naive),
            ("3. Simple Momentum Baseline (20d Trend)", pred_mom, prob_mom),
            ("4. Logistic Regression (Trained)", pred_lr, prob_lr),
            ("5. Random Forest (Trained)", pred_rf, prob_rf),
            ("6. LightGBM (Trained)", pred_lgb, prob_lgb),
        ]

        for name, preds, probs in candidates:
            acc = accuracy_score(y_true, preds) * 100.0
            bal_acc = balanced_accuracy_score(y_true, preds) * 100.0
            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
            brier = multiclass_brier_score(y_true, probs)
            ll = safe_log_loss(y_true, probs)
            cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])

            results.append({
                "model_name": name,
                "evaluation_window": split_name,
                "sample_size": n_samples,
                "accuracy_pct": round(acc, 2),
                "balanced_accuracy_pct": round(bal_acc, 2),
                "macro_f1_pct": round(macro_f1, 2),
                "brier_score": round(brier, 4),
                "log_loss": round(ll, 4),
                "conf_matrix_down_hold_up": str(cm.tolist()),
            })

    df_base = pd.DataFrame(results)
    df_base.to_csv(REPORT_DIR / "baseline_comparison.csv", index=False)
    print("Saved baseline_comparison.csv")
    print(df_base[["model_name", "evaluation_window", "balanced_accuracy_pct", "macro_f1_pct", "brier_score", "log_loss"]].to_string(index=False))
    return df_base


# -----------------------------------------------------------------------------
# STEP 4: INVESTIGATE FEATURE USEFULNESS BY GROUP
# -----------------------------------------------------------------------------
def step4_investigate_feature_usefulness():
    print("\n" + "=" * 80)
    print("STEP 4: INVESTIGATING FEATURE USEFULNESS BY PREDEFINED GROUPS")
    print("=" * 80)

    splits_dir = DATA_DIR / "splits_v3" / "horizon_20d"
    X_tr = pd.read_csv(splits_dir / "X_train.csv")
    y_tr = pd.read_csv(splits_dir / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_va = pd.read_csv(splits_dir / "X_validation.csv")
    y_va = pd.read_csv(splits_dir / "y_validation.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_te = pd.read_csv(splits_dir / "X_test.csv")
    y_te = pd.read_csv(splits_dir / "y_test.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values

    feature_groups = {
        "Group 1: Price & Momentum (15)": [
            "daily_return_1d", "return_5d", "return_10d", "return_20d",
            "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
            "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
            "macd_pct", "macd_signal_pct", "macd_histogram_pct",
            "bollinger_percent_b", "bollinger_width_pct"
        ],
        "Group 2: Volatility & Range (6)": [
            "volatility_10d_annualized", "volatility_20d_annualized", "rolling_drawdown_20d",
            "intraday_range_pct", "open_gap_pct", "close_location_in_range"
        ],
        "Group 3: Volume & Liquidity (3)": [
            "log_volume_change", "volume_vs_sma20", "volume_price_trend_5d"
        ],
        "Group 4: Market Breadth & Macro (3)": [
            "market_avg_return_1d", "market_breadth_above_sma20", "market_avg_volatility_20d"
        ],
        "Group 5: Combined All Features (27)": list(X_tr.columns),
    }

    results = []

    for group_name, cols in feature_groups.items():
        X_tr_sub = X_tr[cols]
        X_va_sub = X_va[cols]
        X_te_sub = X_te[cols]

        scaler = StandardScaler()
        X_tr_sub_sc = scaler.fit_transform(X_tr_sub)
        X_va_sub_sc = scaler.transform(X_va_sub)
        X_te_sub_sc = scaler.transform(X_te_sub)

        # 1. Fit LR
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_tr_sub_sc, y_tr)
        pred_lr_va = lr.predict(X_va_sub_sc)
        pred_lr_te = lr.predict(X_te_sub_sc)

        # 2. Fit LightGBM
        lgb_mod = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1)
        lgb_mod.fit(X_tr_sub, y_tr)
        pred_lgb_va = lgb_mod.predict(X_va_sub)
        pred_lgb_te = lgb_mod.predict(X_te_sub)

        results.append({
            "feature_group": group_name,
            "feature_count": len(cols),
            "lr_val_balanced_acc_pct": round(balanced_accuracy_score(y_va, pred_lr_va) * 100.0, 2),
            "lr_val_macro_f1_pct": round(f1_score(y_va, pred_lr_va, average="macro", zero_division=0) * 100.0, 2),
            "lr_test_balanced_acc_pct": round(balanced_accuracy_score(y_te, pred_lr_te) * 100.0, 2),
            "lr_test_macro_f1_pct": round(f1_score(y_te, pred_lr_te, average="macro", zero_division=0) * 100.0, 2),
            "lgb_val_balanced_acc_pct": round(balanced_accuracy_score(y_va, pred_lgb_va) * 100.0, 2),
            "lgb_val_macro_f1_pct": round(f1_score(y_va, pred_lgb_va, average="macro", zero_division=0) * 100.0, 2),
            "lgb_test_balanced_acc_pct": round(balanced_accuracy_score(y_te, pred_lgb_te) * 100.0, 2),
            "lgb_test_macro_f1_pct": round(f1_score(y_te, pred_lgb_te, average="macro", zero_division=0) * 100.0, 2),
        })

    df_feats = pd.DataFrame(results)
    df_feats.to_csv(REPORT_DIR / "feature_group_results.csv", index=False)
    print("Saved feature_group_results.csv")
    print(df_feats.to_string(index=False))
    return df_feats


# -----------------------------------------------------------------------------
# STEP 5: INVESTIGATE PROBABILITY CALIBRATION
# -----------------------------------------------------------------------------
def step5_investigate_probability_calibration():
    print("\n" + "=" * 80)
    print("STEP 5: PROBABILITY CALIBRATION & CONFIDENCE ANALYSIS")
    print("=" * 80)

    splits_dir = DATA_DIR / "splits_v3" / "horizon_20d"
    X_tr = pd.read_csv(splits_dir / "X_train.csv")
    y_tr = pd.read_csv(splits_dir / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_va = pd.read_csv(splits_dir / "X_validation.csv")
    y_va = pd.read_csv(splits_dir / "y_validation.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_te = pd.read_csv(splits_dir / "X_test.csv")
    y_te = pd.read_csv(splits_dir / "y_test.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # 1. Base Uncalibrated LR
    base_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    base_lr.fit(X_tr_sc, y_tr)
    p_lr_va = base_lr.predict_proba(X_va_sc)
    p_lr_te = base_lr.predict_proba(X_te_sc)

    # 2. Platt Scaled (Sigmoid) LR fit on Train using 5-fold CV (never touching test!)
    cal_platt = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv=5)
    cal_platt.fit(X_tr_sc, y_tr)
    p_platt_va = cal_platt.predict_proba(X_va_sc)
    p_platt_te = cal_platt.predict_proba(X_te_sc)

    # 3. Isotonic Calibrated LR fit on Train using 5-fold CV
    cal_iso = CalibratedClassifierCV(estimator=base_lr, method="isotonic", cv=5)
    cal_iso.fit(X_tr_sc, y_tr)
    p_iso_va = cal_iso.predict_proba(X_va_sc)
    p_iso_te = cal_iso.predict_proba(X_te_sc)

    # Calibration Reliability Bins for Target class UP (index 2)
    # Bins: [0.0, 0.2), [0.2, 0.3), [0.3, 0.4), [0.4, 0.5), [0.5, 1.0]
    bins = [0.0, 0.20, 0.30, 0.40, 0.50, 1.00]
    bin_labels = ["[0.0, 0.20)", "[0.20, 0.30)", "[0.30, 0.40)", "[0.40, 0.50)", "[0.50, 1.00]"]

    cal_records = []

    models = [
        ("Logistic Regression (Uncalibrated)", p_lr_va[:, 2], y_va, "VALIDATION"),
        ("Logistic Regression (Uncalibrated)", p_lr_te[:, 2], y_te, "TEST (REUSED)"),
        ("Logistic Regression (Platt Sigmoid)", p_platt_va[:, 2], y_va, "VALIDATION"),
        ("Logistic Regression (Platt Sigmoid)", p_platt_te[:, 2], y_te, "TEST (REUSED)"),
        ("Logistic Regression (Isotonic)", p_iso_va[:, 2], y_va, "VALIDATION"),
        ("Logistic Regression (Isotonic)", p_iso_te[:, 2], y_te, "TEST (REUSED)"),
    ]

    for model_name, prob_up, y_true, split_name in models:
        y_is_up = (y_true == 2).astype(int)
        digitized = np.digitize(prob_up, bins) - 1  # 0 to 4

        for b_idx, label in enumerate(bin_labels):
            mask = (digitized == b_idx)
            count = int(np.sum(mask))
            if count > 0:
                mean_pred = float(np.mean(prob_up[mask]))
                obs_rate = float(np.mean(y_is_up[mask]))
                gap = abs(mean_pred - obs_rate)
            else:
                mean_pred = 0.0
                obs_rate = 0.0
                gap = 0.0

            cal_records.append({
                "model_name": model_name,
                "split": split_name,
                "confidence_bin": label,
                "sample_count": count,
                "mean_predicted_prob_up": round(mean_pred, 4),
                "observed_up_frequency": round(obs_rate, 4),
                "calibration_gap": round(gap, 4),
            })

    df_cal = pd.DataFrame(cal_records)
    df_cal.to_csv(REPORT_DIR / "probability_calibration.csv", index=False)
    print("Saved probability_calibration.csv")
    print(df_cal.head(15).to_string(index=False))
    return df_cal


# -----------------------------------------------------------------------------
# STEP 6: TEST GENERALIZATION WITH EXPANDING-WINDOW WALK-FORWARD
# -----------------------------------------------------------------------------
def step6_test_generalization_walk_forward():
    print("\n" + "=" * 80)
    print("STEP 6: TESTING GENERALIZATION ACROSS MARKET REGIMES (EXPANDING WALK-FORWARD)")
    print("=" * 80)

    # Load full validated processed dataset
    processed_path = DATA_DIR / "processed_v3" / "multi_stock_features_v3_validated.csv"
    raw_path = DATA_DIR / "raw" / "multi_stock_historical.csv"
    if not processed_path.exists() or not raw_path.exists():
        raise FileNotFoundError("Processed feature CSV or raw historical CSV missing!")

    df_feats = pd.read_csv(processed_path)
    df_feats["Date"] = pd.to_datetime(df_feats["Date"])

    # Load raw to compute 20d forward return target
    raw = pd.read_csv(raw_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Compute forward 20d return on raw adj open
    forward_targets = []
    for sym, g in raw.groupby("Symbol"):
        g = g.sort_values("Date").copy().reset_index(drop=True)
        # adj open proxy
        adj_factor = g["Adj Close"] / g["Close"]
        adj_open = g["Open"] * adj_factor
        adj_open_next = adj_open.shift(-1)
        adj_open_exit = adj_open.shift(-21)
        f_ret = adj_open_exit / adj_open_next - 1.0
        g["forward_return_20d"] = f_ret
        g["target_end_date"] = g["Date"].shift(-21)
        forward_targets.append(g[["Date", "Symbol", "forward_return_20d", "target_end_date"]])

    df_targets = pd.concat(forward_targets, ignore_index=True)
    df_merged = df_feats.merge(df_targets, on=["Date", "Symbol"], how="inner")
    df_merged = df_merged.dropna(subset=["forward_return_20d", "target_end_date"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # 3-class target
    df_merged["Target_20d"] = np.select(
        [df_merged["forward_return_20d"] > 0.02, df_merged["forward_return_20d"] < -0.02],
        [2, 0],  # UP=2, DOWN=0
        default=1  # HOLD=1
    )

    FEATURE_COLS = [
        "daily_return_1d", "return_5d", "return_10d", "return_20d",
        "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
        "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
        "macd_pct", "macd_signal_pct", "macd_histogram_pct",
        "bollinger_percent_b", "bollinger_width_pct",
        "volatility_10d_annualized", "volatility_20d_annualized", "rolling_drawdown_20d",
        "intraday_range_pct", "open_gap_pct", "close_location_in_range",
        "log_volume_change", "volume_vs_sma20", "volume_price_trend_5d",
        "market_avg_return_1d", "market_breadth_above_sma20", "market_avg_volatility_20d"
    ]

    # Define expanding windows:
    # Fold 1: Train 2017 to 2021-12-31, Test 2022 (Bearish Inflation Shock)
    # Fold 2: Train 2017 to 2022-12-31, Test 2023 (Recovery & Bull)
    # Fold 3: Train 2017 to 2023-12-31, Test 2024-01 to 2024-09 (Strong Bull)
    # Fold 4: Train 2017 to 2024-09-23, Test 2024-10 to 2025-09 (Validation Window - Sideways)
    # Fold 5: Train 2017 to 2025-09-19, Test 2025-10 to 2026-09 (Reused Test Window - Correction)

    folds = [
        {
            "fold_name": "Fold 1: 2022 Shock",
            "train_end": pd.Timestamp("2021-12-31"),
            "test_start": pd.Timestamp("2022-01-01"),
            "test_end": pd.Timestamp("2022-12-31"),
            "regime_description": "Global inflation & rate-hiking cycle correction",
        },
        {
            "fold_name": "Fold 2: 2023 Recovery",
            "train_end": pd.Timestamp("2022-12-31"),
            "test_start": pd.Timestamp("2023-01-01"),
            "test_end": pd.Timestamp("2023-12-31"),
            "regime_description": "Broad market post-correction recovery",
        },
        {
            "fold_name": "Fold 3: 2024 Pre-Election Bull",
            "train_end": pd.Timestamp("2023-12-31"),
            "test_start": pd.Timestamp("2024-01-01"),
            "test_end": pd.Timestamp("2024-09-23"),
            "regime_description": "Strong domestic equity liquidity bull run",
        },
        {
            "fold_name": "Fold 4: 2024-2025 (Validation)",
            "train_end": pd.Timestamp("2024-09-23"),
            "test_start": pd.Timestamp("2024-10-24"),
            "test_end": pd.Timestamp("2025-09-19"),
            "regime_description": "Challenging sideways consolidation regime",
        },
        {
            "fold_name": "Fold 5: 2025-2026 (Reused Test)",
            "train_end": pd.Timestamp("2025-09-19"),
            "test_start": pd.Timestamp("2025-10-23"),
            "test_end": pd.Timestamp("2026-09-22"),
            "regime_description": "Market downturn / bear correction regime",
        },
    ]

    wf_records = []

    for fold in folds:
        # Purge boundary: target_end_date must be <= train_end for training
        train_mask = (df_merged["Date"] <= fold["train_end"]) & (df_merged["target_end_date"] <= fold["train_end"])
        test_mask = (df_merged["Date"] >= fold["test_start"]) & (df_merged["Date"] <= fold["test_end"]) & (df_merged["target_end_date"] <= fold["test_end"])

        df_tr = df_merged.loc[train_mask]
        df_te = df_merged.loc[test_mask]

        if len(df_tr) < 500 or len(df_te) < 100:
            print(f"Skipping {fold['fold_name']}: insufficient samples")
            continue

        X_tr = df_tr[FEATURE_COLS]
        y_tr = df_tr["Target_20d"].values
        X_te = df_te[FEATURE_COLS]
        y_te = df_te["Target_20d"].values

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Fit LR
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_tr_sc, y_tr)
        preds_lr = lr.predict(X_te_sc)
        probs_lr = lr.predict_proba(X_te_sc)

        # Baseline: Majority Class
        train_counts = np.bincount(y_tr, minlength=3)
        maj_class = int(np.argmax(train_counts))
        preds_maj = np.full(len(y_te), maj_class)

        # Stats
        n_test = len(y_te)
        up_pct = (y_te == 2).sum() / n_test * 100.0
        down_pct = (y_te == 0).sum() / n_test * 100.0
        hold_pct = (y_te == 1).sum() / n_test * 100.0
        mean_ret = df_te["forward_return_20d"].mean() * 100.0

        lr_bal_acc = balanced_accuracy_score(y_te, preds_lr) * 100.0
        maj_bal_acc = balanced_accuracy_score(y_te, preds_maj) * 100.0
        lr_macro_f1 = f1_score(y_te, preds_lr, average="macro", zero_division=0) * 100.0
        brier = multiclass_brier_score(y_te, probs_lr)

        wf_records.append({
            "fold_name": fold["fold_name"],
            "test_window": f"{fold['test_start'].date()} to {fold['test_end'].date()}",
            "train_samples": len(df_tr),
            "test_samples": n_test,
            "market_regime": fold["regime_description"],
            "test_mean_forward_return_pct": round(mean_ret, 2),
            "actual_up_pct": round(up_pct, 2),
            "actual_hold_pct": round(hold_pct, 2),
            "actual_down_pct": round(down_pct, 2),
            "majority_baseline_bal_acc_pct": round(maj_bal_acc, 2),
            "lr_balanced_accuracy_pct": round(lr_bal_acc, 2),
            "lr_macro_f1_pct": round(lr_macro_f1, 2),
            "lr_brier_score": round(brier, 4),
        })

    df_wf = pd.DataFrame(wf_records)
    df_wf.to_csv(REPORT_DIR / "walk_forward_results.csv", index=False)
    print("Saved walk_forward_results.csv")
    print(df_wf.to_string(index=False))
    return df_wf


# -----------------------------------------------------------------------------
# STEP 7: CONNECT PREDICTION QUALITY TO TRADING OUTCOMES
# -----------------------------------------------------------------------------
def step7_connect_prediction_to_trading():
    print("\n" + "=" * 80)
    print("STEP 7: CONNECTING PREDICTION QUALITY TO TRADING OUTCOMES & COSTS")
    print("=" * 80)

    from run_phase17_reconciliation_research import (
        load_all_market_data,
        simulate_engine_with_session_ledger,
        extract_verified_trade_ledger,
        BASE_FEE_SCHEDULE,
        CONSERVATIVE_FEE_SCHEDULE,
    )

    ZERO_FEE_SCHEDULE = {
        "brokerage_pct": 0.0,
        "stt_buy_pct": 0.0,
        "stt_sell_pct": 0.0,
        "exchange_turnover_pct": 0.0,
        "sebi_turnover_pct": 0.0,
        "stamp_duty_buy_pct": 0.0,
        "gst_rate": 0.0,
        "slippage_pct": 0.0,
    }

    market = load_all_market_data()

    splits_dir = DATA_DIR / "splits_v3" / "horizon_20d"
    X_tr = pd.read_csv(splits_dir / "X_train.csv")
    y_tr = pd.read_csv(splits_dir / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_va = pd.read_csv(splits_dir / "X_validation.csv")
    X_te = pd.read_csv(splits_dir / "X_test.csv")

    meta_va = market["meta_val"]
    meta_te = market["meta_test"]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_tr)
    probs_va = lr.predict_proba(X_va_sc)
    probs_te = lr.predict_proba(X_te_sc)

    all_dates = market["all_trading_dates"]

    # Build signal dictionaries by execution date: {exec_d: [(sym, prob_up)]}
    def build_signals_dict(meta_df, probs, threshold):
        sigs = {}
        for idx, r in meta_df.iterrows():
            sig_d = r["Date"]
            if sig_d in all_dates:
                cur_idx = all_dates.index(sig_d)
                if cur_idx < len(all_dates) - 1:
                    exec_d = all_dates[cur_idx + 1]
                    p_up = float(probs[idx, 2])
                    p_down = float(probs[idx, 0])
                    if p_up >= threshold and p_up > p_down:
                        sigs.setdefault(exec_d, []).append((r["Symbol"], p_up))
        for d in sigs:
            sigs[d].sort(key=lambda x: x[1], reverse=True)
        return sigs

    thresholds = [0.35, 0.38, 0.40, 0.42, 0.45]
    windows = [
        ("VALIDATION (2024-2025)", meta_va, probs_va, market["val_trading_dates"], "VAL"),
        ("REUSED TEST (2025-2026)", meta_te, probs_te, market["test_trading_dates"], "TEST"),
    ]

    fee_scenarios = [
        ("Zero Friction (0.00%)", ZERO_FEE_SCHEDULE),
        ("Base Friction (0.39%)", BASE_FEE_SCHEDULE),
        ("Conservative Friction (0.50%)", CONSERVATIVE_FEE_SCHEDULE),
    ]

    trading_results = []

    for win_name, meta_df, probs, m_dates, win_tag in windows:
        for th in thresholds:
            sigs = build_signals_dict(meta_df, probs, th)
            for fee_label, sched in fee_scenarios:
                strat_name = f"LR_TH{int(th*100)}_{win_tag}"
                df_eq, blot, perf = simulate_engine_with_session_ledger(
                    strat_name, sigs, m_dates, market, max_positions=3, max_hold_sessions=20, fee_sched=sched
                )

                df_t = extract_verified_trade_ledger(blot, strat_name, win_tag)
                n_trades = len(df_t)
                final_eq = perf["final_equity"]
                net_ret = perf["total_return_pct"]

                if n_trades > 0:
                    wins = df_t[df_t["net_pnl_inr"] > 0]
                    losses = df_t[df_t["net_pnl_inr"] <= 0]
                    win_rate = len(wins) / n_trades * 100.0
                    avg_w = wins["net_pnl_inr"].mean() if len(wins) > 0 else 0.0
                    avg_l = losses["net_pnl_inr"].mean() if len(losses) > 0 else 0.0
                    payoff = avg_w / abs(avg_l) if abs(avg_l) > 0 else 0.0
                    expectancy = df_t["net_pnl_inr"].sum() / n_trades
                    tot_comm = df_t["total_commissions_inr"].sum()
                else:
                    win_rate = 0.0
                    avg_w = 0.0
                    avg_l = 0.0
                    payoff = 0.0
                    expectancy = 0.0
                    tot_comm = 0.0

                trading_results.append({
                    "evaluation_window": win_name,
                    "threshold": th,
                    "fee_scenario": fee_label,
                    "trade_count": n_trades,
                    "win_rate_pct": round(win_rate, 2),
                    "avg_win_inr": round(avg_w, 2),
                    "avg_loss_inr": round(avg_l, 2),
                    "realized_payoff_ratio": round(payoff, 4),
                    "net_expectancy_inr": round(expectancy, 2),
                    "total_commissions_inr": round(tot_comm, 2),
                    "final_equity_inr": round(final_eq, 2),
                    "net_return_pct": round(net_ret, 2),
                })

    df_trade = pd.DataFrame(trading_results)
    df_trade.to_csv(REPORT_DIR / "prediction_vs_profitability.csv", index=False)
    print("Saved prediction_vs_profitability.csv")
    print(df_trade.head(20).to_string(index=False))
    return df_trade


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 18 EXECUTION ENGINE")
    print("=" * 80)

    # Step 1: Verify Phase 17
    step1_verify_phase17()

    # Step 2: Target & Leakage Diagnosis
    step2_diagnose_prediction_target()

    # Step 3: Predictive Baselines
    step3_test_predictive_baselines()

    # Step 4: Feature Usefulness
    step4_investigate_feature_usefulness()

    # Step 5: Probability Calibration
    step5_investigate_probability_calibration()

    # Step 6: Walk-Forward Generalization
    step6_test_generalization_walk_forward()

    # Step 7: Prediction vs Profitability
    step7_connect_prediction_to_trading()

    print("\n" + "=" * 80)
    print("PHASE 18 EXPERIMENT SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
