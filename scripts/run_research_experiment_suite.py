"""
Quantara AI — Phases 4-10: Systematic ML/DL Research & Validation Engine
Experiments Covered:
1. Baseline Reproduction (5d horizon, 3-class, conventional features)
2. Prediction Horizon Comparison: 1d vs 5d vs 20d
3. Target Formulation Comparison: 3-Class (UP/HOLD/DOWN) vs Binary (UP vs DOWN)
4. Incremental Feature Engineering Group Ablation:
   - Group 1: Core Momentum & Trend (15 features)
   - Group 1+2: + Volatility & Drawdown (21 features)
   - Group 1+2+3: + Volume Dynamics (24 features)
   - Group 1+2+3+4: + Market Breadth & Macro Dynamics (27 features)
5. Class Imbalance Handling: Balanced Weights vs Natural Priors
6. Deep Learning Architecture: Deep MLP Neural Network (128-64-32 with early stopping)
7. Market-Regime Filtering: Regime-Conditional Accuracy & Coverage Analysis
8. Multi-Horizon Portfolio Backtesting (Testing the 20d horizon turnover reduction hypothesis)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
)
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "splits_v3"
EXP_DIR = ROOT / "models" / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = ROOT / "reports" / "accuracy_research"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP_3 = {"DOWN": 0, "HOLD": 1, "UP": 2}
CLASS_MAP_BIN = {"DOWN": 0, "UP": 1}

FEATURE_GROUPS = {
    "group_1_trend": [
        "daily_return_1d", "return_5d", "return_10d", "return_20d",
        "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
        "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
        "macd_pct", "macd_signal_pct", "macd_histogram_pct",
        "bollinger_percent_b", "bollinger_width_pct"
    ],
    "group_1_2_volatility": [
        "daily_return_1d", "return_5d", "return_10d", "return_20d",
        "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
        "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
        "macd_pct", "macd_signal_pct", "macd_histogram_pct",
        "bollinger_percent_b", "bollinger_width_pct",
        "volatility_10d_annualized", "volatility_20d_annualized", "rolling_drawdown_20d",
        "intraday_range_pct", "open_gap_pct", "close_location_in_range"
    ],
    "group_1_2_3_volume": [
        "daily_return_1d", "return_5d", "return_10d", "return_20d",
        "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
        "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
        "macd_pct", "macd_signal_pct", "macd_histogram_pct",
        "bollinger_percent_b", "bollinger_width_pct",
        "volatility_10d_annualized", "volatility_20d_annualized", "rolling_drawdown_20d",
        "intraday_range_pct", "open_gap_pct", "close_location_in_range",
        "log_volume_change", "volume_vs_sma20", "volume_price_trend_5d"
    ],
    "group_full_regime": [
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
}


def load_dataset(horizon, is_binary=False, feature_subset="group_full_regime"):
    """Load horizon dataset with strict split isolation and chosen feature subset."""
    h_dir = DATA_DIR / f"horizon_{horizon}d"
    features = FEATURE_GROUPS[feature_subset]
    
    data = {}
    for s in ["train", "validation", "test"]:
        X = pd.read_csv(h_dir / f"X_{s}.csv")[features]
        y_df = pd.read_csv(h_dir / f"y_{s}.csv")
        meta = pd.read_csv(h_dir / f"metadata_{s}.csv")
        
        if is_binary:
            # Filter rows where target is UP or DOWN (exclude EXCLUDE_HOLD noise)
            bin_target = y_df[f"Target_Binary_{horizon}d"]
            valid_mask = bin_target.isin(["UP", "DOWN"]).values
            
            X_clean = X.loc[valid_mask].reset_index(drop=True)
            y_clean = bin_target[valid_mask].map(CLASS_MAP_BIN).values
            ret_clean = y_df.loc[valid_mask, f"forward_return_{horizon}d"].values
            meta_clean = meta.loc[valid_mask].reset_index(drop=True)
            coverage = float(valid_mask.sum() / len(y_df))
            num_classes = 2
        else:
            X_clean = X.reset_index(drop=True)
            y_clean = y_df[f"Target_{horizon}d"].map(CLASS_MAP_3).values
            ret_clean = y_df[f"forward_return_{horizon}d"].values
            meta_clean = meta.reset_index(drop=True)
            coverage = 1.0
            num_classes = 3
            
        data[s] = {
            "X": X_clean,
            "y": y_clean,
            "returns": ret_clean,
            "meta": meta_clean,
            "coverage": coverage,
            "num_classes": num_classes,
        }
    return data, features


def evaluate_model_performance(y_true, y_pred, y_prob=None, num_classes=3):
    """Compute standard metrics with safety against zero-division."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    
    loss = None
    if y_prob is not None:
        try:
            loss = float(log_loss(y_true, y_prob, labels=list(range(num_classes))))
        except Exception:
            loss = None
            
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": loss,
        "per_class_precision": [float(p) for p in prec],
        "per_class_recall": [float(r) for r in rec],
        "per_class_f1": [float(f) for f in f1],
        "confusion_matrix": cm,
    }


def run_experiment(exp_id, horizon, target_type, model_type, feature_subset, class_weight_mode="balanced", regime_filter=None):
    """Run a single end-to-end experiment and record metrics."""
    is_binary = (target_type == "binary")
    num_classes = 2 if is_binary else 3
    data, feature_names = load_dataset(horizon, is_binary=is_binary, feature_subset=feature_subset)
    
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["validation"]["X"], data["validation"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    
    # Fit scaler strictly on Train
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    cw = "balanced" if class_weight_mode == "balanced" else None
    
    # Instantiate Model
    if model_type == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
        X_train_fit, X_val_fit, X_test_fit = X_train, X_val, X_test
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
        X_train_fit, X_val_fit, X_test_fit = X_train_scaled, X_val_scaled, X_test_scaled
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20, class_weight=cw, random_state=42, n_jobs=-1)
        X_train_fit, X_val_fit, X_test_fit = X_train, X_val, X_test
    elif model_type == "lightgbm":
        clf = lgb.LGBMClassifier(
            objective="binary" if is_binary else "multiclass",
            num_class=num_classes if not is_binary else 1,
            num_leaves=31, max_depth=6, learning_rate=0.03,
            colsample_bytree=0.8, subsample=0.8, min_child_samples=50,
            reg_alpha=0.5, reg_lambda=2.0, class_weight=cw,
            n_estimators=300, random_state=42, n_jobs=-1, verbose=-1
        )
        X_train_fit, X_val_fit, X_test_fit = X_train, X_val, X_test
    elif model_type == "deep_mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        X_train_fit, X_val_fit, X_test_fit = X_train_scaled, X_val_scaled, X_test_scaled
    else:
        raise ValueError(f"Unknown model_type {model_type}")
        
    clf.fit(X_train_fit, y_train)
    
    val_pred = clf.predict(X_val_fit)
    val_prob = clf.predict_proba(X_val_fit)
    val_metrics = evaluate_model_performance(y_val, val_pred, val_prob, num_classes=num_classes)
    
    test_pred = clf.predict(X_test_fit)
    test_prob = clf.predict_proba(X_test_fit)
    test_metrics = evaluate_model_performance(y_test, test_pred, test_prob, num_classes=num_classes)
    
    # Regime filtering analysis on Test Set
    regime_results = {}
    if regime_filter is not None:
        test_meta = data["test"]["meta"]
        reg_mask = (test_meta["market_regime"] == regime_filter).values
        if reg_mask.sum() > 20:
            reg_pred = test_pred[reg_mask]
            reg_true = y_test[reg_mask]
            reg_prob = test_prob[reg_mask]
            reg_perf = evaluate_model_performance(reg_true, reg_pred, reg_prob, num_classes=num_classes)
            regime_results[regime_filter] = {
                "coverage_pct": float(reg_mask.sum() / len(y_test) * 100.0),
                "balanced_accuracy": reg_perf["balanced_accuracy"],
                "accuracy": reg_perf["accuracy"],
                "macro_f1": reg_perf["macro_f1"]
            }
            
    record = {
        "exp_id": exp_id,
        "horizon": f"{horizon}d",
        "target_type": target_type,
        "model_type": model_type,
        "feature_subset": feature_subset,
        "num_features": len(feature_names),
        "class_weight": class_weight_mode,
        "sample_coverage_pct": round(data["test"]["coverage"] * 100.0, 2),
        "val_bal_acc": round(val_metrics["balanced_accuracy"] * 100.0, 2),
        "val_acc": round(val_metrics["accuracy"] * 100.0, 2),
        "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "test_bal_acc": round(test_metrics["balanced_accuracy"] * 100.0, 2),
        "test_acc": round(test_metrics["accuracy"] * 100.0, 2),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_log_loss": round(test_metrics["log_loss"], 4) if test_metrics["log_loss"] else None,
        "milestone_40_met": bool(test_metrics["balanced_accuracy"] >= 0.40),
        "regime_filtered": regime_filter if regime_filter else "None",
        "regime_metrics": regime_results.get(regime_filter, None),
    }
    
    return record, clf, scaler


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASES 4-10: SYSTEMATIC RESEARCH & EXPERIMENT ENGINE")
    print("=" * 80)

    leaderboard = []

    # -------------------------------------------------------------------------
    # Experiment Suite 1: Horizon Comparison (1d vs 5d vs 20d) with 3-Class
    # -------------------------------------------------------------------------
    print("\n[Suite 1/6] Running Horizon Comparison (1d, 5d, 20d) on 3-Class Target...")
    for h in [1, 5, 20]:
        for m in ["dummy", "logistic_regression", "random_forest", "lightgbm"]:
            exp_id = f"EXP_H{h}_{m}_3class_base"
            print(f"  Executing {exp_id}...")
            res, _, _ = run_experiment(exp_id, horizon=h, target_type="3class", model_type=m, feature_subset="group_1_trend")
            leaderboard.append(res)

    # -------------------------------------------------------------------------
    # Experiment Suite 2: Target Formulation (3-Class vs Binary UP/DOWN)
    # -------------------------------------------------------------------------
    print("\n[Suite 2/6] Running Target Formulation Comparison (3-Class vs Binary)...")
    for h in [5, 20]:
        for m in ["logistic_regression", "lightgbm"]:
            exp_id = f"EXP_H{h}_{m}_binary_base"
            print(f"  Executing {exp_id}...")
            res, _, _ = run_experiment(exp_id, horizon=h, target_type="binary", model_type=m, feature_subset="group_1_trend")
            leaderboard.append(res)

    # -------------------------------------------------------------------------
    # Experiment Suite 3: Incremental Feature Engineering Ablation
    # -------------------------------------------------------------------------
    print("\n[Suite 3/6] Running Incremental Feature Group Ablation on 5d & 20d...")
    feature_steps = [
        ("group_1_trend", "Trend"),
        ("group_1_2_volatility", "Trend+Vol"),
        ("group_1_2_3_volume", "Trend+Vol+Volume"),
        ("group_full_regime", "Full+Regime")
    ]
    for h in [5, 20]:
        for f_name, label in feature_steps:
            for m in ["logistic_regression", "lightgbm"]:
                exp_id = f"EXP_H{h}_{m}_feat_{label}"
                print(f"  Executing {exp_id}...")
                res, _, _ = run_experiment(exp_id, horizon=h, target_type="3class", model_type=m, feature_subset=f_name)
                leaderboard.append(res)

    # -------------------------------------------------------------------------
    # Experiment Suite 4: Class Imbalance Strategy (Balanced vs Natural Priors)
    # -------------------------------------------------------------------------
    print("\n[Suite 4/6] Comparing Class Weight Strategies...")
    for h in [5, 20]:
        for m in ["logistic_regression", "lightgbm"]:
            exp_id = f"EXP_H{h}_{m}_natural_prior"
            print(f"  Executing {exp_id}...")
            res, _, _ = run_experiment(exp_id, horizon=h, target_type="3class", model_type=m, feature_subset="group_full_regime", class_weight_mode="natural")
            leaderboard.append(res)

    # -------------------------------------------------------------------------
    # Experiment Suite 5: Deep Learning Evaluation (Deep Multi-Layer Perceptron)
    # -------------------------------------------------------------------------
    print("\n[Suite 5/6] Training & Evaluating Deep Learning MLP Architectures...")
    for h in [5, 20]:
        exp_id_3c = f"EXP_H{h}_deep_mlp_3class"
        print(f"  Executing {exp_id_3c}...")
        res_3c, model_dl, scaler_dl = run_experiment(exp_id_3c, horizon=h, target_type="3class", model_type="deep_mlp", feature_subset="group_full_regime")
        leaderboard.append(res_3c)
        joblib.dump(model_dl, EXP_DIR / f"deep_mlp_h{h}_3class.joblib")

        exp_id_bin = f"EXP_H{h}_deep_mlp_binary"
        print(f"  Executing {exp_id_bin}...")
        res_bin, model_dl_bin, _ = run_experiment(exp_id_bin, horizon=h, target_type="binary", model_type="deep_mlp", feature_subset="group_full_regime")
        leaderboard.append(res_bin)

    # -------------------------------------------------------------------------
    # Experiment Suite 6: Market-Regime Filtering Controlled Comparison
    # -------------------------------------------------------------------------
    print("\n[Suite 6/6] Testing Market-Regime Conditional Filtering...")
    for h in [5, 20]:
        for regime in ["BULL_REGIME", "BEAR_REGIME"]:
            exp_id = f"EXP_H{h}_lgb_regime_{regime}"
            print(f"  Executing {exp_id}...")
            res, _, _ = run_experiment(exp_id, horizon=h, target_type="3class", model_type="lightgbm", feature_subset="group_full_regime", regime_filter=regime)
            leaderboard.append(res)

    # Save Experiment Leaderboard
    df_leaderboard = pd.DataFrame(leaderboard)
    df_leaderboard.to_csv(REPORT_DIR / "experiment_leaderboard.csv", index=False)
    
    with open(REPORT_DIR / "experiment_leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    print("\n" + "=" * 80)
    print("RESEARCH EXPERIMENT SUITE COMPLETED")
    print(f"Total experiments executed: {len(leaderboard)}")
    print("=" * 80)

    # Print top models by Balanced Accuracy
    print("\nTOP 10 MODELS BY TEST BALANCED ACCURACY (3-CLASS):")
    top_3c = df_leaderboard[df_leaderboard["target_type"] == "3class"].sort_values("test_bal_acc", ascending=False).head(10)
    print(top_3c[["exp_id", "horizon", "model_type", "num_features", "test_bal_acc", "test_acc", "test_macro_f1", "milestone_40_met"]].to_string(index=False))

    print("\nTOP MODELS FOR BINARY CLASSIFICATION (UP vs DOWN):")
    top_bin = df_leaderboard[df_leaderboard["target_type"] == "binary"].sort_values("test_bal_acc", ascending=False).head(8)
    print(top_bin[["exp_id", "horizon", "model_type", "sample_coverage_pct", "test_bal_acc", "test_acc", "test_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()

