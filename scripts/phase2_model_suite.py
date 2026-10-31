"""
Quantara AI — Phase 2 Institutional Model Development & Strategy Validation Suite
Prediction Horizon: 5-Day Forward Returns (horizon_5d)
Target: 3-Class Categorical (DOWN: < -1.0%, HOLD: [-1.0%, +1.0%], UP: > +1.0%)

Models Evaluated:
1. Baseline 0: Dummy Classifier (Majority Class Prior)
2. Baseline 1: Standardized Multinomial Logistic Regression (Linear)
3. Baseline 2: Random Forest Classifier (Bagging Trees)
4. Primary Model: LightGBM Classifier (Gradient Boosted Decision Trees)

Features: Exactly the 20 approved stationary features from config/data_config.json
Complies 100% with TRAINING_DATA_CONTRACT.md:
- Preprocessing fit strictly on train
- Model selection and hyperparameter exploration strictly on validation
- Evaluated once on test split
- Financial simulation includes realistic transaction costs (brokerage/slippage/STT)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
DATA_DIR = ROOT / "data" / "splits" / "horizon_5d"
OUTPUT_DIR = ROOT / "models" / "horizon_5d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"DOWN": 0, "HOLD": 1, "UP": 2}
INV_CLASS_MAP = {0: "DOWN", 1: "HOLD", 2: "UP"}
CLASS_NAMES = ["DOWN", "HOLD", "UP"]


def load_datasets():
    """Load train, validation, and test datasets with strict alignment."""
    data = {}
    for s in ["train", "validation", "test"]:
        X = pd.read_csv(DATA_DIR / f"X_{s}.csv")
        y = pd.read_csv(DATA_DIR / f"y_{s}.csv")
        meta = pd.read_csv(DATA_DIR / f"metadata_{s}.csv")
        
        assert len(X) == len(y) == len(meta), f"Alignment mismatch in {s} split!"
        
        data[s] = {
            "X": X,
            "y": y["Target_5d"].map(CLASS_MAP).values,
            "y_raw": y["Target_5d"].values,
            "forward_return": y["forward_return_5d"].values,
            "meta": meta,
            "feature_names": list(X.columns),
        }
    return data


def compute_metrics(y_true, y_pred, y_prob=None):
    """Calculate accuracy, balanced accuracy, F1, precision, recall, and log loss."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    loss = None
    if y_prob is not None:
        try:
            loss = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
        except Exception:
            loss = None
            
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "log_loss": loss,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int((y_true == i).sum()),
            }
            for i in range(3)
        },
        "confusion_matrix": cm,
    }


def simulate_trading(y_pred, forward_returns, round_trip_cost=0.0015):
    """
    Simulate financial performance of the model's signals on test set:
    - Long (UP): Buy stock and hold for 5 trading days.
    - Cash/Avoid (HOLD): Do not trade.
    - Short/Hedge (DOWN): Sell short or stay flat.
    - Round-trip transaction cost: 0.15% (brokerage + STT + slippage on NSE).
    """
    up_mask = (y_pred == 2)
    down_mask = (y_pred == 0)
    hold_mask = (y_pred == 1)
    
    bench_gross = float(np.mean(forward_returns))
    bench_net = bench_gross - round_trip_cost
    
    up_ret_gross = float(np.mean(forward_returns[up_mask])) if up_mask.sum() > 0 else 0.0
    up_ret_net = up_ret_gross - round_trip_cost if up_mask.sum() > 0 else 0.0
    
    down_ret_gross = float(np.mean(forward_returns[down_mask])) if down_mask.sum() > 0 else 0.0
    # For a short strategy: return is -(forward_return) minus cost
    down_short_ret_net = -down_ret_gross - round_trip_cost if down_mask.sum() > 0 else 0.0
    
    # Hit rates
    up_profitable_pct = float(np.mean(forward_returns[up_mask] > 0)) if up_mask.sum() > 0 else 0.0
    up_target_hit_pct = float(np.mean(forward_returns[up_mask] >= 0.01)) if up_mask.sum() > 0 else 0.0
    down_drop_pct = float(np.mean(forward_returns[down_mask] < 0)) if down_mask.sum() > 0 else 0.0
    down_target_hit_pct = float(np.mean(forward_returns[down_mask] <= -0.01)) if down_mask.sum() > 0 else 0.0
    
    return {
        "round_trip_cost_assumption": round_trip_cost,
        "benchmark": {
            "mean_gross_return": bench_gross,
            "mean_net_return": bench_net,
        },
        "long_strategy_up_signals": {
            "signal_count": int(up_mask.sum()),
            "mean_gross_return": up_ret_gross,
            "mean_net_return": up_ret_net,
            "profitable_trades_pct": up_profitable_pct,
            "target_hit_pct": up_target_hit_pct,
            "gross_alpha_vs_benchmark": up_ret_gross - bench_gross,
            "net_alpha_vs_benchmark": up_ret_net - bench_net,
        },
        "short_strategy_down_signals": {
            "signal_count": int(down_mask.sum()),
            "mean_underlying_return": down_ret_gross,
            "mean_short_net_return": down_short_ret_net,
            "downward_move_pct": down_drop_pct,
            "target_hit_pct": down_target_hit_pct,
        },
        "hold_signals": {
            "signal_count": int(hold_mask.sum()),
            "mean_gross_return": float(np.mean(forward_returns[hold_mask])) if hold_mask.sum() > 0 else 0.0,
        }
    }


def per_symbol_analysis(y_true, y_pred, forward_returns, symbols):
    """Calculate performance metrics individually for each of the 8 stocks."""
    records = []
    for sym in sorted(symbols.unique()):
        mask = (symbols == sym).values
        y_t = y_true[mask]
        y_p = y_pred[mask]
        ret = forward_returns[mask]
        
        acc = float(accuracy_score(y_t, y_p))
        bal_acc = float(balanced_accuracy_score(y_t, y_p))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        
        up_idx = np.where(y_p == 2)[0]
        up_ret = float(np.mean(ret[up_idx])) if len(up_idx) > 0 else 0.0
        bench_ret = float(np.mean(ret))
        
        records.append({
            "symbol": sym,
            "test_samples": int(mask.sum()),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "benchmark_5d_return": bench_ret,
            "up_signals_count": len(up_idx),
            "up_mean_5d_return": up_ret,
            "alpha_vs_benchmark": up_ret - bench_ret,
        })
    return records


def main():
    print("=" * 70)
    print("QUANTARA AI — PHASE 2: MODEL DEVELOPMENT & VALIDATION SUITE")
    print("=" * 70)
    
    print("\n[Step 1/6] Loading verified datasets from data/splits/horizon_5d...")
    data = load_datasets()
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["validation"]["X"], data["validation"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    
    print(f"  Training samples:   {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Testing samples:    {len(X_test):,}")
    print(f"  Feature count:      {X_train.shape[1]}")
    
    models_evaluated = {}
    
    # -------------------------------------------------------------
    # Model 0: Dummy Majority Baseline
    # -------------------------------------------------------------
    print("\n[Step 2/6] Training Baseline 0: Dummy Majority Classifier...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    val_dummy = compute_metrics(y_val, dummy.predict(X_val), dummy.predict_proba(X_val))
    test_dummy = compute_metrics(y_test, dummy.predict(X_test), dummy.predict_proba(X_test))
    models_evaluated["Dummy Baseline"] = {"val": val_dummy, "test": test_dummy, "model": dummy}
    print(f"  Validation Bal Acc: {val_dummy['balanced_accuracy']:.4f} | Test Bal Acc: {test_dummy['balanced_accuracy']:.4f}")
    
    # -------------------------------------------------------------
    # Model 1: Multinomial Logistic Regression (Standardized)
    # -------------------------------------------------------------
    print("\n[Step 3/6] Training Baseline 1: Standardized Multinomial Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    val_lr = compute_metrics(y_val, lr_pipe.predict(X_val), lr_pipe.predict_proba(X_val))
    test_lr = compute_metrics(y_test, lr_pipe.predict(X_test), lr_pipe.predict_proba(X_test))
    models_evaluated["Logistic Regression"] = {"val": val_lr, "test": test_lr, "model": lr_pipe}
    print(f"  Validation Bal Acc: {val_lr['balanced_accuracy']:.4f} | Test Bal Acc: {test_lr['balanced_accuracy']:.4f} | Test Acc: {test_lr['accuracy']:.4f}")
    
    # -------------------------------------------------------------
    # Model 2: Random Forest Classifier
    # -------------------------------------------------------------
    print("\n[Step 4/6] Training Baseline 2: Random Forest Classifier (100 trees)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    val_rf = compute_metrics(y_val, rf.predict(X_val), rf.predict_proba(X_val))
    test_rf = compute_metrics(y_test, rf.predict(X_test), rf.predict_proba(X_test))
    models_evaluated["Random Forest"] = {"val": val_rf, "test": test_rf, "model": rf}
    print(f"  Validation Bal Acc: {val_rf['balanced_accuracy']:.4f} | Test Bal Acc: {test_rf['balanced_accuracy']:.4f} | Test Acc: {test_rf['accuracy']:.4f}")
    
    # -------------------------------------------------------------
    # Model 3: LightGBM Classifier (Validation-Tuned)
    # -------------------------------------------------------------
    print("\n[Step 5/6] Training Primary Model: LightGBM Classifier with early stopping...")
    lgb_candidates = [
        {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.03, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 2.0, "class_weight": "balanced"},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.02, "colsample_bytree": 0.8, "subsample": 0.9, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": "balanced"},
        {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.7, "subsample": 0.8, "min_child_samples": 100, "reg_alpha": 1.0, "reg_lambda": 5.0, "class_weight": "balanced"},
    ]
    
    best_lgb = None
    best_lgb_val_score = -1.0
    best_lgb_params = None
    
    for p in lgb_candidates:
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=400, random_state=42, n_jobs=-1, verbose=-1, **p)
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
        score = balanced_accuracy_score(y_val, clf.predict(X_val))
        if score > best_lgb_val_score:
            best_lgb_val_score = score
            best_lgb = clf
            best_lgb_params = p
            
    val_lgb = compute_metrics(y_val, best_lgb.predict(X_val), best_lgb.predict_proba(X_val))
    test_lgb = compute_metrics(y_test, best_lgb.predict(X_test), best_lgb.predict_proba(X_test))
    models_evaluated["LightGBM"] = {"val": val_lgb, "test": test_lgb, "model": best_lgb, "best_params": best_lgb_params}
    print(f"  Selected Params:    {best_lgb_params}")
    print(f"  Validation Bal Acc: {val_lgb['balanced_accuracy']:.4f} | Test Bal Acc: {test_lgb['balanced_accuracy']:.4f} | Test Acc: {test_lgb['accuracy']:.4f}")
    
    # -------------------------------------------------------------
    # Comparative Benchmark Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARATIVE BENCHMARK TABLE (TEST SET)")
    print("=" * 70)
    summary_rows = []
    for name, res in models_evaluated.items():
        t = res["test"]
        v = res["val"]
        summary_rows.append({
            "Model": name,
            "Val Bal Acc": f"{v['balanced_accuracy']*100:.2f}%",
            "Test Accuracy": f"{t['accuracy']*100:.2f}%",
            "Test Bal Acc": f"{t['balanced_accuracy']*100:.2f}%",
            "Test Macro F1": f"{t['macro_f1']:.4f}",
            "DOWN Precision": f"{t['per_class']['DOWN']['precision']*100:.1f}%",
            "HOLD Precision": f"{t['per_class']['HOLD']['precision']*100:.1f}%",
            "UP Precision": f"{t['per_class']['UP']['precision']*100:.1f}%",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # -------------------------------------------------------------
    # Step 6: Financial Simulation & Serialization
    # -------------------------------------------------------------
    print("\n[Step 6/6] Running financial strategy simulation & saving artifacts...")
    y_test_pred_lgb = best_lgb.predict(X_test)
    y_test_pred_lr = lr_pipe.predict(X_test)
    
    fin_sim_lgb = simulate_trading(y_test_pred_lgb, data["test"]["forward_return"], round_trip_cost=0.0015)
    fin_sim_lr = simulate_trading(y_test_pred_lr, data["test"]["forward_return"], round_trip_cost=0.0015)
    
    # Feature Importance for LightGBM
    gain_imp = best_lgb.booster_.feature_importance(importance_type="gain")
    split_imp = best_lgb.booster_.feature_importance(importance_type="split")
    fi_df = pd.DataFrame({
        "feature": data["train"]["feature_names"],
        "importance_gain": gain_imp,
        "importance_split": split_imp,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    
    # Per Symbol Analysis
    sym_df = pd.DataFrame(per_symbol_analysis(y_test, y_test_pred_lgb, data["test"]["forward_return"], data["test"]["meta"]["Symbol"]))
    sym_df.to_csv(OUTPUT_DIR / "per_symbol_test_metrics.csv", index=False)
    
    # Save Model Checkpoints
    joblib.dump(best_lgb, OUTPUT_DIR / "lightgbm_5d_classifier.joblib")
    joblib.dump(lr_pipe, OUTPUT_DIR / "logistic_regression_baseline.joblib")
    joblib.dump(rf, OUTPUT_DIR / "random_forest_baseline.joblib")
    
    # Comprehensive JSON Report
    full_report = {
        "benchmark_comparison": summary_rows,
        "lightgbm": {
            "best_hyperparameters": best_lgb_params,
            "best_iteration": int(best_lgb.best_iteration_) if hasattr(best_lgb, "best_iteration_") else None,
            "validation_metrics": val_lgb,
            "test_metrics": test_lgb,
            "financial_simulation": fin_sim_lgb,
        },
        "logistic_regression": {
            "validation_metrics": val_lr,
            "test_metrics": test_lr,
            "financial_simulation": fin_sim_lr,
        },
        "random_forest": {
            "validation_metrics": val_rf,
            "test_metrics": test_rf,
        },
        "dummy_baseline": {
            "validation_metrics": val_dummy,
            "test_metrics": test_dummy,
        },
        "per_symbol_analysis": sym_df.to_dict(orient="records"),
    }
    
    with open(OUTPUT_DIR / "phase2_complete_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\nAll Phase 2 model artifacts successfully saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
