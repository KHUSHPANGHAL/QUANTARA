"""
Quantara AI — Phase 3.4: Versioned Model Suite & Validation Threshold Optimization
Prediction Horizon: 5-Day Forward Returns (horizon_5d_v2)
Target: 3-Class Categorical (DOWN: < -1.0%, HOLD: [-1.0%, +1.0%], UP: > +1.0%)
Target Basis: Adj_Open(t+6) / Adj_Open(t+1) - 1 (Execution-Aligned)

Key Methodology:
1. Strict Split Isolation:
   - Preprocessing / Scaler fit strictly on X_train_v2.
   - Model selection, early stopping, and decision threshold search performed strictly on validation set.
   - Test set evaluated strictly once with frozen models and frozen decision thresholds.
2. Models Evaluated:
   - Baseline 0: Dummy Majority Classifier
   - Baseline 1: Standardized Multinomial Logistic Regression
   - Baseline 2: Random Forest Classifier
   - Primary: LightGBM Classifier (GBDT)
3. Threshold Grid Search (Validation Only):
   - P(UP) candidate thresholds: [0.333, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]
   - Evaluated for trade count, precision, hit rate, and net return after statutory round-trip cost (0.3938%).
4. Artifacts saved strictly to models/horizon_5d_v2/.
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
DATA_DIR = ROOT / "data" / "splits_v2" / "horizon_5d"
OUTPUT_DIR = ROOT / "models" / "horizon_5d_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"DOWN": 0, "HOLD": 1, "UP": 2}
INV_CLASS_MAP = {0: "DOWN", 1: "HOLD", 2: "UP"}
CLASS_NAMES = ["DOWN", "HOLD", "UP"]

# Realistic Round-Trip Transaction Cost (NSE Delivery Schedule + Slippage):
# Buy statutory: 0.1544%, Sell statutory: 0.1394%, Slippage: 0.100% round-trip -> Total = 0.3938%
ROUND_TRIP_COST = 0.003938


def load_v2_datasets():
    """Load train, validation, and test datasets from v2 pipeline."""
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
    """Calculate accuracy, balanced accuracy, macro/weighted F1, log loss, and confusion matrix."""
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


def evaluate_threshold_grid_on_validation(model, X_val, y_val, forward_returns_val, scaler=None):
    """
    Evaluate candidate decision thresholds for P(UP) strictly on the validation set.
    Candidate thresholds: [0.333, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50].
    """
    if scaler is not None:
        X_val_scaled = scaler.transform(X_val)
        probs = model.predict_proba(X_val_scaled)
    else:
        probs = model.predict_proba(X_val)
        
    p_up = probs[:, 2]
    thresholds = [0.333, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]
    
    val_benchmark_gross = float(np.mean(forward_returns_val))
    val_benchmark_net = val_benchmark_gross - ROUND_TRIP_COST
    
    grid_results = []
    for th in thresholds:
        # A sample is flagged as candidate BUY if P(UP) >= th AND P(UP) > P(DOWN)
        signal_mask = (p_up >= th) & (p_up > probs[:, 0])
        trade_count = int(signal_mask.sum())
        
        if trade_count > 0:
            trade_returns = forward_returns_val[signal_mask]
            gross_mean = float(np.mean(trade_returns))
            net_mean = gross_mean - ROUND_TRIP_COST
            win_rate = float(np.mean(trade_returns > 0)) * 100.0
            target_hit_rate = float(np.mean(trade_returns >= 0.010)) * 100.0  # Hit 1.0% target
            
            # Cumulative compounded return simulation on validation trades
            cum_net = float(np.prod(1.0 + trade_returns - ROUND_TRIP_COST) - 1.0)
            
            # Precision of UP class under this threshold
            # True UP is y_val == 2
            precision_up = float(np.mean(y_val[signal_mask] == 2)) * 100.0
        else:
            gross_mean = 0.0
            net_mean = 0.0
            win_rate = 0.0
            target_hit_rate = 0.0
            cum_net = 0.0
            precision_up = 0.0
            
        grid_results.append({
            "threshold": th,
            "validation_trades": trade_count,
            "trade_frequency_pct": float(trade_count / len(y_val) * 100.0),
            "precision_up_pct": precision_up,
            "target_hit_pct": target_hit_rate,
            "win_rate_pct": win_rate,
            "mean_gross_return_pct": gross_mean * 100.0,
            "mean_net_return_pct": net_mean * 100.0,
            "alpha_net_vs_benchmark_pct": (net_mean - val_benchmark_net) * 100.0,
        })
        
    return pd.DataFrame(grid_results)


def simulate_trading(probs, forward_returns, threshold=0.40, round_trip_cost=ROUND_TRIP_COST):
    """Simulate trade-level financial metrics under a specific threshold."""
    p_down, p_hold, p_up = probs[:, 0], probs[:, 1], probs[:, 2]
    
    # BUY if P(UP) >= threshold and P(UP) > P(DOWN)
    up_mask = (p_up >= threshold) & (p_up > p_down)
    # SHORT/EXIT if P(DOWN) > P(UP) and P(DOWN) >= threshold
    down_mask = (p_down >= threshold) & (p_down > p_up)
    
    bench_gross = float(np.mean(forward_returns))
    bench_net = bench_gross - round_trip_cost
    
    up_ret_gross = float(np.mean(forward_returns[up_mask])) if up_mask.sum() > 0 else 0.0
    up_ret_net = up_ret_gross - round_trip_cost if up_mask.sum() > 0 else 0.0
    
    down_ret_gross = float(np.mean(forward_returns[down_mask])) if down_mask.sum() > 0 else 0.0
    down_short_ret_net = -down_ret_gross - round_trip_cost if down_mask.sum() > 0 else 0.0
    
    up_profitable_pct = float(np.mean(forward_returns[up_mask] > 0)) if up_mask.sum() > 0 else 0.0
    up_target_hit_pct = float(np.mean(forward_returns[up_mask] >= 0.010)) if up_mask.sum() > 0 else 0.0
    
    return {
        "round_trip_cost_applied": round_trip_cost,
        "selected_p_up_threshold": threshold,
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
        "short_signals": {
            "signal_count": int(down_mask.sum()),
            "mean_underlying_return": down_ret_gross,
            "mean_short_net_return": down_short_ret_net,
        }
    }


def per_symbol_analysis(y_true, probs, forward_returns, symbols, threshold=0.40):
    """Calculate test metrics by individual ticker symbol."""
    records = []
    p_up = probs[:, 2]
    p_down = probs[:, 0]
    preds = np.argmax(probs, axis=1)
    
    for sym in sorted(symbols.unique()):
        mask = (symbols == sym).values
        y_t = y_true[mask]
        y_p = preds[mask]
        ret = forward_returns[mask]
        
        acc = float(accuracy_score(y_t, y_p))
        bal_acc = float(balanced_accuracy_score(y_t, y_p))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        
        sym_p_up = p_up[mask]
        sym_p_down = p_down[mask]
        buy_mask = (sym_p_up >= threshold) & (sym_p_up > sym_p_down)
        buy_ret = float(np.mean(ret[buy_mask])) if buy_mask.sum() > 0 else 0.0
        bench_ret = float(np.mean(ret))
        
        records.append({
            "symbol": sym,
            "test_samples": int(mask.sum()),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "benchmark_5d_return": bench_ret,
            "buy_signals_count": int(buy_mask.sum()),
            "buy_mean_5d_return": buy_ret,
            "alpha_vs_benchmark": buy_ret - bench_ret,
        })
    return records


def main():
    print("=" * 70)
    print("QUANTARA AI — PHASE 3.4: MODEL SUITE & THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    data = load_v2_datasets()
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["validation"]["X"], data["validation"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    
    print(f"  Training samples:   {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Testing samples:    {len(X_test):,}")
    print(f"  Feature count:      {X_train.shape[1]}")
    
    # Preprocessing Scaler fit STRICTLY on Training Set
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    
    models_evaluated = {}
    
    # -----------------------------------------------------------------
    # Baseline 0: Dummy Majority Classifier
    # -----------------------------------------------------------------
    print("\n[Step 1/5] Training Baseline 0: Dummy Majority Classifier...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    val_dummy = compute_metrics(y_val, dummy.predict(X_val), dummy.predict_proba(X_val))
    test_dummy = compute_metrics(y_test, dummy.predict(X_test), dummy.predict_proba(X_test))
    models_evaluated["Dummy Baseline"] = {"val": val_dummy, "test": test_dummy, "model": dummy}
    print(f"  Val Bal Acc: {val_dummy['balanced_accuracy']:.4f} | Test Bal Acc: {test_dummy['balanced_accuracy']:.4f}")
    
    # -----------------------------------------------------------------
    # Baseline 1: Standardized Multinomial Logistic Regression
    # -----------------------------------------------------------------
    print("\n[Step 2/5] Training Baseline 1: Standardized Multinomial Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    val_lr = compute_metrics(y_val, lr.predict(X_val_scaled), lr.predict_proba(X_val_scaled))
    test_lr = compute_metrics(y_test, lr.predict(X_test_scaled), lr.predict_proba(X_test_scaled))
    models_evaluated["Logistic Regression"] = {"val": val_lr, "test": test_lr, "model": lr, "scaler": scaler}
    print(f"  Val Bal Acc: {val_lr['balanced_accuracy']:.4f} | Test Bal Acc: {test_lr['balanced_accuracy']:.4f} | Test Acc: {test_lr['accuracy']:.4f}")
    
    # -----------------------------------------------------------------
    # Baseline 2: Random Forest Classifier
    # -----------------------------------------------------------------
    print("\n[Step 3/5] Training Baseline 2: Random Forest Classifier (100 trees)...")
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
    print(f"  Val Bal Acc: {val_rf['balanced_accuracy']:.4f} | Test Bal Acc: {test_rf['balanced_accuracy']:.4f} | Test Acc: {test_rf['accuracy']:.4f}")
    
    # -----------------------------------------------------------------
    # Primary Model: LightGBM Classifier (Validation Hyperparameter Search)
    # -----------------------------------------------------------------
    print("\n[Step 4/5] Training Primary Model: LightGBM Classifier (Tuned on Validation)...")
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
    print(f"  Val Bal Acc: {val_lgb['balanced_accuracy']:.4f} | Test Bal Acc: {test_lgb['balanced_accuracy']:.4f} | Test Acc: {test_lgb['accuracy']:.4f}")

    # -----------------------------------------------------------------
    # Threshold Grid Evaluation (STRICTLY ON VALIDATION SET)
    # -----------------------------------------------------------------
    print("\n[Step 5/5] Evaluating Decision Threshold Grid Strictly on Validation Set...")
    print("Evaluating LightGBM validation threshold grid:")
    lgb_grid_val = evaluate_threshold_grid_on_validation(best_lgb, X_val, y_val, data["validation"]["forward_return"])
    print(lgb_grid_val.to_string(index=False))
    lgb_grid_val.to_csv(OUTPUT_DIR / "validation_threshold_grid_lgb.csv", index=False)

    print("\nEvaluating Logistic Regression validation threshold grid:")
    lr_grid_val = evaluate_threshold_grid_on_validation(lr, X_val, y_val, data["validation"]["forward_return"], scaler=scaler)
    print(lr_grid_val.to_string(index=False))
    lr_grid_val.to_csv(OUTPUT_DIR / "validation_threshold_grid_lr.csv", index=False)

    # -----------------------------------------------------------------
    # Threshold Selection Decision on Validation Set
    # -----------------------------------------------------------------
    # We select the threshold that achieves maximum positive net alpha on validation
    # while maintaining statistically meaningful trade frequency (> 50 trades).
    # If all positive alpha thresholds have low trade count, select the threshold with the highest net return.
    valid_lgb_candidates = lgb_grid_val[lgb_grid_val["validation_trades"] >= 40]
    if len(valid_lgb_candidates) > 0:
        best_lgb_th_row = valid_lgb_candidates.sort_values("mean_net_return_pct", ascending=False).iloc[0]
        selected_lgb_threshold = float(best_lgb_th_row["threshold"])
    else:
        selected_lgb_threshold = 0.38

    valid_lr_candidates = lr_grid_val[lr_grid_val["validation_trades"] >= 40]
    if len(valid_lr_candidates) > 0:
        best_lr_th_row = valid_lr_candidates.sort_values("mean_net_return_pct", ascending=False).iloc[0]
        selected_lr_threshold = float(best_lr_th_row["threshold"])
    else:
        selected_lr_threshold = 0.38

    print(f"\nFrozen Decision Thresholds (Selected from Validation Set Only):")
    print(f"  LightGBM Selected Threshold:            P(UP) >= {selected_lgb_threshold:.3f}")
    print(f"  Logistic Regression Selected Threshold: P(UP) >= {selected_lr_threshold:.3f}")

    # -----------------------------------------------------------------
    # TEST SET EVALUATION (FROZEN MODELS & FROZEN THRESHOLDS)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST SET BENCHMARK SUMMARY (EVALUATED ONCE)")
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

    # Test Set Probability Predictions
    test_lgb_probs = best_lgb.predict_proba(X_test)
    test_lr_probs = lr.predict_proba(X_test_scaled)

    # Test Financial Simulation using Frozen Thresholds
    fin_sim_lgb = simulate_trading(test_lgb_probs, data["test"]["forward_return"], threshold=selected_lgb_threshold)
    fin_sim_lr = simulate_trading(test_lr_probs, data["test"]["forward_return"], threshold=selected_lr_threshold)

    # Feature Importance for LightGBM
    gain_imp = best_lgb.booster_.feature_importance(importance_type="gain")
    split_imp = best_lgb.booster_.feature_importance(importance_type="split")
    fi_df = pd.DataFrame({
        "feature": data["train"]["feature_names"],
        "importance_gain": gain_imp,
        "importance_split": split_imp,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Per-Symbol Analysis on Test Set
    sym_df = pd.DataFrame(per_symbol_analysis(
        y_test, test_lgb_probs, data["test"]["forward_return"],
        data["test"]["meta"]["Symbol"], threshold=selected_lgb_threshold
    ))
    sym_df.to_csv(OUTPUT_DIR / "per_symbol_test_metrics.csv", index=False)

    # Model Checkpoints Serialization
    joblib.dump(best_lgb, OUTPUT_DIR / "lightgbm_5d_classifier.joblib")
    joblib.dump(lr, OUTPUT_DIR / "logistic_regression_baseline.joblib")
    joblib.dump(rf, OUTPUT_DIR / "random_forest_baseline.joblib")

    # Complete Metrics JSON
    full_report = {
        "pipeline_version": "v2_execution_aligned",
        "benchmark_comparison": summary_rows,
        "selected_decision_thresholds": {
            "lightgbm_p_up_threshold": selected_lgb_threshold,
            "logistic_regression_p_up_threshold": selected_lr_threshold,
            "threshold_selection_criterion": "highest validation net return with >= 40 trades",
        },
        "lightgbm": {
            "best_hyperparameters": best_lgb_params,
            "best_iteration": int(best_lgb.best_iteration_) if hasattr(best_lgb, "best_iteration_") else None,
            "validation_metrics": val_lgb,
            "test_metrics": test_lgb,
            "validation_threshold_grid": lgb_grid_val.to_dict(orient="records"),
            "test_financial_simulation": fin_sim_lgb,
        },
        "logistic_regression": {
            "validation_metrics": val_lr,
            "test_metrics": test_lr,
            "validation_threshold_grid": lr_grid_val.to_dict(orient="records"),
            "test_financial_simulation": fin_sim_lr,
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

    with open(OUTPUT_DIR / "phase3_4_model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nAll Phase 3.4 model artifacts saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

