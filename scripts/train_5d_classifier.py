"""
Quantara AI — 5-Day Horizon Classifier Training & Evaluation Pipeline
Complies strictly with TRAINING_DATA_CONTRACT.md:
- Uses horizon_5d split data only
- Strictly preserves row order across X, y, and metadata
- Uses only the 20 approved numeric features in X
- Fits preprocessing (scalers) strictly on train
- Selects models/hyperparameters using validation split
- Evaluates test split once at the end
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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
    classification_report,
)
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "splits" / "horizon_5d"
OUTPUT_DIR = ROOT / "models" / "horizon_5d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"DOWN": 0, "HOLD": 1, "UP": 2}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = ["DOWN", "HOLD", "UP"]


def load_data():
    """Load train, validation, and test sets from data/splits/horizon_5d."""
    splits = {}
    for s in ["train", "validation", "test"]:
        X = pd.read_csv(DATA_DIR / f"X_{s}.csv")
        y = pd.read_csv(DATA_DIR / f"y_{s}.csv")
        meta = pd.read_csv(DATA_DIR / f"metadata_{s}.csv")
        
        # Verify strict alignment
        assert len(X) == len(y) == len(meta), f"Row count mismatch in {s} split!"
        
        y_encoded = y["Target_5d"].map(CLASS_MAP).values
        forward_return = y["forward_return_5d"].values
        
        splits[s] = {
            "X": X,
            "y_raw": y["Target_5d"].values,
            "y": y_encoded,
            "forward_return": forward_return,
            "meta": meta,
            "feature_names": list(X.columns),
        }
    return splits


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            name: {
                "precision": float(prec_per_class[i]),
                "recall": float(rec_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int((y_true == i).sum()),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }
    
    if y_prob is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
        except Exception:
            metrics["log_loss"] = None
            
    return metrics


def train_baselines(data):
    """Train Dummy Classifier and Multinomial Logistic Regression baselines."""
    results = {}
    
    # 1. Dummy Baseline (majority class)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(data["train"]["X"], data["train"]["y"])
    
    y_pred_val = dummy.predict(data["validation"]["X"])
    y_prob_val = dummy.predict_proba(data["validation"]["X"])
    val_metrics = compute_metrics(data["validation"]["y"], y_pred_val, y_prob_val)
    
    y_pred_test = dummy.predict(data["test"]["X"])
    y_prob_test = dummy.predict_proba(data["test"]["X"])
    test_metrics = compute_metrics(data["test"]["y"], y_pred_test, y_prob_test)
    
    results["dummy"] = {
        "model": dummy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    
    # 2. Logistic Regression (Multinomial with StandardScaler fit ONLY on train)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    lr_pipe.fit(data["train"]["X"], data["train"]["y"])
    
    y_pred_val_lr = lr_pipe.predict(data["validation"]["X"])
    y_prob_val_lr = lr_pipe.predict_proba(data["validation"]["X"])
    val_metrics_lr = compute_metrics(data["validation"]["y"], y_pred_val_lr, y_prob_val_lr)
    
    y_pred_test_lr = lr_pipe.predict(data["test"]["X"])
    y_prob_test_lr = lr_pipe.predict_proba(data["test"]["X"])
    test_metrics_lr = compute_metrics(data["test"]["y"], y_pred_test_lr, y_prob_test_lr)
    
    results["logistic_regression"] = {
        "model": lr_pipe,
        "val_metrics": val_metrics_lr,
        "test_metrics": test_metrics_lr,
    }
    
    return results


def train_lightgbm_grid(data):
    """Explore hyperparameter configurations on validation split and select best model."""
    candidates = [
        {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.03, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": "balanced"},
        {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.03, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 2.0, "class_weight": "balanced"},
        {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.7, "subsample": 0.8, "min_child_samples": 100, "reg_alpha": 1.0, "reg_lambda": 5.0, "class_weight": "balanced"},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.02, "colsample_bytree": 0.8, "subsample": 0.9, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": "balanced"},
        {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.03, "colsample_bytree": 0.8, "subsample": 0.8, "min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 2.0, "class_weight": None},
    ]
    
    best_model = None
    best_val_score = -1.0
    best_params = None
    best_val_metrics = None
    all_runs = []
    
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["validation"]["X"], data["validation"]["y"]
    
    for idx, params in enumerate(candidates):
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params
        )
        
        # Fit with early stopping on validation split
        callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)]
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        
        y_pred_val = clf.predict(X_val)
        y_prob_val = clf.predict_proba(X_val)
        val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)
        
        score = val_metrics["balanced_accuracy"]
        all_runs.append({
            "candidate_id": idx,
            "params": params,
            "best_iteration": int(clf.best_iteration_) if hasattr(clf, "best_iteration_") else None,
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        })
        
        if score > best_val_score:
            best_val_score = score
            best_model = clf
            best_params = params
            best_val_metrics = val_metrics
            
    return best_model, best_params, best_val_metrics, all_runs


def financial_simulation(y_pred, forward_returns, threshold=0.01):
    """
    Simulate financial performance of the model's signals on test set:
    - Long (UP): Buy stock and hold for 5 days
    - Short/Avoid (DOWN): Short stock or stay flat
    - Benchmark: Equal-weighted buy-and-hold forward return
    """
    up_idx = np.where(y_pred == 2)[0]
    down_idx = np.where(y_pred == 0)[0]
    hold_idx = np.where(y_pred == 1)[0]
    
    avg_benchmark = float(np.mean(forward_returns))
    avg_up_return = float(np.mean(forward_returns[up_idx])) if len(up_idx) > 0 else 0.0
    avg_down_return = float(np.mean(forward_returns[down_idx])) if len(down_idx) > 0 else 0.0
    avg_hold_return = float(np.mean(forward_returns[hold_idx])) if len(hold_idx) > 0 else 0.0
    
    # Hit rate: percentage of UP calls that actually gained >= threshold (+1.0%)
    up_hit_rate = float(np.mean(forward_returns[up_idx] > threshold)) if len(up_idx) > 0 else 0.0
    # Positive rate: percentage of UP calls that had positive return (> 0)
    up_positive_rate = float(np.mean(forward_returns[up_idx] > 0)) if len(up_idx) > 0 else 0.0
    
    # DOWN hit rate: percentage of DOWN calls that actually lost <= -threshold (-1.0%)
    down_hit_rate = float(np.mean(forward_returns[down_idx] < -threshold)) if len(down_idx) > 0 else 0.0
    down_negative_rate = float(np.mean(forward_returns[down_idx] < 0)) if len(down_idx) > 0 else 0.0
    
    return {
        "benchmark_mean_return": avg_benchmark,
        "up_signals_count": len(up_idx),
        "up_signals_mean_return": avg_up_return,
        "up_signals_hit_rate_target": up_hit_rate,
        "up_signals_positive_rate": up_positive_rate,
        "down_signals_count": len(down_idx),
        "down_signals_mean_return": avg_down_return,
        "down_signals_hit_rate_target": down_hit_rate,
        "down_signals_negative_rate": down_negative_rate,
        "hold_signals_count": len(hold_idx),
        "hold_signals_mean_return": avg_hold_return,
        "long_excess_over_benchmark": avg_up_return - avg_benchmark,
    }


def per_symbol_breakdown(y_true, y_pred, forward_returns, symbols):
    """Break down model metrics and forward returns by each individual stock."""
    records = []
    unique_symbols = sorted(symbols.unique())
    for sym in unique_symbols:
        mask = (symbols == sym).values
        y_t = y_true[mask]
        y_p = y_pred[mask]
        f_ret = forward_returns[mask]
        
        acc = accuracy_score(y_t, y_p)
        bal_acc = balanced_accuracy_score(y_t, y_p)
        macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        
        up_idx = np.where(y_p == 2)[0]
        avg_up_ret = float(np.mean(f_ret[up_idx])) if len(up_idx) > 0 else 0.0
        avg_bench = float(np.mean(f_ret))
        
        records.append({
            "symbol": sym,
            "samples": int(mask.sum()),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "benchmark_5d_return": avg_bench,
            "up_predicted_count": len(up_idx),
            "up_predicted_5d_return": avg_up_ret,
            "up_alpha": avg_up_ret - avg_bench,
        })
    return records


def main():
    print("=" * 60)
    print("QUANTARA AI — 5-DAY HORIZON CLASSIFIER PIPELINE")
    print("=" * 60)
    
    print("\n[1/6] Loading horizon_5d datasets...")
    data = load_data()
    print(f"  Train samples:      {len(data['train']['X'])}")
    print(f"  Validation samples: {len(data['validation']['X'])}")
    print(f"  Test samples:       {len(data['test']['X'])}")
    print(f"  Features:           {len(data['train']['feature_names'])}")
    
    print("\n[2/6] Training baseline models (Dummy, Logistic Regression)...")
    baseline_res = train_baselines(data)
    dummy_test = baseline_res["dummy"]["test_metrics"]
    lr_test = baseline_res["logistic_regression"]["test_metrics"]
    print(f"  Dummy Baseline Test Acc:        {dummy_test['accuracy']:.4f} (Bal Acc: {dummy_test['balanced_accuracy']:.4f})")
    print(f"  Logistic Regression Test Acc:   {lr_test['accuracy']:.4f} (Bal Acc: {lr_test['balanced_accuracy']:.4f})")
    
    print("\n[3/6] Tuning and training LightGBM on validation split...")
    best_lgb, best_params, val_metrics_lgb, all_runs = train_lightgbm_grid(data)
    print(f"  Selected Best Params: {best_params}")
    print(f"  Validation Balanced Acc: {val_metrics_lgb['balanced_accuracy']:.4f}, Macro F1: {val_metrics_lgb['macro_f1']:.4f}")
    
    print("\n[4/6] Evaluating best LightGBM model on final Test split...")
    y_pred_test = best_lgb.predict(data["test"]["X"])
    y_prob_test = best_lgb.predict_proba(data["test"]["X"])
    test_metrics_lgb = compute_metrics(data["test"]["y"], y_pred_test, y_prob_test)
    
    y_pred_train = best_lgb.predict(data["train"]["X"])
    y_prob_train = best_lgb.predict_proba(data["train"]["X"])
    train_metrics_lgb = compute_metrics(data["train"]["y"], y_pred_train, y_prob_train)
    
    print(f"  Train Acc:      {train_metrics_lgb['accuracy']:.4f} (Bal Acc: {train_metrics_lgb['balanced_accuracy']:.4f})")
    print(f"  Validation Acc: {val_metrics_lgb['accuracy']:.4f} (Bal Acc: {val_metrics_lgb['balanced_accuracy']:.4f})")
    print(f"  Test Acc:       {test_metrics_lgb['accuracy']:.4f} (Bal Acc: {test_metrics_lgb['balanced_accuracy']:.4f})")
    print(f"  Test Macro F1:  {test_metrics_lgb['macro_f1']:.4f}")
    
    print("\n[5/6] Feature importance & Financial sanity check...")
    features = data["train"]["feature_names"]
    gain_importance = best_lgb.booster_.feature_importance(importance_type="gain")
    split_importance = best_lgb.booster_.feature_importance(importance_type="split")
    
    fi_df = pd.DataFrame({
        "feature": features,
        "importance_gain": gain_importance,
        "importance_split": split_importance,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    
    # Financial simulation on test set
    fin_sim = financial_simulation(y_pred_test, data["test"]["forward_return"], threshold=0.01)
    print(f"  Test Benchmark Return:      {fin_sim['benchmark_mean_return']*100:+.2f}%")
    print(f"  Predicted UP Mean Return:   {fin_sim['up_signals_mean_return']*100:+.2f}% (Count: {fin_sim['up_signals_count']})")
    print(f"  Predicted DOWN Mean Return: {fin_sim['down_signals_mean_return']*100:+.2f}% (Count: {fin_sim['down_signals_count']})")
    print(f"  Long Alpha over Benchmark:  {fin_sim['long_excess_over_benchmark']*100:+.2f}%")
    
    # Per symbol breakdown
    sym_breakdown = per_symbol_breakdown(
        data["test"]["y"], y_pred_test, data["test"]["forward_return"], data["test"]["meta"]["Symbol"]
    )
    pd.DataFrame(sym_breakdown).to_csv(OUTPUT_DIR / "per_symbol_test_metrics.csv", index=False)
    
    print("\n[6/6] Saving model and serialization artifacts...")
    joblib.dump(best_lgb, OUTPUT_DIR / "lightgbm_5d_classifier.joblib")
    joblib.dump(baseline_res["logistic_regression"]["model"], OUTPUT_DIR / "logistic_regression_baseline.joblib")
    
    # Save metrics JSON
    summary_metrics = {
        "horizon": "5d",
        "model_type": "LightGBM Classifier",
        "best_hyperparameters": best_params,
        "best_iteration": int(best_lgb.best_iteration_) if hasattr(best_lgb, "best_iteration_") else None,
        "baseline_dummy": {"test": dummy_test},
        "baseline_logistic_regression": {
            "validation": baseline_res["logistic_regression"]["val_metrics"],
            "test": lr_test,
        },
        "lightgbm": {
            "train": train_metrics_lgb,
            "validation": val_metrics_lgb,
            "test": test_metrics_lgb,
        },
        "financial_simulation_test": fin_sim,
        "per_symbol_test": sym_breakdown,
        "hyperparameter_search_log": all_runs,
    }
    
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
        
    print(f"\nAll artifacts successfully saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
