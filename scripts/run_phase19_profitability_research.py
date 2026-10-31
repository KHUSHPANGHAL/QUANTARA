"""
QUANTARA AI — PHASE 19
PROFITABILITY-ORIENTED STRATEGY RESEARCH WITH A 40–50% ACCURACY TARGET

Comprehensive research engine covering:
1. Baseline verification across horizons & models
2. Feature group investigation (Regime, Relative Strength, Breadth, Momentum, Fundamental disclosure)
3. Prediction horizon comparison (1d, 5d, 20d)
4. Profitability at 40–50% accuracy (Expectancy, Payoff ratio, Break-even win rate, Wilson CIs)
5. Selective trading & risk management (Regime filters, Asymmetric SL/TP)
6. Walk-forward generalization across 5 market cycles (2022–2026)
7. Cost sensitivity (Zero, Base 0.3937%, Conservative 0.5000%)
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
sys.path.append(str(ROOT / "scripts"))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

from run_phase17_reconciliation_research import (
    load_all_market_data,
    simulate_engine_with_session_ledger,
    extract_verified_trade_ledger,
    BASE_FEE_SCHEDULE,
    CONSERVATIVE_FEE_SCHEDULE,
    compute_metrics,
)

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "phase19_profitability_strategy"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.00

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


def wilson_score_interval(k, n, confidence=0.95):
    """Compute Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959964  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std_dev = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return max(0.0, float(lower)), min(1.0, float(upper))


def build_signals_dict(meta_df, probs, threshold, all_dates, filter_mask=None):
    """Build signals dictionary keyed by execution date."""
    sigs = {}
    for idx, r in meta_df.iterrows():
        if filter_mask is not None and not filter_mask[idx]:
            continue
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


# -----------------------------------------------------------------------------
# STEP 2: INVESTIGATE FEATURE GROUPS & RELATIVE STRENGTH
# -----------------------------------------------------------------------------
def run_feature_group_investigation(market):
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE GROUP & RELATIVE STRENGTH INVESTIGATION")
    print("=" * 80)

    splits_dir = DATA_DIR / "splits_v3" / "horizon_20d"
    X_tr = pd.read_csv(splits_dir / "X_train.csv")
    y_tr = pd.read_csv(splits_dir / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_va = pd.read_csv(splits_dir / "X_validation.csv")
    y_va = pd.read_csv(splits_dir / "y_validation.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_te = pd.read_csv(splits_dir / "X_test.csv")
    y_te = pd.read_csv(splits_dir / "y_test.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values

    meta_tr = pd.read_csv(splits_dir / "metadata_train.csv")
    meta_va = pd.read_csv(splits_dir / "metadata_validation.csv")
    meta_te = pd.read_csv(splits_dir / "metadata_test.csv")

    # Add engineered Relative Strength Alpha: return_20d - market_avg_return_1d*20 (proxy)
    for df in [X_tr, X_va, X_te]:
        df["relative_strength_vs_market_20d"] = df["return_20d"] - (df["market_avg_return_1d"] * 20.0)
        df["momentum_breadth_interaction"] = df["rsi_14"] * df["market_breadth_above_sma20"]

    groups = {
        "A. Market Regime & Volatility": [
            "volatility_10d_annualized", "volatility_20d_annualized", "rolling_drawdown_20d",
            "market_avg_volatility_20d", "market_breadth_above_sma20"
        ],
        "B. Stock Relative Strength vs Market Proxy": [
            "relative_strength_vs_market_20d", "return_20d", "return_10d", "return_5d"
        ],
        "C. Market Breadth & Broad Market Conditions": [
            "market_avg_return_1d", "market_breadth_above_sma20", "market_avg_volatility_20d"
        ],
        "D. Price Momentum & Trend": [
            "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
            "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14", "macd_pct",
            "macd_signal_pct", "bollinger_percent_b"
        ],
        "E. Fundamental Information": [],  # Verified unavailable
        "All Combined Stationary Features": list(X_tr.columns),
    }

    results = []

    for g_name, cols in groups.items():
        if len(cols) == 0:
            results.append({
                "feature_group": g_name,
                "feature_count": 0,
                "data_availability": "UNAVAILABLE (Disclosed strictly per rules, zero synthetic data)",
                "lr_val_balanced_acc_pct": np.nan,
                "lr_val_macro_f1_pct": np.nan,
                "lr_test_balanced_acc_pct": np.nan,
                "lr_test_macro_f1_pct": np.nan,
                "lgb_val_balanced_acc_pct": np.nan,
                "lgb_test_balanced_acc_pct": np.nan,
            })
            continue

        X_tr_sub = X_tr[cols]
        X_va_sub = X_va[cols]
        X_te_sub = X_te[cols]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sub)
        X_va_sc = scaler.transform(X_va_sub)
        X_te_sc = scaler.transform(X_te_sub)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_tr_sc, y_tr)
        p_lr_va = lr.predict(X_va_sc)
        p_lr_te = lr.predict(X_te_sc)

        lgb_mod = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1)
        lgb_mod.fit(X_tr_sub, y_tr)
        p_lgb_va = lgb_mod.predict(X_va_sub)
        p_lgb_te = lgb_mod.predict(X_te_sub)

        results.append({
            "feature_group": g_name,
            "feature_count": len(cols),
            "data_availability": "VERIFIED AVAILABLE (Historical OHLCV)",
            "lr_val_balanced_acc_pct": round(balanced_accuracy_score(y_va, p_lr_va) * 100.0, 2),
            "lr_val_macro_f1_pct": round(f1_score(y_va, p_lr_va, average="macro", zero_division=0) * 100.0, 2),
            "lr_test_balanced_acc_pct": round(balanced_accuracy_score(y_te, p_lr_te) * 100.0, 2),
            "lr_test_macro_f1_pct": round(f1_score(y_te, p_lr_te, average="macro", zero_division=0) * 100.0, 2),
            "lgb_val_balanced_acc_pct": round(balanced_accuracy_score(y_va, p_lgb_va) * 100.0, 2),
            "lgb_test_balanced_acc_pct": round(balanced_accuracy_score(y_te, p_lgb_te) * 100.0, 2),
        })

    df_feats = pd.DataFrame(results)
    df_feats.to_csv(REPORT_DIR / "feature_group_results.csv", index=False)
    print("Saved feature_group_results.csv")
    print(df_feats.to_string(index=False))
    return df_feats


# -----------------------------------------------------------------------------
# STEP 3: PREDICTION HORIZON COMPARISON (1d, 5d, 20d)
# -----------------------------------------------------------------------------
def run_horizon_comparison(market):
    print("\n" + "=" * 80)
    print("STEP 3: PREDICTION HORIZON COMPARISON (1d, 5d, 20d)")
    print("=" * 80)

    all_dates = market["all_trading_dates"]
    meta_va_master = market["meta_val"]
    meta_te_master = market["meta_test"]

    horizon_records = []

    for h in [1, 5, 20]:
        splits_dir = DATA_DIR / "splits_v3" / f"horizon_{h}d"
        X_tr = pd.read_csv(splits_dir / "X_train.csv")
        y_tr = pd.read_csv(splits_dir / "y_train.csv")[f"Target_{h}d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
        X_va = pd.read_csv(splits_dir / "X_validation.csv")
        y_va = pd.read_csv(splits_dir / "y_validation.csv")[f"Target_{h}d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
        X_te = pd.read_csv(splits_dir / "X_test.csv")
        y_te = pd.read_csv(splits_dir / "y_test.csv")[f"Target_{h}d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values

        meta_va = pd.read_csv(splits_dir / "metadata_validation.csv")
        meta_te = pd.read_csv(splits_dir / "metadata_test.csv")
        meta_va["Date"] = pd.to_datetime(meta_va["Date"])
        meta_te["Date"] = pd.to_datetime(meta_te["Date"])

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_tr_sc, y_tr)
        probs_va = lr.predict_proba(X_va_sc)
        probs_te = lr.predict_proba(X_te_sc)
        preds_va = lr.predict(X_va_sc)
        preds_te = lr.predict(X_te_sc)

        # Baseline: Majority Class
        train_counts = np.bincount(y_tr, minlength=3)
        maj_class = int(np.argmax(train_counts))
        preds_maj_va = np.full(len(y_va), maj_class)
        preds_maj_te = np.full(len(y_te), maj_class)

        # Backtest Cash Portfolio with default threshold 0.38 (or 0.35)
        # Using base friction (0.3937%)
        for split_name, meta_df, probs, preds, y_true, preds_maj, m_dates, win_tag in [
            ("VALIDATION (2024-2025)", meta_va, probs_va, preds_va, y_va, preds_maj_va, market["val_trading_dates"], "VAL"),
            ("REUSED TEST (2025-2026)", meta_te, probs_te, preds_te, y_te, preds_maj_te, market["test_trading_dates"], "TEST")
        ]:
            sigs = build_signals_dict(meta_df, probs, threshold=0.38, all_dates=all_dates)
            strat_name = f"LR_H{h}D_{win_tag}"
            df_eq, blot, perf = simulate_engine_with_session_ledger(
                strat_name, sigs, m_dates, market, max_positions=3, max_hold_sessions=h, fee_sched=BASE_FEE_SCHEDULE
            )

            df_t = extract_verified_trade_ledger(blot, strat_name, win_tag)
            n_t = len(df_t)
            wins = df_t[df_t["net_pnl_inr"] > 0]
            losses = df_t[df_t["net_pnl_inr"] <= 0]
            w_rate = (len(wins) / n_t * 100.0) if n_t > 0 else 0.0
            avg_w = wins["net_pnl_inr"].mean() if len(wins) > 0 else 0.0
            avg_l = losses["net_pnl_inr"].mean() if len(losses) > 0 else 0.0
            payoff = (avg_w / abs(avg_l)) if abs(avg_l) > 0 else 0.0
            net_exp = (df_t["net_pnl_inr"].sum() / n_t) if n_t > 0 else 0.0
            comm = df_t["total_commissions_inr"].sum() if n_t > 0 else 0.0

            # Classification metrics
            acc = accuracy_score(y_true, preds) * 100.0
            bal_acc = balanced_accuracy_score(y_true, preds) * 100.0
            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
            maj_bal_acc = balanced_accuracy_score(y_true, preds_maj) * 100.0

            horizon_records.append({
                "horizon_sessions": h,
                "horizon_label": f"{h}-Day Forward Target",
                "evaluation_window": split_name,
                "directional_accuracy_pct": round(acc, 2),
                "balanced_accuracy_pct": round(bal_acc, 2),
                "majority_baseline_bal_acc_pct": round(maj_bal_acc, 2),
                "macro_f1_pct": round(macro_f1, 2),
                "trades_count": n_t,
                "trade_win_rate_pct": round(w_rate, 2),
                "avg_win_inr": round(avg_w, 2),
                "avg_loss_inr": round(avg_l, 2),
                "realized_payoff_ratio": round(payoff, 4),
                "net_expectancy_inr": round(net_exp, 2),
                "total_commissions_inr": round(comm, 2),
                "final_equity_inr": round(perf["final_equity"], 2),
                "net_return_pct": round(perf["total_return_pct"], 2),
                "max_drawdown_pct": round(perf["max_drawdown_pct"], 2),
            })

    df_hor = pd.DataFrame(horizon_records)
    df_hor.to_csv(REPORT_DIR / "horizon_comparison.csv", index=False)
    print("Saved horizon_comparison.csv")
    print(df_hor[["horizon_label", "evaluation_window", "directional_accuracy_pct", "trades_count", "trade_win_rate_pct", "net_return_pct", "total_commissions_inr"]].to_string(index=False))
    return df_hor


# -----------------------------------------------------------------------------
# STEP 4 & 5: DESIGN FOR PROFITABILITY AT 40-50% ACCURACY & SELECTIVE TRADING
# -----------------------------------------------------------------------------
def run_profitability_and_selective_strategies(market):
    print("\n" + "=" * 80)
    print("STEP 4 & 5: PROFITABILITY AT 40-50% ACCURACY & SELECTIVE RISK MANAGEMENT")
    print("=" * 80)

    all_dates = market["all_trading_dates"]
    meta_va = market["meta_val"]
    meta_te = market["meta_test"]

    # Load 20d models
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

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_tr)
    probs_va = lr.predict_proba(X_va_sc)
    probs_te = lr.predict_proba(X_te_sc)

    # Strategy Configurations to Test
    strategies = [
        {
            "id": "STRAT_BASE_20D_TH038",
            "name": "1. Baseline LR (Th 0.38, 20d Hold, No Filters)",
            "threshold": 0.38,
            "filter": None,
            "sl": None,
            "tp": None,
            "hold": 20,
        },
        {
            "id": "STRAT_ASYM_SL4_TP8",
            "name": "2. Asymmetric Payoff A (4% SL + 8% TP, 1:2 Target)",
            "threshold": 0.38,
            "filter": None,
            "sl": 0.04,
            "tp": 0.08,
            "hold": 20,
        },
        {
            "id": "STRAT_ASYM_SL5_TP10",
            "name": "3. Asymmetric Payoff B (5% SL + 10% TP, 1:2 Target)",
            "threshold": 0.38,
            "filter": None,
            "sl": 0.05,
            "tp": 0.10,
            "hold": 20,
        },
        {
            "id": "STRAT_ASYM_SL6_TP12",
            "name": "4. Asymmetric Payoff C (6% SL + 12% TP, 1:2 Target)",
            "threshold": 0.38,
            "filter": None,
            "sl": 0.06,
            "tp": 0.12,
            "hold": 20,
        },
        {
            "id": "STRAT_REGIME_SMA50",
            "name": "5. Trend Selective (Stock > SMA50 Filter)",
            "threshold": 0.38,
            "filter": "SMA50",
            "sl": 0.07,
            "tp": None,
            "hold": 20,
        },
        {
            "id": "STRAT_REGIME_BREADTH",
            "name": "6. Macro Breadth Selective (Market Breadth >= 50%)",
            "threshold": 0.38,
            "filter": "BREADTH50",
            "sl": 0.07,
            "tp": None,
            "hold": 20,
        },
        {
            "id": "STRAT_SELECTIVE_COMBINED",
            "name": "7. Combined Selective Alpha (Stock > SMA50 + Breadth >= 50% + 5% SL + 12% TP)",
            "threshold": 0.38,
            "filter": "COMBINED",
            "sl": 0.05,
            "tp": 0.12,
            "hold": 20,
        },
    ]

    strategy_metric_rows = []
    expectancy_rows = []

    for strat in strategies:
        for split_name, meta_df, probs, X_df, y_true, m_dates, win_tag in [
            ("VALIDATION (2024-2025)", meta_va, probs_va, X_va, y_va, market["val_trading_dates"], "VAL"),
            ("REUSED TEST (2025-2026)", meta_te, probs_te, X_te, y_te, market["test_trading_dates"], "TEST")
        ]:
            # Apply Filter Mask
            if strat["filter"] == "SMA50":
                f_mask = (X_df["adj_close_vs_sma50"] > 0).values
            elif strat["filter"] == "BREADTH50":
                f_mask = (X_df["market_breadth_above_sma20"] >= 0.50).values
            elif strat["filter"] == "COMBINED":
                f_mask = ((X_df["adj_close_vs_sma50"] > 0) & (X_df["market_breadth_above_sma20"] >= 0.50)).values
            else:
                f_mask = None

            sigs = build_signals_dict(meta_df, probs, strat["threshold"], all_dates, filter_mask=f_mask)
            strat_id_tagged = f"{strat['id']}_{win_tag}"

            df_eq, blot, perf = simulate_engine_with_session_ledger(
                strat_id_tagged, sigs, m_dates, market, max_positions=3, max_hold_sessions=strat["hold"],
                stop_loss_pct=strat["sl"], take_profit_pct=strat["tp"], fee_sched=BASE_FEE_SCHEDULE
            )

            df_t = extract_verified_trade_ledger(blot, strat_id_tagged, win_tag)
            n_t = len(df_t)
            wins = df_t[df_t["net_pnl_inr"] > 0]
            losses = df_t[df_t["net_pnl_inr"] <= 0]
            n_w = len(wins)
            n_l = len(losses)
            w_rate = (n_w / n_t * 100.0) if n_t > 0 else 0.0
            avg_w = wins["net_pnl_inr"].mean() if n_w > 0 else 0.0
            avg_l = losses["net_pnl_inr"].mean() if n_l > 0 else 0.0
            gross_w = wins["gross_pnl_inr"].mean() if n_w > 0 else 0.0
            gross_l = losses["gross_pnl_inr"].mean() if n_l > 0 else 0.0

            realized_payoff = (avg_w / abs(avg_l)) if abs(avg_l) > 0 else 0.0
            net_expectancy = (df_t["net_pnl_inr"].sum() / n_t) if n_t > 0 else 0.0
            gross_expectancy = (df_t["gross_pnl_inr"].sum() / n_t) if n_t > 0 else 0.0

            # Break-even win rate: W_be = |L| / (W + |L|)
            if (avg_w + abs(avg_l)) > 0:
                be_win_rate = (abs(avg_l) / (avg_w + abs(avg_l))) * 100.0
            else:
                be_win_rate = 50.0

            # Profit Factor
            tot_win_inr = wins["net_pnl_inr"].sum()
            tot_loss_inr = abs(losses["net_pnl_inr"].sum())
            profit_factor = (tot_win_inr / tot_loss_inr) if tot_loss_inr > 0 else (99.0 if tot_win_inr > 0 else 0.0)

            # Wilson Score CI for trade win rate
            ci_low, ci_high = wilson_score_interval(n_w, n_t, confidence=0.95)
            ci_str = f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]"

            # Time in cash %
            cash_sessions = int((df_eq["open_positions_count"] == 0).sum())
            time_in_cash_pct = (cash_sessions / len(df_eq)) * 100.0
            turnover_mult = (df_t["cost_basis_inr"].sum() / STARTING_CAPITAL) if n_t > 0 else 0.0

            # Directional Classification Accuracy of Model (Full Split)
            preds_split = lr.predict(scaler.transform(X_df))
            dir_acc = accuracy_score(y_true, preds_split) * 100.0
            bal_acc = balanced_accuracy_score(y_true, preds_split) * 100.0

            strategy_metric_rows.append({
                "strategy_id": strat["id"],
                "strategy_name": strat["name"],
                "evaluation_window": split_name,
                "directional_accuracy_pct": round(dir_acc, 2),
                "balanced_accuracy_pct": round(bal_acc, 2),
                "executed_trades": n_t,
                "trade_win_rate_pct": round(w_rate, 2),
                "win_rate_95pct_ci": ci_str,
                "avg_win_inr": round(avg_w, 2),
                "avg_loss_inr": round(avg_l, 2),
                "realized_payoff_ratio": round(realized_payoff, 4),
                "net_expectancy_per_trade_inr": round(net_expectancy, 2),
                "profit_factor": round(profit_factor, 2),
                "max_drawdown_pct": round(perf["max_drawdown_pct"], 2),
                "time_in_cash_pct": round(time_in_cash_pct, 2),
                "turnover_multiple": round(turnover_mult, 2),
                "total_commissions_inr": round(df_t["total_commissions_inr"].sum(), 2) if n_t > 0 else 0.0,
                "final_equity_inr": round(perf["final_equity"], 2),
                "net_return_pct": round(perf["total_return_pct"], 2),
            })

            expectancy_rows.append({
                "strategy_id": strat["id"],
                "strategy_name": strat["name"],
                "evaluation_window": split_name,
                "executed_trades": n_t,
                "trade_win_rate_pct": round(w_rate, 2),
                "break_even_win_rate_pct": round(be_win_rate, 2),
                "win_rate_surplus_deficit_pct": round(w_rate - be_win_rate, 2),
                "avg_win_inr": round(avg_w, 2),
                "avg_loss_inr": round(avg_l, 2),
                "realized_payoff_ratio": round(realized_payoff, 4),
                "gross_expectancy_inr": round(gross_expectancy, 2),
                "avg_friction_per_trade_inr": round(gross_expectancy - net_expectancy, 2),
                "net_expectancy_per_trade_inr": round(net_expectancy, 2),
                "in_40_to_50_accuracy_target": "YES" if (40.0 <= w_rate <= 50.0) else "NO",
                "is_net_expectancy_positive": "YES" if net_expectancy > 0 else "NO",
            })

    df_strat = pd.DataFrame(strategy_metric_rows)
    df_strat.to_csv(REPORT_DIR / "strategy_metrics.csv", index=False)
    print("Saved strategy_metrics.csv")

    df_exp = pd.DataFrame(expectancy_rows)
    df_exp.to_csv(REPORT_DIR / "expectancy_and_break_even.csv", index=False)
    print("Saved expectancy_and_break_even.csv")

    print(df_exp[["strategy_name", "evaluation_window", "trade_win_rate_pct", "break_even_win_rate_pct", "realized_payoff_ratio", "net_expectancy_per_trade_inr", "is_net_expectancy_positive"]].to_string(index=False))
    return df_strat, df_exp


# -----------------------------------------------------------------------------
# STEP 6: EXPANDING WALK-FORWARD GENERALIZATION (2022-2026)
# -----------------------------------------------------------------------------
def run_walk_forward_evaluation(market):
    print("\n" + "=" * 80)
    print("STEP 6: WALK-FORWARD GENERALIZATION ACROSS 5 MARKET CYCLES (2022-2026)")
    print("=" * 80)

    # Use Phase 18 expanding walk-forward data
    from run_phase18_diagnosis_research import step6_test_generalization_walk_forward
    df_wf = step6_test_generalization_walk_forward()
    df_wf.to_csv(REPORT_DIR / "walk_forward_results.csv", index=False)
    print("Saved walk_forward_results.csv")
    return df_wf


# -----------------------------------------------------------------------------
# STEP 7: COST SENSITIVITY MATRIX
# -----------------------------------------------------------------------------
def run_cost_sensitivity(market):
    print("\n" + "=" * 80)
    print("STEP 7: COST SENSITIVITY ANALYSIS (Zero, Base 0.39%, Conservative 0.50%)")
    print("=" * 80)

    # Use Phase 18 prediction vs profitability data
    from run_phase18_diagnosis_research import step7_connect_prediction_to_trading
    df_cost = step7_connect_prediction_to_trading()
    df_cost.to_csv(REPORT_DIR / "cost_sensitivity.csv", index=False)
    print("Saved cost_sensitivity.csv")
    return df_cost


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 19 RESEARCH ENGINE")
    print("=" * 80)

    market = load_all_market_data()

    # Step 2: Feature Groups & Relative Strength
    run_feature_group_investigation(market)

    # Step 3: Horizon Comparison (1d, 5d, 20d)
    run_horizon_comparison(market)

    # Step 4 & 5: Profitability at 40-50% & Selective Risk Management
    run_profitability_and_selective_strategies(market)

    # Step 6: Walk-Forward Generalization
    run_walk_forward_evaluation(market)

    # Step 7: Cost Sensitivity
    run_cost_sensitivity(market)

    print("\n" + "=" * 80)
    print("PHASE 19 EXPERIMENT SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

