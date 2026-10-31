"""
Quantara AI — Phase 16: Profitability-First Model Research
Comprehensive comparison of Logistic Regression, Random Forest, and LightGBM across
chronological Validation (2024-2025) and Test (2025-2026) splits. Evaluates directional
prediction accuracy, probability calibration, selective thresholds, reward-to-risk exit
structures, and portfolio net profitability after realistic NSE statutory costs and slippage.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

# -----------------------------------------------------------------------------
# DIRECTORIES & CONFIGURATION
# -----------------------------------------------------------------------------
ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "phase16_profitability_research"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.0

BASE_FEE_SCHEDULE = {
    "brokerage_pct": 0.00030,
    "stt_buy_pct": 0.00100,
    "stt_sell_pct": 0.00100,
    "exchange_turnover_pct": 0.0000325,
    "sebi_turnover_pct": 0.000001,
    "stamp_duty_buy_pct": 0.00015,
    "gst_rate": 0.18,
    "slippage_pct": 0.00050,
}

CONSERVATIVE_FEE_SCHEDULE = {
    "brokerage_pct": 0.00040,
    "stt_buy_pct": 0.00125,
    "stt_sell_pct": 0.00125,
    "exchange_turnover_pct": 0.000035,
    "sebi_turnover_pct": 0.000001,
    "stamp_duty_buy_pct": 0.00015,
    "gst_rate": 0.18,
    "slippage_pct": 0.00100,
}


def get_fee_rates(fee_sched=BASE_FEE_SCHEDULE):
    gst_base = fee_sched["brokerage_pct"] + fee_sched["exchange_turnover_pct"] + fee_sched["sebi_turnover_pct"]
    buy_fee = (fee_sched["brokerage_pct"] + fee_sched["stt_buy_pct"] + fee_sched["exchange_turnover_pct"]
               + fee_sched["sebi_turnover_pct"] + fee_sched["stamp_duty_buy_pct"] + fee_sched["gst_rate"] * gst_base)
    sell_fee = (fee_sched["brokerage_pct"] + fee_sched["stt_sell_pct"] + fee_sched["exchange_turnover_pct"]
                + fee_sched["sebi_turnover_pct"] + fee_sched["gst_rate"] * gst_base)
    slippage = fee_sched["slippage_pct"]
    return buy_fee, sell_fee, slippage


def extract_dividends(raw_df):
    df = raw_df.sort_values(["Symbol", "Date"]).copy().reset_index(drop=True)
    df["adj_ratio"] = df["Adj Close"] / df["Close"]
    div_records = []
    for sym, grp in df.groupby("Symbol"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        grp["ratio_diff"] = grp["adj_ratio"] - grp["adj_ratio"].shift(1)
        events = grp[grp["ratio_diff"] > 1e-4]
        for idx, r in events.iterrows():
            if idx == 0:
                continue
            prev_row = grp.loc[idx - 1]
            div_yield = 1.0 - (prev_row["adj_ratio"] / r["adj_ratio"])
            div_records.append({
                "Symbol": sym,
                "Ex_Date": r["Date"],
                "div_per_share": float(div_yield * prev_row["Close"]),
            })
    return pd.DataFrame(div_records)


def load_all_market_data():
    raw = pd.read_csv(DATA_DIR / "raw" / "multi_stock_historical.csv")
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    raw["sma50"] = raw.groupby("Symbol")["Close"].transform(lambda x: x.rolling(50, min_periods=50).mean())
    sma50_lookup = {(row["Date"], row["Symbol"]): bool(row["Close"] > row["sma50"]) for _, row in raw.dropna(subset=["sma50"]).iterrows()}

    meta_val = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "metadata_validation.csv")
    meta_val["Date"] = pd.to_datetime(meta_val["Date"])
    val_signal_dates = sorted(meta_val["Date"].unique())

    meta_test = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "metadata_test.csv")
    meta_test["Date"] = pd.to_datetime(meta_test["Date"])
    test_signal_dates = sorted(meta_test["Date"].unique())

    all_trading_dates = sorted(raw["Date"].unique())

    idx_v_start = all_trading_dates.index(val_signal_dates[0])
    idx_v_end_sig = all_trading_dates.index(val_signal_dates[-1])
    val_trading_dates = all_trading_dates[idx_v_start: idx_v_end_sig + 22]

    idx_t_start = all_trading_dates.index(test_signal_dates[0])
    test_trading_dates = all_trading_dates[idx_t_start:]

    opens = raw.pivot(index="Date", columns="Symbol", values="Open")
    closes = raw.pivot(index="Date", columns="Symbol", values="Close")
    lows = raw.pivot(index="Date", columns="Symbol", values="Low")
    highs = raw.pivot(index="Date", columns="Symbol", values="High")

    div_df = extract_dividends(raw)
    dividend_lookup = {(row["Ex_Date"], row["Symbol"]): row["div_per_share"] for _, row in div_df.iterrows()} if len(div_df) > 0 else {}

    return {
        "raw": raw,
        "all_trading_dates": all_trading_dates,
        "val_trading_dates": val_trading_dates,
        "test_trading_dates": test_trading_dates,
        "meta_val": meta_val,
        "meta_test": meta_test,
        "opens": opens,
        "closes": closes,
        "lows": lows,
        "highs": highs,
        "sma50_lookup": sma50_lookup,
        "dividend_lookup": dividend_lookup,
        "symbols": sorted(raw["Symbol"].unique()),
    }


def compute_metrics(equity_series, daily_returns, trade_blotter):
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    n_days = len(equity_series)
    cagr = ((1.0 + total_ret) ** (252.0 / n_days)) - 1.0 if n_days > 0 and (1.0 + total_ret) > 0 else -1.0

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = abs(drawdown.min())

    ann_vol = daily_returns.std() * np.sqrt(252.0)
    rf_daily = 0.065 / 252.0
    excess_ret = daily_returns - rf_daily
    sharpe = np.sqrt(252.0) * excess_ret.mean() / daily_returns.std() if daily_returns.std() > 1e-7 else 0.0

    closed_trades = [t for t in trade_blotter if t["action"] == "SELL"]
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t["net_pnl"] > 0]
    losing_trades = [t for t in closed_trades if t["net_pnl"] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(t["net_pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["net_pnl"] for t in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    tot_comm = sum(t["commission"] for t in trade_blotter)
    tot_div = sum(t["net_pnl"] for t in trade_blotter if t["action"] == "DIVIDEND")
    avg_win = gross_profit / len(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss = -gross_loss / len(losing_trades) if len(losing_trades) > 0 else 0.0
    payoff_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > 0 else np.nan
    net_pnl_total = gross_profit - gross_loss + tot_div
    expectancy_per_trade = net_pnl_total / total_trades if total_trades > 0 else 0.0
    total_turnover = sum(t["shares"] * t["exec_price"] for t in trade_blotter if t["action"] == "BUY") / STARTING_CAPITAL

    return {
        "final_equity": float(equity_series.iloc[-1]),
        "total_return_pct": float(total_ret * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "annualized_volatility_pct": float(ann_vol * 100.0),
        "sharpe_ratio": float(sharpe),
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": float(win_rate * 100.0),
        "avg_win_inr": float(avg_win),
        "avg_loss_inr": float(avg_loss),
        "payoff_ratio": float(payoff_ratio) if not np.isnan(payoff_ratio) else 0.0,
        "profit_factor": float(profit_factor),
        "expectancy_per_trade_inr": float(expectancy_per_trade),
        "commissions_paid": float(tot_comm),
        "dividends_received": float(tot_div),
        "turnover_multiple": float(total_turnover),
    }


def simulate_advanced_engine(
    strategy_name,
    signals_by_date,
    market_dates,
    market_obj,
    max_positions=3,
    max_hold_sessions=20,
    stop_loss_pct=None,
    take_profit_pct=None,
    fee_sched=BASE_FEE_SCHEDULE
):
    """
    Generalized event-driven execution simulator supporting realistic gap stops and take profits.
    """
    dates = market_dates
    opens = market_obj["opens"]
    closes = market_obj["closes"]
    lows = market_obj["lows"]
    highs = market_obj["highs"]
    div_lookup = market_obj["dividend_lookup"]
    buy_fee, sell_fee, slippage = get_fee_rates(fee_sched)

    cash = STARTING_CAPITAL
    open_positions = {}
    trade_blotter = []
    daily_records = []
    stopped_cooldown = {}

    for i, current_date in enumerate(dates):
        # 1. Update cooldowns
        for s in list(stopped_cooldown.keys()):
            stopped_cooldown[s] -= 1
            if stopped_cooldown[s] <= 0:
                del stopped_cooldown[s]

        # 2. CHECK PENDING SCHEDULED EXITS & GAP EXITS AT OPEN
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            raw_open = opens.loc[current_date, sym]
            p_stop = pos["entry_price"] * (1.0 - stop_loss_pct) if stop_loss_pct is not None else -1e9
            p_target = pos["entry_price"] * (1.0 + take_profit_pct) if take_profit_pct is not None else 1e9

            is_gap_stop = (stop_loss_pct is not None and raw_open <= p_stop and pos["entry_date"] < current_date)
            is_gap_tp = (take_profit_pct is not None and raw_open >= p_target and pos["entry_date"] < current_date)
            is_sched_exit = pos.get("pending_exit", False)

            if is_gap_stop or is_gap_tp or is_sched_exit:
                if (current_date, sym) in div_lookup and pos["entry_date"] < current_date:
                    div_payout = pos["shares"] * div_lookup[(current_date, sym)]
                    cash += div_payout
                    trade_blotter.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": float(div_lookup[(current_date, sym)]),
                        "commission": 0.0,
                        "net_pnl": div_payout,
                        "cash_after": float(cash),
                    })

                exec_price = raw_open * (1.0 - slippage)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * sell_fee
                cash += (gross_proceeds - fee)
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + buy_fee)
                net_pnl = gross_proceeds - fee - total_cost

                reason = "GAP_DOWN_STOP" if is_gap_stop else ("GAP_UP_TP" if is_gap_tp else pos.get("exit_reason", "SCHEDULED_EXIT"))
                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_pnl),
                    "reason": reason,
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })

                if is_gap_stop:
                    stopped_cooldown[sym] = max(1, max_hold_sessions - pos["sessions_held"])
                del open_positions[sym]

        # 3. EXECUTE CANDIDATE BUYS AT OPEN
        cands = signals_by_date.get(current_date, [])
        for sym, score in cands:
            if len(open_positions) < max_positions and sym not in open_positions and sym not in stopped_cooldown:
                avail_slots = max_positions - len(open_positions)
                slot_budget = cash / avail_slots

                raw_open = opens.loc[current_date, sym]
                exec_price = raw_open * (1.0 + slippage)
                cost_per_share = exec_price * (1.0 + buy_fee)
                shares = int(np.floor(slot_budget / cost_per_share))

                if shares > 0:
                    gross_cost = shares * exec_price
                    fee = gross_cost * buy_fee
                    total_outflow = gross_cost + fee
                    if cash >= total_outflow:
                        cash -= total_outflow
                        open_positions[sym] = {
                            "shares": shares,
                            "entry_price": float(exec_price),
                            "entry_date": current_date,
                            "sessions_held": 0,
                            "pending_exit": False,
                        }
                        trade_blotter.append({
                            "date": current_date.strftime("%Y-%m-%d"),
                            "symbol": sym,
                            "action": "BUY",
                            "shares": shares,
                            "exec_price": float(exec_price),
                            "commission": float(fee),
                            "net_pnl": 0.0,
                            "cash_after": float(cash),
                        })

        # 4. AGE POSITIONS & PROCESS INTRADAY EXITS & DIVIDENDS
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            pos["sessions_held"] += 1

            raw_low = lows.loc[current_date, sym]
            raw_high = highs.loc[current_date, sym]
            p_stop = pos["entry_price"] * (1.0 - stop_loss_pct) if stop_loss_pct is not None else -1e9
            p_target = pos["entry_price"] * (1.0 + take_profit_pct) if take_profit_pct is not None else 1e9

            # Take Profit takes precedence if reached intraday
            if take_profit_pct is not None and raw_high >= p_target:
                exec_price = p_target * (1.0 - slippage)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * sell_fee
                cash += (gross_proceeds - fee)
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + buy_fee)
                net_pnl = gross_proceeds - fee - total_cost

                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_pnl),
                    "reason": f"TAKE_PROFIT_{take_profit_pct*100:.0f}PCT_INTRADAY",
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
                del open_positions[sym]
                continue

            elif stop_loss_pct is not None and raw_low <= p_stop:
                exec_price = p_stop * (1.0 - slippage)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * sell_fee
                cash += (gross_proceeds - fee)
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + buy_fee)
                net_pnl = gross_proceeds - fee - total_cost

                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_pnl),
                    "reason": f"STOP_LOSS_{stop_loss_pct*100:.0f}PCT_INTRADAY",
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
                stopped_cooldown[sym] = max(1, max_hold_sessions - pos["sessions_held"])
                del open_positions[sym]
                continue

            # Process Dividends
            if (current_date, sym) in div_lookup and pos["entry_date"] < current_date:
                credited = pos.get("dividend_credited_dates", set())
                if current_date not in credited:
                    div_payout = pos["shares"] * div_lookup[(current_date, sym)]
                    cash += div_payout
                    trade_blotter.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": float(div_lookup[(current_date, sym)]),
                        "commission": 0.0,
                        "net_pnl": div_payout,
                        "cash_after": float(cash),
                    })
                    pos["dividend_credited_dates"] = credited | {current_date}

        # 5. MARK-TO-MARKET & LIQUIDATE ON FINAL DAY
        val = sum(pos["shares"] * closes.loc[current_date, sym] for sym, pos in open_positions.items())
        total_equity = cash + val

        if i == len(dates) - 1:
            for sym, pos in list(open_positions.items()):
                raw_close = closes.loc[current_date, sym]
                exec_price = raw_close * (1.0 - slippage)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * sell_fee
                cash += (gross_proceeds - fee)
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + buy_fee)
                net_pnl = gross_proceeds - fee - total_cost

                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_pnl),
                    "reason": "FINAL_DAY_LIQUIDATION",
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
            open_positions.clear()
            val = 0.0
            total_equity = cash

        daily_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "total_equity": float(total_equity),
        })

        # 6. SCHEDULE MAX HOLD EXITS FOR NEXT OPEN
        for sym, pos in open_positions.items():
            if pos["sessions_held"] >= max_hold_sessions:
                pos["pending_exit"] = True
                pos["exit_reason"] = f"MAX_HOLD_{max_hold_sessions}D"

    df_eq = pd.DataFrame(daily_records)
    df_eq["daily_return"] = df_eq["total_equity"].pct_change().fillna(0.0)
    perf = compute_metrics(df_eq["total_equity"], df_eq["daily_return"], trade_blotter)
    return df_eq, trade_blotter, perf


def extract_round_trips(trade_blotter, strategy_name, window_name):
    trades = []
    open_pos = {}
    trade_id = 0

    for b in trade_blotter:
        sym = b["symbol"]
        act = b["action"]
        if act == "BUY":
            open_pos[sym] = {
                "entry_date": b["date"],
                "shares": b["shares"],
                "entry_price": b["exec_price"],
                "buy_comm": b["commission"],
                "dividends": 0.0,
            }
        elif act == "DIVIDEND":
            if sym in open_pos:
                open_pos[sym]["dividends"] += b["net_pnl"]
        elif act == "SELL":
            op = open_pos.pop(sym, None)
            if op:
                trade_id += 1
                cost_basis = op["shares"] * op["entry_price"] + op["buy_comm"]
                gross_proceeds = b["shares"] * b["exec_price"]
                sell_comm = b["commission"]
                net_pnl = b["net_pnl"] + op["dividends"]
                ret_pct = (net_pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0

                trades.append({
                    "trade_id": f"T{trade_id:02d}",
                    "strategy": strategy_name,
                    "window": window_name,
                    "symbol": sym,
                    "entry_date": op["entry_date"],
                    "exit_date": b["date"],
                    "shares": b["shares"],
                    "entry_price": round(op["entry_price"], 2),
                    "exit_price": round(b["exec_price"], 2),
                    "cost_basis_inr": round(cost_basis, 2),
                    "gross_proceeds_inr": round(gross_proceeds, 2),
                    "commissions_inr": round(op["buy_comm"] + sell_comm, 2),
                    "dividends_inr": round(op["dividends"], 2),
                    "net_pnl_inr": round(net_pnl, 2),
                    "net_return_pct": round(ret_pct, 2),
                    "exit_reason": b.get("reason", "SCHEDULED_EXIT"),
                    "sessions_held": b.get("sessions_held", 20),
                })
    return pd.DataFrame(trades)


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 16: PROFITABILITY-FIRST MODEL RESEARCH")
    print("=" * 80)

    market = load_all_market_data()
    all_dates = market["all_trading_dates"]
    val_dates = market["val_trading_dates"]
    test_dates = market["test_trading_dates"]
    sma50 = market["sma50_lookup"]

    print(f"Validation window: {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')} ({len(val_dates)} sessions)")
    print(f"Test window:       {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} sessions)")

    # Load 20d Horizon Data
    p_splits = DATA_DIR / "splits_v3" / "horizon_20d"
    X_tr = pd.read_csv(p_splits / "X_train.csv")
    y_tr = pd.read_csv(p_splits / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_va = pd.read_csv(p_splits / "X_validation.csv")
    y_va = pd.read_csv(p_splits / "y_validation.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_te = pd.read_csv(p_splits / "X_test.csv")
    y_te = pd.read_csv(p_splits / "y_test.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values

    meta_va = market["meta_val"]
    meta_te = market["meta_test"]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # -------------------------------------------------------------------------
    # 1. TRAIN MODELS
    # -------------------------------------------------------------------------
    print("\n--- Training Model Families ---")
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_tr)
    probs_lr_va = lr.predict_proba(X_va_sc)
    probs_lr_te = lr.predict_proba(X_te_sc)
    preds_lr_va = lr.predict(X_va_sc)
    preds_lr_te = lr.predict(X_te_sc)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    probs_rf_va = rf.predict_proba(X_va)
    probs_rf_te = rf.predict_proba(X_te)
    preds_rf_va = rf.predict(X_va)
    preds_rf_te = rf.predict(X_te)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.03, class_weight="balanced", random_state=42, verbose=-1)
    lgb_model.fit(X_tr, y_tr)
    probs_lgb_va = lgb_model.predict_proba(X_va)
    probs_lgb_te = lgb_model.predict_proba(X_te)
    preds_lgb_va = lgb_model.predict(X_va)
    preds_lgb_te = lgb_model.predict(X_te)

    # -------------------------------------------------------------------------
    # 2. DELIVERABLE: model_accuracy_comparison.csv
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Model Accuracy Across All Observations and Selected Trades ---")
    accuracy_records = []

    model_dict = {
        "Logistic Regression": {"preds_va": preds_lr_va, "preds_te": preds_lr_te, "probs_va": probs_lr_va, "probs_te": probs_lr_te},
        "Random Forest": {"preds_va": preds_rf_va, "preds_te": preds_rf_te, "probs_va": probs_rf_va, "probs_te": probs_rf_te},
        "LightGBM": {"preds_va": preds_lgb_va, "preds_te": preds_lgb_te, "probs_va": probs_lgb_va, "probs_te": probs_lgb_te},
    }

    for name, m_data in model_dict.items():
        for win_name, y_true, preds, probs, meta_df in [("VALIDATION", y_va, m_data["preds_va"], m_data["probs_va"], meta_va),
                                                        ("TEST", y_te, m_data["preds_te"], m_data["probs_te"], meta_te)]:
            # 1. Overall evaluation on all observations
            acc = accuracy_score(y_true, preds) * 100.0
            bal_acc = balanced_accuracy_score(y_true, preds) * 100.0
            prec_up = precision_score(y_true, preds, labels=[2], average="micro", zero_division=0) * 100.0
            rec_up = recall_score(y_true, preds, labels=[2], average="micro", zero_division=0) * 100.0
            f1_up = f1_score(y_true, preds, labels=[2], average="micro", zero_division=0) * 100.0
            brier = brier_score_loss((y_true == 2).astype(int), probs[:, 2])

            accuracy_records.append({
                "model_name": name,
                "window": win_name,
                "evaluation_scope": "ALL_OBSERVATIONS_3CLASS",
                "sample_count": len(y_true),
                "coverage_pct": 100.0,
                "ordinary_accuracy_pct": round(acc, 2),
                "balanced_accuracy_pct": round(bal_acc, 2),
                "precision_up_pct": round(prec_up, 2),
                "recall_up_pct": round(rec_up, 2),
                "f1_score_up_pct": round(f1_up, 2),
                "brier_score": round(brier, 4),
            })

            # 2. Selected high-conviction trades (Threshold 0.40)
            is_selected = (probs[:, 2] >= 0.40) & (probs[:, 2] > probs[:, 0])
            n_sel = int(is_selected.sum())
            if n_sel > 0:
                y_sel_true = y_true[is_selected]
                # Win rate is true UP rate on selected trades
                win_rate_sel = float((y_sel_true == 2).mean() * 100.0)
                accuracy_records.append({
                    "model_name": name,
                    "window": win_name,
                    "evaluation_scope": "SELECTED_TRADES_P_UP_GE_0.40",
                    "sample_count": n_sel,
                    "coverage_pct": round((n_sel / len(y_true)) * 100.0, 2),
                    "ordinary_accuracy_pct": round(win_rate_sel, 2),
                    "balanced_accuracy_pct": round(win_rate_sel, 2),
                    "precision_up_pct": round(win_rate_sel, 2),
                    "recall_up_pct": round((((y_sel_true == 2).sum()) / ((y_true == 2).sum())) * 100.0, 2),
                    "f1_score_up_pct": np.nan,
                    "brier_score": round(brier_score_loss((y_sel_true == 2).astype(int), probs[is_selected, 2]), 4),
                })

    df_acc = pd.DataFrame(accuracy_records)
    df_acc.to_csv(REPORT_DIR / "model_accuracy_comparison.csv", index=False)
    print("Saved: model_accuracy_comparison.csv")
    print(df_acc[df_acc["window"] == "TEST"].to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. DELIVERABLE: risk_reward_analysis.csv
    # -------------------------------------------------------------------------
    print("\n--- Testing Reward-to-Risk and Exit Structures ---")
    rr_records = []

    # Build LR test signals for Th 0.40
    test_sigs_040 = {}
    for idx, row in meta_te.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            p_up = float(probs_lr_te[idx, 2])
            p_down = float(probs_lr_te[idx, 0])
            if p_up >= 0.40 and p_up > p_down:
                test_sigs_040.setdefault(exec_d, []).append((row["Symbol"], p_up))
    for d in test_sigs_040:
        test_sigs_040[d].sort(key=lambda x: x[1], reverse=True)

    val_sigs_040 = {}
    for idx, row in meta_va.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        exec_d = all_dates[cur_idx + 1]
        p_up = float(probs_lr_va[idx, 2])
        p_down = float(probs_lr_va[idx, 0])
        if p_up >= 0.40 and p_up > p_down:
            val_sigs_040.setdefault(exec_d, []).append((row["Symbol"], p_up))
    for d in val_sigs_040:
        val_sigs_040[d].sort(key=lambda x: x[1], reverse=True)

    exit_variants = [
        {"name": "Fixed 20-Day Hold (No SL / No TP)", "sl": None, "tp": None, "ratio": "N/A"},
        {"name": "7% Stop-Loss Only", "sl": 0.07, "tp": None, "ratio": "N/A"},
        {"name": "5% Stop-Loss Only", "sl": 0.05, "tp": None, "ratio": "N/A"},
        {"name": "7% Stop + 10% TP Target", "sl": 0.07, "tp": 0.10, "ratio": "1.43 : 1"},
        {"name": "7% Stop + 12% TP Target", "sl": 0.07, "tp": 0.12, "ratio": "1.71 : 1"},
        {"name": "7% Stop + 15% TP Target", "sl": 0.07, "tp": 0.15, "ratio": "2.14 : 1"},
        {"name": "5% Stop + 8% TP Target", "sl": 0.05, "tp": 0.08, "ratio": "1.60 : 1"},
        {"name": "5% Stop + 10% TP Target", "sl": 0.05, "tp": 0.10, "ratio": "2.00 : 1"},
    ]

    for ev in exit_variants:
        for win_name, sigs, m_dates in [("VALIDATION", val_sigs_040, val_dates), ("TEST", test_sigs_040, test_dates)]:
            _, blot, perf = simulate_advanced_engine(
                ev["name"], sigs, m_dates, market, max_positions=3, max_hold_sessions=20,
                stop_loss_pct=ev["sl"], take_profit_pct=ev["tp"], fee_sched=BASE_FEE_SCHEDULE
            )
            rr_records.append({
                "window": win_name,
                "exit_rule_name": ev["name"],
                "target_reward_to_risk_ratio": ev["ratio"],
                "stop_loss_pct": f"{ev['sl']*100:.0f}%" if ev["sl"] else "None",
                "take_profit_pct": f"{ev['tp']*100:.0f}%" if ev["tp"] else "None",
                "final_equity_inr": round(perf["final_equity"], 2),
                "net_return_pct": round(perf["total_return_pct"], 2),
                "max_drawdown_pct": round(perf["max_drawdown_pct"], 2),
                "total_trades": perf["total_trades"],
                "win_rate_pct": round(perf["win_rate_pct"], 2),
                "avg_win_inr": round(perf["avg_win_inr"], 2),
                "avg_loss_inr": round(perf["avg_loss_inr"], 2),
                "realized_payoff_ratio": round(perf["payoff_ratio"], 2),
                "profit_factor": round(perf["profit_factor"], 2),
                "net_expectancy_per_trade_inr": round(perf["expectancy_per_trade_inr"], 2),
                "total_friction_inr": round(perf["commissions_paid"], 2),
            })

    df_rr = pd.DataFrame(rr_records)
    df_rr.to_csv(REPORT_DIR / "risk_reward_analysis.csv", index=False)
    print("Saved: risk_reward_analysis.csv")
    print(df_rr[df_rr["window"] == "TEST"].to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. DELIVERABLE: net_profitability_comparison.csv & trade_by_trade_ledger.csv
    # -------------------------------------------------------------------------
    print("\n--- Running Master Profitability Suite Across All Candidate Models & Benchmarks ---")
    master_profit_records = []
    trade_ledger_records = []

    # Define all candidate models and benchmark strategies
    candidate_strategies = [
        # Benchmarks
        {"id": "BENCH_01_CASH", "family": "Benchmark", "name": "Cash Benchmark (6.5% Annual Yield)", "type": "cash"},
        {"id": "BENCH_02_BUY_HOLD", "family": "Benchmark", "name": "Buy & Hold Benchmark (Equal Weight 8 Stocks)", "type": "buy_hold"},
        {"id": "BENCH_03_SMA50_TREND", "family": "Benchmark", "name": "Classical SMA-50 Trend Following", "type": "sma50_trend"},
        # Logistic Regression Suites
        {"id": "LR_01_BASE", "family": "LogisticRegression", "name": "LR Baseline (Th 0.35, No SMA, 20d Hold)", "model": "lr", "th": 0.35, "sma": False, "sl": None, "tp": None},
        {"id": "LR_02_TH038_SMA", "family": "LogisticRegression", "name": "LR Comb A (Th 0.38 + SMA50)", "model": "lr", "th": 0.38, "sma": True, "sl": None, "tp": None},
        {"id": "LR_03_TH038_SMA_STOP7", "family": "LogisticRegression", "name": "LR Comb C (Th 0.38 + SMA50 + 7% Stop)", "model": "lr", "th": 0.38, "sma": True, "sl": 0.07, "tp": None},
        {"id": "LR_04_TH040_HOLD20", "family": "LogisticRegression", "name": "LR High Conviction (Th 0.40, 20d Hold)", "model": "lr", "th": 0.40, "sma": False, "sl": None, "tp": None},
        {"id": "LR_05_TH040_STOP7_TP10", "family": "LogisticRegression", "name": "LR Asymmetric (Th 0.40 + 7% Stop + 10% TP)", "model": "lr", "th": 0.40, "sma": False, "sl": 0.07, "tp": 0.10},
        {"id": "LR_06_TH042_HOLD20", "family": "LogisticRegression", "name": "LR Ultra Selective (Th 0.42, 20d Hold)", "model": "lr", "th": 0.42, "sma": False, "sl": None, "tp": None},
        # Random Forest Suites
        {"id": "RF_01_BASE", "family": "RandomForest", "name": "Random Forest Base (Th 0.35, 20d Hold)", "model": "rf", "th": 0.35, "sma": False, "sl": None, "tp": None},
        {"id": "RF_02_TH038_SMA", "family": "RandomForest", "name": "Random Forest (Th 0.38 + SMA50)", "model": "rf", "th": 0.38, "sma": True, "sl": None, "tp": None},
        {"id": "RF_03_TH040_HOLD20", "family": "RandomForest", "name": "Random Forest High Conviction (Th 0.40)", "model": "rf", "th": 0.40, "sma": False, "sl": None, "tp": None},
        # LightGBM Suites
        {"id": "LGB_01_BASE", "family": "LightGBM", "name": "LightGBM Base (Th 0.35, 20d Hold)", "model": "lgb", "th": 0.35, "sma": False, "sl": None, "tp": None},
        {"id": "LGB_02_TH038_HOLD20", "family": "LightGBM", "name": "LightGBM Stronger (Th 0.38, 20d Hold)", "model": "lgb", "th": 0.38, "sma": False, "sl": None, "tp": None},
        {"id": "LGB_03_TH038_SMA", "family": "LightGBM", "name": "LightGBM (Th 0.38 + SMA50)", "model": "lgb", "th": 0.38, "sma": True, "sl": None, "tp": None},
    ]

    for strat in candidate_strategies:
        s_id = strat["id"]
        s_fam = strat["family"]
        s_name = strat["name"]

        for win_name, m_dates in [("VALIDATION", val_dates), ("TEST", test_dates)]:
            for cost_name, sched in [("Base (0.39%)", BASE_FEE_SCHEDULE), ("Conservative (0.50%)", CONSERVATIVE_FEE_SCHEDULE)]:
                # 1. Cash Benchmark
                if strat.get("type") == "cash":
                    daily_rate = (1.0 + 0.065) ** (1.0 / 252.0) - 1.0
                    cash_curve = pd.Series([STARTING_CAPITAL * ((1.0 + daily_rate) ** d_idx) for d_idx in range(len(m_dates))])
                    perf = compute_metrics(cash_curve, cash_curve.pct_change().fillna(daily_rate), [])
                    df_eq = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in m_dates], "total_equity": cash_curve})
                    blot = []

                # 2. Buy and Hold Benchmark
                elif strat.get("type") == "buy_hold":
                    closes_sub = market["closes"].loc[m_dates]
                    bh_daily_returns = closes_sub.pct_change().mean(axis=1).fillna(0.0)
                    bh_curve = STARTING_CAPITAL * (1.0 + bh_daily_returns).cumprod()
                    perf = compute_metrics(bh_curve, bh_daily_returns, [])
                    df_eq = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in m_dates], "total_equity": bh_curve})
                    blot = []

                # 3. SMA50 Trend Benchmark
                elif strat.get("type") == "sma50_trend":
                    trend_sigs = {}
                    meta_df = meta_va if win_name == "VALIDATION" else meta_te
                    for idx, row in meta_df.iterrows():
                        sig_d = row["Date"]
                        cur_idx = all_dates.index(sig_d)
                        if cur_idx < len(all_dates) - 1:
                            exec_d = all_dates[cur_idx + 1]
                            if sma50.get((sig_d, row["Symbol"]), False):
                                trend_sigs.setdefault(exec_d, []).append((row["Symbol"], 1.0))
                    df_eq, blot, perf = simulate_advanced_engine(
                        s_name, trend_sigs, m_dates, market, max_positions=3, max_hold_sessions=20, fee_sched=sched
                    )

                # 4. ML Model Strategies
                else:
                    m_key = strat["model"]
                    probs_active = model_dict["Logistic Regression"]["probs_va" if win_name == "VALIDATION" else "probs_te"] if m_key == "lr" else (
                        model_dict["Random Forest"]["probs_va" if win_name == "VALIDATION" else "probs_te"] if m_key == "rf" else
                        model_dict["LightGBM"]["probs_va" if win_name == "VALIDATION" else "probs_te"]
                    )
                    th = strat["th"]
                    use_sma = strat["sma"]
                    sl = strat["sl"]
                    tp = strat["tp"]
                    meta_df = meta_va if win_name == "VALIDATION" else meta_te

                    strat_sigs = {}
                    for idx, row in meta_df.iterrows():
                        sig_d = row["Date"]
                        cur_idx = all_dates.index(sig_d)
                        if cur_idx < len(all_dates) - 1:
                            exec_d = all_dates[cur_idx + 1]
                            p_up = float(probs_active[idx, 2])
                            p_down = float(probs_active[idx, 0])
                            if p_up >= th and p_up > p_down:
                                if not use_sma or sma50.get((sig_d, row["Symbol"]), False):
                                    strat_sigs.setdefault(exec_d, []).append((row["Symbol"], p_up))
                    for d in strat_sigs:
                        strat_sigs[d].sort(key=lambda x: x[1], reverse=True)

                    df_eq, blot, perf = simulate_advanced_engine(
                        s_name, strat_sigs, m_dates, market, max_positions=3, max_hold_sessions=20,
                        stop_loss_pct=sl, take_profit_pct=tp, fee_sched=sched
                    )

                master_profit_records.append({
                    "strategy_id": s_id,
                    "model_family": s_fam,
                    "strategy_name": s_name,
                    "window": win_name,
                    "cost_scenario": cost_name,
                    "entry_threshold": strat.get("th", np.nan),
                    "exit_rule": f"SL {strat.get('sl')*100:.0f}%" if strat.get("sl") else ("Fixed 20d" if "LR" in s_id or "RF" in s_id or "LGB" in s_id else "Benchmark"),
                    "final_equity_inr": round(perf["final_equity"], 2),
                    "net_return_pct": round(perf["total_return_pct"], 2),
                    "cagr_pct": round(perf["cagr_pct"], 2),
                    "max_drawdown_pct": round(perf["max_drawdown_pct"], 2),
                    "annualized_volatility_pct": round(perf["annualized_volatility_pct"], 2),
                    "sharpe_ratio": round(perf["sharpe_ratio"], 2),
                    "total_trades": perf["total_trades"],
                    "win_rate_pct": round(perf["win_rate_pct"], 2),
                    "avg_win_inr": round(perf["avg_win_inr"], 2),
                    "avg_loss_inr": round(perf["avg_loss_inr"], 2),
                    "payoff_ratio": round(perf["payoff_ratio"], 2),
                    "profit_factor": round(perf["profit_factor"], 2),
                    "net_expectancy_per_trade_inr": round(perf["expectancy_per_trade_inr"], 2),
                    "turnover_multiple": round(perf["turnover_multiple"], 2),
                    "total_friction_inr": round(perf["commissions_paid"], 2),
                })

                # Save trade ledger for key test strategies under Base costs
                if cost_name == "Base (0.39%)" and s_id in ["LR_01_BASE", "LR_02_TH038_SMA", "LR_04_TH040_HOLD20", "LR_05_TH040_STOP7_TP10", "LR_06_TH042_HOLD20", "LGB_02_TH038_HOLD20"]:
                    df_rt = extract_round_trips(blot, s_id, win_name)
                    trade_ledger_records.append(df_rt)

    df_master_profit = pd.DataFrame(master_profit_records)
    df_master_profit.to_csv(REPORT_DIR / "net_profitability_comparison.csv", index=False)
    print("Saved: net_profitability_comparison.csv")

    if trade_ledger_records:
        df_ledger_all = pd.concat(trade_ledger_records, ignore_index=True)
        df_ledger_all.to_csv(REPORT_DIR / "trade_by_trade_ledger.csv", index=False)
        print(f"Saved: trade_by_trade_ledger.csv ({len(df_ledger_all)} trades recorded)")

    print("\n" + "=" * 95)
    print("MASTER PROFITABILITY COMPARISON (TEST WINDOW, BASE COSTS)")
    print("=" * 95)
    test_view = df_master_profit[(df_master_profit["window"] == "TEST") & (df_master_profit["cost_scenario"] == "Base (0.39%)")][
        ["strategy_name", "final_equity_inr", "net_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades", "win_rate_pct", "payoff_ratio", "profit_factor", "net_expectancy_per_trade_inr"]
    ]
    print(test_view.to_string(index=False))

    print("\n" + "=" * 95)
    print("MASTER PROFITABILITY COMPARISON (VALIDATION WINDOW, BASE COSTS)")
    print("=" * 95)
    val_view = df_master_profit[(df_master_profit["window"] == "VALIDATION") & (df_master_profit["cost_scenario"] == "Base (0.39%)")][
        ["strategy_name", "final_equity_inr", "net_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades", "win_rate_pct", "payoff_ratio", "profit_factor", "net_expectancy_per_trade_inr"]
    ]
    print(val_view.to_string(index=False))

    print("\nPhase 16 Profitability-First Research Script Finished Successfully!")


if __name__ == "__main__":
    main()

