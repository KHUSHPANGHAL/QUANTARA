"""
Quantara AI — Phase 15B: Controlled Strategy Improvement
Exhaustive, rigorous experimental suite testing SMA50 trend filter, stop-loss rules,
probability calibration, threshold selection, and controlled combinations.
Preserves all existing data, models, configurations, and prior outputs.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# DIRECTORIES & CONFIGURATION
# -----------------------------------------------------------------------------
ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "phase15b_controlled_improvement"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.0

# Base Fee Schedule (0.3937% Round-Trip + 0.05% slippage each way)
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

# Conservative Fee Schedule (0.50% statutory + 0.10% slippage each way)
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

    # 50-day SMA computed strictly historically
    raw["sma50"] = raw.groupby("Symbol")["Close"].transform(lambda x: x.rolling(50, min_periods=50).mean())
    sma50_lookup = {(row["Date"], row["Symbol"]): bool(row["Close"] > row["sma50"]) for _, row in raw.dropna(subset=["sma50"]).iterrows()}

    # Splits
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
        "profit_factor": float(profit_factor),
        "commissions_paid": float(tot_comm),
        "dividends_received": float(tot_div),
        "turnover_multiple": float(total_turnover),
    }


def simulate_realistic_engine(
    strategy_name,
    signals_by_date,
    market_dates,
    market_obj,
    max_positions=3,
    max_hold_sessions=20,
    stop_loss_pct=None,
    fee_sched=BASE_FEE_SCHEDULE
):
    """
    Event-driven simulation with realistic gap execution and stop loss rules.
    """
    dates = market_dates
    opens = market_obj["opens"]
    closes = market_obj["closes"]
    lows = market_obj["lows"]
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

        # 2. CHECK PENDING SCHEDULED EXITS & GAP DOWN STOPS AT OPEN
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            raw_open = opens.loc[current_date, sym]
            p_stop = pos["entry_price"] * (1.0 - stop_loss_pct) if stop_loss_pct is not None else -1e9

            # Check gap down below stop
            is_gap_stop = (stop_loss_pct is not None and raw_open <= p_stop and pos["entry_date"] < current_date)
            is_sched_exit = pos.get("pending_exit", False)

            if is_gap_stop or is_sched_exit:
                # Credit dividend if ex-date
                div_key = (current_date, sym)
                if div_key in div_lookup and pos["entry_date"] < current_date:
                    div_payout = pos["shares"] * div_lookup[div_key]
                    cash += div_payout
                    trade_blotter.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": float(div_lookup[div_key]),
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

                reason = "STOP_LOSS_GAP_DOWN" if is_gap_stop else pos.get("exit_reason", "SCHEDULED_EXIT")
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

        # 4. AGE POSITIONS & PROCESS INTRADAY STOPS & DIVIDENDS
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            pos["sessions_held"] += 1

            # Check intraday stop
            if stop_loss_pct is not None:
                p_stop = pos["entry_price"] * (1.0 - stop_loss_pct)
                raw_low = lows.loc[current_date, sym]
                if raw_low <= p_stop:
                    # Intraday breach executed at stop price minus slippage
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
            div_key = (current_date, sym)
            if div_key in div_lookup and pos["entry_date"] < current_date:
                credited = pos.get("dividend_credited_dates", set())
                if current_date not in credited:
                    div_payout = pos["shares"] * div_lookup[div_key]
                    cash += div_payout
                    trade_blotter.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": float(div_lookup[div_key]),
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
    print("QUANTARA AI — PHASE 15B: CONTROLLED STRATEGY IMPROVEMENT")
    print("=" * 80)

    market = load_all_market_data()
    all_dates = market["all_trading_dates"]
    val_dates = market["val_trading_dates"]
    test_dates = market["test_trading_dates"]
    sma50 = market["sma50lookup"] = market["sma50_lookup"]

    print(f"Total historical trading sessions: {len(all_dates)}")
    print(f"Validation window: {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')} ({len(val_dates)} sessions)")
    print(f"Test window:       {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} sessions)")

    # Load 20d Datasets (Splits V3)
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

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_tr)
    probs_va = lr.predict_proba(X_va_sc)
    probs_te = lr.predict_proba(X_te_sc)

    # Pre-generate uncalibrated next-day signals
    signals_va_raw = {}
    for idx, row in meta_va.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        exec_d = all_dates[cur_idx + 1]
        p_up = float(probs_va[idx, 2])
        p_down = float(probs_va[idx, 0])
        signals_va_raw.setdefault(exec_d, []).append((row["Symbol"], p_up, p_down, sig_d))

    signals_te_raw = {}
    for idx, row in meta_te.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            p_up = float(probs_te[idx, 2])
            p_down = float(probs_te[idx, 0])
            signals_te_raw.setdefault(exec_d, []).append((row["Symbol"], p_up, p_down, sig_d))

    # Platt Calibration fit on Validation set ONLY
    p_va_up = probs_va[:, 2].reshape(-1, 1)
    is_up_va = (y_va == 2).astype(int)
    platt = LogisticRegression()
    platt.fit(p_va_up, is_up_va)

    probs_va_cal_up = platt.predict_proba(p_va_up)[:, 1]
    p_te_up = probs_te[:, 2].reshape(-1, 1)
    probs_te_cal_up = platt.predict_proba(p_te_up)[:, 1]

    signals_va_cal = {}
    for idx, row in meta_va.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        exec_d = all_dates[cur_idx + 1]
        p_cal_up = float(probs_va_cal_up[idx])
        p_down = float(probs_va[idx, 0])
        signals_va_cal.setdefault(exec_d, []).append((row["Symbol"], p_cal_up, p_down, sig_d))

    signals_te_cal = {}
    for idx, row in meta_te.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            p_cal_up = float(probs_te_cal_up[idx])
            p_down = float(probs_te[idx, 0])
            signals_te_cal.setdefault(exec_d, []).append((row["Symbol"], p_cal_up, p_down, sig_d))

    # -------------------------------------------------------------------------
    # EXPERIMENT A: SMA50 TREND FILTER
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT A: SMA50 TREND FILTER")
    print("=" * 60)

    trend_filter_records = []
    for win_name, sigs_raw, m_dates, meta_df in [("VALIDATION", signals_va_raw, val_dates, meta_va),
                                                ("TEST", signals_te_raw, test_dates, meta_te)]:
        # Filter candidate signals
        cand_signals = 0
        accepted_signals = 0
        rejected_signals = 0

        sigs_base = {}
        sigs_sma = {}

        for exec_d, cand_list in sigs_raw.items():
            b_list = []
            s_list = []
            for sym, p_up, p_down, sig_d in cand_list:
                if p_up >= 0.35 and p_up > p_down:
                    cand_signals += 1
                    b_list.append((sym, p_up))
                    is_above = sma50.get((sig_d, sym), False)
                    if is_above:
                        accepted_signals += 1
                        s_list.append((sym, p_up))
                    else:
                        rejected_signals += 1
            if b_list:
                b_list.sort(key=lambda x: x[1], reverse=True)
                sigs_base[exec_d] = b_list
            if s_list:
                s_list.sort(key=lambda x: x[1], reverse=True)
                sigs_sma[exec_d] = s_list

        _, b_blot, p_base = simulate_realistic_engine("Baseline", sigs_base, m_dates, market)
        _, s_blot, p_sma = simulate_realistic_engine("SMA50_Filter", sigs_sma, m_dates, market)

        trend_filter_records.append({
            "window": win_name,
            "variant": "Baseline (No SMA Filter)",
            "candidate_signals": cand_signals,
            "executed_trades": p_base["total_trades"],
            "rejected_signals": 0,
            "final_equity_inr": round(p_base["final_equity"], 2),
            "net_return_pct": round(p_base["total_return_pct"], 2),
            "max_drawdown_pct": round(p_base["max_drawdown_pct"], 2),
            "win_rate_pct": round(p_base["win_rate_pct"], 2),
            "profit_factor": round(p_base["profit_factor"], 2),
        })
        trend_filter_records.append({
            "window": win_name,
            "variant": "SMA50 Filter (Longs only above SMA50)",
            "candidate_signals": cand_signals,
            "executed_trades": p_sma["total_trades"],
            "rejected_signals": rejected_signals,
            "final_equity_inr": round(p_sma["final_equity"], 2),
            "net_return_pct": round(p_sma["total_return_pct"], 2),
            "max_drawdown_pct": round(p_sma["max_drawdown_pct"], 2),
            "win_rate_pct": round(p_sma["win_rate_pct"], 2),
            "profit_factor": round(p_sma["profit_factor"], 2),
        })

    df_tf = pd.DataFrame(trend_filter_records)
    df_tf.to_csv(REPORT_DIR / "trend_filter_comparison.csv", index=False)
    print("Saved: trend_filter_comparison.csv")
    print(df_tf.to_string(index=False))

    # -------------------------------------------------------------------------
    # EXPERIMENT B: STOP-LOSS RULES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT B: STOP-LOSS RULES (ISOLATED ON BASELINE SIGNALS)")
    print("=" * 60)

    stop_loss_records = []
    # Build baseline signals for both windows
    sigs_val_base = {}
    for exec_d, cand_list in signals_va_raw.items():
        filt = [(sym, p_up) for sym, p_up, p_down, _ in cand_list if p_up >= 0.35 and p_up > p_down]
        if filt:
            filt.sort(key=lambda x: x[1], reverse=True)
            sigs_val_base[exec_d] = filt

    sigs_test_base = {}
    for exec_d, cand_list in signals_te_raw.items():
        filt = [(sym, p_up) for sym, p_up, p_down, _ in cand_list if p_up >= 0.35 and p_up > p_down]
        if filt:
            filt.sort(key=lambda x: x[1], reverse=True)
            sigs_test_base[exec_d] = filt

    for win_name, sigs, m_dates in [("VALIDATION", sigs_val_base, val_dates), ("TEST", sigs_test_base, test_dates)]:
        for sl_label, sl_val in [("No Stop-Loss (20d Hold)", None), ("5% Stop-Loss", 0.05), ("7% Stop-Loss", 0.07)]:
            for cost_name, sched in [("Base (0.39%)", BASE_FEE_SCHEDULE), ("Conservative (0.50%)", CONSERVATIVE_FEE_SCHEDULE)]:
                _, blot, perf = simulate_realistic_engine(
                    f"Stop_{sl_label}", sigs, m_dates, market, max_positions=3, max_hold_sessions=20,
                    stop_loss_pct=sl_val, fee_sched=sched
                )
                stop_loss_records.append({
                    "window": win_name,
                    "stop_loss_rule": sl_label,
                    "cost_scenario": cost_name,
                    "final_equity_inr": round(perf["final_equity"], 2),
                    "net_return_pct": round(perf["total_return_pct"], 2),
                    "max_drawdown_pct": round(perf["max_drawdown_pct"], 2),
                    "sharpe_ratio": round(perf["sharpe_ratio"], 2),
                    "total_trades": perf["total_trades"],
                    "win_rate_pct": round(perf["win_rate_pct"], 2),
                    "avg_win_inr": round(perf["avg_win_inr"], 2),
                    "avg_loss_inr": round(perf["avg_loss_inr"], 2),
                    "profit_factor": round(perf["profit_factor"], 2),
                    "commissions_paid_inr": round(perf["commissions_paid"], 2),
                })

    df_sl = pd.DataFrame(stop_loss_records)
    df_sl.to_csv(REPORT_DIR / "stop_loss_comparison.csv", index=False)
    print("Saved: stop_loss_comparison.csv")
    print(df_sl[df_sl["cost_scenario"] == "Base (0.39%)"].to_string(index=False))

    # -------------------------------------------------------------------------
    # EXPERIMENT C: PROBABILITY CALIBRATION & THRESHOLD SELECTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT C: PROBABILITY CALIBRATION & THRESHOLD GRID")
    print("=" * 60)

    calib_records = []
    # Evaluate calibration quality
    is_up_te = (y_te == 2).astype(int)
    brier_uncal = np.mean((probs_te[:, 2] - is_up_te) ** 2)
    brier_cal = np.mean((probs_te_cal_up - is_up_te) ** 2)

    # Threshold grid evaluation across both windows
    for win_name, sigs_raw_dict, sigs_cal_dict, m_dates in [("VALIDATION", signals_va_raw, signals_va_cal, val_dates),
                                                            ("TEST", signals_te_raw, signals_te_cal, test_dates)]:
        for cal_mode, active_sigs in [("Uncalibrated", sigs_raw_dict), ("Platt_Calibrated", sigs_cal_dict)]:
            for th in [0.35, 0.38, 0.40, 0.42, 0.45]:
                th_sigs = {}
                n_eligible_signals = 0
                for exec_d, cand_list in active_sigs.items():
                    filt = []
                    for sym, p_up, p_down, _ in cand_list:
                        if p_up >= th and p_up > p_down:
                            n_eligible_signals += 1
                            filt.append((sym, p_up))
                    if filt:
                        filt.sort(key=lambda x: x[1], reverse=True)
                        th_sigs[exec_d] = filt

                _, blot, perf = simulate_realistic_engine(
                    f"Th_{th}", th_sigs, m_dates, market, max_positions=3, max_hold_sessions=20
                )

                calib_records.append({
                    "window": win_name,
                    "calibration_model": cal_mode,
                    "entry_threshold": th,
                    "eligible_signals": n_eligible_signals,
                    "executed_trades": perf["total_trades"],
                    "final_equity_inr": round(perf["final_equity"], 2),
                    "net_return_pct": round(perf["total_return_pct"], 2),
                    "max_drawdown_pct": round(perf["max_drawdown_pct"], 2),
                    "win_rate_pct": round(perf["win_rate_pct"], 2),
                    "profit_factor": round(perf["profit_factor"], 2),
                    "brier_score_test": round(brier_cal if cal_mode == "Platt_Calibrated" else brier_uncal, 4) if win_name == "TEST" else np.nan,
                })

    df_cal = pd.DataFrame(calib_records)
    df_cal.to_csv(REPORT_DIR / "probability_calibration_results.csv", index=False)
    print("Saved: probability_calibration_results.csv")
    print(df_cal[(df_cal["calibration_model"] == "Uncalibrated") & (df_cal["window"] == "TEST")].to_string(index=False))

    # -------------------------------------------------------------------------
    # EXPERIMENT D: CONTROLLED COMBINATIONS & MASTER COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT D: CONTROLLED COMBINATIONS & MASTER COMPARISON")
    print("=" * 60)

    combinations_to_test = [
        # Baseline
        {"id": "STRAT_01_BASELINE", "name": "Baseline (Th 0.35, No SMA, No Stop)", "th": 0.35, "sma": False, "sl": None},
        # Individual improvements
        {"id": "STRAT_02_SMA50_ONLY", "name": "SMA50 Filter Only (Th 0.35)", "th": 0.35, "sma": True, "sl": None},
        {"id": "STRAT_03_STOP7PCT_ONLY", "name": "7% Stop-Loss Only (Th 0.35)", "th": 0.35, "sma": False, "sl": 0.07},
        {"id": "STRAT_04_TH038_ONLY", "name": "Stronger Threshold Only (Th 0.38)", "th": 0.38, "sma": False, "sl": None},
        {"id": "STRAT_05_TH040_ONLY", "name": "High Conviction Only (Th 0.40)", "th": 0.40, "sma": False, "sl": None},
        # Combinations supported by validation evidence
        {"id": "STRAT_06_COMB_TH038_SMA", "name": "Comb A: Th 0.38 + SMA50", "th": 0.38, "sma": True, "sl": None},
        {"id": "STRAT_07_COMB_TH038_STOP7", "name": "Comb B: Th 0.38 + 7% Stop", "th": 0.38, "sma": False, "sl": 0.07},
        {"id": "STRAT_08_COMB_TH038_SMA_STOP7", "name": "Comb C: Th 0.38 + SMA50 + 7% Stop", "th": 0.38, "sma": True, "sl": 0.07},
        {"id": "STRAT_09_COMB_TH040_STOP7", "name": "Comb D: Th 0.40 + 7% Stop", "th": 0.40, "sma": False, "sl": 0.07},
        {"id": "STRAT_10_COMB_TH042_STOP7", "name": "Comb E: Th 0.42 + 7% Stop", "th": 0.42, "sma": False, "sl": 0.07},
    ]

    master_records = []
    trade_comparison_records = []

    for c in combinations_to_test:
        c_name = c["name"]
        th = c["th"]
        use_sma = c["sma"]
        sl = c["sl"]

        for win_name, sigs_raw, m_dates in [("VALIDATION", signals_va_raw, val_dates), ("TEST", signals_te_raw, test_dates)]:
            # Build signals
            comb_sigs = {}
            for exec_d, cand_list in sigs_raw.items():
                filt = []
                for sym, p_up, p_down, sig_d in cand_list:
                    if p_up >= th and p_up > p_down:
                        if not use_sma or sma50.get((sig_d, sym), False):
                            filt.append((sym, p_up))
                if filt:
                    filt.sort(key=lambda x: x[1], reverse=True)
                    comb_sigs[exec_d] = filt

            for cost_name, sched in [("Base (0.39%)", BASE_FEE_SCHEDULE), ("Conservative (0.50%)", CONSERVATIVE_FEE_SCHEDULE)]:
                df_eq, blot, perf = simulate_realistic_engine(
                    c["id"], comb_sigs, m_dates, market, max_positions=3, max_hold_sessions=20,
                    stop_loss_pct=sl, fee_sched=sched
                )

                master_records.append({
                    "strategy_id": c["id"],
                    "strategy_name": c_name,
                    "window": win_name,
                    "cost_scenario": cost_name,
                    "entry_threshold": th,
                    "trend_filter": "SMA50" if use_sma else "None",
                    "stop_loss": f"{sl*100:.0f}%" if sl else "None",
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
                    "profit_factor": round(perf["profit_factor"], 2),
                    "turnover_multiple": round(perf["turnover_multiple"], 2),
                    "commissions_paid_inr": round(perf["commissions_paid"], 2),
                    "dividends_received_inr": round(perf["dividends_received"], 2),
                })

                # Save round trips for Base scenario
                if cost_name == "Base (0.39%)" and c["id"] in ["STRAT_01_BASELINE", "STRAT_03_STOP7PCT_ONLY", "STRAT_06_COMB_TH038_SMA", "STRAT_07_COMB_TH038_STOP7"]:
                    df_rt = extract_round_trips(blot, c["id"], win_name)
                    trade_comparison_records.append(df_rt)

    df_master = pd.DataFrame(master_records)
    df_master.to_csv(REPORT_DIR / "strategy_comparison.csv", index=False)
    print("Saved: strategy_comparison.csv")

    if trade_comparison_records:
        df_trades_all = pd.concat(trade_comparison_records, ignore_index=True)
        df_trades_all.to_csv(REPORT_DIR / "trade_by_trade_comparison.csv", index=False)
        print(f"Saved: trade_by_trade_comparison.csv ({len(df_trades_all)} trades recorded)")

    print("\n--- MASTER STRATEGY COMPARISON TABLE (TEST WINDOW, BASE COSTS) ---")
    test_base = df_master[(df_master["window"] == "TEST") & (df_master["cost_scenario"] == "Base (0.39%)")][
        ["strategy_name", "final_equity_inr", "net_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades", "win_rate_pct", "profit_factor"]
    ]
    print(test_base.to_string(index=False))

    print("\n--- MASTER STRATEGY COMPARISON TABLE (VALIDATION WINDOW, BASE COSTS) ---")
    val_base = df_master[(df_master["window"] == "VALIDATION") & (df_master["cost_scenario"] == "Base (0.39%)")][
        ["strategy_name", "final_equity_inr", "net_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades", "win_rate_pct", "profit_factor"]
    ]
    print(val_base.to_string(index=False))

    print("\nPhase 15B Script Execution Finished Successfully!")


if __name__ == "__main__":
    main()

