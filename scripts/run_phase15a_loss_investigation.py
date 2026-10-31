"""
Quantara AI — Phase 15A: Losing Trade Root-Cause Investigation
Exhaustive, strictly diagnostic analysis of why the Quantara AI trading strategies lose money.
Preserves all existing data, models, configurations, and outputs.
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
REPORT_DIR = ROOT / "reports" / "phase15a_loss_investigation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.0

# Exact Phase 14 NSE Statutory Cash Delivery Fee Structure
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

    meta_test = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "metadata_test.csv")
    meta_test["Date"] = pd.to_datetime(meta_test["Date"])
    start_date = meta_test["Date"].min()

    raw_test = raw[raw["Date"] >= start_date].sort_values(["Date", "Symbol"]).reset_index(drop=True)
    all_trading_dates = sorted(raw_test["Date"].unique())

    opens = raw_test.pivot(index="Date", columns="Symbol", values="Open")
    closes = raw_test.pivot(index="Date", columns="Symbol", values="Close")
    adj_closes = raw_test.pivot(index="Date", columns="Symbol", values="Adj Close")

    div_df = extract_dividends(raw)
    test_divs = div_df[(div_df["Ex_Date"] >= start_date) & (div_df["Ex_Date"] <= all_trading_dates[-1])].copy()
    dividend_lookup = {(row["Ex_Date"], row["Symbol"]): row["div_per_share"] for _, row in test_divs.iterrows()} if len(div_df) > 0 else {}

    return {
        "raw": raw,
        "raw_test": raw_test,
        "all_trading_dates": all_trading_dates,
        "test_signal_dates": sorted(meta_test["Date"].unique()),
        "opens": opens,
        "closes": closes,
        "adj_closes": adj_closes,
        "symbols": sorted(raw["Symbol"].unique()),
        "dividend_lookup": dividend_lookup,
    }


def compute_comprehensive_portfolio_metrics(equity_series, daily_returns, trade_blotter, is_cash=False):
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
    avg_loss = gross_loss / len(losing_trades) if len(losing_trades) > 0 else 0.0
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
        "avg_winning_trade": float(avg_win),
        "avg_losing_trade": float(avg_loss),
        "profit_factor": float(profit_factor),
        "commissions_paid": float(tot_comm),
        "dividends_received": float(tot_div),
        "turnover_multiple": float(total_turnover),
    }


def simulate_with_round_trips(strategy_name, signals_by_date, market, max_positions=3, max_hold_sessions=20):
    dates = market["all_trading_dates"]
    opens = market["opens"]
    closes = market["closes"]
    div_lookup = market["dividend_lookup"]
    buy_fee, sell_fee, slippage = get_fee_rates(BASE_FEE_SCHEDULE)

    cash = STARTING_CAPITAL
    open_positions = {}
    trade_blotter = []
    daily_records = []
    pending_exits = set()

    round_trip_trades = []
    trade_counter = 0

    for i, current_date in enumerate(dates):
        # 1. EXECUTE PENDING EXITS AT OPEN
        for sym in list(pending_exits):
            if sym in open_positions:
                pos = open_positions[sym]
                # Dividend credit on ex-date exit
                div_key = (current_date, sym)
                if div_key in div_lookup and pos["entry_date"] < current_date:
                    div_payout = pos["shares"] * div_lookup[div_key]
                    cash += div_payout
                    pos["dividends_received"] += div_payout
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

                raw_open = opens.loc[current_date, sym]
                exec_price = raw_open * (1.0 - slippage)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * sell_fee
                net_proceeds = gross_proceeds - fee
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + buy_fee)
                net_pnl = net_proceeds - total_cost

                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_pnl),
                    "reason": pos.get("exit_reason", "SCHEDULED_EXIT"),
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })

                # Capture full lifecycle for diagnostics
                trade_counter += 1
                gross_pnl = (exec_price - pos["entry_price"]) * pos["shares"]
                slippage_cost = (pos["shares"] * pos["raw_entry_open"] * slippage) + (pos["shares"] * raw_open * slippage)
                tot_comm = pos["buy_commission"] + fee
                final_trade_net_pnl = net_pnl + pos["dividends_received"]

                round_trip_trades.append({
                    "trade_id": f"T{trade_counter:02d}",
                    "strategy": strategy_name,
                    "symbol": sym,
                    "direction": "LONG",
                    "signal_date": pos["signal_date"].strftime("%Y-%m-%d"),
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "sessions_held": pos["sessions_held"],
                    "shares": pos["shares"],
                    "raw_entry_price": float(pos["raw_entry_open"]),
                    "entry_exec_price": float(pos["entry_price"]),
                    "raw_exit_price": float(raw_open),
                    "exit_exec_price": float(exec_price),
                    "cost_basis_inr": float(total_cost),
                    "gross_proceeds_inr": float(gross_proceeds),
                    "gross_pnl_inr": float(gross_pnl),
                    "gross_return_pct": float(((exec_price / pos["entry_price"]) - 1.0) * 100.0),
                    "buy_commission_inr": float(pos["buy_commission"]),
                    "sell_commission_inr": float(fee),
                    "total_commissions_inr": float(tot_comm),
                    "slippage_cost_inr": float(slippage_cost),
                    "dividend_credited_inr": float(pos["dividends_received"]),
                    "net_pnl_inr": float(final_trade_net_pnl),
                    "net_return_pct": float((final_trade_net_pnl / total_cost) * 100.0),
                    "exit_reason": pos.get("exit_reason", "SCHEDULED_EXIT"),
                    "model_p_up": float(pos["p_up"]),
                    "model_p_down": float(pos["p_down"]),
                    "model_conviction": float(pos["p_up"] - pos["p_down"]),
                })
                del open_positions[sym]
        pending_exits.clear()

        # 2. EXECUTE CANDIDATE BUYS AT OPEN
        candidate_buys = signals_by_date.get(current_date, [])
        for sym, p_up, p_down, sig_d in candidate_buys:
            if len(open_positions) < max_positions and sym not in open_positions:
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
                            "raw_entry_open": float(raw_open),
                            "entry_price": float(exec_price),
                            "entry_date": current_date,
                            "signal_date": sig_d,
                            "sessions_held": 0,
                            "exit_reason": "SCHEDULED_EXIT",
                            "buy_commission": float(fee),
                            "p_up": float(p_up),
                            "p_down": float(p_down),
                            "dividends_received": 0.0,
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

        # 3. AGE POSITIONS & PROCESS INTRADAY DIVIDENDS
        for sym in open_positions:
            open_positions[sym]["sessions_held"] += 1

        for sym, pos in list(open_positions.items()):
            div_key = (current_date, sym)
            if div_key in div_lookup and pos["entry_date"] < current_date:
                credited = pos.get("dividend_credited_dates", set())
                if current_date not in credited:
                    div_payout = pos["shares"] * div_lookup[div_key]
                    cash += div_payout
                    pos["dividends_received"] += div_payout
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

        # 4. MARK-TO-MARKET & LIQUIDATE ON FINAL DAY
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
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })

                trade_counter += 1
                gross_pnl = (exec_price - pos["entry_price"]) * pos["shares"]
                slippage_cost = (pos["shares"] * pos["raw_entry_open"] * slippage) + (pos["shares"] * raw_close * slippage)
                tot_comm = pos["buy_commission"] + fee
                final_trade_net_pnl = net_pnl + pos["dividends_received"]

                round_trip_trades.append({
                    "trade_id": f"T{trade_counter:02d}",
                    "strategy": strategy_name,
                    "symbol": sym,
                    "direction": "LONG",
                    "signal_date": pos["signal_date"].strftime("%Y-%m-%d"),
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "sessions_held": pos["sessions_held"],
                    "shares": pos["shares"],
                    "raw_entry_price": float(pos["raw_entry_open"]),
                    "entry_exec_price": float(pos["entry_price"]),
                    "raw_exit_price": float(raw_close),
                    "exit_exec_price": float(exec_price),
                    "cost_basis_inr": float(total_cost),
                    "gross_proceeds_inr": float(gross_proceeds),
                    "gross_pnl_inr": float(gross_pnl),
                    "gross_return_pct": float(((exec_price / pos["entry_price"]) - 1.0) * 100.0),
                    "buy_commission_inr": float(pos["buy_commission"]),
                    "sell_commission_inr": float(fee),
                    "total_commissions_inr": float(tot_comm),
                    "slippage_cost_inr": float(slippage_cost),
                    "dividend_credited_inr": float(pos["dividends_received"]),
                    "net_pnl_inr": float(final_trade_net_pnl),
                    "net_return_pct": float((final_trade_net_pnl / total_cost) * 100.0),
                    "exit_reason": "FINAL_DAY_LIQUIDATION",
                    "model_p_up": float(pos["p_up"]),
                    "model_p_down": float(pos["p_down"]),
                    "model_conviction": float(pos["p_up"] - pos["p_down"]),
                })

            open_positions.clear()
            val = 0.0
            total_equity = cash

        daily_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "cash": float(cash),
            "positions_val": float(val),
            "total_equity": float(total_equity),
            "open_positions": len(open_positions),
        })

        # 5. PLAN EXITS FOR NEXT OPEN
        if i < len(dates) - 1:
            for sym, pos in open_positions.items():
                if pos["sessions_held"] >= max_hold_sessions:
                    pos["exit_reason"] = f"MAX_HOLD_{max_hold_sessions}D"
                    pending_exits.add(sym)

    df_eq = pd.DataFrame(daily_records)
    df_eq["daily_return"] = df_eq["total_equity"].pct_change().fillna(0.0)
    perf = compute_comprehensive_portfolio_metrics(df_eq["total_equity"], df_eq["daily_return"], trade_blotter)
    return df_eq, trade_blotter, pd.DataFrame(round_trip_trades), perf


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 15A: LOSING TRADE ROOT-CAUSE INVESTIGATION")
    print("=" * 80)

    market = load_all_market_data()
    all_dates = market["all_trading_dates"]
    print(f"Loaded {len(all_dates)} trading sessions across {len(market['symbols'])} stocks.")

    raw_sorted = market["raw"].sort_values(["Symbol", "Date"]).copy().reset_index(drop=True)
    raw_sorted["ret_1d"] = raw_sorted.groupby("Symbol")["Adj Close"].pct_change()
    raw_sorted["vol_20d"] = raw_sorted.groupby("Symbol")["ret_1d"].transform(lambda x: x.rolling(20, min_periods=20).std() * np.sqrt(252))
    vol_dict = {(row["Date"], row["Symbol"]): float(row["vol_20d"]) for _, row in raw_sorted.dropna().iterrows()}

    # Market Regime lookup
    raw_sorted["sma20"] = raw_sorted.groupby("Symbol")["Adj Close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    raw_sorted["above_sma20"] = (raw_sorted["Adj Close"] > raw_sorted["sma20"]).astype(float)
    market_breadth = raw_sorted.groupby("Date")["above_sma20"].mean().to_dict()

    # SMA50 for stock trend lookup
    raw_sorted["sma50"] = raw_sorted.groupby("Symbol")["Adj Close"].transform(lambda x: x.rolling(50, min_periods=50).mean())
    stock_sma50_dict = {(row["Date"], row["Symbol"]): (row["Adj Close"] > row["sma50"]) for _, row in raw_sorted.dropna().iterrows()}

    # Load 20d ML Models
    X_train_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "X_train.csv")
    y_train_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_test_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "X_test.csv")
    y_test_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "y_test.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    meta_test_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "metadata_test.csv")
    meta_test_20d["Date"] = pd.to_datetime(meta_test_20d["Date"])

    scaler_20d = StandardScaler()
    scaler_20d.fit(X_train_20d)
    X_tr_sc = scaler_20d.transform(X_train_20d)
    X_te_sc = scaler_20d.transform(X_test_20d)

    lr_20d = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_20d.fit(X_tr_sc, y_train_20d)
    probs_lr_test = lr_20d.predict_proba(X_te_sc)

    # Pre-generate Next-Day Signals
    ml_signals_by_exec_date = {}
    for idx, row in meta_test_20d.iterrows():
        sig_d = row["Date"]
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            p_up = float(probs_lr_test[idx, 2])
            p_down = float(probs_lr_test[idx, 0])
            if exec_d not in ml_signals_by_exec_date:
                ml_signals_by_exec_date[exec_d] = []
            ml_signals_by_exec_date[exec_d].append((row["Symbol"], p_up, p_down, sig_d))

    signals_ml_all = {}
    for exec_d, cands in ml_signals_by_exec_date.items():
        filt = [(sym, p_up, p_down, sig_d) for sym, p_up, p_down, sig_d in cands if p_up >= 0.35 and p_up > p_down]
        filt.sort(key=lambda x: x[1], reverse=True)
        signals_ml_all[exec_d] = filt

    # Run Baseline Simulation (Strategy 5: 20d ML All Eligible Base)
    df_eq, blotter, df_trades, perf = simulate_with_round_trips(
        "20d_ML_All_Eligible", signals_ml_all, market, max_positions=3, max_hold_sessions=20
    )

    print("\n--- BASELINE REPRODUCTION VERIFICATION ---")
    print(f"Starting Capital: INR {STARTING_CAPITAL:,.2f}")
    print(f"Final Equity:     INR {perf['final_equity']:,.2f}  (Expected: INR 82,538.93)")
    print(f"Total Return:     {perf['total_return_pct']:+.2f}%       (Expected: -17.46%)")
    print(f"Max Drawdown:     {perf['max_drawdown_pct']:.2f}%        (Expected: 20.66%)")
    print(f"Total Trades:     {perf['total_trades']}               (Expected: 27)")
    print(f"Win Rate:         {perf['win_rate_pct']:.1f}%")
    print(f"Profit Factor:    {perf['profit_factor']:.2f}")

    assert abs(perf["final_equity"] - 82538.93) < 0.05, f"Final equity mismatch! {perf['final_equity']}"
    assert abs(perf["total_return_pct"] - (-17.46)) < 0.05, f"Return mismatch! {perf['total_return_pct']}"
    assert abs(perf["max_drawdown_pct"] - 20.66) < 0.05, f"Drawdown mismatch! {perf['max_drawdown_pct']}"
    assert perf["total_trades"] == 27, f"Trades mismatch! {perf['total_trades']}"
    print(">>> ZERO DELTA CONFIRMED! Baseline is 100% reproduced.\n")

    # Add Regime and Trend metadata to trades
    regimes = []
    breadths = []
    alignments = []
    stock_trends = []
    stock_vols = []

    for _, row in df_trades.iterrows():
        sig_d = pd.to_datetime(row["signal_date"])
        sym = row["symbol"]
        br = market_breadth.get(sig_d, 0.5)
        breadths.append(br)
        if br <= 0.30:
            reg = "BEAR"
            align = False
        elif br >= 0.70:
            reg = "BULL"
            align = True
        else:
            reg = "NEUTRAL"
            align = True
        regimes.append(reg)
        alignments.append(align)

        st_trend = "BULL_ABOVE_SMA50" if stock_sma50_dict.get((sig_d, sym), False) else "BEAR_BELOW_SMA50"
        stock_trends.append(st_trend)
        st_vol = vol_dict.get((sig_d, sym), np.nan)
        stock_vols.append(st_vol)

    df_trades["market_breadth_at_entry"] = [round(b, 4) for b in breadths]
    df_trades["market_regime_at_entry"] = regimes
    df_trades["market_alignment"] = alignments
    df_trades["stock_trend_at_entry"] = stock_trends
    df_trades["stock_vol_annualized"] = [round(v, 4) if not np.isnan(v) else np.nan for v in stock_vols]

    # Save Diagnostic Deliverable 1: trade_by_trade_diagnostics.csv
    trade_diag_path = REPORT_DIR / "trade_by_trade_diagnostics.csv"
    df_trades.to_csv(trade_diag_path, index=False)
    print(f"Saved: {trade_diag_path} ({len(df_trades)} trades)")

    # -------------------------------------------------------------------------
    # DIAGNOSTIC DELIVERABLE 2: stock_loss_breakdown.csv
    # -------------------------------------------------------------------------
    stock_records = []
    tot_portfolio_loss = abs(sum(r["net_pnl_inr"] for _, r in df_trades.iterrows() if r["net_pnl_inr"] < 0))

    for sym, grp in df_trades.groupby("symbol"):
        n_t = len(grp)
        wins = grp[grp["net_pnl_inr"] > 0]
        losses = grp[grp["net_pnl_inr"] <= 0]
        n_w = len(wins)
        n_l = len(losses)
        win_rate = (n_w / n_t) * 100.0 if n_t > 0 else 0.0

        gp = wins["net_pnl_inr"].sum()
        gl = abs(losses["net_pnl_inr"].sum())
        net_pnl = grp["net_pnl_inr"].sum()
        tot_comm = grp["total_commissions_inr"].sum()
        tot_div = grp["dividend_credited_inr"].sum()

        avg_pnl = net_pnl / n_t if n_t > 0 else 0.0
        avg_win = gp / n_w if n_w > 0 else 0.0
        avg_loss = -gl / n_l if n_l > 0 else 0.0
        payoff_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > 0 else np.nan
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
        max_loss = grp["net_pnl_inr"].min()
        loss_share = (gl / tot_portfolio_loss) * 100.0 if tot_portfolio_loss > 0 else 0.0

        stock_records.append({
            "symbol": sym,
            "total_trades": n_t,
            "winning_trades": n_w,
            "losing_trades": n_l,
            "win_rate_pct": round(win_rate, 2),
            "gross_pnl_inr": round(grp["gross_pnl_inr"].sum(), 2),
            "total_commissions_inr": round(tot_comm, 2),
            "dividend_credited_inr": round(tot_div, 2),
            "net_pnl_inr": round(net_pnl, 2),
            "avg_pnl_per_trade_inr": round(avg_pnl, 2),
            "avg_win_inr": round(avg_win, 2),
            "avg_loss_inr": round(avg_loss, 2),
            "payoff_ratio": round(payoff_ratio, 2) if not np.isnan(payoff_ratio) else np.nan,
            "profit_factor": round(pf, 2),
            "max_single_trade_loss_inr": round(max_loss, 2),
            "share_of_total_losses_pct": round(loss_share, 2),
        })

    df_stock = pd.DataFrame(stock_records).sort_values("net_pnl_inr", ascending=True)
    stock_breakdown_path = REPORT_DIR / "stock_loss_breakdown.csv"
    df_stock.to_csv(stock_breakdown_path, index=False)
    print(f"Saved: {stock_breakdown_path}")
    print("\nStock Loss Breakdown:")
    print(df_stock.to_string(index=False))

    # -------------------------------------------------------------------------
    # DIAGNOSTIC DELIVERABLE 3: market_regime_breakdown.csv
    # -------------------------------------------------------------------------
    regime_records = []
    for reg, grp in df_trades.groupby("market_regime_at_entry"):
        n_t = len(grp)
        wins = grp[grp["net_pnl_inr"] > 0]
        losses = grp[grp["net_pnl_inr"] <= 0]
        n_w = len(wins)
        n_l = len(losses)
        win_rate = (n_w / n_t) * 100.0 if n_t > 0 else 0.0

        gp = wins["net_pnl_inr"].sum()
        gl = abs(losses["net_pnl_inr"].sum())
        net_pnl = grp["net_pnl_inr"].sum()
        tot_comm = grp["total_commissions_inr"].sum()
        avg_pnl = net_pnl / n_t if n_t > 0 else 0.0
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
        loss_share = (gl / tot_portfolio_loss) * 100.0 if tot_portfolio_loss > 0 else 0.0

        regime_records.append({
            "market_regime": reg,
            "total_trades": n_t,
            "winning_trades": n_w,
            "losing_trades": n_l,
            "win_rate_pct": round(win_rate, 2),
            "gross_pnl_inr": round(grp["gross_pnl_inr"].sum(), 2),
            "commissions_inr": round(tot_comm, 2),
            "net_pnl_inr": round(net_pnl, 2),
            "avg_pnl_per_trade_inr": round(avg_pnl, 2),
            "profit_factor": round(pf, 2),
            "share_of_total_losses_pct": round(loss_share, 2),
        })

    df_regime = pd.DataFrame(regime_records).sort_values("net_pnl_inr", ascending=True)
    regime_path = REPORT_DIR / "market_regime_breakdown.csv"
    df_regime.to_csv(regime_path, index=False)
    print(f"\nSaved: {regime_path}")
    print("\nMarket Regime Breakdown:")
    print(df_regime.to_string(index=False))

    # -------------------------------------------------------------------------
    # DIAGNOSTIC DELIVERABLE 4: prediction_error_analysis.csv
    # -------------------------------------------------------------------------
    df_test_eval = meta_test_20d.copy()
    df_test_eval["y_true"] = y_test_20d
    df_test_eval["p_down"] = probs_lr_test[:, 0]
    df_test_eval["p_hold"] = probs_lr_test[:, 1]
    df_test_eval["p_up"] = probs_lr_test[:, 2]
    df_test_eval["is_up_true"] = (df_test_eval["y_true"] == 2).astype(int)

    bins = [0.0, 0.35, 0.40, 0.45, 0.50, 1.0]
    labels = ["< 0.35", "[0.35, 0.40)", "[0.40, 0.45)", "[0.45, 0.50)", ">= 0.50"]
    df_test_eval["conf_band"] = pd.cut(df_test_eval["p_up"], bins=bins, labels=labels, right=False)

    calib_records = []
    for band, grp in df_test_eval.groupby("conf_band", observed=False):
        n_samples = len(grp)
        if n_samples == 0:
            continue
        actual_up = grp["is_up_true"].sum()
        actual_up_rate = (actual_up / n_samples) * 100.0
        mean_p_up = grp["p_up"].mean() * 100.0
        calib_gap = mean_p_up - actual_up_rate

        trades_in_band = df_trades[df_trades["model_p_up"].apply(lambda p: (band == "< 0.35" and p < 0.35) or
                                                                            (band == "[0.35, 0.40)" and 0.35 <= p < 0.40) or
                                                                            (band == "[0.40, 0.45)" and 0.40 <= p < 0.45) or
                                                                            (band == "[0.45, 0.50)" and 0.45 <= p < 0.50) or
                                                                            (band == ">= 0.50" and p >= 0.50))]
        n_exec = len(trades_in_band)
        exec_win_rate = (len(trades_in_band[trades_in_band["net_pnl_inr"] > 0]) / n_exec * 100.0) if n_exec > 0 else 0.0
        exec_net_pnl = trades_in_band["net_pnl_inr"].sum() if n_exec > 0 else 0.0

        brier = np.mean((grp["p_up"] - grp["is_up_true"]) ** 2)

        calib_records.append({
            "confidence_band": str(band),
            "test_samples": n_samples,
            "mean_model_p_up_pct": round(mean_p_up, 2),
            "empirical_up_rate_pct": round(actual_up_rate, 2),
            "overconfidence_bias_pct": round(calib_gap, 2),
            "brier_score": round(brier, 4),
            "executed_trades": n_exec,
            "executed_win_rate_pct": round(exec_win_rate, 2),
            "executed_net_pnl_inr": round(exec_net_pnl, 2),
        })

    df_calib = pd.DataFrame(calib_records)
    calib_path = REPORT_DIR / "prediction_error_analysis.csv"
    df_calib.to_csv(calib_path, index=False)
    print(f"\nSaved: {calib_path}")
    print("\nPrediction Error & Calibration Breakdown:")
    print(df_calib.to_string(index=False))

    # -------------------------------------------------------------------------
    # RECONCILIATION & ACCOUNTING INTEGRITY CHECKS
    # -------------------------------------------------------------------------
    print("\n--- ACCOUNTING INTEGRITY VERIFICATION ---")
    sum_gross_pnl = df_trades["gross_pnl_inr"].sum()
    sum_buy_comm = df_trades["buy_commission_inr"].sum()
    sum_sell_comm = df_trades["sell_commission_inr"].sum()
    sum_comm = df_trades["total_commissions_inr"].sum()
    sum_div = df_trades["dividend_credited_inr"].sum()
    sum_net_pnl = df_trades["net_pnl_inr"].sum()
    recomputed_final_equity = STARTING_CAPITAL + sum_net_pnl

    print(f"Gross PnL Sum:        INR {sum_gross_pnl:,.2f}")
    print(f"Total Commissions:    INR {sum_comm:,.2f} (Buy: {sum_buy_comm:,.2f}, Sell: {sum_sell_comm:,.2f})")
    print(f"Dividends Credited:   INR {sum_div:,.2f}")
    print(f"Net PnL Sum:          INR {sum_net_pnl:,.2f}")
    print(f"Recomputed Equity:    INR {recomputed_final_equity:,.2f}")
    print(f"Blotter Final Equity: INR {perf['final_equity']:,.2f}")
    accounting_delta = abs(recomputed_final_equity - perf["final_equity"])
    print(f"Accounting Delta:     INR {accounting_delta:.10f}")
    assert accounting_delta < 0.01, f"Accounting delta detected! {accounting_delta}"
    print(">>> ZERO ACCOUNTING LEAKAGE CONFIRMED.")

    # Save summary metadata JSON
    summary_meta = {
        "baseline_reproduction": {
            "starting_capital": STARTING_CAPITAL,
            "final_equity": perf["final_equity"],
            "total_return_pct": perf["total_return_pct"],
            "max_drawdown_pct": perf["max_drawdown_pct"],
            "total_trades": perf["total_trades"],
            "winning_trades": perf["winning_trades"],
            "losing_trades": perf["losing_trades"],
            "win_rate_pct": perf["win_rate_pct"],
            "profit_factor": perf["profit_factor"],
            "total_commissions_inr": perf["commissions_paid"],
            "total_dividends_inr": perf["dividends_received"],
        },
        "accounting_reconciliation": {
            "sum_gross_pnl": sum_gross_pnl,
            "sum_commissions": sum_comm,
            "sum_dividends": sum_div,
            "sum_net_pnl": sum_net_pnl,
            "recomputed_equity": recomputed_final_equity,
            "delta": accounting_delta,
        },
        "top_losing_stocks": df_stock.head(3).to_dict(orient="records"),
        "regime_distribution": df_regime.to_dict(orient="records"),
    }
    with open(REPORT_DIR / "phase15a_summary_metrics.json", "w") as f:
        json.dump(summary_meta, f, indent=2)

    print("\nPhase 15A Diagnostics Script Completed Successfully!")


if __name__ == "__main__":
    main()
