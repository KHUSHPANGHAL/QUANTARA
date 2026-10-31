"""
Quantara AI — Phase 9: Multi-Horizon & Regime-Filtered Portfolio Backtest Engine
Compares:
1. Horizon 5d Active Strategy (120-190 trades, high turnover drag)
2. Horizon 20d Active Strategy (Hold 20 sessions; ~75% lower turnover)
3. Horizon 20d + Regime Filter (Mandates 100% Cash during BEAR_REGIME)
4. Passive Buy-and-Hold Benchmark
5. Cash Benchmark (6.50% Assumed Annual Yield)
All starting from INR 100,000.00 with realistic statutory NSE delivery fees (0.3937% RT).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "accuracy_research"
OUTPUT_DIR = ROOT / "models" / "horizon_20d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.0
MAX_POSITIONS = 3
CASH_YIELD_ANNUAL = 0.065

FEE_SCHEDULE = {
    "brokerage_pct": 0.00030,
    "stt_buy_pct": 0.00100,
    "stt_sell_pct": 0.00100,
    "exchange_turnover_pct": 0.0000325,
    "sebi_turnover_pct": 0.000001,
    "stamp_duty_buy_pct": 0.00015,
    "gst_rate": 0.18,
    "slippage_pct": 0.00050,
}

_gst_base_buy = FEE_SCHEDULE["brokerage_pct"] + FEE_SCHEDULE["exchange_turnover_pct"] + FEE_SCHEDULE["sebi_turnover_pct"]
_gst_buy = FEE_SCHEDULE["gst_rate"] * _gst_base_buy
BUY_STATUTORY_FEE = (
    FEE_SCHEDULE["brokerage_pct"]
    + FEE_SCHEDULE["stt_buy_pct"]
    + FEE_SCHEDULE["exchange_turnover_pct"]
    + FEE_SCHEDULE["sebi_turnover_pct"]
    + FEE_SCHEDULE["stamp_duty_buy_pct"]
    + _gst_buy
)

_gst_base_sell = FEE_SCHEDULE["brokerage_pct"] + FEE_SCHEDULE["exchange_turnover_pct"] + FEE_SCHEDULE["sebi_turnover_pct"]
_gst_sell = FEE_SCHEDULE["gst_rate"] * _gst_base_sell
SELL_STATUTORY_FEE = (
    FEE_SCHEDULE["brokerage_pct"]
    + FEE_SCHEDULE["stt_sell_pct"]
    + FEE_SCHEDULE["exchange_turnover_pct"]
    + FEE_SCHEDULE["sebi_turnover_pct"]
    + _gst_sell
)
SLIPPAGE = FEE_SCHEDULE["slippage_pct"]


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
            div_amt = div_yield * prev_row["Close"]
            div_records.append({
                "Symbol": sym,
                "Ex_Date": r["Date"],
                "div_per_share": round(float(div_amt), 4),
            })
    return pd.DataFrame(div_records)


def load_market_environment(horizon=20):
    raw = pd.read_csv(DATA_DIR / "raw" / "multi_stock_historical.csv")
    raw["Date"] = pd.to_datetime(raw["Date"])
    
    meta_test = pd.read_csv(DATA_DIR / "splits_v3" / f"horizon_{horizon}d" / "metadata_test.csv")
    meta_test["Date"] = pd.to_datetime(meta_test["Date"])
    X_test = pd.read_csv(DATA_DIR / "splits_v3" / f"horizon_{horizon}d" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "splits_v3" / f"horizon_{horizon}d" / "y_test.csv")
    
    start_date = meta_test["Date"].min()
    raw_test = raw[raw["Date"] >= start_date].sort_values(["Date", "Symbol"]).reset_index(drop=True)
    all_trading_dates = sorted(raw_test["Date"].unique())
    
    open_prices = raw_test.pivot(index="Date", columns="Symbol", values="Open")
    close_prices = raw_test.pivot(index="Date", columns="Symbol", values="Close")
    
    div_df = extract_dividends(raw)
    test_divs = div_df[(div_df["Ex_Date"] >= start_date) & (div_df["Ex_Date"] <= all_trading_dates[-1])].copy()
    dividend_lookup = {(row["Ex_Date"], row["Symbol"]): row["div_per_share"] for _, row in test_divs.iterrows()}
    
    return {
        "all_trading_dates": all_trading_dates,
        "test_signal_dates": sorted(meta_test["Date"].unique()),
        "open_prices": open_prices,
        "close_prices": close_prices,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
        "symbols": sorted(raw["Symbol"].unique()),
        "dividend_lookup": dividend_lookup,
    }


def compute_metrics(equity_series, daily_returns, trade_blotter, is_cash=False):
    initial_val = STARTING_CAPITAL
    final_val = equity_series.iloc[-1]
    n_days = len(equity_series)
    
    tot_ret = (final_val - initial_val) / initial_val
    cagr = (final_val / initial_val) ** (252.0 / n_days) - 1.0 if final_val > 0 else -1.0
    
    peaks = equity_series.cummax()
    max_dd = float(((peaks - equity_series) / peaks).max())
    
    rf_daily = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    excess = daily_returns - rf_daily
    ret_std = float(daily_returns.std())
    sharpe = 0.0 if is_cash or ret_std < 1e-6 else float(np.sqrt(252.0) * (excess.mean() / ret_std))
    
    closed_trades = [t for t in trade_blotter if t["action"] == "SELL"]
    profitable = [t for t in closed_trades if t["net_pnl"] > 0]
    unprofitable = [t for t in closed_trades if t["net_pnl"] < 0]
    win_rate = len(profitable) / len(closed_trades) if len(closed_trades) > 0 else 0.0
    
    gross_gains = sum(t["net_pnl"] for t in profitable)
    gross_losses = sum(abs(t["net_pnl"]) for t in unprofitable)
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else (999.0 if gross_gains > 0 else 0.0)
    
    tot_comm = sum(t["commission"] for t in trade_blotter)
    tot_div = sum(t["net_pnl"] for t in trade_blotter if t["action"] == "DIVIDEND")
    
    return {
        "final_equity": float(final_val),
        "total_return_pct": float(tot_ret * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "annualized_volatility_pct": float(ret_std * np.sqrt(252.0) * 100.0),
        "sharpe_ratio": float(sharpe),
        "total_trades": len(closed_trades),
        "win_rate_pct": float(win_rate * 100.0),
        "profit_factor": float(profit_factor),
        "commissions_paid": float(tot_comm),
        "dividends_received": float(tot_div),
    }


def run_horizon_portfolio(model_name, model, market, max_hold_sessions=20, p_up_threshold=0.35, use_regime_filter=False, scaler=None):
    dates = market["all_trading_dates"]
    signal_dates = set(market["test_signal_dates"])
    opens = market["open_prices"]
    closes = market["close_prices"]
    div_lookup = market["dividend_lookup"]
    
    meta = market["meta_test"]
    X = market["X_test"]
    X_eval = scaler.transform(X) if scaler is not None else X
    
    probs = model.predict_proba(X_eval)
    preds = np.argmax(probs, axis=1)
    
    daily_signals = {}
    for idx, row in meta.iterrows():
        d = row["Date"]
        sym = row["Symbol"]
        reg = row.get("market_regime", "NEUTRAL_REGIME")
        if d not in daily_signals:
            daily_signals[d] = {"regime": reg, "symbols": {}}
        daily_signals[d]["symbols"][sym] = {
            "pred": int(preds[idx]),
            "p_up": float(probs[idx, 2]),
            "p_down": float(probs[idx, 0]),
        }
        
    cash = STARTING_CAPITAL
    open_positions = {}
    pending_buys = []
    pending_exits = set()
    trade_blotter = []
    daily_records = []
    
    for i, current_date in enumerate(dates):
        # 1. MORNING EXITS
        for sym in list(pending_exits):
            if sym in open_positions:
                pos = open_positions[sym]
                # Dividend entitlement check
                div_key = (current_date, sym)
                if div_key in div_lookup and pos["entry_date"] < current_date:
                    credited = pos.get("dividend_credited_dates", set())
                    if current_date not in credited:
                        div_per_share = div_lookup[div_key]
                        div_payout = pos["shares"] * div_per_share
                        cash += div_payout
                        trade_blotter.append({
                            "date": current_date.strftime("%Y-%m-%d"),
                            "symbol": sym,
                            "action": "DIVIDEND",
                            "shares": pos["shares"],
                            "exec_price": float(div_per_share),
                            "commission": 0.0,
                            "net_pnl": float(div_payout),
                            "cash_after": float(cash),
                        })
                        pos["dividend_credited_dates"] = credited | {current_date}
                        
                raw_open = opens.loc[current_date, sym]
                exec_price = raw_open * (1.0 - SLIPPAGE)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * SELL_STATUTORY_FEE
                net_proceeds = gross_proceeds - fee
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + BUY_STATUTORY_FEE)
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
                del open_positions[sym]
        pending_exits.clear()
        
        # 2. MORNING BUYS
        pending_buys.sort(key=lambda x: x[1], reverse=True)
        for sym, p_up in pending_buys:
            if len(open_positions) < MAX_POSITIONS and sym not in open_positions:
                avail_slots = MAX_POSITIONS - len(open_positions)
                slot_budget = cash / avail_slots
                raw_open = opens.loc[current_date, sym]
                exec_price = raw_open * (1.0 + SLIPPAGE)
                cost_per_share = exec_price * (1.0 + BUY_STATUTORY_FEE)
                shares = int(np.floor(slot_budget / cost_per_share))
                if shares > 0:
                    gross_cost = shares * exec_price
                    fee = gross_cost * BUY_STATUTORY_FEE
                    total_outflow = gross_cost + fee
                    if cash >= total_outflow:
                        cash -= total_outflow
                        open_positions[sym] = {
                            "shares": shares,
                            "entry_price": float(exec_price),
                            "entry_date": current_date,
                            "sessions_held": 0,
                            "exit_reason": "SCHEDULED_EXIT",
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
        pending_buys.clear()
        
        # 3. AGE POSITIONS & DIVIDENDS
        for sym in open_positions:
            open_positions[sym]["sessions_held"] += 1
            
        for sym, pos in list(open_positions.items()):
            div_key = (current_date, sym)
            if div_key in div_lookup and pos["entry_date"] < current_date:
                credited = pos.get("dividend_credited_dates", set())
                if current_date not in credited:
                    div_per_share = div_lookup[div_key]
                    div_payout = pos["shares"] * div_per_share
                    cash += div_payout
                    trade_blotter.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": float(div_per_share),
                        "commission": 0.0,
                        "net_pnl": float(div_payout),
                        "cash_after": float(cash),
                    })
                    pos["dividend_credited_dates"] = credited | {current_date}
                    
        # 4. MARK-TO-MARKET & FINAL LIQUIDATION
        val = sum(pos["shares"] * closes.loc[current_date, sym] for sym, pos in open_positions.items())
        total_equity = cash + val
        
        if i == len(dates) - 1:
            for sym, pos in list(open_positions.items()):
                raw_close = closes.loc[current_date, sym]
                exec_price = raw_close * (1.0 - SLIPPAGE)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * SELL_STATUTORY_FEE
                net_proceeds = gross_proceeds - fee
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + BUY_STATUTORY_FEE)
                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_proceeds - total_cost),
                    "reason": "END_OF_TEST_LIQUIDATION",
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
            open_positions.clear()
            val = 0.0
            total_equity = cash
            
        daily_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "cash": float(cash),
            "positions_value": float(val),
            "total_equity": float(total_equity),
            "open_positions": len(open_positions),
        })
        
        # 5. SIGNAL PLANNING FOR NEXT DAY
        if i < len(dates) - 1:
            has_signals = current_date in signal_dates
            sig_info = daily_signals.get(current_date, {})
            current_regime = sig_info.get("regime", "NEUTRAL_REGIME")
            sym_sigs = sig_info.get("symbols", {})
            
            # Check exits
            for sym, pos in open_positions.items():
                if has_signals and sym in sym_sigs and sym_sigs[sym]["pred"] == 0:
                    pos["exit_reason"] = "DOWN_RISK_EXIT"
                    pending_exits.add(sym)
                elif pos["sessions_held"] >= max_hold_sessions:
                    pos["exit_reason"] = f"MAX_HOLD_{max_hold_sessions}D_EXIT"
                    pending_exits.add(sym)
                    
            # Check candidate buys
            if has_signals:
                # If regime filter is active: refuse new buys if in BEAR_REGIME!
                regime_permits_buying = True
                if use_regime_filter and current_regime == "BEAR_REGIME":
                    regime_permits_buying = False
                    
                if regime_permits_buying:
                    for sym, s_data in sym_sigs.items():
                        if s_data["p_up"] >= p_up_threshold and s_data["p_up"] > s_data["p_down"]:
                            if sym not in open_positions and sym not in pending_exits:
                                pending_buys.append((sym, s_data["p_up"]))
                                
    df_eq = pd.DataFrame(daily_records)
    df_eq["daily_return"] = df_eq["total_equity"].pct_change().fillna(0.0)
    perf = compute_metrics(df_eq["total_equity"], df_eq["daily_return"], trade_blotter)
    return df_eq, trade_blotter, perf


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 9: MULTI-HORIZON & REGIME PORTFOLIO BACKTEST ENGINE")
    print("=" * 80)

    # 1. Load Market for 20-Day Horizon
    market_20d = load_market_environment(horizon=20)
    
    # Train Logistic Regression on Horizon 20d (Group Full Regime)
    X_train_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "X_train.csv")
    y_train_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    
    scaler_20d = StandardScaler()
    scaler_20d.fit(X_train_20d)
    X_train_20d_sc = scaler_20d.transform(X_train_20d)
    
    lr_20d = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_20d.fit(X_train_20d_sc, y_train_20d)
    
    # Train LightGBM on Horizon 20d
    lgb_20d = lgb.LGBMClassifier(num_leaves=31, max_depth=6, learning_rate=0.03, class_weight="balanced", n_estimators=300, random_state=42, verbose=-1, n_jobs=-1)
    lgb_20d.fit(X_train_20d, y_train_20d)
    
    # Portfolio 1: Horizon 20d - Logistic Regression (Unfiltered)
    print("\n[Strategy 1/4] Running Horizon 20d Active Strategy (Unfiltered)...")
    df_eq_20d, blotter_20d, perf_20d = run_horizon_portfolio(
        "LR_20d", lr_20d, market_20d, max_hold_sessions=20, p_up_threshold=0.35, use_regime_filter=False, scaler=scaler_20d
    )
    
    # Portfolio 2: Horizon 20d - LightGBM with Regime Filter (Cash in Bear Regime)
    print("\n[Strategy 2/4] Running Horizon 20d Strategy + Macro Regime Filter...")
    df_eq_reg, blotter_reg, perf_reg = run_horizon_portfolio(
        "LGB_20d_Regime", lgb_20d, market_20d, max_hold_sessions=20, p_up_threshold=0.333, use_regime_filter=True
    )
    
    # Compare with Horizon 5d Baseline from Phase 3.4
    h5_metrics = json.load(open(ROOT / "models" / "horizon_5d_v2" / "portfolio_backtest_metrics.json"))
    perf_5d = h5_metrics["logistic_regression"]
    perf_bh = h5_metrics["buy_and_hold_benchmark"]
    perf_cash = h5_metrics["cash_benchmark"]
    
    # Summary Table
    comparison = [
        {
            "Strategy / Horizon": "Horizon 5d Active (LR Baseline)",
            "Final Equity": f"INR {perf_5d['final_equity']:,.2f}",
            "Return": f"{perf_5d['total_return_pct']:+.2f}%",
            "CAGR": f"{perf_5d['cagr_pct']:+.2f}%",
            "Max DD": f"{perf_5d['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{perf_5d['sharpe_ratio']:.2f}",
            "Trades": perf_5d["total_trades_completed"],
            "Commissions": f"INR {perf_5d['total_commissions_paid']:,.2f}",
        },
        {
            "Strategy / Horizon": "Horizon 20d Active (Unfiltered)",
            "Final Equity": f"INR {perf_20d['final_equity']:,.2f}",
            "Return": f"{perf_20d['total_return_pct']:+.2f}%",
            "CAGR": f"{perf_20d['cagr_pct']:+.2f}%",
            "Max DD": f"{perf_20d['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{perf_20d['sharpe_ratio']:.2f}",
            "Trades": perf_20d["total_trades"],
            "Commissions": f"INR {perf_20d['commissions_paid']:,.2f}",
        },
        {
            "Strategy / Horizon": "Horizon 20d Active + Regime Filter",
            "Final Equity": f"INR {perf_reg['final_equity']:,.2f}",
            "Return": f"{perf_reg['total_return_pct']:+.2f}%",
            "CAGR": f"{perf_reg['cagr_pct']:+.2f}%",
            "Max DD": f"{perf_reg['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{perf_reg['sharpe_ratio']:.2f}",
            "Trades": perf_reg["total_trades"],
            "Commissions": f"INR {perf_reg['commissions_paid']:,.2f}",
        },
        {
            "Strategy / Horizon": "Buy & Hold Benchmark",
            "Final Equity": f"INR {perf_bh['final_equity']:,.2f}",
            "Return": f"{perf_bh['total_return_pct']:+.2f}%",
            "CAGR": f"{perf_bh['cagr_pct']:+.2f}%",
            "Max DD": f"{perf_bh['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{perf_bh['sharpe_ratio']:.2f}",
            "Trades": perf_bh["total_trades_completed"],
            "Commissions": f"INR {perf_bh['total_commissions_paid']:,.2f}",
        },
        {
            "Strategy / Horizon": "Cash Benchmark (6.5% Yield)",
            "Final Equity": f"INR {perf_cash['final_equity']:,.2f}",
            "Return": f"{perf_cash['total_return_pct']:+.2f}%",
            "CAGR": f"{perf_cash['cagr_pct']:+.2f}%",
            "Max DD": f"{perf_cash['max_drawdown_pct']:.2f}%",
            "Sharpe": "0.00",
            "Trades": 0,
            "Commissions": "INR 0.00",
        },
    ]
    
    df_summary = pd.DataFrame(comparison)
    print("\n" + "=" * 80)
    print("CROSS-HORIZON & REGIME FILTER ECONOMIC IMPACT TABLE (INR 100,000 BASE)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    
    # Save artifacts
    pd.DataFrame(blotter_20d).to_csv(OUTPUT_DIR / "trade_blotter_20d_unfiltered.csv", index=False)
    pd.DataFrame(blotter_reg).to_csv(OUTPUT_DIR / "trade_blotter_20d_regime_filtered.csv", index=False)
    df_eq_20d.to_csv(OUTPUT_DIR / "equity_curve_20d_unfiltered.csv", index=False)
    df_eq_reg.to_csv(OUTPUT_DIR / "equity_curve_20d_regime_filtered.csv", index=False)
    
    full_backtest_report = {
        "horizon_5d_baseline": perf_5d,
        "horizon_20d_unfiltered": perf_20d,
        "horizon_20d_regime_filtered": perf_reg,
        "buy_and_hold_benchmark": perf_bh,
        "cash_benchmark": perf_cash,
        "comparison_table": comparison
    }
    with open(OUTPUT_DIR / "multi_horizon_backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_backtest_report, f, indent=2)


if __name__ == "__main__":
    main()
