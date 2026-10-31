"""
Quantara AI — Phase 14: Profitability-Focused Strategy Research Engine
Evaluates:
1. Cash Benchmark (6.5% assumed annual yield)
2. Buy-and-Hold Benchmark (equal weight all 8 stocks)
3. Classical 20-day Cross-Sectional Momentum (Top 2 stocks by 20d return)
4. Classical SMA-50 Trend Following (Long when Close > SMA-50)
5. 5-day Machine Learning Strategy (All eligible signals, max 3 positions)
6. 20-day Machine Learning Strategy (All eligible signals, max 3 positions)
7. 20-day Selective Cross-Sectional Ranking (Top 2 highest conviction stocks by P(UP))
8. 20-day Selective Cross-Sectional Ranking (Top 1 highest conviction stock)
9. 20-day Selective Top 2 + Market-Regime Filter (100% Cash during BEAR_REGIME)
10. 20-day Selective Top 2 + Regime Filter + Volatility-Aware Inverse-Vol Sizing
11. 20-day Selective Top 2 + Regime Filter + 5% Dynamic Stop-Loss Protection
12. Transaction Cost Sensitivity (Base 0.3937%, Elevated 0.50%, High 0.60%)

All models and thresholds tuned strictly on Train and Validation data.
Evaluated on the 248-session Test period (2025-10-23 to 2026-10-23).
Starting Capital: INR 100,000.00.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "phase14_profitability"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = ROOT / "models" / "phase14_profitability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.0
CASH_YIELD_ANNUAL = 0.065

# NSE Statutory Cash Delivery Fee Structure
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
            if idx == 0: continue
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
    
    # 20d v3 split data
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
    initial_val = STARTING_CAPITAL
    final_val = equity_series.iloc[-1]
    n_days = len(equity_series)
    
    total_ret = (final_val - initial_val) / initial_val
    cagr = (final_val / initial_val) ** (252.0 / n_days) - 1.0 if final_val > 0 else -1.0
    
    peaks = equity_series.cummax()
    drawdowns = (peaks - equity_series) / peaks
    max_dd = float(drawdowns.max())
    
    rf_daily = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    excess = daily_returns - rf_daily
    ret_std = float(daily_returns.std())
    sharpe = 0.0 if is_cash or ret_std < 1e-6 else float(np.sqrt(252.0) * (excess.mean() / ret_std))
    
    closed_trades = [t for t in trade_blotter if t["action"] == "SELL"]
    profitable = [t for t in closed_trades if t["net_pnl"] > 0]
    unprofitable = [t for t in closed_trades if t["net_pnl"] < 0]
    
    win_rate = len(profitable) / len(closed_trades) if len(closed_trades) > 0 else 0.0
    avg_win = float(np.mean([t["net_pnl"] for t in profitable])) if len(profitable) > 0 else 0.0
    avg_loss = float(np.mean([abs(t["net_pnl"]) for t in unprofitable])) if len(unprofitable) > 0 else 0.0
    
    gross_gains = sum(t["net_pnl"] for t in profitable)
    gross_losses = sum(abs(t["net_pnl"]) for t in unprofitable)
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else (999.0 if gross_gains > 0 else 0.0)
    
    tot_comm = sum(t["commission"] for t in trade_blotter)
    tot_div = sum(t["net_pnl"] for t in trade_blotter if t["action"] == "DIVIDEND")
    
    # Turnover calculation (total buy turnover / starting capital)
    total_turnover = sum(t["shares"] * t["exec_price"] for t in trade_blotter if t["action"] == "BUY") / initial_val
    
    return {
        "final_equity": float(final_val),
        "total_return_pct": float(total_ret * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "annualized_volatility_pct": float(ret_std * np.sqrt(252.0) * 100.0),
        "sharpe_ratio": float(sharpe),
        "total_trades": len(closed_trades),
        "win_rate_pct": float(win_rate * 100.0),
        "avg_winning_trade": float(avg_win),
        "avg_losing_trade": float(avg_loss),
        "profit_factor": float(profit_factor),
        "commissions_paid": float(tot_comm),
        "dividends_received": float(tot_div),
        "turnover_multiple": float(total_turnover),
    }


def simulate_strategy_engine(
    strategy_name,
    signals_by_date,
    market,
    max_positions=3,
    max_hold_sessions=20,
    regime_filter=False,
    vol_sizing=False,
    stop_loss_pct=None,
    fee_sched=BASE_FEE_SCHEDULE,
    vol_dict=None
):
    """
    Generalized event-driven execution simulator.
    signals_by_date: dict of date -> list of (symbol, score) sorted by conviction
    """
    dates = market["all_trading_dates"]
    opens = market["opens"]
    closes = market["closes"]
    div_lookup = market["dividend_lookup"]
    buy_fee, sell_fee, slippage = get_fee_rates(fee_sched)
    
    cash = STARTING_CAPITAL
    open_positions = {}
    trade_blotter = []
    daily_records = []
    pending_exits = set()
    
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
                    trade_blotter.append({"date": current_date.strftime("%Y-%m-%d"), "symbol": sym, "action": "DIVIDEND", "shares": pos["shares"], "exec_price": float(div_lookup[div_key]), "commission": 0.0, "net_pnl": div_payout, "cash_after": float(cash)})
                
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
                del open_positions[sym]
        pending_exits.clear()
        
        # 2. EXECUTE CANDIDATE BUYS AT OPEN
        candidate_buys = signals_by_date.get(current_date, [])
        for sym, score in candidate_buys:
            if len(open_positions) < max_positions and sym not in open_positions:
                avail_slots = max_positions - len(open_positions)
                
                # Sizing method
                if vol_sizing and vol_dict is not None and (current_date, sym) in vol_dict:
                    sym_vol = vol_dict[(current_date, sym)]
                    # Volatility-weighted sizing: inversely proportional to rolling 20d vol
                    # baseline 20% vol -> factor 1.0
                    vol_factor = np.clip(0.20 / max(sym_vol, 0.05), 0.5, 1.5)
                    slot_budget = (cash / avail_slots) * vol_factor
                    slot_budget = min(slot_budget, cash)
                else:
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
                    trade_blotter.append({"date": current_date.strftime("%Y-%m-%d"), "symbol": sym, "action": "DIVIDEND", "shares": pos["shares"], "exec_price": float(div_lookup[div_key]), "commission": 0.0, "net_pnl": div_payout, "cash_after": float(cash)})
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
                trade_blotter.append({"date": current_date.strftime("%Y-%m-%d"), "symbol": sym, "action": "SELL", "shares": pos["shares"], "exec_price": float(exec_price), "commission": fee, "net_pnl": gross_proceeds - fee - pos["shares"] * pos["entry_price"] * (1.0 + buy_fee), "sessions_held": pos["sessions_held"], "cash_after": float(cash)})
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
                cur_close = closes.loc[current_date, sym]
                unrealized_ret = (cur_close - pos["entry_price"]) / pos["entry_price"]
                
                # Stop loss check
                if stop_loss_pct is not None and unrealized_ret <= -stop_loss_pct:
                    pos["exit_reason"] = f"STOP_LOSS_{stop_loss_pct*100:.0f}PCT"
                    pending_exits.add(sym)
                elif pos["sessions_held"] >= max_hold_sessions:
                    pos["exit_reason"] = f"MAX_HOLD_{max_hold_sessions}D"
                    pending_exits.add(sym)
                    
    df_eq = pd.DataFrame(daily_records)
    df_eq["daily_return"] = df_eq["total_equity"].pct_change().fillna(0.0)
    perf = compute_comprehensive_portfolio_metrics(df_eq["total_equity"], df_eq["daily_return"], trade_blotter)
    return df_eq, trade_blotter, perf


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 14: PROFITABILITY-FOCUSED RESEARCH SUITE")
    print("=" * 80)

    market = load_all_market_data()
    all_dates = market["all_trading_dates"]
    signal_dates = market["test_signal_dates"]
    print(f"Loaded {len(all_dates)} trading sessions across {len(market['symbols'])} stocks.")

    # Volatility lookup for inverse-vol sizing
    raw_sorted = market["raw"].sort_values(["Symbol", "Date"]).copy().reset_index(drop=True)
    raw_sorted["ret_1d"] = raw_sorted.groupby("Symbol")["Adj Close"].pct_change()
    raw_sorted["vol_20d"] = raw_sorted.groupby("Symbol")["ret_1d"].transform(lambda x: x.rolling(20, min_periods=20).std() * np.sqrt(252))
    vol_dict = {(row["Date"], row["Symbol"]): float(row["vol_20d"]) for _, row in raw_sorted.dropna().iterrows()}

    # Market Regime lookup
    raw_sorted["sma20"] = raw_sorted.groupby("Symbol")["Adj Close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    raw_sorted["above_sma20"] = (raw_sorted["Adj Close"] > raw_sorted["sma20"]).astype(float)
    market_breadth = raw_sorted.groupby("Date")["above_sma20"].mean().to_dict()

    # Load Trained Models
    # Horizon 20d ML Models (from v3 pipeline)
    X_train_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "X_train.csv")
    y_train_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "y_train.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_val_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "X_validation.csv")
    y_val_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "y_validation.csv")["Target_20d"].map({"DOWN": 0, "HOLD": 1, "UP": 2}).values
    X_test_20d = pd.read_csv(DATA_DIR / "splits_v3" / "horizon_20d" / "X_test.csv")
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
    # Map (signal_date, symbol) -> P(UP)
    # Signal generated at close of day t executes at open of day t+1
    ml_signals_by_exec_date = {}
    for idx, row in meta_test_20d.iterrows():
        sig_d = row["Date"]
        # Find next trading date
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            p_up = float(probs_lr_test[idx, 2])
            p_down = float(probs_lr_test[idx, 0])
            if exec_d not in ml_signals_by_exec_date:
                ml_signals_by_exec_date[exec_d] = []
            ml_signals_by_exec_date[exec_d].append((row["Symbol"], p_up, p_down, sig_d))

    # Classical Momentum 20d Signals
    raw_sorted["ret_20d"] = raw_sorted.groupby("Symbol")["Adj Close"].pct_change(20)
    mom_dict = {(row["Date"], row["Symbol"]): float(row["ret_20d"]) for _, row in raw_sorted.dropna().iterrows()}
    mom_signals_by_exec_date = {}
    for sig_d in signal_dates:
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            cands = []
            for s in market["symbols"]:
                if (sig_d, s) in mom_dict:
                    cands.append((s, mom_dict[(sig_d, s)]))
            cands.sort(key=lambda x: x[1], reverse=True)
            mom_signals_by_exec_date[exec_d] = cands

    # Classical SMA50 Trend Signals
    raw_sorted["sma50"] = raw_sorted.groupby("Symbol")["Adj Close"].transform(lambda x: x.rolling(50, min_periods=50).mean())
    trend_dict = {(row["Date"], row["Symbol"]): (row["Adj Close"] > row["sma50"]) for _, row in raw_sorted.dropna().iterrows()}
    trend_signals_by_exec_date = {}
    for sig_d in signal_dates:
        cur_idx = all_dates.index(sig_d)
        if cur_idx < len(all_dates) - 1:
            exec_d = all_dates[cur_idx + 1]
            cands = [s for s in market["symbols"] if trend_dict.get((sig_d, s), False)]
            trend_signals_by_exec_date[exec_d] = [(s, 1.0) for s in cands]

    # Build Signal Collections for different strategy variants
    # 1. 20d ML All Eligible (threshold P(UP) >= 0.35, P(UP) > P(DOWN))
    signals_ml_all = {}
    for exec_d, cands in ml_signals_by_exec_date.items():
        filt = [(sym, p_up) for sym, p_up, p_down, _ in cands if p_up >= 0.35 and p_up > p_down]
        filt.sort(key=lambda x: x[1], reverse=True)
        signals_ml_all[exec_d] = filt

    # 2. 20d ML Selective Top 2 (rank by P(UP), take at most top 2)
    signals_ml_top2 = {}
    for exec_d, cands in ml_signals_by_exec_date.items():
        filt = [(sym, p_up) for sym, p_up, p_down, _ in cands if p_up >= 0.35 and p_up > p_down]
        filt.sort(key=lambda x: x[1], reverse=True)
        signals_ml_top2[exec_d] = filt[:2]

    # 3. 20d ML Selective Top 1 (only highest conviction)
    signals_ml_top1 = {}
    for exec_d, cands in ml_signals_by_exec_date.items():
        filt = [(sym, p_up) for sym, p_up, p_down, _ in cands if p_up >= 0.35 and p_up > p_down]
        filt.sort(key=lambda x: x[1], reverse=True)
        signals_ml_top1[exec_d] = filt[:1]

    # 4. 20d ML Selective Top 2 + Regime Filter (100% Cash when market breadth <= 30%)
    signals_ml_top2_regime = {}
    for exec_d, cands in ml_signals_by_exec_date.items():
        sig_d = cands[0][3] if len(cands) > 0 else None
        breadth = market_breadth.get(sig_d, 0.5) if sig_d else 0.5
        if breadth <= 0.30:  # BEAR REGIME -> REFUSE ALL BUY SIGNALS (STAY IN CASH)
            signals_ml_top2_regime[exec_d] = []
        else:
            filt = [(sym, p_up) for sym, p_up, p_down, _ in cands if p_up >= 0.35 and p_up > p_down]
            filt.sort(key=lambda x: x[1], reverse=True)
            signals_ml_top2_regime[exec_d] = filt[:2]

    # -------------------------------------------------------------------------
    # RUN STRATEGY SUITE
    # -------------------------------------------------------------------------
    all_experiments = []

    # Strategy 1: Cash Benchmark
    daily_rate = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    cash_curve = [STARTING_CAPITAL * ((1.0 + daily_rate) ** d_idx) for d_idx in range(len(all_dates))]
    df_cash = pd.DataFrame({"total_equity": cash_curve})
    df_cash["daily_return"] = df_cash["total_equity"].pct_change().fillna(daily_rate)
    perf_cash = compute_comprehensive_portfolio_metrics(df_cash["total_equity"], df_cash["daily_return"], [], is_cash=True)
    all_experiments.append({"name": "1. Cash Benchmark (6.5% Yield)", "perf": perf_cash, "curve": df_cash["total_equity"]})

    # Strategy 2: Buy & Hold Benchmark
    bh_metrics = json.load(open(ROOT / "models" / "horizon_5d_v2" / "portfolio_backtest_metrics.json"))["buy_and_hold_benchmark"]
    all_experiments.append({"name": "2. Buy & Hold Benchmark", "perf": bh_metrics, "curve": None})

    # Strategy 3: Classical 20d Momentum (Top 2)
    df_eq_mom, blot_mom, perf_mom = simulate_strategy_engine(
        "Classical_Momentum_20d", {k: v[:2] for k, v in mom_signals_by_exec_date.items()}, market, max_positions=2, max_hold_sessions=20
    )
    all_experiments.append({"name": "3. Classical 20d Momentum (Top 2)", "perf": perf_mom, "curve": df_eq_mom["total_equity"]})

    # Strategy 4: Classical SMA-50 Trend Following
    df_eq_trend, blot_trend, perf_trend = simulate_strategy_engine(
        "Classical_SMA50_Trend", trend_signals_by_exec_date, market, max_positions=3, max_hold_sessions=20
    )
    all_experiments.append({"name": "4. Classical SMA-50 Trend", "perf": perf_trend, "curve": df_eq_trend["total_equity"]})

    # Strategy 5: 20d ML All Eligible (Baseline)
    df_eq_ml_all, blot_ml_all, perf_ml_all = simulate_strategy_engine(
        "20d_ML_All_Eligible", signals_ml_all, market, max_positions=3, max_hold_sessions=20
    )
    all_experiments.append({"name": "5. 20d ML All Eligible (Base)", "perf": perf_ml_all, "curve": df_eq_ml_all["total_equity"]})

    # Strategy 6: 20d ML Selective Top 2 (Cross-Sectional Ranking)
    df_eq_top2, blot_top2, perf_top2 = simulate_strategy_engine(
        "20d_ML_Selective_Top2", signals_ml_top2, market, max_positions=2, max_hold_sessions=20
    )
    all_experiments.append({"name": "6. 20d ML Selective Top 2", "perf": perf_top2, "curve": df_eq_top2["total_equity"]})

    # Strategy 7: 20d ML Selective Top 1 (Highest Conviction Only)
    df_eq_top1, blot_top1, perf_top1 = simulate_strategy_engine(
        "20d_ML_Selective_Top1", signals_ml_top1, market, max_positions=1, max_hold_sessions=20
    )
    all_experiments.append({"name": "7. 20d ML Selective Top 1", "perf": perf_top1, "curve": df_eq_top1["total_equity"]})

    # Strategy 8: 20d ML Selective Top 2 + Market-Regime Filter (Cash in Bear Market)
    df_eq_reg, blot_reg, perf_reg = simulate_strategy_engine(
        "20d_ML_Top2_RegimeFilter", signals_ml_top2_regime, market, max_positions=2, max_hold_sessions=20
    )
    all_experiments.append({"name": "8. 20d ML Top 2 + Regime Filter", "perf": perf_reg, "curve": df_eq_reg["total_equity"]})

    # Strategy 9: 20d ML Selective Top 2 + Regime Filter + Volatility Sizing
    df_eq_vol, blot_vol, perf_vol = simulate_strategy_engine(
        "20d_ML_Top2_Regime_VolSized", signals_ml_top2_regime, market, max_positions=2, max_hold_sessions=20, vol_sizing=True, vol_dict=vol_dict
    )
    all_experiments.append({"name": "9. 20d ML Top 2 + Regime + Vol-Size", "perf": perf_vol, "curve": df_eq_vol["total_equity"]})

    # Strategy 10: 20d ML Selective Top 2 + Regime Filter + 5% Stop-Loss
    df_eq_sl, blot_sl, perf_sl = simulate_strategy_engine(
        "20d_ML_Top2_Regime_StopLoss5pct", signals_ml_top2_regime, market, max_positions=2, max_hold_sessions=20, stop_loss_pct=0.05
    )
    all_experiments.append({"name": "10. 20d ML Top 2 + Regime + 5% Stop", "perf": perf_sl, "curve": df_eq_sl["total_equity"]})

    # -------------------------------------------------------------------------
    # COST SENSITIVITY TESTING (Strategy 8: Top 2 + Regime Filter)
    # -------------------------------------------------------------------------
    cost_sensitivity_results = []
    for friction_label, f_mult in [("Base (0.39%)", 1.0), ("Elevated (0.50%)", 1.27), ("High (0.60%)", 1.52)]:
        sched = {k: v * f_mult for k, v in BASE_FEE_SCHEDULE.items()}
        sched["gst_rate"] = 0.18
        _, _, p_sens = simulate_strategy_engine(
            f"Sens_{friction_label}", signals_ml_top2_regime, market, max_positions=2, max_hold_sessions=20, fee_sched=sched
        )
        cost_sensitivity_results.append({
            "Cost Scenario": friction_label,
            "Final Equity": f"INR {p_sens['final_equity']:,.2f}",
            "Total Return": f"{p_sens['total_return_pct']:+.2f}%",
            "Max Drawdown": f"{p_sens['max_drawdown_pct']:.2f}%",
            "Commissions Paid": f"INR {p_sens['commissions_paid']:,.2f}",
        })

    # -------------------------------------------------------------------------
    # FORMATTING & SERIALIZATION
    # -------------------------------------------------------------------------
    table_rows = []
    for exp in all_experiments:
        p = exp["perf"]
        table_rows.append({
            "Strategy Name": exp["name"],
            "Final Equity": f"INR {p.get('final_equity', 0):,.2f}",
            "Return": f"{p.get('total_return_pct', 0):+.2f}%",
            "CAGR": f"{p.get('cagr_pct', 0):+.2f}%",
            "Max DD": f"{p.get('max_drawdown_pct', 0):.2f}%",
            "Sharpe": f"{p.get('sharpe_ratio', 0):.2f}",
            "Trades": p.get("total_trades", 0),
            "Win Rate": f"{p.get('win_rate_pct', 0):.1f}%",
            "Profit Factor": f"{p.get('profit_factor', 0):.2f}",
            "Commissions": f"INR {p.get('commissions_paid', 0):,.2f}",
            "Turnover": f"{p.get('turnover_multiple', 0):.2f}x",
        })

    df_summary = pd.DataFrame(table_rows)
    print("\n" + "=" * 90)
    print("PHASE 14 PROFITABILITY RESEARCH COMPARISON TABLE (INR 100,000 BASE)")
    print("=" * 90)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 90)
    print("COST SENSITIVITY TABLE (20d ML Top 2 + Regime Filter)")
    print("=" * 90)
    print(pd.DataFrame(cost_sensitivity_results).to_string(index=False))

    # Save artifacts
    df_curves = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in all_dates],
        "Cash_Benchmark": df_cash["total_equity"],
        "Momentum_20d": df_eq_mom["total_equity"],
        "SMA50_Trend": df_eq_trend["total_equity"],
        "ML_20d_Base": df_eq_ml_all["total_equity"],
        "ML_20d_Top2": df_eq_top2["total_equity"],
        "ML_20d_Top2_Regime": df_eq_reg["total_equity"],
        "ML_20d_Top2_Regime_VolSized": df_eq_vol["total_equity"],
        "ML_20d_Top2_Regime_StopLoss": df_eq_sl["total_equity"],
    })
    df_curves.to_csv(OUTPUT_DIR / "phase14_equity_curves.csv", index=False)

    pd.DataFrame(blot_reg).to_csv(OUTPUT_DIR / "blotter_top2_regime_filtered.csv", index=False)
    pd.DataFrame(blot_top2).to_csv(OUTPUT_DIR / "blotter_top2_selective.csv", index=False)

    master_phase14 = {
        "summary_table": table_rows,
        "cost_sensitivity": cost_sensitivity_results,
        "individual_performances": {exp["name"]: exp["perf"] for exp in all_experiments}
    }
    with open(REPORT_DIR / "phase14_profitability_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_phase14, f, indent=2)

    print(f"\nAll Phase 14 research artifacts saved to:\n  {OUTPUT_DIR}\n  {REPORT_DIR}")


if __name__ == "__main__":
    main()

