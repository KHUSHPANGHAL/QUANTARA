"""
Quantara AI — Phase 3.4: Cash-Equity Portfolio Backtest Engine (v2 Pipeline)
Execution-Aligned Target & Frozen Validation Decision Thresholds

Key Features of v2 Backtest:
1. Data Pipeline: Uses data/splits_v2/horizon_5d/ and models/horizon_5d_v2/.
2. Decision Thresholds Frozen from Validation Set:
   - Logistic Regression: P(UP) >= 0.350
   - LightGBM Classifier: P(UP) >= 0.333
3. Target-Execution Alignment:
   - Signal at close of session t.
   - Entry at open of session t+1 at Open * (1 + Slippage).
   - Intraday sessions_held incremented at close.
   - Scheduled exit at open of session t+6 after exactly 5 completed trading sessions.
   - Down-risk exit at open of session t+1 if DOWN signal triggered at session t close.
4. Complete Statutory NSE Delivery Schedule:
   - Brokerage: 0.030%
   - STT: 0.100% buy, 0.100% sell
   - Exchange: 0.00325%
   - SEBI: 0.0001%
   - Stamp Duty: 0.015% (Buy only)
   - GST: 18% on (Brokerage + Exchange + SEBI)
   - Slippage: 0.050% on market open executions
5. Full Dividend Entitlement:
   - Positions held across cum-date close are credited dividend cash on ex-date.
   - Positions sold at ex-date morning open receive dividend credit prior to position closure.
6. Unified Base Capital: INR 100,000.00 for all strategies.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models" / "horizon_5d_v2"

STARTING_CAPITAL = 100000.0
MAX_POSITIONS = 3
MAX_HOLD_SESSIONS = 5
CASH_YIELD_ANNUAL = 0.065  # Assumed annual yield rate (based on RBI repo benchmark; not guaranteed)

# -----------------------------------------------------------------------------
# Transaction Cost Structure (NSE Cash Equity Delivery Schedule)
# -----------------------------------------------------------------------------
FEE_SCHEDULE = {
    "brokerage_pct": 0.00030,          # 0.030% delivery brokerage
    "stt_buy_pct": 0.00100,            # 0.100% STT on delivery buy
    "stt_sell_pct": 0.00100,           # 0.100% STT on delivery sell
    "exchange_turnover_pct": 0.0000325,# 0.00325% NSE turnover charge
    "sebi_turnover_pct": 0.000001,     # 0.0001% SEBI charge (Rs 10/crore)
    "stamp_duty_buy_pct": 0.00015,     # 0.015% Stamp duty on buy only
    "gst_rate": 0.18,                  # 18% GST on (Brokerage + Exchange + SEBI)
    "slippage_pct": 0.00050,           # 0.050% execution slippage on open price
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
    """Extract historical dividend events directly from Adj Close vs Close price adjustment factors."""
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
                "div_yield_pct": round(float(div_yield * 100.0), 3),
            })
            
    return pd.DataFrame(div_records)


def load_market_data():
    """Load raw daily prices, test split features/metadata from v2 pipeline, and corporate actions."""
    raw = pd.read_csv(DATA_DIR / "raw" / "multi_stock_historical.csv")
    raw["Date"] = pd.to_datetime(raw["Date"])
    
    X_test = pd.read_csv(DATA_DIR / "splits_v2" / "horizon_5d" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "splits_v2" / "horizon_5d" / "y_test.csv")
    meta_test = pd.read_csv(DATA_DIR / "splits_v2" / "horizon_5d" / "metadata_test.csv")
    meta_test["Date"] = pd.to_datetime(meta_test["Date"])
    
    test_dates = sorted(meta_test["Date"].unique())
    start_date = test_dates[0]
    
    # Filter raw prices from start_date to end of dataset
    raw_test = raw[raw["Date"] >= start_date].sort_values(["Date", "Symbol"]).reset_index(drop=True)
    all_trading_dates = sorted(raw_test["Date"].unique())
    
    open_prices = raw_test.pivot(index="Date", columns="Symbol", values="Open")
    close_prices = raw_test.pivot(index="Date", columns="Symbol", values="Close")
    
    # Extract dividends across test period
    div_df = extract_dividends(raw)
    test_divs = div_df[(div_df["Ex_Date"] >= start_date) & (div_df["Ex_Date"] <= all_trading_dates[-1])].copy()
    
    dividend_lookup = {}
    for _, row in test_divs.iterrows():
        dividend_lookup[(row["Ex_Date"], row["Symbol"])] = row["div_per_share"]
        
    return {
        "raw_test": raw_test,
        "all_trading_dates": all_trading_dates,
        "test_signal_dates": test_dates,
        "open_prices": open_prices,
        "close_prices": close_prices,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
        "symbols": sorted(raw["Symbol"].unique()),
        "dividend_lookup": dividend_lookup,
        "dividend_events": test_divs,
    }


def compute_portfolio_metrics(equity_series, daily_returns, trade_blotter, is_cash_benchmark=False):
    """Compute financial statistics: Total Return, CAGR, Max Drawdown, Sharpe, Volatility."""
    initial_val = STARTING_CAPITAL
    final_val = equity_series.iloc[-1]
    n_days = len(equity_series)
    
    total_return = (final_val - initial_val) / initial_val
    cagr = (final_val / initial_val) ** (252.0 / n_days) - 1.0 if final_val > 0 else -1.0
    
    peaks = equity_series.cummax()
    drawdowns = (peaks - equity_series) / peaks
    max_drawdown = float(drawdowns.max())
    
    rf_daily = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - rf_daily
    ret_std = float(daily_returns.std())
    
    if is_cash_benchmark or ret_std < 1e-6:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(252.0) * (excess_returns.mean() / ret_std))
        
    closed_trades = [t for t in trade_blotter if t["action"] == "SELL"]
    profitable_trades = [t for t in closed_trades if t["net_pnl"] > 0]
    unprofitable_trades = [t for t in closed_trades if t["net_pnl"] < 0]
    
    win_rate = len(profitable_trades) / len(closed_trades) if len(closed_trades) > 0 else 0.0
    
    gross_gains = sum(t["net_pnl"] for t in profitable_trades)
    gross_losses = sum(abs(t["net_pnl"]) for t in unprofitable_trades)
    profit_factor = gross_gains / gross_losses if gross_losses > 0 else (999.0 if gross_gains > 0 else 0.0)
    
    total_commissions = sum(t["commission"] for t in trade_blotter)
    total_dividends = sum(t["net_pnl"] for t in trade_blotter if t["action"] == "DIVIDEND")
    
    return {
        "starting_capital": initial_val,
        "final_equity": float(final_val),
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "annualized_volatility_pct": float(ret_std * np.sqrt(252.0) * 100.0),
        "sharpe_ratio": float(sharpe),
        "total_trades_completed": len(closed_trades),
        "win_rate_pct": float(win_rate * 100.0),
        "profit_factor": float(profit_factor),
        "total_commissions_paid": float(total_commissions),
        "total_dividends_received": float(total_dividends),
    }


def run_model_simulation_v2(model_name, model, market, p_up_threshold=0.35, scaler=None):
    """Run event-driven portfolio simulation for a model using frozen decision threshold."""
    dates = market["all_trading_dates"]
    signal_dates = set(market["test_signal_dates"])
    opens = market["open_prices"]
    closes = market["close_prices"]
    div_lookup = market["dividend_lookup"]
    
    meta = market["meta_test"]
    X = market["X_test"]
    
    if scaler is not None:
        X_eval = scaler.transform(X)
    else:
        X_eval = X
        
    probs = model.predict_proba(X_eval)
    preds = np.argmax(probs, axis=1)
    
    daily_signals = {}
    for idx, row in meta.iterrows():
        d = row["Date"]
        sym = row["Symbol"]
        if d not in daily_signals:
            daily_signals[d] = {}
        daily_signals[d][sym] = {
            "pred": int(preds[idx]),
            "p_down": float(probs[idx, 0]),
            "p_hold": float(probs[idx, 1]),
            "p_up": float(probs[idx, 2]),
        }
        
    cash = STARTING_CAPITAL
    open_positions = {}
    pending_buys = []
    pending_exits = set()
    
    trade_blotter = []
    daily_records = []
    
    for i, current_date in enumerate(dates):
        # -------------------------------------------------------------
        # 1. MORNING (Market Open): Execute Pending Orders at Open
        # -------------------------------------------------------------
        # A. Execute Pending Exits first
        exits_to_process = list(pending_exits)
        for sym in exits_to_process:
            if sym in open_positions:
                pos = open_positions[sym]
                
                # DIVIDEND ENTITLEMENT CHECK (EX-DATE MORNING EXIT):
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
                            "reason": f"EX_DIVIDEND_PAYOUT_ON_EXIT_RS_{div_per_share:.2f}",
                            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                            "exit_date": current_date.strftime("%Y-%m-%d"),
                            "sessions_held": pos["sessions_held"],
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
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
                del open_positions[sym]
        pending_exits.clear()
        
        # B. Execute Pending Buys
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
                            "reason": f"BUY_SIGNAL_P_UP_{p_up:.3f}",
                            "entry_date": current_date.strftime("%Y-%m-%d"),
                            "exit_date": "",
                            "sessions_held": 0,
                            "cash_after": float(cash),
                        })
        pending_buys.clear()
        
        # -------------------------------------------------------------
        # 2. INTRADAY: Age active positions
        # -------------------------------------------------------------
        for sym in open_positions:
            open_positions[sym]["sessions_held"] += 1
            
        # -------------------------------------------------------------
        # 3. DIVIDEND ACCOUNTING: Credit dividends on Ex-Dividend Day
        # -------------------------------------------------------------
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
                        "reason": f"EX_DIVIDEND_PAYOUT_RS_{div_per_share:.2f}",
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": "",
                        "sessions_held": pos["sessions_held"],
                        "cash_after": float(cash),
                    })
                    pos["dividend_credited_dates"] = credited | {current_date}
                
        # -------------------------------------------------------------
        # 4. POST-MARKET (Close): Mark-to-Market & Liquidation Check
        # -------------------------------------------------------------
        positions_value = sum(
            pos["shares"] * closes.loc[current_date, sym]
            for sym, pos in open_positions.items()
        )
        total_equity = cash + positions_value
        
        # Liquidate on final day of backtest if still open
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
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
            open_positions.clear()
            positions_value = 0.0
            total_equity = cash
            
        daily_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "cash": float(cash),
            "positions_value": float(positions_value),
            "total_equity": float(total_equity),
            "open_positions_count": len(open_positions),
        })
        
        # Plan orders for NEXT open (t+1)
        if i < len(dates) - 1:
            has_signals = current_date in signal_dates
            sigs = daily_signals.get(current_date, {})
            
            # Check for exits on currently held positions
            for sym, pos in open_positions.items():
                # Early DOWN exit: if stock receives DOWN signal (p_down > p_up and p_down >= 0.333)
                if has_signals and sym in sigs and sigs[sym]["pred"] == 0:
                    pos["exit_reason"] = "DOWN_RISK_EXIT"
                    pending_exits.add(sym)
                elif pos["sessions_held"] >= MAX_HOLD_SESSIONS:
                    pos["exit_reason"] = "MAX_HOLD_5D_EXIT"
                    pending_exits.add(sym)
                        
            # Check for candidate buys (strictly using frozen validation threshold)
            if has_signals:
                for sym, s_data in sigs.items():
                    # Qualifies if P(UP) >= frozen threshold AND P(UP) > P(DOWN)
                    if s_data["p_up"] >= p_up_threshold and s_data["p_up"] > s_data["p_down"]:
                        if sym not in open_positions and sym not in pending_exits:
                            pending_buys.append((sym, s_data["p_up"]))
                        
    df_equity = pd.DataFrame(daily_records)
    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(0.0)
    
    metrics = compute_portfolio_metrics(df_equity["total_equity"], df_equity["daily_return"], trade_blotter)
    return df_equity, trade_blotter, metrics


def run_buy_and_hold_benchmark(market):
    """Run passive Equal-Weighted Buy-and-Hold benchmark across all 8 stocks with dividends."""
    dates = market["all_trading_dates"]
    opens = market["open_prices"]
    closes = market["close_prices"]
    symbols = market["symbols"]
    div_lookup = market["dividend_lookup"]
    
    cash = STARTING_CAPITAL
    slot_budget = cash / len(symbols)
    positions = {}
    trade_blotter = []
    
    start_date = dates[0]
    for sym in symbols:
        raw_open = opens.loc[start_date, sym]
        exec_price = raw_open * (1.0 + SLIPPAGE)
        cost_per_share = exec_price * (1.0 + BUY_STATUTORY_FEE)
        shares = int(np.floor(slot_budget / cost_per_share))
        
        gross_cost = shares * exec_price
        fee = gross_cost * BUY_STATUTORY_FEE
        total_outflow = gross_cost + fee
        cash -= total_outflow
        
        positions[sym] = {
            "shares": shares,
            "entry_price": float(exec_price),
            "entry_date": start_date,
            "sessions_held": 0,
        }
        trade_blotter.append({
            "date": start_date.strftime("%Y-%m-%d"),
            "symbol": sym,
            "action": "BUY",
            "shares": shares,
            "exec_price": float(exec_price),
            "commission": float(fee),
            "net_pnl": 0.0,
            "reason": "BENCHMARK_BUY",
            "entry_date": start_date.strftime("%Y-%m-%d"),
            "exit_date": "",
            "sessions_held": 0,
            "cash_after": float(cash),
        })
        
    daily_records = []
    for i, d in enumerate(dates):
        for sym in positions:
            positions[sym]["sessions_held"] += 1
            
        for sym, pos in positions.items():
            div_key = (d, sym)
            if div_key in div_lookup and pos["entry_date"] < d:
                credited = pos.get("dividend_credited_dates", set())
                if d not in credited:
                    div_per_share = div_lookup[div_key]
                    div_payout = pos["shares"] * div_per_share
                    cash += div_payout
                    trade_blotter.append({
                        "date": d.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": float(div_per_share),
                        "commission": 0.0,
                        "net_pnl": float(div_payout),
                        "reason": f"EX_DIVIDEND_PAYOUT_RS_{div_per_share:.2f}",
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": "",
                        "sessions_held": pos["sessions_held"],
                        "cash_after": float(cash),
                    })
                    pos["dividend_credited_dates"] = credited | {d}
                
        if i == len(dates) - 1:
            for sym, pos in list(positions.items()):
                raw_close = closes.loc[d, sym]
                exec_price = raw_close * (1.0 - SLIPPAGE)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * SELL_STATUTORY_FEE
                net_proceeds = gross_proceeds - fee
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + BUY_STATUTORY_FEE)
                trade_blotter.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_proceeds - total_cost),
                    "reason": "BENCHMARK_FINAL_EXIT",
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": d.strftime("%Y-%m-%d"),
                    "sessions_held": pos["sessions_held"],
                    "cash_after": float(cash),
                })
            positions.clear()
            val = 0.0
            tot = cash
        else:
            val = sum(pos["shares"] * closes.loc[d, sym] for sym, pos in positions.items())
            tot = cash + val
            
        daily_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "cash": float(cash),
            "positions_value": float(val),
            "total_equity": float(tot),
        })
        
    df_equity = pd.DataFrame(daily_records)
    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(
        (df_equity["total_equity"].iloc[0] - STARTING_CAPITAL) / STARTING_CAPITAL
    )
    metrics = compute_portfolio_metrics(df_equity["total_equity"], df_equity["daily_return"], trade_blotter)
    return df_equity, trade_blotter, metrics


def run_cash_benchmark(market):
    """Run cash benchmark with assumed 6.50% annualized yield rate."""
    dates = market["all_trading_dates"]
    daily_rate = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    
    cash = STARTING_CAPITAL
    daily_records = []
    for d in dates:
        cash *= (1.0 + daily_rate)
        daily_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "cash": float(cash),
            "positions_value": 0.0,
            "total_equity": float(cash),
        })
        
    df_equity = pd.DataFrame(daily_records)
    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(daily_rate)
    metrics = compute_portfolio_metrics(df_equity["total_equity"], df_equity["daily_return"], [], is_cash_benchmark=True)
    return df_equity, metrics


def validate_simulation_integrity(df_equity, trade_blotter, model_name):
    """Run comprehensive automated audit and integrity assertions."""
    print(f"\n[Validation Assertions: {model_name}]")
    
    min_cash = df_equity["cash"].min()
    assert min_cash >= -1e-4, f"FAIL: Negative cash balance detected: {min_cash}"
    print(f"  PASS: Cash non-negativity maintained (Min Cash: INR {min_cash:,.2f})")
    
    if "open_positions_count" in df_equity.columns:
        max_pos = df_equity["open_positions_count"].max()
        assert max_pos <= MAX_POSITIONS, f"FAIL: Max positions exceeded: {max_pos}"
        print(f"  PASS: Maximum concurrent positions satisfied (Peak: {max_pos}/{MAX_POSITIONS})")
        
    trades_only = [t for t in trade_blotter if t["action"] in ["BUY", "SELL"]]
    for t in trades_only:
        assert t["shares"] == int(t["shares"]) and t["shares"] > 0, f"FAIL: Fractional or zero shares: {t}"
    print(f"  PASS: Integer share constraint validated across {len(trades_only)} trades")
    
    assert all(t["action"] in ["BUY", "SELL", "DIVIDEND"] for t in trade_blotter), "FAIL: Invalid trade actions"
    print(f"  PASS: Zero overnight shorting verified")
    
    final_eq = df_equity.iloc[-1]["total_equity"]
    final_cash = df_equity.iloc[-1]["cash"]
    assert abs(final_eq - final_cash) < 1e-4, f"FAIL: Final equity != final cash ({final_eq} vs {final_cash})"
    print(f"  PASS: 100% clean final liquidation to realized cash (Final Cash: INR {final_cash:,.2f})")
    
    sell_trades = [t for t in trade_blotter if t["action"] == "SELL"]
    for t in sell_trades:
        reason = t["reason"]
        sessions = t["sessions_held"]
        if reason == "MAX_HOLD_5D_EXIT":
            assert sessions == MAX_HOLD_SESSIONS, f"FAIL: Scheduled exit held {sessions} != {MAX_HOLD_SESSIONS}: {t}"
        elif reason == "DOWN_RISK_EXIT":
            assert 1 <= sessions <= MAX_HOLD_SESSIONS, f"FAIL: Down risk exit held {sessions} not in [1, {MAX_HOLD_SESSIONS}]: {t}"
        elif reason == "END_OF_TEST_LIQUIDATION":
            assert 1 <= sessions <= MAX_HOLD_SESSIONS, f"FAIL: Liquidation held {sessions} > {MAX_HOLD_SESSIONS}: {t}"
        elif reason == "BENCHMARK_FINAL_EXIT":
            assert sessions == len(df_equity), f"FAIL: Benchmark exit held {sessions} != {len(df_equity)}: {t}"
    print(f"  PASS: Exact holding-period constraints validated across {len(sell_trades)} exits")
    
    div_trades = [t for t in trade_blotter if t["action"] == "DIVIDEND"]
    for dt in div_trades:
        assert dt["net_pnl"] > 0.0, f"FAIL: Non-positive dividend amount: {dt}"
        assert dt["commission"] == 0.0, f"FAIL: Non-zero commission on dividend: {dt}"
    print(f"  PASS: Dividend cash flows validated ({len(div_trades)} payouts credited)")


def main():
    print("=" * 80)
    print("QUANTARA AI — PHASE 3.4: PORTFOLIO BACKTEST ENGINE (v2 PIPELINE)")
    print(f"Starting Capital: INR {STARTING_CAPITAL:,.2f} | Max Concurrent Positions: {MAX_POSITIONS}")
    print(f"Regulatory Buy Fee: {BUY_STATUTORY_FEE:.4%} | Regulatory Sell Fee: {SELL_STATUTORY_FEE:.4%}")
    print(f"Execution Slippage: {SLIPPAGE:.3%} | Effective Round-Trip Friction: {BUY_STATUTORY_FEE + SELL_STATUTORY_FEE + 2*SLIPPAGE:.4%}")
    print("=" * 80)
    
    market = load_market_data()
    print(f"Loaded {len(market['all_trading_dates'])} trading sessions ({market['all_trading_dates'][0].date()} to {market['all_trading_dates'][-1].date()})")
    print(f"Evaluating 8 NSE stocks: {', '.join(market['symbols'])}")
    print(f"Identified {len(market['dividend_events'])} corporate dividend distributions during test period")
    
    # Load Pre-Trained v2 Models & Preprocessing
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_baseline.joblib")
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_5d_classifier.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    
    # Load Frozen Validation Thresholds from model evaluation report
    eval_report = json.loads((MODELS_DIR / "phase3_4_model_evaluation.json").read_text(encoding="utf-8"))
    lr_threshold = eval_report["selected_decision_thresholds"]["logistic_regression_p_up_threshold"]
    lgb_threshold = eval_report["selected_decision_thresholds"]["lightgbm_p_up_threshold"]
    
    print(f"Loaded frozen validation decision thresholds:")
    print(f"  Logistic Regression: P(UP) >= {lr_threshold:.3f}")
    print(f"  LightGBM Classifier: P(UP) >= {lgb_threshold:.3f}")
    
    # 1. Logistic Regression Simulation
    print("\n[Strategy 1/4] Running Standardized Logistic Regression...")
    df_lr, blotter_lr, metrics_lr = run_model_simulation_v2(
        "Logistic Regression", lr_model, market, p_up_threshold=lr_threshold, scaler=scaler
    )
    validate_simulation_integrity(df_lr, blotter_lr, "Logistic Regression")
    
    # 2. LightGBM Simulation
    print("\n[Strategy 2/4] Running Tuned LightGBM Classifier...")
    df_lgb, blotter_lgb, metrics_lgb = run_model_simulation_v2(
        "LightGBM", lgb_model, market, p_up_threshold=lgb_threshold
    )
    validate_simulation_integrity(df_lgb, blotter_lgb, "LightGBM")
    
    # 3. Buy and Hold Benchmark
    print("\n[Strategy 3/4] Running Equal-Weighted Buy-and-Hold Benchmark...")
    df_bh, blotter_bh, metrics_bh = run_buy_and_hold_benchmark(market)
    validate_simulation_integrity(df_bh, blotter_bh, "Buy-and-Hold Benchmark")
    
    # 4. Cash Yield Benchmark
    print("\n[Strategy 4/4] Running Cash Benchmark (6.50% Assumed Annual Yield)...")
    df_cash, metrics_cash = run_cash_benchmark(market)
    validate_simulation_integrity(df_cash, [], "Cash Benchmark")
    
    # Save Trade Blotters
    pd.DataFrame(blotter_lr).to_csv(MODELS_DIR / "trade_blotter_logistic_regression.csv", index=False)
    pd.DataFrame(blotter_lgb).to_csv(MODELS_DIR / "trade_blotter_lightgbm.csv", index=False)
    pd.DataFrame(blotter_bh).to_csv(MODELS_DIR / "trade_blotter_buy_and_hold.csv", index=False)
    
    # Save Equity Curves
    df_comparison = pd.DataFrame({
        "Date": df_lr["date"],
        "Logistic_Regression_Equity": df_lr["total_equity"],
        "LightGBM_Equity": df_lgb["total_equity"],
        "Buy_and_Hold_Equity": df_bh["total_equity"],
        "Cash_Yield_Equity": df_cash["total_equity"],
    })
    df_comparison.to_csv(MODELS_DIR / "equity_curve_comparison.csv", index=False)
    
    # Save Metrics JSON
    all_metrics = {
        "pipeline_version": "v2_execution_aligned",
        "parameters": {
            "starting_capital": STARTING_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "max_hold_sessions": MAX_HOLD_SESSIONS,
            "slippage_pct": SLIPPAGE,
            "buy_statutory_fee_pct": round(float(BUY_STATUTORY_FEE), 6),
            "sell_statutory_fee_pct": round(float(SELL_STATUTORY_FEE), 6),
            "round_trip_friction_pct": round(float(BUY_STATUTORY_FEE + SELL_STATUTORY_FEE + 2 * SLIPPAGE), 6),
            "fee_schedule_details": FEE_SCHEDULE,
            "cash_yield_annual_assumed": CASH_YIELD_ANNUAL,
            "cash_benchmark_note": "Assumed 6.50% annualized yield (based on RBI repo benchmark; actual retail cash rates vary)",
            "selected_thresholds": {
                "logistic_regression": lr_threshold,
                "lightgbm": lgb_threshold,
            },
            "dividend_accounting_status": "UNVERIFIED (Primary Exchange Records Unavailable)",
            "dividend_accounting_methodology": "Exact ex-date ownership entitlement. Shares held across cum-date close (entry_date < ex_date) are credited dividend cash on ex-date, including positions sold at ex-date open.",
            "dividend_data_limitation": "Dividends are mathematically inferred from unadjusted vs adjusted price step-changes in multi_stock_historical.csv. Primary exchange filings (NSE/BSE notices, official payment dates) are unavailable offline. Cash availability lag between ex-date and payment date is unmodeled.",
            "start_date": str(market["all_trading_dates"][0].date()),
            "end_date": str(market["all_trading_dates"][-1].date()),
            "total_trading_sessions": len(market["all_trading_dates"]),
        },
        "logistic_regression": metrics_lr,
        "lightgbm": metrics_lgb,
        "buy_and_hold_benchmark": metrics_bh,
        "cash_benchmark": metrics_cash,
    }
    
    with open(MODELS_DIR / "portfolio_backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        
    print("\n" + "=" * 80)
    print("PHASE 3.4: FINAL RECONCILED PORTFOLIO PERFORMANCE SUMMARY (INR 100,000 BASE)")
    print("=" * 80)
    summary_table = [
        {
            "Strategy": f"Logistic Regression (th={lr_threshold:.3f})",
            "Final Equity": f"INR {metrics_lr['final_equity']:,.2f}",
            "Return": f"{metrics_lr['total_return_pct']:+.2f}%",
            "CAGR": f"{metrics_lr['cagr_pct']:+.2f}%",
            "Max DD": f"{metrics_lr['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_lr['sharpe_ratio']:.2f}",
            "Trades": metrics_lr["total_trades_completed"],
            "Win Rate": f"{metrics_lr['win_rate_pct']:.1f}%",
            "Dividends": f"INR {metrics_lr['total_dividends_received']:,.2f}",
            "Commissions": f"INR {metrics_lr['total_commissions_paid']:,.2f}",
        },
        {
            "Strategy": f"LightGBM Classifier (th={lgb_threshold:.3f})",
            "Final Equity": f"INR {metrics_lgb['final_equity']:,.2f}",
            "Return": f"{metrics_lgb['total_return_pct']:+.2f}%",
            "CAGR": f"{metrics_lgb['cagr_pct']:+.2f}%",
            "Max DD": f"{metrics_lgb['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_lgb['sharpe_ratio']:.2f}",
            "Trades": metrics_lgb["total_trades_completed"],
            "Win Rate": f"{metrics_lgb['win_rate_pct']:.1f}%",
            "Dividends": f"INR {metrics_lgb['total_dividends_received']:,.2f}",
            "Commissions": f"INR {metrics_lgb['total_commissions_paid']:,.2f}",
        },
        {
            "Strategy": "Buy & Hold Benchmark",
            "Final Equity": f"INR {metrics_bh['final_equity']:,.2f}",
            "Return": f"{metrics_bh['total_return_pct']:+.2f}%",
            "CAGR": f"{metrics_bh['cagr_pct']:+.2f}%",
            "Max DD": f"{metrics_bh['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_bh['sharpe_ratio']:.2f}",
            "Trades": metrics_bh["total_trades_completed"],
            "Win Rate": f"{metrics_bh['win_rate_pct']:.1f}%",
            "Dividends": f"INR {metrics_bh['total_dividends_received']:,.2f}",
            "Commissions": f"INR {metrics_bh['total_commissions_paid']:,.2f}",
        },
        {
            "Strategy": "Cash Benchmark (6.5%)",
            "Final Equity": f"INR {metrics_cash['final_equity']:,.2f}",
            "Return": f"{metrics_cash['total_return_pct']:+.2f}%",
            "CAGR": f"{metrics_cash['cagr_pct']:+.2f}%",
            "Max DD": f"{metrics_cash['max_drawdown_pct']:.2f}%",
            "Sharpe": "0.00 (Benchmark)",
            "Trades": 0,
            "Win Rate": "N/A",
            "Dividends": "INR 0.00",
            "Commissions": "INR 0.00",
        },
    ]
    print(pd.DataFrame(summary_table).to_string(index=False))
    print("=" * 80)
    print(f"All Phase 3.4 blotters, metrics, and curves saved to:\n{MODELS_DIR}")


if __name__ == "__main__":
    main()

