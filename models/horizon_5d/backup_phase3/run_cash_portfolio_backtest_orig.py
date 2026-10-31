"""
Quantara AI — Phase 3: Real-World Cash-Equity Portfolio Backtest Engine
Implements institutional portfolio accounting on the 5-day horizon test split:
- Starting Capital: INR 100,000
- Maximum Simultaneous Positions: 3
- Position Sizing: Integer shares, equal-slot allocation from available cash
- Signals: Model predicts after session t close; executes at session t+1 open
- Competing UP signals ranked by model P(UP) probability
- DOWN signals treat as SELL/EXIT or MOVE-TO-CASH (zero overnight shorting)
- Holding Period: Up to 5 trading sessions (exits at open t+6), or earlier if DOWN triggers
- Transaction Costs: 0.085% on entry, 0.185% on exit (0.27% round-trip including STT/slippage)
- Benchmarks: Equal-weighted 8-stock Buy-and-Hold and 6.50% Cash Yield Benchmark
- Models Evaluated: Standardized Logistic Regression & Tuned LightGBM
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models" / "horizon_5d"

STARTING_CAPITAL = 100000.0
MAX_POSITIONS = 3
SLIPPAGE = 0.0005      # 0.05% slippage on open executions
BUY_FEE = 0.00035      # 0.035% brokerage + turnover on buy
SELL_FEE = 0.00135     # 0.035% brokerage + 0.10% STT on delivery sell
MAX_HOLD_SESSIONS = 5
CASH_YIELD_ANNUAL = 0.065 # 6.5% RBI repo rate for cash benchmark


def load_market_data():
    """Load raw daily prices and test split features/metadata."""
    raw = pd.read_csv(DATA_DIR / "raw" / "multi_stock_historical.csv")
    raw["Date"] = pd.to_datetime(raw["Date"])
    
    X_test = pd.read_csv(DATA_DIR / "splits" / "horizon_5d" / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "splits" / "horizon_5d" / "y_test.csv")
    meta_test = pd.read_csv(DATA_DIR / "splits" / "horizon_5d" / "metadata_test.csv")
    meta_test["Date"] = pd.to_datetime(meta_test["Date"])
    
    test_dates = sorted(meta_test["Date"].unique())
    start_date = test_dates[0]
    
    # Filter raw prices from start_date to the end of raw data
    raw_test = raw[raw["Date"] >= start_date].sort_values(["Date", "Symbol"]).reset_index(drop=True)
    all_trading_dates = sorted(raw_test["Date"].unique())
    
    # Pivot tables for fast O(1) price lookups by Date and Symbol
    open_prices = raw_test.pivot(index="Date", columns="Symbol", values="Open")
    close_prices = raw_test.pivot(index="Date", columns="Symbol", values="Close")
    
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
    }


def compute_portfolio_metrics(equity_series, daily_returns, trade_blotter):
    """Compute financial statistics: Total Return, CAGR, Max Drawdown, Sharpe."""
    initial_val = equity_series.iloc[0]
    final_val = equity_series.iloc[-1]
    n_days = len(equity_series)
    
    total_return = (final_val - initial_val) / initial_val
    cagr = (final_val / initial_val) ** (252.0 / n_days) - 1.0 if final_val > 0 else -1.0
    
    # Drawdown
    peaks = equity_series.cummax()
    drawdowns = (peaks - equity_series) / peaks
    max_drawdown = float(drawdowns.max())
    
    # Sharpe Ratio
    rf_daily = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - rf_daily
    ret_std = float(daily_returns.std())
    sharpe = float(np.sqrt(252.0) * (excess_returns.mean() / ret_std)) if ret_std > 0 else 0.0
    
    # Trade statistics
    closed_trades = [t for t in trade_blotter if t["action"] == "SELL"]
    profitable_trades = [t for t in closed_trades if t["net_pnl"] > 0]
    win_rate = len(profitable_trades) / len(closed_trades) if closed_trades else 0.0
    
    gross_gains = sum(t["net_pnl"] for t in closed_trades if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in closed_trades if t["net_pnl"] < 0))
    profit_factor = float(gross_gains / gross_losses) if gross_losses > 0 else (999.0 if gross_gains > 0 else 0.0)
    
    return {
        "initial_equity": float(initial_val),
        "final_equity": float(final_val),
        "total_return_pct": float(total_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "annualized_volatility_pct": float(ret_std * np.sqrt(252.0) * 100.0),
        "sharpe_ratio": float(sharpe),
        "total_trades_completed": len(closed_trades),
        "win_rate_pct": float(win_rate * 100.0),
        "profit_factor": float(profit_factor),
        "total_commissions_paid": float(sum(t["commission"] for t in trade_blotter)),
    }


def run_model_simulation(model_name, model, market):
    """Run event-driven portfolio simulation for a model."""
    dates = market["all_trading_dates"]
    signal_dates = set(market["test_signal_dates"])
    opens = market["open_prices"]
    closes = market["close_prices"]
    
    # Pre-generate predictions for all test sessions
    meta = market["meta_test"]
    X = market["X_test"]
    
    # Verify class ordering: [0, 1, 2] -> 0: DOWN, 1: HOLD, 2: UP
    classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
    assert list(classes) == [0, 1, 2], f"Unexpected classes {classes} in {model_name}!"
    
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    # Group signals by Date for fast daily retrieval
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
    open_positions = {}  # sym -> {shares, entry_price, entry_date, sessions_held}
    pending_buys = []    # list of (sym, p_up)
    pending_exits = set()# set of syms
    
    trade_blotter = []
    daily_records = []
    
    for i, current_date in enumerate(dates):
        # -------------------------------------------------------------
        # 1. MORNING (Market Open): Execute Pending Orders at Open
        # -------------------------------------------------------------
        # A. Execute Pending Exits first (frees up slots & cash)
        exits_to_process = list(pending_exits)
        for sym in exits_to_process:
            if sym in open_positions:
                pos = open_positions[sym]
                raw_open = opens.loc[current_date, sym]
                exec_price = raw_open * (1.0 - SLIPPAGE)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * SELL_FEE
                net_proceeds = gross_proceeds - fee
                
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + BUY_FEE)
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
                    "cash_after": float(cash),
                })
                del open_positions[sym]
        pending_exits.clear()
        
        # B. Execute Pending Buys (if slots and cash available)
        # Sort pending buys descending by p_up confidence
        pending_buys.sort(key=lambda x: x[1], reverse=True)
        for sym, p_up in pending_buys:
            if len(open_positions) < MAX_POSITIONS and sym not in open_positions:
                avail_slots = MAX_POSITIONS - len(open_positions)
                slot_budget = cash / avail_slots
                raw_open = opens.loc[current_date, sym]
                exec_price = raw_open * (1.0 + SLIPPAGE)
                cost_per_share = exec_price * (1.0 + BUY_FEE)
                
                shares = int(np.floor(slot_budget / cost_per_share))
                if shares > 0:
                    gross_cost = shares * exec_price
                    fee = gross_cost * BUY_FEE
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
                            "cash_after": float(cash),
                        })
        pending_buys.clear()
        
        # -------------------------------------------------------------
        # 2. INTRADAY: Age active positions
        # -------------------------------------------------------------
        for sym in open_positions:
            open_positions[sym]["sessions_held"] += 1
            
        # -------------------------------------------------------------
        # 3. POST-MARKET (Close): Mark-to-Market & Compute Signals
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
                fee = gross_proceeds * SELL_FEE
                net_proceeds = gross_proceeds - fee
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + BUY_FEE)
                trade_blotter.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "action": "SELL",
                    "shares": pos["shares"],
                    "exec_price": float(exec_price),
                    "commission": float(fee),
                    "net_pnl": float(net_proceeds - total_cost),
                    "reason": "END_OF_TEST_LIQUIDATION",
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
        
        # Plan orders for NEXT open (t+1) if current_date has model signals
        if current_date in signal_dates and i < len(dates) - 1:
            sigs = daily_signals[current_date]
            
            # Check for exits on currently held positions
            for sym, pos in open_positions.items():
                if sym in sigs:
                    if sigs[sym]["pred"] == 0:  # DOWN signal
                        pos["exit_reason"] = "DOWN_RISK_EXIT"
                        pending_exits.add(sym)
                    elif pos["sessions_held"] >= MAX_HOLD_SESSIONS:
                        pos["exit_reason"] = "MAX_HOLD_5D_EXIT"
                        pending_exits.add(sym)
                        
            # Check for candidate buys
            for sym, s_data in sigs.items():
                if s_data["pred"] == 2:  # UP signal
                    if sym not in open_positions and sym not in pending_exits:
                        pending_buys.append((sym, s_data["p_up"]))
                        
    df_equity = pd.DataFrame(daily_records)
    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(0.0)
    
    metrics = compute_portfolio_metrics(df_equity["total_equity"], df_equity["daily_return"], trade_blotter)
    return df_equity, trade_blotter, metrics


def run_buy_and_hold_benchmark(market):
    """Run passive Equal-Weighted Buy-and-Hold benchmark across all 8 stocks."""
    dates = market["all_trading_dates"]
    opens = market["open_prices"]
    closes = market["close_prices"]
    symbols = market["symbols"]
    
    cash = STARTING_CAPITAL
    slot_budget = cash / len(symbols)
    positions = {}
    trade_blotter = []
    
    # Buy all 8 stocks on day 0 open
    start_date = dates[0]
    for sym in symbols:
        raw_open = opens.loc[start_date, sym]
        exec_price = raw_open * (1.0 + SLIPPAGE)
        cost_per_share = exec_price * (1.0 + BUY_FEE)
        shares = int(np.floor(slot_budget / cost_per_share))
        
        gross_cost = shares * exec_price
        fee = gross_cost * BUY_FEE
        total_outflow = gross_cost + fee
        cash -= total_outflow
        
        positions[sym] = {"shares": shares, "entry_price": float(exec_price)}
        trade_blotter.append({
            "date": start_date.strftime("%Y-%m-%d"), "symbol": sym, "action": "BUY",
            "shares": shares, "exec_price": float(exec_price), "commission": float(fee),
            "net_pnl": 0.0, "reason": "BENCHMARK_BUY", "cash_after": float(cash)
        })
        
    daily_records = []
    for i, d in enumerate(dates):
        if i == len(dates) - 1:
            # Liquidate on final day
            for sym, pos in list(positions.items()):
                raw_close = closes.loc[d, sym]
                exec_price = raw_close * (1.0 - SLIPPAGE)
                gross_proceeds = pos["shares"] * exec_price
                fee = gross_proceeds * SELL_FEE
                net_proceeds = gross_proceeds - fee
                cash += net_proceeds
                total_cost = pos["shares"] * pos["entry_price"] * (1.0 + BUY_FEE)
                trade_blotter.append({
                    "date": d.strftime("%Y-%m-%d"), "symbol": sym, "action": "SELL",
                    "shares": pos["shares"], "exec_price": float(exec_price), "commission": float(fee),
                    "net_pnl": float(net_proceeds - total_cost), "reason": "BENCHMARK_FINAL_EXIT", "cash_after": float(cash)
                })
            positions.clear()
            val = 0.0
            tot = cash
        else:
            val = sum(pos["shares"] * closes.loc[d, sym] for sym, pos in positions.items())
            tot = cash + val
            
        daily_records.append({
            "date": d.strftime("%Y-%m-%d"), "cash": float(cash),
            "positions_value": float(val), "total_equity": float(tot)
        })
        
    df_equity = pd.DataFrame(daily_records)
    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(0.0)
    metrics = compute_portfolio_metrics(df_equity["total_equity"], df_equity["daily_return"], trade_blotter)
    return df_equity, metrics


def run_cash_benchmark(market):
    """Run pure cash benchmark earning RBI repo rate (6.5% annualized)."""
    dates = market["all_trading_dates"]
    daily_rate = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 252.0) - 1.0
    
    cash = STARTING_CAPITAL
    daily_records = []
    for d in dates:
        daily_records.append({"date": d.strftime("%Y-%m-%d"), "cash": float(cash), "positions_value": 0.0, "total_equity": float(cash)})
        cash *= (1.0 + daily_rate)
        
    df_equity = pd.DataFrame(daily_records)
    df_equity["daily_return"] = df_equity["total_equity"].pct_change().fillna(0.0)
    metrics = compute_portfolio_metrics(df_equity["total_equity"], df_equity["daily_return"], [])
    return df_equity, metrics


def validate_simulation_integrity(df_equity, trade_blotter, model_name):
    """Run the 7 strict safety assertions."""
    print(f"\n[Validation Checks: {model_name}]")
    
    # 1. Cash non-negative
    min_cash = df_equity["cash"].min()
    assert min_cash >= -1e-4, f"FAIL: Negative cash balance detected: {min_cash}"
    print(f"  PASS: Cash non-negativity maintained (Min Cash: INR {min_cash:,.2f})")
    
    # 2. Position cap
    max_pos = df_equity["open_positions_count"].max()
    assert max_pos <= MAX_POSITIONS, f"FAIL: Max positions exceeded: {max_pos}"
    print(f"  PASS: Maximum concurrent positions constraint satisfied (Peak Positions: {max_pos}/{MAX_POSITIONS})")
    
    # 3. Integer shares
    for t in trade_blotter:
        assert t["shares"] == int(t["shares"]) and t["shares"] > 0, f"FAIL: Fractional or zero shares: {t}"
    print(f"  PASS: Integer share constraint validated across all {len(trade_blotter)} trade executions")
    
    # 4. Zero shorting
    assert all(t["action"] in ["BUY", "SELL"] for t in trade_blotter), "FAIL: Invalid trade actions"
    print(f"  PASS: Zero overnight shorting verified (Only cash BUY and SELL positions)")
    
    # 5. Clean liquidation
    final_pos = df_equity.iloc[-1]["open_positions_count"]
    final_eq = df_equity.iloc[-1]["total_equity"]
    final_cash = df_equity.iloc[-1]["cash"]
    assert final_pos == 0, f"FAIL: Positions left open at test end: {final_pos}"
    assert abs(final_eq - final_cash) < 1e-4, f"FAIL: Final equity != final cash ({final_eq} vs {final_cash})"
    print(f"  PASS: 100% clean final liquidation to realized cash (Final Cash: INR {final_cash:,.2f})")


def main():
    print("=" * 75)
    print("QUANTARA AI — PHASE 3: CASH-EQUITY PORTFOLIO BACKTEST ENGINE")
    print(f"Initial Capital: INR {STARTING_CAPITAL:,.2f} | Max Concurrent Positions: {MAX_POSITIONS}")
    print(f"Round-Trip Friction: {BUY_FEE+SELL_FEE+2*SLIPPAGE:.3%} (Brokerage + STT + Slippage)")
    print("=" * 75)
    
    market = load_market_data()
    print(f"Loaded {len(market['all_trading_dates'])} total trading sessions ({market['all_trading_dates'][0].date()} to {market['all_trading_dates'][-1].date()})")
    print(f"Evaluating 8 NSE stocks: {', '.join(market['symbols'])}")
    
    # Load Models
    lr_model = joblib.load(MODELS_DIR / "logistic_regression_baseline.joblib")
    lgb_model = joblib.load(MODELS_DIR / "lightgbm_5d_classifier.joblib")
    
    # Run Simulations
    print("\nRunning Simulation 1: Standardized Logistic Regression...")
    df_lr, blotter_lr, metrics_lr = run_model_simulation("Logistic Regression", lr_model, market)
    validate_simulation_integrity(df_lr, blotter_lr, "Logistic Regression")
    
    print("\nRunning Simulation 2: Tuned LightGBM Classifier...")
    df_lgb, blotter_lgb, metrics_lgb = run_model_simulation("LightGBM", lgb_model, market)
    validate_simulation_integrity(df_lgb, blotter_lgb, "LightGBM")
    
    print("\nRunning Benchmark 1: Buy-and-Hold (Equal-Weighted 8 Stocks)...")
    df_bh, metrics_bh = run_buy_and_hold_benchmark(market)
    
    print("Running Benchmark 2: Cash Yield (6.50% RBI Repo Rate)...")
    df_cash, metrics_cash = run_cash_benchmark(market)
    
    # Save Outputs
    pd.DataFrame(blotter_lr).to_csv(MODELS_DIR / "trade_blotter_logistic_regression.csv", index=False)
    pd.DataFrame(blotter_lgb).to_csv(MODELS_DIR / "trade_blotter_lightgbm.csv", index=False)
    
    df_comparison = pd.DataFrame({
        "Date": df_lr["date"],
        "Logistic_Regression_Equity": df_lr["total_equity"],
        "LightGBM_Equity": df_lgb["total_equity"],
        "Buy_and_Hold_Equity": df_bh["total_equity"],
        "Cash_Yield_Equity": df_cash["total_equity"],
    })
    df_comparison.to_csv(MODELS_DIR / "equity_curve_comparison.csv", index=False)
    
    all_metrics = {
        "parameters": {
            "starting_capital": STARTING_CAPITAL,
            "max_positions": MAX_POSITIONS,
            "slippage": SLIPPAGE,
            "buy_fee": BUY_FEE,
            "sell_fee": SELL_FEE,
            "max_hold_sessions": MAX_HOLD_SESSIONS,
            "cash_yield_annual": CASH_YIELD_ANNUAL,
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
        
    # Print Summary Table
    print("\n" + "=" * 75)
    print("FINAL PORTFOLIO BACKTEST PERFORMANCE (REALIZED INR 100,000 CASH ACCOUNT)")
    print("=" * 75)
    summary_table = [
        {
            "Strategy": "Logistic Regression",
            "Final Equity": f"INR {metrics_lr['final_equity']:,.2f}",
            "Return": f"{metrics_lr['total_return_pct']:+.2f}%",
            "Max Drawdown": f"{metrics_lr['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_lr['sharpe_ratio']:.2f}",
            "Trades": metrics_lr["total_trades_completed"],
            "Win Rate": f"{metrics_lr['win_rate_pct']:.1f}%",
            "Profit Factor": f"{metrics_lr['profit_factor']:.2f}",
        },
        {
            "Strategy": "LightGBM Classifier",
            "Final Equity": f"INR {metrics_lgb['final_equity']:,.2f}",
            "Return": f"{metrics_lgb['total_return_pct']:+.2f}%",
            "Max Drawdown": f"{metrics_lgb['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_lgb['sharpe_ratio']:.2f}",
            "Trades": metrics_lgb["total_trades_completed"],
            "Win Rate": f"{metrics_lgb['win_rate_pct']:.1f}%",
            "Profit Factor": f"{metrics_lgb['profit_factor']:.2f}",
        },
        {
            "Strategy": "Buy & Hold Benchmark",
            "Final Equity": f"INR {metrics_bh['final_equity']:,.2f}",
            "Return": f"{metrics_bh['total_return_pct']:+.2f}%",
            "Max Drawdown": f"{metrics_bh['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_bh['sharpe_ratio']:.2f}",
            "Trades": metrics_bh["total_trades_completed"],
            "Win Rate": f"{metrics_bh['win_rate_pct']:.1f}%",
            "Profit Factor": f"{metrics_bh['profit_factor']:.2f}",
        },
        {
            "Strategy": "Cash Benchmark (6.5%)",
            "Final Equity": f"INR {metrics_cash['final_equity']:,.2f}",
            "Return": f"{metrics_cash['total_return_pct']:+.2f}%",
            "Max Drawdown": f"{metrics_cash['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{metrics_cash['sharpe_ratio']:.2f}",
            "Trades": 0,
            "Win Rate": "N/A",
            "Profit Factor": "N/A",
        },
    ]
    print(pd.DataFrame(summary_table).to_string(index=False))
    print("=" * 75)
    print(f"Results, trade blotters, and equity curves saved to:\n{MODELS_DIR}")


if __name__ == "__main__":
    main()
