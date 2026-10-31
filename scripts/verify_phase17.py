import pandas as pd
import numpy as np

# Load Phase 17 files
ledger_path = 'reports/phase17_portfolio_reconciliation/trade_ledger_reconciled.csv'
reconciliation_path = 'reports/phase17_portfolio_reconciliation/accounting_reconciliation.csv'
equity_path = 'reports/phase17_portfolio_reconciliation/equity_curve_reconciled.csv'
comparison_path = 'reports/phase17_portfolio_reconciliation/strategy_comparison.csv'
reward_risk_path = 'reports/phase17_portfolio_reconciliation/reward_risk_analysis.csv'

df_trades = pd.read_csv(ledger_path)
df_reconcil = pd.read_csv(reconciliation_path)
df_equity = pd.read_csv(equity_path)
df_strat = pd.read_csv(comparison_path)
df_rr = pd.read_csv(reward_risk_path)

print("=== 1. TRADE LEDGER DETAILED AUDIT ===")
print(df_trades[['trade_id', 'symbol', 'entry_date', 'exit_date', 'shares', 'cost_basis_inr', 'gross_proceeds_inr', 'gross_pnl_inr', 'total_commissions_inr', 'dividends_credited_inr', 'net_pnl_inr']])

# Arithmetic verification per trade
for idx, r in df_trades.iterrows():
    # Cost basis includes buy commission (allow up to 1 INR due to 2-decimal rounded price in CSV)
    expected_cost_basis = (r['shares'] * r['entry_exec_price']) + r['buy_commission_inr']
    assert abs(r['cost_basis_inr'] - expected_cost_basis) < 1.0, f"Mismatch in cost basis for trade {r['trade_id']}: {r['cost_basis_inr']} vs {expected_cost_basis}"
    
    expected_gross_proceeds = r['shares'] * r['exit_exec_price']
    assert abs(r['gross_proceeds_inr'] - expected_gross_proceeds) < 1.0, f"Mismatch in gross proceeds for trade {r['trade_id']}"
    
    expected_gross_pnl = expected_gross_proceeds - (r['shares'] * r['entry_exec_price'])
    assert abs(r['gross_pnl_inr'] - expected_gross_pnl) < 1.0, f"Mismatch in gross pnl for trade {r['trade_id']}"
    
    expected_net_pnl = r['gross_pnl_inr'] - r['total_commissions_inr'] + r['dividends_credited_inr']
    assert abs(r['net_pnl_inr'] - expected_net_pnl) < 0.05, f"Mismatch in net pnl for trade {r['trade_id']}"

print("All trade-level arithmetic verified: Gross = Proceeds - Share Cost, Net = Gross - Comm + Div.")

# Recalculate portfolio level metrics
total_trades = len(df_trades)
gross_pnl_sum = df_trades['gross_pnl_inr'].sum()
commissions_sum = df_trades['total_commissions_inr'].sum()
dividends_sum = df_trades['dividends_credited_inr'].sum()
net_pnl_sum = df_trades['net_pnl_inr'].sum()

wins = df_trades[df_trades['net_pnl_inr'] > 0]
losses = df_trades[df_trades['net_pnl_inr'] < 0]
win_count = len(wins)
loss_count = len(losses)
win_rate = win_count / total_trades

avg_win = wins['net_pnl_inr'].mean()
avg_loss = losses['net_pnl_inr'].mean()
payoff_ratio = avg_win / abs(avg_loss)
net_expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

initial_equity = 100000.00
final_equity_reconstructed = initial_equity + net_pnl_sum
final_equity_curve = df_equity['total_portfolio_equity_inr'].iloc[-1]

print("\n=== 2. INDEPENDENT RECALCULATION RESULTS ===")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {win_count} ({win_rate:.4%})")
print(f"Losing Trades: {loss_count} ({1 - win_rate:.4%})")
print(f"Sum of Gross PnL: INR {gross_pnl_sum:.2f}")
print(f"Sum of Total Commissions: INR {commissions_sum:.2f}")
print(f"Sum of Dividends Credited: INR {dividends_sum:.2f}")
print(f"Sum of Net PnL: INR {net_pnl_sum:.2f}")
print(f"Average Winning Trade: INR {avg_win:.2f}")
print(f"Average Losing Trade: INR {avg_loss:.2f}")
print(f"Realized Payoff Ratio: {payoff_ratio:.4f}")
print(f"Net Expectancy per trade: INR {net_expectancy:.2f}")
print(f"Starting Capital: INR {initial_equity:.2f}")
print(f"Final Equity (Ledger Sum): INR {final_equity_reconstructed:.2f}")
print(f"Final Equity (Equity Curve): INR {final_equity_curve:.2f}")
print(f"Discrepancy (Ledger vs Equity Curve): INR {abs(final_equity_reconstructed - final_equity_curve):.5f}")

# Cross-check with strategy_comparison.csv
row_p17_strat = df_strat[df_strat['strategy_id'] == 'STRAT_P15B_TH042'].iloc[0]
print("\n=== 3. CROSS-CHECK WITH strategy_comparison.csv ===")
print(f"Reported Final Equity: {row_p17_strat['final_equity_inr']}")
print(f"Reported Net Return: {row_p17_strat['net_return_pct']}%")
print(f"Reported Total Trades: {row_p17_strat['total_trades']}")
print(f"Reported Win Rate: {row_p17_strat['win_rate_pct']}%")
print(f"Reported Commissions: {row_p17_strat['commissions_inr']}")
print(f"Reported Dividends: {row_p17_strat['dividends_inr']}")

# Cross-check with reward_risk_analysis.csv
row_p17_rr = df_rr[df_rr['strategy_id'] == 'STRAT_P15B_TH042'].iloc[0]
print("\n=== 4. CROSS-CHECK WITH reward_risk_analysis.csv ===")
print(f"Reported Avg Win: {row_p17_rr['avg_win_inr']}")
print(f"Reported Avg Loss: {row_p17_rr['avg_loss_inr']}")
print(f"Reported Payoff Ratio: {row_p17_rr['realized_payoff_ratio']}")
print(f"Reported Net Expectancy: {row_p17_rr['net_expectancy_per_trade_inr']}")

print("\n=== STEP 1 VERIFICATION COMPLETE ===")
