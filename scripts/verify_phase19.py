import pandas as pd
import numpy as np

strat_metrics_path = 'reports/phase19_profitability_strategy/strategy_metrics.csv'
expectancy_path = 'reports/phase19_profitability_strategy/expectancy_and_break_even.csv'

df_metrics = pd.read_csv(strat_metrics_path)
df_exp = pd.read_csv(expectancy_path)

print("=== 1. STRATEGY METRICS SUMMARY (PHASE 19) ===")
print(df_metrics[['strategy_id', 'evaluation_window', 'executed_trades', 'trade_win_rate_pct', 'avg_win_inr', 'avg_loss_inr', 'realized_payoff_ratio', 'net_expectancy_per_trade_inr', 'final_equity_inr', 'net_return_pct']])

print("\n=== 2. EXPECTANCY & BREAK-EVEN SUMMARY (PHASE 19) ===")
print(df_exp[['strategy_id', 'evaluation_window', 'executed_trades', 'trade_win_rate_pct', 'break_even_win_rate_pct', 'win_rate_surplus_deficit_pct', 'realized_payoff_ratio', 'net_expectancy_per_trade_inr', 'is_net_expectancy_positive']])

# Verify arithmetic consistency between strategy_metrics and expectancy_and_break_even
for idx, r in df_metrics.iterrows():
    match = df_exp[(df_exp['strategy_id'] == r['strategy_id']) & (df_exp['evaluation_window'] == r['evaluation_window'])].iloc[0]
    assert r['executed_trades'] == match['executed_trades'], f"Mismatch in trade count for {r['strategy_id']}"
    assert r['trade_win_rate_pct'] == match['trade_win_rate_pct'], f"Mismatch in win rate for {r['strategy_id']}"
    assert abs(r['net_expectancy_per_trade_inr'] - match['net_expectancy_per_trade_inr']) < 0.01, f"Mismatch in expectancy for {r['strategy_id']}"
    assert abs(r['realized_payoff_ratio'] - match['realized_payoff_ratio']) < 0.01, f"Mismatch in payoff ratio for {r['strategy_id']}"

print("\n=== ALL PHASE 19 SUMMARY METRICS INTERNALLY CONSISTENT ===")

