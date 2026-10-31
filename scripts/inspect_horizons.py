import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/splits_v3')
for h in [1, 5, 20]:
    p = DATA_DIR / f'horizon_{h}d'
    print(f'=== HORIZON {h}D ===')
    for s in ['train', 'validation', 'test']:
        y = pd.read_csv(p / f'y_{s}.csv')
        counts = y[f'Target_{h}d'].value_counts(normalize=True).round(4) * 100
        mean_ret = y[f'forward_return_{h}d'].mean() * 100.0
        up_pct = counts.get('UP', 0.0)
        hold_pct = counts.get('HOLD', 0.0)
        down_pct = counts.get('DOWN', 0.0)
        print(f'{s.upper()} (N={len(y)}): UP={up_pct:.1f}%, HOLD={hold_pct:.1f}%, DOWN={down_pct:.1f}%, Mean Ret={mean_ret:.2f}%')

