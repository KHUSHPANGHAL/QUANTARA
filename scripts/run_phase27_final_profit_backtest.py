"""
QUANTARA AI — PHASE 27: FINAL OUT-OF-SAMPLE PROFIT AND LOSS BACKTEST
Script: scripts/run_phase27_final_profit_backtest.py
Author: Lead Quantitative Verification Engineer
Objective:
  1. Implement a genuinely tradable 3-class target (DOWN: < -2%, HOLD: [-2%, +2%], UP: > +2%) across ALL eligible dates.
  2. Train fresh fold-specific 3-class models (Logistic Regression, LSTM, GRU) using strict chronological walk-forward boundaries.
  3. Execute realistic long-only portfolio trading simulation:
     - Starting Capital: Rs 1,00,000 per strategy.
     - 8 Stocks, equal max allocation (12.5% max per stock, max 100% total equity exposure).
     - Signal at Close t, entry at Open t+1, exit at Open t+21 (20 completed sessions).
     - Full transaction cost schedule (brokerage, STT, turnover fees, stamp duty, GST, slippage).
     - Cash management, dividend credits, and corporate actions.
  4. Benchmark comparison against Buy-and-Hold and Idle Cash.
  5. Accounting reconciliation and transaction cost sensitivity analysis.
  6. Export all 10 required CSV/MD deliverables to reports/phase27_final_profit/ and models to models/phase27/.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Paths
ROOT = Path("c:/Project Data/Quantara_AI_Audited_Data")
REPORT_DIR = ROOT / "reports" / "phase27_final_profit"
MODEL_DIR = ROOT / "models" / "phase27"
PROCESSED_CSV = ROOT / "data" / "processed_v3" / "multi_stock_features_v3_validated.csv"
RAW_CSV = ROOT / "data" / "raw" / "multi_stock_historical.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100000.0  # Rs 1,00,000
STOCKS = ["BHARTIARTL.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS"]
MAX_ALLOC_PER_STOCK = 1.0 / len(STOCKS)  # 12.5%
HORIZON_SESSIONS = 20
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURE_COLS = [
    "daily_return_1d", "return_5d", "return_10d", "return_20d",
    "adj_close_vs_sma20", "adj_close_vs_sma50", "adj_close_vs_ema12",
    "adj_close_vs_ema20", "adj_close_vs_ema26", "rsi_14",
    "macd_pct", "macd_signal_pct", "macd_histogram_pct",
    "bollinger_percent_b", "bollinger_width_pct"
]
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Statutory Indian Equity Delivery Fee Schedule
BASE_FEE_SCHEDULE = {
    "brokerage_pct": 0.00040,         # 0.04%
    "stt_buy_pct": 0.00125,          # 0.125%
    "stt_sell_pct": 0.00125,         # 0.125%
    "exchange_turnover_pct": 0.000035,# 0.0035%
    "sebi_turnover_pct": 0.000001,   # 0.0001%
    "stamp_duty_buy_pct": 0.00015,   # 0.015%
    "gst_rate": 0.18,                # 18% on (brokerage + exchange + sebi)
    "slippage_pct": 0.00050,         # 0.05% per leg (0.10% round-trip)
}


def get_fee_rates(fee_sched=BASE_FEE_SCHEDULE):
    gst_base = fee_sched["brokerage_pct"] + fee_sched["exchange_turnover_pct"] + fee_sched["sebi_turnover_pct"]
    buy_fee = (fee_sched["brokerage_pct"] + fee_sched["stt_buy_pct"] + fee_sched["exchange_turnover_pct"]
               + fee_sched["sebi_turnover_pct"] + fee_sched["stamp_duty_buy_pct"] + fee_sched["gst_rate"] * gst_base)
    sell_fee = (fee_sched["brokerage_pct"] + fee_sched["stt_sell_pct"] + fee_sched["exchange_turnover_pct"]
                + fee_sched["sebi_turnover_pct"] + fee_sched["gst_rate"] * gst_base)
    slippage = fee_sched["slippage_pct"]
    return buy_fee, sell_fee, slippage


def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# 3-CLASS PYTORCH MODEL ARCHITECTURES
# -----------------------------------------------------------------------------
class StockLSTMClassifier3Class(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


class StockGRUClassifier3Class(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, hn = self.gru(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


# -----------------------------------------------------------------------------
# STEP 1: PREPARE COMPLETE 3-CLASS DATASET & MARKET DATA
# -----------------------------------------------------------------------------
def load_and_prepare_complete_market():
    print("=" * 80)
    print("STEP 1: PREPARING UNFILTERED 3-CLASS DATASET & MARKET ENGINE")
    print("=" * 80)

    raw = pd.read_csv(RAW_CSV)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Dividend extraction
    raw["adj_ratio"] = raw["Adj Close"] / raw["Close"]
    div_records = []
    for sym, grp in raw.groupby("Symbol"):
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
    div_df = pd.DataFrame(div_records)
    div_lookup = {(r["Ex_Date"], r["Symbol"]): r["div_per_share"] for _, r in div_df.iterrows()}
    print(f"Extracted {len(div_df)} authentic dividend events across {len(STOCKS)} stocks.")

    # Forward 20-session return and 3-class target calculation
    fwd_records = []
    for sym, g in raw.groupby("Symbol"):
        g = g.sort_values("Date").copy().reset_index(drop=True)
        adj_factor = g["Adj Close"] / g["Close"]
        adj_open = g["Open"] * adj_factor
        adj_open_next = adj_open.shift(-1)
        adj_open_exit = adj_open.shift(-21)
        f_ret = adj_open_exit / adj_open_next - 1.0

        g["forward_return_20d"] = f_ret
        g["signal_date"] = g["Date"]
        g["entry_date"] = g["Date"].shift(-1)
        g["exit_date"] = g["Date"].shift(-21)
        fwd_records.append(g[["signal_date", "Symbol", "Open", "Close", "forward_return_20d", "entry_date", "exit_date"]])

    fwd_df = pd.concat(fwd_records, ignore_index=True)
    fwd_df = fwd_df.dropna(subset=["forward_return_20d", "entry_date", "exit_date"]).sort_values(["signal_date", "Symbol"]).reset_index(drop=True)

    # 3-Class Target: 0=DOWN (< -2%), 1=HOLD ([-2%, +2%]), 2=UP (> +2%)
    fwd_df["Target_3Class"] = np.select(
        [fwd_df["forward_return_20d"] > 0.02, fwd_df["forward_return_20d"] < -0.02],
        [2, 0],
        default=1
    )

    print(f"Total eligible 3-class observations: {len(fwd_df)}")
    counts = fwd_df["Target_3Class"].value_counts().to_dict()
    print(f"Class Breakdown: DOWN (0)={counts.get(0, 0)}, HOLD (1)={counts.get(1, 0)}, UP (2)={counts.get(2, 0)}")

    # Merge with technical features
    feats = pd.read_csv(PROCESSED_CSV)
    feats["Date"] = pd.to_datetime(feats["Date"])
    merged = pd.merge(feats, fwd_df, left_on=["Date", "Symbol"], right_on=["signal_date", "Symbol"], how="inner")
    merged = merged.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Price tables for simulation
    raw_pivot_open = raw.pivot(index="Date", columns="Symbol", values="Open")
    raw_pivot_close = raw.pivot(index="Date", columns="Symbol", values="Close")
    all_trading_dates = sorted(raw["Date"].unique())

    market_obj = {
        "raw_df": raw,
        "merged_df": merged,
        "opens": raw_pivot_open,
        "closes": raw_pivot_close,
        "all_trading_dates": all_trading_dates,
        "div_lookup": div_lookup,
    }
    return market_obj


# -----------------------------------------------------------------------------
# STEP 2: WALK-FORWARD FOLD SPECIFICATIONS
# -----------------------------------------------------------------------------
def get_phase27_folds():
    return [
        {
            "fold_id": 1,
            "fold_name": "Fold 1: 2023 Recovery",
            "market_regime": "Broad Post-Correction Recovery",
            "train_start": "2017-01-03", "train_end": "2022-09-30",
            "val_start": "2022-11-01", "val_end": "2022-12-30",
            "test_start": "2023-01-30", "test_end": "2023-12-29",
        },
        {
            "fold_id": 2,
            "fold_name": "Fold 2: 2024 Bull Run",
            "market_regime": "Strong Domestic Equity Liquidity",
            "train_start": "2017-01-03", "train_end": "2023-09-29",
            "val_start": "2023-11-01", "val_end": "2023-12-29",
            "test_start": "2024-02-01", "test_end": "2024-09-23",
        },
        {
            "fold_id": 3,
            "fold_name": "Fold 3: 2024-2025 Val",
            "market_regime": "Sideways Consolidation",
            "train_start": "2017-01-03", "train_end": "2024-06-28",
            "val_start": "2024-08-01", "val_end": "2024-09-23",
            "test_start": "2024-10-24", "test_end": "2025-09-19",
        },
        {
            "fold_id": 4,
            "fold_name": "Fold 4: 2025-2026 Test",
            "market_regime": "Severe Large-Cap Downtrend",
            "train_start": "2017-01-03", "train_end": "2024-09-23",
            "val_start": "2024-10-24", "val_end": "2025-09-19",
            "test_start": "2025-10-23", "test_end": "2026-09-22",
        },
    ]


# -----------------------------------------------------------------------------
# STEP 3: TRAIN 3-CLASS MODELS & GENERATE OUT-OF-SAMPLE PREDICTIONS
# -----------------------------------------------------------------------------
def train_and_predict_3class_fold(market_obj, fold_def):
    f_id = fold_def["fold_id"]
    f_name = fold_def["fold_name"]
    df = market_obj["merged_df"]

    tr_mask = (df["Date"] >= fold_def["train_start"]) & (df["Date"] <= fold_def["train_end"])
    va_mask = (df["Date"] >= fold_def["val_start"]) & (df["Date"] <= fold_def["val_end"])
    te_mask = (df["Date"] >= fold_def["test_start"]) & (df["Date"] <= fold_def["test_end"])

    tr_df = df[tr_mask].copy().reset_index(drop=True)
    va_df = df[va_mask].copy().reset_index(drop=True)
    te_df = df[te_mask].copy().reset_index(drop=True)

    # 1. Fit Scaler strictly on train features
    scaler = StandardScaler()
    scaler.fit(tr_df[FEATURE_COLS])
    joblib.dump(scaler, MODEL_DIR / f"scaler_3class_fold{f_id}.joblib")

    # Scaled feature matrices
    feat_all = df.copy()
    scaled_vals = scaler.transform(feat_all[FEATURE_COLS])
    for i, c in enumerate(FEATURE_COLS):
        feat_all[f"sc_{c}"] = scaled_vals[:, i]
    sc_cols = [f"sc_{c}" for c in FEATURE_COLS]

    stock_groups = {sym: grp.sort_values("Date").reset_index(drop=True) for sym, grp in feat_all.groupby("Symbol")}

    def build_sequences(slice_df):
        seq_list = []
        for _, row in slice_df.iterrows():
            sym = row["Symbol"]
            d = row["Date"]
            grp = stock_groups[sym]
            pos = grp.index[grp["Date"] == d][0]
            if pos >= SEQ_LEN - 1:
                seq_arr = grp.loc[pos - SEQ_LEN + 1 : pos, sc_cols].values
            else:
                available = grp.loc[:pos, sc_cols].values
                pad_len = SEQ_LEN - len(available)
                pad_arr = np.repeat(available[0:1, :], pad_len, axis=0)
                seq_arr = np.vstack([pad_arr, available])
            seq_list.append(seq_arr)
        return np.array(seq_list, dtype=np.float32)

    X_tr_seq = build_sequences(tr_df)
    X_va_seq = build_sequences(va_df)
    X_te_seq = build_sequences(te_df)

    y_tr = tr_df["Target_3Class"].values.astype(np.int64)
    y_va = va_df["Target_3Class"].values.astype(np.int64)
    y_te = te_df["Target_3Class"].values.astype(np.int64)

    X_tr_flat = X_tr_seq[:, -1, :]
    X_va_flat = X_va_seq[:, -1, :]
    X_te_flat = X_te_seq[:, -1, :]

    print(f"\n--- Fold {f_id} 3-Class Training: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)} ---")

    # 2. Train Logistic Regression (Multinomial 3-Class)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED + f_id)
    lr.fit(X_tr_flat, y_tr)
    lr_preds = lr.predict(X_te_flat)
    lr_probs = lr.predict_proba(X_te_flat)

    # 3. Compute 3-class training class weights for PyTorch
    classes, counts = np.unique(y_tr, return_counts=True)
    class_weights = len(y_tr) / (len(classes) * counts.astype(np.float32))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # 4. Train Fresh 3-Class LSTM
    print(f"Training Fresh 3-Class LSTM (Fold {f_id})...")
    torch.manual_seed(SEED + f_id * 100)
    lstm = StockLSTMClassifier3Class(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.2, num_classes=3)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    tr_ds = TensorDataset(torch.tensor(X_tr_seq, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(X_va_seq, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_ba = 0.0
    best_lstm_w = None
    p_cnt = 0
    for ep in range(1, EPOCHS + 1):
        lstm.train()
        for bx, by in tr_loader:
            optimizer.zero_grad()
            loss = criterion(lstm(bx), by)
            loss.backward()
            optimizer.step()

        lstm.eval()
        va_p, va_t = [], []
        with torch.no_grad():
            for bx, by in va_loader:
                preds = torch.argmax(lstm(bx), dim=1).cpu().numpy()
                va_p.extend(preds)
                va_t.extend(by.cpu().numpy())
        ba = balanced_accuracy_score(va_t, va_p)
        if ba > best_ba:
            best_ba = ba
            best_lstm_w = lstm.state_dict().copy()
            p_cnt = 0
        else:
            p_cnt += 1
            if p_cnt >= PATIENCE:
                break

    if best_lstm_w is not None:
        lstm.load_state_dict(best_lstm_w)
    torch.save(lstm.state_dict(), MODEL_DIR / f"lstm_3class_fold{f_id}.pt")

    # Predict LSTM
    lstm.eval()
    te_ds = TensorDataset(torch.tensor(X_te_seq, dtype=torch.float32))
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
    lstm_probs_list = []
    with torch.no_grad():
        for (bx,) in te_loader:
            probs = torch.softmax(lstm(bx), dim=1).cpu().numpy()
            lstm_probs_list.append(probs)
    lstm_probs = np.vstack(lstm_probs_list)
    lstm_preds = np.argmax(lstm_probs, axis=1)

    # 5. Train Fresh 3-Class GRU
    print(f"Training Fresh 3-Class GRU (Fold {f_id})...")
    torch.manual_seed(SEED + f_id * 200)
    gru = StockGRUClassifier3Class(input_size=len(FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.2, num_classes=3)
    optimizer_gru = torch.optim.Adam(gru.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_ba_gru = 0.0
    best_gru_w = None
    p_cnt = 0
    for ep in range(1, EPOCHS + 1):
        gru.train()
        for bx, by in tr_loader:
            optimizer_gru.zero_grad()
            loss = criterion(gru(bx), by)
            loss.backward()
            optimizer_gru.step()

        gru.eval()
        va_p, va_t = [], []
        with torch.no_grad():
            for bx, by in va_loader:
                preds = torch.argmax(gru(bx), dim=1).cpu().numpy()
                va_p.extend(preds)
                va_t.extend(by.cpu().numpy())
        ba = balanced_accuracy_score(va_t, va_p)
        if ba > best_ba_gru:
            best_ba_gru = ba
            best_gru_w = gru.state_dict().copy()
            p_cnt = 0
        else:
            p_cnt += 1
            if p_cnt >= PATIENCE:
                break

    if best_gru_w is not None:
        gru.load_state_dict(best_gru_w)
    torch.save(gru.state_dict(), MODEL_DIR / f"gru_3class_fold{f_id}.pt")

    # Predict GRU
    gru.eval()
    gru_probs_list = []
    with torch.no_grad():
        for (bx,) in te_loader:
            probs = torch.softmax(gru(bx), dim=1).cpu().numpy()
            gru_probs_list.append(probs)
    gru_probs = np.vstack(gru_probs_list)
    gru_preds = np.argmax(gru_probs, axis=1)

    # 6. Corrected Training-Prior Majority Baseline (3-Class)
    p_tr = [float((y_tr == c).sum() / len(y_tr)) for c in [0, 1, 2]]
    maj_probs = np.tile(p_tr, (len(y_te), 1))
    maj_pred_class = int(np.argmax(p_tr))
    maj_preds = np.full(len(y_te), maj_pred_class, dtype=np.int64)

    # Compile predictions dataframe for evaluation
    pred_df = te_df[["signal_date", "Symbol", "Open", "Close", "forward_return_20d", "entry_date", "exit_date", "Target_3Class"]].copy()
    pred_df["fold_id"] = f_id
    pred_df["fold_name"] = f_name
    pred_df["lr_pred"] = lr_preds
    pred_df["lr_prob_up"] = lr_probs[:, 2]
    pred_df["lstm_pred"] = lstm_preds
    pred_df["lstm_prob_up"] = lstm_probs[:, 2]
    pred_df["gru_pred"] = gru_preds
    pred_df["gru_prob_up"] = gru_probs[:, 2]
    pred_df["majority_pred"] = maj_preds
    pred_df["majority_prob_up"] = maj_probs[:, 2]

    return pred_df


# -----------------------------------------------------------------------------
# STEP 4: REALISTIC TRADING SIMULATION ENGINE (LONG-ONLY, EQUAL WEIGHT)
# -----------------------------------------------------------------------------
def run_trading_simulation(
    pred_df,
    market_obj,
    strategy_model_col,
    start_capital=STARTING_CAPITAL,
    fee_sched=BASE_FEE_SCHEDULE,
    eval_dates=None,
):
    """
    Simulates long-only equity portfolio based on model predictions.
    Rules:
      - Predict UP (class 2): Target 12.5% allocation for that stock.
      - Predict HOLD or DOWN (class 1 or 0): Target 0% (cash).
      - Max 1 position per stock; max 8 positions total (max 100% equity).
      - Signal at close t -> Enter at Open t+1 -> Exit at Open t+21 (20 completed sessions).
      - Transaction costs: buy fee, sell fee, slippage.
      - Dividends credited on ex-date.
    """
    opens = market_obj["opens"]
    closes = market_obj["closes"]
    div_lookup = market_obj["div_lookup"]
    all_dates = market_obj["all_trading_dates"]
    buy_fee, sell_fee, slippage = get_fee_rates(fee_sched)

    # Filter evaluation dates
    if eval_dates is None:
        eval_dates = sorted(pred_df["signal_date"].unique())

    start_date = eval_dates[0]
    end_date = eval_dates[-1]

    # Sub-slice of trading dates from start_date to end of evaluation
    date_slice = [d for d in all_dates if d >= start_date and d <= end_date]

    # Pre-index predictions by (signal_date, Symbol)
    pred_lookup = pred_df.set_index(["signal_date", "Symbol"])[strategy_model_col].to_dict()

    cash = float(start_capital)
    open_positions = {}  # sym -> dict(shares, entry_price, entry_date, exit_date, cost_basis)
    trade_blotter = []
    daily_equity_records = []
    total_costs_paid = 0.0
    total_dividends_earned = 0.0

    peak_equity = float(start_capital)

    for current_date in date_slice:
        # A. Process scheduled exits at OPEN
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            if current_date >= pos["exit_date"]:
                # Exit at Open
                exit_price = float(opens.loc[current_date, sym])
                shares = pos["shares"]
                gross_proceeds = shares * exit_price
                sell_friction = gross_proceeds * (sell_fee + slippage)
                net_proceeds = gross_proceeds - sell_friction
                total_costs_paid += sell_friction

                pnl = net_proceeds - pos["cost_basis"] + pos["dividends_accum"]
                ret_pct = (pnl / pos["cost_basis"]) * 100.0

                trade_blotter.append({
                    "strategy": strategy_model_col,
                    "symbol": sym,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "shares": shares,
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "cost_basis": round(pos["cost_basis"], 2),
                    "gross_proceeds": round(gross_proceeds, 2),
                    "net_proceeds": round(net_proceeds, 2),
                    "trade_costs": round(pos["buy_friction"] + sell_friction, 2),
                    "dividends_credited": round(pos["dividends_accum"], 2),
                    "net_pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "win": bool(pnl > 0),
                })

                cash += net_proceeds
                del open_positions[sym]

        # B. Credit Dividends on ex-date during holding period
        for sym, pos in open_positions.items():
            if (current_date, sym) in div_lookup and current_date > pos["entry_date"]:
                div_payout = pos["shares"] * div_lookup[(current_date, sym)]
                cash += div_payout
                pos["dividends_accum"] += div_payout
                total_dividends_earned += div_payout

        # C. Process New Entries at OPEN (signals generated at prior session close t-1)
        prev_idx = all_dates.index(current_date) - 1
        if prev_idx >= 0:
            prior_date = all_dates[prev_idx]
            # Check if any stock had an UP signal (class 2) on prior date close
            for sym in STOCKS:
                sig = pred_lookup.get((prior_date, sym), 1)  # Default HOLD if not found
                if sig == 2 and sym not in open_positions:
                    # Current portfolio equity at open
                    current_equity = cash + sum(pos["shares"] * opens.loc[current_date, s] for s, pos in open_positions.items())
                    target_alloc = current_equity * MAX_ALLOC_PER_STOCK
                    alloc_cash = min(cash, target_alloc)

                    open_price = float(opens.loc[current_date, sym])
                    effective_price = open_price * (1.0 + buy_fee + slippage)
                    shares = int(alloc_cash // effective_price)

                    if shares > 0:
                        gross_cost = shares * open_price
                        buy_friction = gross_cost * (buy_fee + slippage)
                        total_cost = gross_cost + buy_friction

                        # Schedule exit date: 20 trading sessions after entry date
                        entry_idx = all_dates.index(current_date)
                        exit_idx = min(entry_idx + HORIZON_SESSIONS, len(all_dates) - 1)
                        exit_date = all_dates[exit_idx]

                        cash -= total_cost
                        total_costs_paid += buy_friction

                        open_positions[sym] = {
                            "shares": shares,
                            "entry_price": open_price,
                            "entry_date": current_date,
                            "exit_date": exit_date,
                            "cost_basis": total_cost,
                            "buy_friction": buy_friction,
                            "dividends_accum": 0.0,
                        }

        # D. Daily Valuation at CLOSE
        holdings_val = sum(pos["shares"] * float(closes.loc[current_date, s]) for s, pos in open_positions.items())
        total_equity = cash + holdings_val
        equity_exposure = holdings_val / total_equity if total_equity > 0 else 0.0

        if total_equity > peak_equity:
            peak_equity = total_equity
        drawdown = (total_equity - peak_equity) / peak_equity

        daily_equity_records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "strategy": strategy_model_col,
            "cash": round(cash, 2),
            "holdings_value": round(holdings_val, 2),
            "total_equity": round(total_equity, 2),
            "equity_exposure_pct": round(equity_exposure * 100.0, 2),
            "active_positions": len(open_positions),
            "drawdown_pct": round(drawdown * 100.0, 2),
        })

    # Liquidate remaining open positions at final close
    final_date = date_slice[-1]
    for sym, pos in list(open_positions.items()):
        exit_price = float(closes.loc[final_date, sym])
        gross_proceeds = pos["shares"] * exit_price
        sell_friction = gross_proceeds * (sell_fee + slippage)
        net_proceeds = gross_proceeds - sell_friction
        total_costs_paid += sell_friction

        pnl = net_proceeds - pos["cost_basis"] + pos["dividends_accum"]
        ret_pct = (pnl / pos["cost_basis"]) * 100.0

        trade_blotter.append({
            "strategy": strategy_model_col,
            "symbol": sym,
            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": final_date.strftime("%Y-%m-%d"),
            "shares": pos["shares"],
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "cost_basis": round(pos["cost_basis"], 2),
            "gross_proceeds": round(gross_proceeds, 2),
            "net_proceeds": round(net_proceeds, 2),
            "trade_costs": round(pos["buy_friction"] + sell_friction, 2),
            "dividends_credited": round(pos["dividends_accum"], 2),
            "net_pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
            "win": bool(pnl > 0),
        })
        cash += net_proceeds

    final_equity = cash
    net_pnl = final_equity - start_capital
    net_return_pct = (final_equity / start_capital - 1.0) * 100.0

    return {
        "strategy": strategy_model_col,
        "start_capital": start_capital,
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "net_return_pct": net_return_pct,
        "total_costs_paid": total_costs_paid,
        "total_dividends_earned": total_dividends_earned,
        "trade_blotter": trade_blotter,
        "daily_equity": daily_equity_records,
    }


# -----------------------------------------------------------------------------
# STEP 5: BUY-AND-HOLD & IDLE CASH BENCHMARKS
# -----------------------------------------------------------------------------
def run_buy_and_hold_benchmark(market_obj, start_capital=STARTING_CAPITAL, fee_sched=BASE_FEE_SCHEDULE, eval_dates=None):
    opens = market_obj["opens"]
    closes = market_obj["closes"]
    div_lookup = market_obj["div_lookup"]
    all_dates = market_obj["all_trading_dates"]
    buy_fee, sell_fee, slippage = get_fee_rates(fee_sched)

    if eval_dates is None:
        eval_dates = sorted(market_obj["merged_df"]["signal_date"].unique())

    start_date = eval_dates[0]
    end_date = eval_dates[-1]
    date_slice = [d for d in all_dates if d >= start_date and d <= end_date]

    alloc_per_stock = start_capital / len(STOCKS)
    cash = float(start_capital)
    positions = {}
    total_costs = 0.0
    total_divs = 0.0

    # Enter all stocks on day 1 at OPEN
    first_date = date_slice[0]
    for sym in STOCKS:
        op = float(opens.loc[first_date, sym])
        eff_price = op * (1.0 + buy_fee + slippage)
        shares = int(alloc_per_stock // eff_price)
        gross = shares * op
        friction = gross * (buy_fee + slippage)
        cost = gross + friction
        cash -= cost
        total_costs += friction
        positions[sym] = {"shares": shares, "entry_price": op, "cost_basis": cost}

    daily_records = []
    peak_eq = float(start_capital)

    for d in date_slice:
        # Credit dividends
        for sym, pos in positions.items():
            if (d, sym) in div_lookup and d > first_date:
                div_p = pos["shares"] * div_lookup[(d, sym)]
                cash += div_p
                total_divs += div_p

        hold_val = sum(pos["shares"] * float(closes.loc[d, s]) for s, pos in positions.items())
        eq = cash + hold_val
        if eq > peak_eq:
            peak_eq = eq
        dd = (eq - peak_eq) / peak_eq
        daily_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "strategy": "Buy_and_Hold",
            "cash": round(cash, 2),
            "holdings_value": round(hold_val, 2),
            "total_equity": round(eq, 2),
            "equity_exposure_pct": round(hold_val / eq * 100.0, 2),
            "active_positions": len(positions),
            "drawdown_pct": round(dd * 100.0, 2),
        })

    # Liquidate at final close
    final_d = date_slice[-1]
    for sym, pos in positions.items():
        cp = float(closes.loc[final_d, sym])
        gross = pos["shares"] * cp
        friction = gross * (sell_fee + slippage)
        net = gross - friction
        total_costs += friction
        cash += net

    final_equity = cash
    return {
        "strategy": "Buy_and_Hold",
        "start_capital": start_capital,
        "final_equity": final_equity,
        "net_pnl": final_equity - start_capital,
        "net_return_pct": (final_equity / start_capital - 1.0) * 100.0,
        "total_costs_paid": total_costs,
        "total_dividends_earned": total_divs,
        "trade_blotter": [],
        "daily_equity": daily_records,
    }


# -----------------------------------------------------------------------------
# STEP 6: EXECUTE FULL SUITE & SENSITIVITY ANALYSIS
# -----------------------------------------------------------------------------
def execute_phase27_backtest():
    market_obj = load_and_prepare_complete_market()
    folds = get_phase27_folds()

    fold_predictions = []
    for f in folds:
        f_preds = train_and_predict_3class_fold(market_obj, f)
        fold_predictions.append(f_preds)

    all_preds_df = pd.concat(fold_predictions, ignore_index=True)
    all_preds_df.to_csv(REPORT_DIR / "phase27_prediction_and_execution_checks.csv", index=False)
    print("Saved phase27_prediction_and_execution_checks.csv")

    strategies = [
        ("Logistic Regression", "lr_pred"),
        ("LSTM", "lstm_pred"),
        ("GRU", "gru_pred"),
    ]

    # 1. Continuous Multi-Year Simulation (Pooled Out-of-Sample: 2023 to 2026)
    print("\n" + "=" * 80)
    print("STEP 2: CONTINUOUS MULTI-YEAR PORTFOLIO BACKTEST (2023 TO 2026)")
    print("=" * 80)

    pooled_results = {}
    all_trade_blotters = []
    all_daily_curves = []

    for strat_label, col in strategies:
        sim = run_trading_simulation(all_preds_df, market_obj, col, start_capital=STARTING_CAPITAL)
        pooled_results[strat_label] = sim
        for t in sim["trade_blotter"]:
            t["strategy_name"] = strat_label
            all_trade_blotters.append(t)
        all_daily_curves.extend(sim["daily_equity"])

    # Buy and Hold Benchmark
    bh_sim = run_buy_and_hold_benchmark(market_obj, start_capital=STARTING_CAPITAL)
    pooled_results["Buy-and-hold"] = bh_sim
    all_daily_curves.extend(bh_sim["daily_equity"])

    # Idle Cash Benchmark
    idle_daily = []
    for r in bh_sim["daily_equity"]:
        idle_daily.append({
            "date": r["date"],
            "strategy": "Idle_Cash",
            "cash": STARTING_CAPITAL,
            "holdings_value": 0.0,
            "total_equity": STARTING_CAPITAL,
            "equity_exposure_pct": 0.0,
            "active_positions": 0,
            "drawdown_pct": 0.0,
        })
    all_daily_curves.extend(idle_daily)
    pooled_results["Idle cash"] = {
        "strategy": "Idle cash",
        "start_capital": STARTING_CAPITAL,
        "final_equity": STARTING_CAPITAL,
        "net_pnl": 0.0,
        "net_return_pct": 0.0,
        "total_costs_paid": 0.0,
        "total_dividends_earned": 0.0,
        "trade_blotter": [],
        "daily_equity": idle_daily,
    }

    pd.DataFrame(all_trade_blotters).to_csv(REPORT_DIR / "phase27_trade_ledger.csv", index=False)
    print(f"Saved phase27_trade_ledger.csv ({len(all_trade_blotters)} simulated trades)")

    pd.DataFrame(all_daily_curves).to_csv(REPORT_DIR / "phase27_daily_equity_curves.csv", index=False)
    print("Saved phase27_daily_equity_curves.csv")

    # 2. Strategy Comparison Summary Table
    comparison_rows = []
    for s_name, res in pooled_results.items():
        tb = [t for t in all_trade_blotters if t.get("strategy_name") == s_name]
        n_trades = len(tb)
        n_win = sum(1 for t in tb if t["win"])
        n_loss = n_trades - n_win
        win_rate = (n_win / n_trades * 100.0) if n_trades > 0 else 0.0

        win_pnls = [t["net_pnl"] for t in tb if t["win"]]
        loss_pnls = [t["net_pnl"] for t in tb if not t["win"]]
        avg_win = np.mean(win_pnls) if len(win_pnls) > 0 else 0.0
        avg_loss = np.mean(loss_pnls) if len(loss_pnls) > 0 else 0.0
        expectancy = np.mean([t["net_pnl"] for t in tb]) if n_trades > 0 else 0.0

        daily_df = pd.DataFrame(res["daily_equity"])
        max_dd = daily_df["drawdown_pct"].min() if len(daily_df) > 0 else 0.0
        avg_exp = daily_df["equity_exposure_pct"].mean() if len(daily_df) > 0 else 0.0

        comparison_rows.append({
            "Strategy": s_name,
            "Starting capital": round(res["start_capital"], 2),
            "Final equity": round(res["final_equity"], 2),
            "Net profit/loss": round(res["net_pnl"], 2),
            "Net return": f"{res['net_return_pct']:.2f}%",
            "Max drawdown": f"{max_dd:.2f}%",
            "Trades": n_trades,
            "Win rate": f"{win_rate:.2f}%",
            "Avg win (Rs)": round(avg_win, 2),
            "Avg loss (Rs)": round(avg_loss, 2),
            "Expectancy per trade (Rs)": round(expectancy, 2),
            "Total costs paid (Rs)": round(res["total_costs_paid"], 2),
            "Dividends credited (Rs)": round(res["total_dividends_earned"], 2),
            "Avg equity exposure": f"{avg_exp:.2f}%",
        })

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(REPORT_DIR / "phase27_strategy_comparison.csv", index=False)
    print("Saved phase27_strategy_comparison.csv")
    print(df_comp[["Strategy", "Starting capital", "Final equity", "Net profit/loss", "Net return", "Max drawdown", "Trades"]].to_string(index=False))

    # 3. Fold-by-Fold Profitability Breakdown
    print("\n" + "=" * 80)
    print("STEP 3: FOLD-BY-FOLD PROFITABILITY BREAKDOWN (RS 1,00,000 PER FOLD)")
    print("=" * 80)
    fold_profit_rows = []

    for f in folds:
        f_id = f["fold_id"]
        f_name = f["fold_name"]
        regime = f["market_regime"]
        f_preds = all_preds_df[all_preds_df["fold_id"] == f_id]
        f_dates = sorted(f_preds["signal_date"].unique())

        for strat_label, col in strategies:
            sim = run_trading_simulation(f_preds, market_obj, col, start_capital=STARTING_CAPITAL, eval_dates=f_dates)
            tb = sim["trade_blotter"]
            n_t = len(tb)
            win_r = (sum(1 for t in tb if t["win"]) / n_t * 100.0) if n_t > 0 else 0.0
            daily_df = pd.DataFrame(sim["daily_equity"])
            max_dd = daily_df["drawdown_pct"].min() if len(daily_df) > 0 else 0.0

            fold_profit_rows.append({
                "fold_id": f_id,
                "fold_name": f_name,
                "market_regime": regime,
                "strategy": strat_label,
                "start_capital": STARTING_CAPITAL,
                "final_equity": round(sim["final_equity"], 2),
                "net_pnl": round(sim["net_pnl"], 2),
                "net_return_pct": round(sim["net_return_pct"], 2),
                "max_drawdown_pct": round(max_dd, 2),
                "trades": n_t,
                "win_rate_pct": round(win_r, 2),
                "costs_paid": round(sim["total_costs_paid"], 2),
                "dividends": round(sim["total_dividends_earned"], 2),
            })

        # Buy and Hold on this fold
        bh_f = run_buy_and_hold_benchmark(market_obj, start_capital=STARTING_CAPITAL, eval_dates=f_dates)
        daily_df_bh = pd.DataFrame(bh_f["daily_equity"])
        max_dd_bh = daily_df_bh["drawdown_pct"].min() if len(daily_df_bh) > 0 else 0.0
        fold_profit_rows.append({
            "fold_id": f_id,
            "fold_name": f_name,
            "market_regime": regime,
            "strategy": "Buy-and-hold",
            "start_capital": STARTING_CAPITAL,
            "final_equity": round(bh_f["final_equity"], 2),
            "net_pnl": round(bh_f["net_pnl"], 2),
            "net_return_pct": round(bh_f["net_return_pct"], 2),
            "max_drawdown_pct": round(max_dd_bh, 2),
            "trades": 8,
            "win_rate_pct": 100.0 if bh_f["net_pnl"] > 0 else 0.0,
            "costs_paid": round(bh_f["total_costs_paid"], 2),
            "dividends": round(bh_f["total_dividends_earned"], 2),
        })

    df_fold_profit = pd.DataFrame(fold_profit_rows)
    df_fold_profit.to_csv(REPORT_DIR / "phase27_fold_profitability.csv", index=False)
    print("Saved phase27_fold_profitability.csv")

    # 4. Transaction Cost Breakdown Ledger
    cost_rows = []
    for strat_label, col in strategies:
        sim = pooled_results[strat_label]
        gross_pnl = sim["net_pnl"] + sim["total_costs_paid"] - sim["total_dividends_earned"]
        cost_rows.append({
            "strategy": strat_label,
            "gross_trading_pnl": round(gross_pnl, 2),
            "total_transaction_costs": round(sim["total_costs_paid"], 2),
            "total_dividends_credited": round(sim["total_dividends_earned"], 2),
            "net_pnl": round(sim["net_pnl"], 2),
            "cost_drag_pct_of_capital": round(sim["total_costs_paid"] / STARTING_CAPITAL * 100.0, 2),
        })
    df_costs = pd.DataFrame(cost_rows)
    df_costs.to_csv(REPORT_DIR / "phase27_transaction_costs.csv", index=False)
    print("Saved phase27_transaction_costs.csv")

    # 5. Benchmark Comparison Table
    bench_rows = [
        {
            "benchmark_name": "Buy-and-hold (Equal Weight 8 Stocks)",
            "final_equity": round(pooled_results["Buy-and-hold"]["final_equity"], 2),
            "net_pnl": round(pooled_results["Buy-and-hold"]["net_pnl"], 2),
            "net_return_pct": round(pooled_results["Buy-and-hold"]["net_return_pct"], 2),
            "max_drawdown_pct": round(pd.DataFrame(pooled_results["Buy-and-hold"]["daily_equity"])["drawdown_pct"].min(), 2),
            "description": "Equal-weight portfolio held passively across 2023-2026",
        },
        {
            "benchmark_name": "Idle Cash (Zero Yield)",
            "final_equity": STARTING_CAPITAL,
            "net_pnl": 0.0,
            "net_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "description": "100% Cash holding with zero risk and zero nominal return",
        },
        {
            "benchmark_name": "Verified Liquid Fund (5.20% Net Annual Yield)",
            "final_equity": round(STARTING_CAPITAL * ((1.0 + 0.0520) ** (len(bh_sim["daily_equity"]) / 252.0)), 2),
            "net_pnl": round(STARTING_CAPITAL * (((1.0 + 0.0520) ** (len(bh_sim["daily_equity"]) / 252.0)) - 1.0), 2),
            "net_return_pct": round((((1.0 + 0.0520) ** (len(bh_sim["daily_equity"]) / 252.0)) - 1.0) * 100.0, 2),
            "max_drawdown_pct": 0.0,
            "description": "Phase 20 audited net liquid fund yield (5.20% p.a. compound)",
        },
    ]
    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv(REPORT_DIR / "phase27_benchmark_comparison.csv", index=False)
    print("Saved phase27_benchmark_comparison.csv")

    # 6. Sensitivity Analysis across Slippage & Friction Scenarios
    print("\n" + "=" * 80)
    print("STEP 4: TRANSACTION COST & SLIPPAGE SENSITIVITY ANALYSIS")
    print("=" * 80)
    scenarios = [
        ("Zero Friction (Theoretical Gross)", 0.0, 0.0),
        ("Statutory Only (0.3937% No Slippage)", 0.3937 / 200.0, 0.0),
        ("Baseline (0.3937% + 0.05% Slippage / Leg)", 0.3937 / 200.0, 0.0005),
        ("High Friction (0.3937% + 0.15% Slippage / Leg)", 0.3937 / 200.0, 0.0015),
        ("Stressed Friction (1.00% Round-Trip)", 0.0040, 0.0010),
    ]

    sens_rows = []
    for sc_name, fee_leg, slip_leg in scenarios:
        sched = {
            "brokerage_pct": fee_leg * 0.2,
            "stt_buy_pct": fee_leg * 0.6,
            "stt_sell_pct": fee_leg * 0.6,
            "exchange_turnover_pct": 0.000035,
            "sebi_turnover_pct": 0.000001,
            "stamp_duty_buy_pct": 0.00015,
            "gst_rate": 0.18,
            "slippage_pct": slip_leg,
        }
        for strat_label, col in strategies:
            sim = run_trading_simulation(all_preds_df, market_obj, col, start_capital=STARTING_CAPITAL, fee_sched=sched)
            sens_rows.append({
                "friction_scenario": sc_name,
                "strategy": strat_label,
                "final_equity": round(sim["final_equity"], 2),
                "net_pnl": round(sim["net_pnl"], 2),
                "net_return_pct": round(sim["net_return_pct"], 2),
                "total_costs": round(sim["total_costs_paid"], 2),
            })

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(REPORT_DIR / "phase27_sensitivity_analysis.csv", index=False)
    print("Saved phase27_sensitivity_analysis.csv")

    # 7. Accounting Reconciliation
    recon_rows = []
    for strat_label, col in strategies:
        sim = pooled_results[strat_label]
        tb = [t for t in all_trade_blotters if t.get("strategy_name") == strat_label]

        sum_net_pnl = sum(t["net_pnl"] for t in tb)
        sum_trade_costs = sum(t["trade_costs"] for t in tb)
        sum_trade_divs = sum(t["dividends_credited"] for t in tb)

        expected_final = STARTING_CAPITAL + sum_net_pnl
        actual_final = sim["final_equity"]
        discrepancy = abs(expected_final - actual_final)

        recon_rows.append({
            "strategy": strat_label,
            "starting_capital": STARTING_CAPITAL,
            "sum_of_trade_net_pnl": round(sum_net_pnl, 2),
            "expected_final_equity": round(expected_final, 2),
            "actual_final_equity": round(actual_final, 2),
            "discrepancy": round(discrepancy, 4),
            "audit_verdict": "PERFECT_RECONCILIATION" if discrepancy < 0.01 else "DISCREPANCY_DETECTED",
        })

    df_recon = pd.DataFrame(recon_rows)
    df_recon.to_csv(REPORT_DIR / "phase27_accounting_reconciliation.csv", index=False)
    print("Saved phase27_accounting_reconciliation.csv")
    print(df_recon.to_string(index=False))


if __name__ == "__main__":
    execute_phase27_backtest()
