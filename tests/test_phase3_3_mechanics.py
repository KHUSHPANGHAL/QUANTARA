"""
Quantara AI — Phase 3.3 Unit Test Suite
Covers:
1. Dividend entitlement rules:
   - Position held through prior session and sold at ex-date open (entitled).
   - Position bought on ex-date (not entitled).
   - Position sold before ex-date (not entitled).
   - Multiple shares and dividend amounts.
   - Prevention of duplicate dividend credits.
   - Non-entitlement condition negative tests.
2. Open-gap feature consistency:
   - Behavior on ex-dividend dates vs ordinary trading dates.
   - Corporate-action adjusted gap vs raw gap comparison.
"""

import unittest
import numpy as np
import pandas as pd


class MockBacktestPortfolio:
    """Mock portfolio engine testing exact dividend entitlement logic."""
    def __init__(self, div_lookup):
        self.div_lookup = div_lookup  # (Date, Symbol) -> div_per_share
        self.cash = 100000.0
        self.open_positions = {}
        self.trade_blotter = []

    def buy(self, date, symbol, shares, price):
        outflow = shares * price
        self.cash -= outflow
        self.open_positions[symbol] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": pd.Timestamp(date),
            "sessions_held": 0,
            "dividend_credited_dates": set(),
        }
        self.trade_blotter.append({
            "date": date, "symbol": symbol, "action": "BUY", "shares": shares, "net_pnl": 0.0
        })

    def process_morning_exits(self, current_date, exits):
        current_date = pd.Timestamp(current_date)
        for sym in list(exits):
            if sym in self.open_positions:
                pos = self.open_positions[sym]
                
                # Check dividend entitlement for exiting position
                div_key = (current_date, sym)
                if div_key in self.div_lookup and pos["entry_date"] < current_date:
                    credited = pos.get("dividend_credited_dates", set())
                    if current_date not in credited:
                        div_per_share = self.div_lookup[div_key]
                        div_payout = pos["shares"] * div_per_share
                        self.cash += div_payout
                        self.trade_blotter.append({
                            "date": str(current_date.date()),
                            "symbol": sym,
                            "action": "DIVIDEND",
                            "shares": pos["shares"],
                            "exec_price": div_per_share,
                            "net_pnl": div_payout,
                            "reason": "EX_DIVIDEND_PAYOUT_ON_EXIT",
                        })
                        pos["dividend_credited_dates"] = credited | {current_date}
                        
                # Execute exit
                gross_proceeds = pos["shares"] * pos["entry_price"]
                self.cash += gross_proceeds
                self.trade_blotter.append({
                    "date": str(current_date.date()), "symbol": sym, "action": "SELL",
                    "shares": pos["shares"], "net_pnl": 0.0
                })
                del self.open_positions[sym]

    def process_intraday_dividends(self, current_date):
        current_date = pd.Timestamp(current_date)
        for sym, pos in list(self.open_positions.items()):
            div_key = (current_date, sym)
            if div_key in self.div_lookup and pos["entry_date"] < current_date:
                credited = pos.get("dividend_credited_dates", set())
                if current_date not in credited:
                    div_per_share = self.div_lookup[div_key]
                    div_payout = pos["shares"] * div_per_share
                    self.cash += div_payout
                    self.trade_blotter.append({
                        "date": str(current_date.date()),
                        "symbol": sym,
                        "action": "DIVIDEND",
                        "shares": pos["shares"],
                        "exec_price": div_per_share,
                        "net_pnl": div_payout,
                        "reason": "EX_DIVIDEND_PAYOUT",
                    })
                    pos["dividend_credited_dates"] = credited | {current_date}


class TestPhase33DividendEntitlement(unittest.TestCase):
    def setUp(self):
        self.ex_date = pd.Timestamp("2026-06-10")
        self.prior_date = pd.Timestamp("2026-06-03")
        self.symbol = "INFY.NS"
        self.div_per_share = 25.0
        self.div_lookup = {(self.ex_date, self.symbol): self.div_per_share}

    def test_held_through_prior_session_and_sold_at_ex_date_open(self):
        """Case 1: Position entered prior to ex-date and sold at ex-date open MUST receive dividend."""
        pf = MockBacktestPortfolio(self.div_lookup)
        pf.buy(self.prior_date, self.symbol, shares=28, price=1180.0)
        initial_cash_before_exit = pf.cash
        
        # Morning exit on ex-date
        pf.process_morning_exits(self.ex_date, exits=[self.symbol])
        
        div_events = [t for t in pf.trade_blotter if t["action"] == "DIVIDEND"]
        self.assertEqual(len(div_events), 1, "Exiting position must receive exactly 1 dividend credit")
        self.assertEqual(div_events[0]["net_pnl"], 28 * 25.0, "Dividend payout must equal shares * div_per_share")
        self.assertNotIn(self.symbol, pf.open_positions, "Position must be closed after exit")

    def test_position_bought_on_ex_date(self):
        """Case 2: Position bought on ex-date morning MUST NOT receive dividend."""
        pf = MockBacktestPortfolio(self.div_lookup)
        pf.buy(self.ex_date, self.symbol, shares=20, price=1160.0)
        
        # Intraday check on ex-date
        pf.process_intraday_dividends(self.ex_date)
        
        div_events = [t for t in pf.trade_blotter if t["action"] == "DIVIDEND"]
        self.assertEqual(len(div_events), 0, "Position bought on ex-date must NOT receive dividend")

    def test_position_sold_before_ex_date(self):
        """Case 3: Position sold prior to ex-date MUST NOT receive dividend on ex-date."""
        pf = MockBacktestPortfolio(self.div_lookup)
        pf.buy(self.prior_date, self.symbol, shares=15, price=1180.0)
        
        # Sold on day before ex-date
        pf.process_morning_exits(self.prior_date + pd.Timedelta(days=2), exits=[self.symbol])
        
        # Ex-date arrives
        pf.process_morning_exits(self.ex_date, exits=[])
        pf.process_intraday_dividends(self.ex_date)
        
        div_events = [t for t in pf.trade_blotter if t["action"] == "DIVIDEND"]
        self.assertEqual(len(div_events), 0, "Position sold before ex-date must NOT receive dividend")

    def test_multiple_shares_and_dividend_amounts(self):
        """Case 4: Correct scaling with various share counts and dividend amounts."""
        custom_lookup = {(self.ex_date, "TCS.NS"): 57.0}
        pf = MockBacktestPortfolio(custom_lookup)
        pf.buy(self.prior_date, "TCS.NS", shares=100, price=3200.0)
        
        pf.process_morning_exits(self.ex_date, exits=["TCS.NS"])
        
        div_events = [t for t in pf.trade_blotter if t["action"] == "DIVIDEND"]
        self.assertEqual(len(div_events), 1)
        self.assertEqual(div_events[0]["net_pnl"], 100 * 57.0)

    def test_no_duplicate_dividend_credits(self):
        """Case 5: Position held through ex-date must receive dividend ONCE, not twice."""
        pf = MockBacktestPortfolio(self.div_lookup)
        pf.buy(self.prior_date, self.symbol, shares=28, price=1180.0)
        
        # Ex-date morning (not exiting)
        pf.process_morning_exits(self.ex_date, exits=[])
        # Ex-date intraday (credited once)
        pf.process_intraday_dividends(self.ex_date)
        # Ex-date intraday run again accidentally
        pf.process_intraday_dividends(self.ex_date)
        # Next morning exit on day after ex-date
        next_day = self.ex_date + pd.Timedelta(days=1)
        pf.process_morning_exits(next_day, exits=[self.symbol])
        
        div_events = [t for t in pf.trade_blotter if t["action"] == "DIVIDEND"]
        self.assertEqual(len(div_events), 1, "Position must receive dividend exactly once")
        self.assertEqual(div_events[0]["net_pnl"], 28 * 25.0)

    def test_no_credit_when_entitlement_conditions_not_met(self):
        """Case 6: Different symbol or non-ex-date must not receive any payout."""
        pf = MockBacktestPortfolio(self.div_lookup)
        pf.buy(self.prior_date, "RELIANCE.NS", shares=20, price=1300.0)
        
        pf.process_morning_exits(self.ex_date, exits=[])
        pf.process_intraday_dividends(self.ex_date)
        
        div_events = [t for t in pf.trade_blotter if t["action"] == "DIVIDEND"]
        self.assertEqual(len(div_events), 0, "Non-dividend symbol must not receive credit")


class TestOpenGapFeatureConsistency(unittest.TestCase):
    def test_gap_calculation_on_ex_dividend_date(self):
        """Test that corporate-action adjusted open gap eliminates artificial drops on ex-dividend dates."""
        prev_close_raw = 799.0
        dividend_amt = 13.0
        # Ex-dividend adjusted baseline is 786.0
        # If stock opens at 788.7, it opened +0.34% HIGHER than the ex-dividend price
        open_raw = 788.7
        
        raw_gap = (open_raw / prev_close_raw) - 1.0
        # Raw gap falsely claims a drop of -1.29%
        self.assertAlmostEqual(raw_gap, -0.012891, places=5)
        
        # Adjusted gap calculation
        adj_prev_close = prev_close_raw - dividend_amt
        adj_gap = (open_raw / adj_prev_close) - 1.0
        # Adjusted gap correctly reflects true market sentiment (+0.34%)
        self.assertAlmostEqual(adj_gap, 0.003435, places=5)
        self.assertGreater(adj_gap, 0.0, "Adjusted gap must correctly identify the positive market open")

    def test_gap_calculation_on_ordinary_date(self):
        """Test that on ordinary dates (dividend_amt = 0), adjusted gap equals raw gap."""
        prev_close_raw = 1000.0
        open_raw = 1010.0
        dividend_amt = 0.0
        
        raw_gap = (open_raw / prev_close_raw) - 1.0
        adj_gap = (open_raw / (prev_close_raw - dividend_amt)) - 1.0
        self.assertEqual(raw_gap, adj_gap, "On non-ex-dates, adjusted gap must equal raw gap exactly")


if __name__ == "__main__":
    unittest.main()

