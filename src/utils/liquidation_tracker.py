"""
Liquidation Tracker Module
Tracks open positions against maintained margin so the environment can refuse or
unwind trades before the simulated account would be liquidated.
"""

import numpy as np

class LiquidationTracker:
    """Margin bookkeeping for a set of open trades.

    Holds its own balance and trade book, separate from TradingEnvironment's.
    Every price argument is keyed by trade id so a single call can value the
    whole book at once.

    Note that open_trade() records only a magnitude, with no side, and
    calculate_unrealized_pnl() applies the long formula to it. See the "Known
    problems" section of the repository README.
    """

    def __init__(self, initial_balance, margin_rate, liquidation_threshold):
        """
        Args:
            initial_balance: Starting account balance.
            margin_rate: Fraction of notional held as margin, i.e. 1 / leverage.
            liquidation_threshold: Margin level below which liquidation is
                treated as imminent (for example 1.1 for a 110% margin level).
        """
        self.balance = initial_balance
        self.margin_rate = margin_rate
        self.liquidation_threshold = liquidation_threshold
        self.open_trades = {}

    def open_trade(self, trade_id, position_size, entry_price):
        """Record a new open position under trade_id."""
        self.open_trades[trade_id] = {
            'position_size': position_size,
            'entry_price': entry_price,
        }

    def close_trade(self, trade_id, exit_price):
        """Close trade_id at exit_price, credit the P&L, and return it.

        Returns 0 if trade_id is not open.
        """
        if trade_id in self.open_trades:
            realized_pnl = (exit_price - self.open_trades[trade_id]['entry_price']) * self.open_trades[trade_id]['position_size']
            self.balance += realized_pnl
            del self.open_trades[trade_id]
            return realized_pnl
        return 0

    def calculate_unrealized_pnl(self, current_prices):
        """Sum unrealized P&L over the open book.

        Args:
            current_prices: Mapping of trade id to that trade's current price.
        """
        unrealized_pnl = 0
        for trade_id, trade in self.open_trades.items():
            unrealized_pnl += (current_prices[trade_id] - trade['entry_price']) * trade['position_size']
        return unrealized_pnl

    def calculate_margin_level(self, current_prices):
        """Return equity / used margin, or infinity when nothing is open."""
        equity = self.balance + self.calculate_unrealized_pnl(current_prices)
        used_margin = self.calculate_used_margin()
        if used_margin == 0:
            return float('inf')
        return equity / used_margin

    def calculate_used_margin(self):
        """Return the margin currently tied up by the open book."""
        used_margin = 0
        for trade in self.open_trades.values():
            used_margin += (trade['position_size'] * trade['entry_price']) * self.margin_rate
        return used_margin

    def is_liquidation_imminent(self, current_prices):
        """Return True when the margin level has fallen below the threshold."""
        margin_level = self.calculate_margin_level(current_prices)
        return margin_level < self.liquidation_threshold

    def get_trades_to_close(self, current_prices):
        """Return ids of the losing open trades, worst loss first.

        Used to pick which positions to unwind when liquidation is imminent.
        """
        trades_with_pnl = []
        for trade_id, trade in self.open_trades.items():
            unrealized_pnl = (current_prices[trade_id] - trade['entry_price']) * trade['position_size']
            if unrealized_pnl < 0:
                trades_with_pnl.append((trade_id, unrealized_pnl))
        
        trades_with_pnl.sort(key=lambda x: x[1])
        
        return [trade[0] for trade in trades_with_pnl]
