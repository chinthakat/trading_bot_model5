"""
Financial Calculations Module

This module provides encapsulated functions for calculating profit/loss, net worth,
and other financial metrics used in the trading system.

Enhanced with production-ready features:
- Dual commission handling (entry + exit)
- Liquidation price calculations
- Improved floating-point precision
- Position sizing helpers
- Comprehensive error handling and logging
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Tuple, Optional

# Configure logging for financial calculations
logger = logging.getLogger(__name__)


def clean_numeric(value: Union[float, int, np.number], default: float = 0.0, warn_on_invalid: bool = False) -> float:
    """
    Clean numeric values by handling NaN and infinite values.
    
    Args:
        value: The numeric value to clean
        default: Default value to return for invalid inputs
        warn_on_invalid: Whether to log warnings for invalid values
        
    Returns:
        Cleaned float value
    """
    if pd.isna(value) or np.isinf(value):
        if warn_on_invalid:
            logger.warning(f"Invalid numeric value detected: {value}, using default: {default}")
        return default
    
    # Check for negative prices and warn
    if warn_on_invalid and value < 0 and default >= 0:
        logger.warning(f"Negative value detected: {value}, using default: {default}")
        return default
        
    return float(value)


def calculate_net_worth(balance: float, unrealized_pnl: float, precision: int = None) -> float:
    """
    Calculate net worth (total portfolio value) with optional precision control.
    
    Net Worth = Cash Balance + Unrealized P&L
    
    Args:
        balance: Current cash balance
        unrealized_pnl: Current unrealized profit/loss
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        
    Returns:
        Net worth value, optionally rounded to specified precision
    """
    balance = clean_numeric(balance, 0.0)
    unrealized_pnl = clean_numeric(unrealized_pnl, 0.0)
    
    net_worth = balance + unrealized_pnl
    
    if precision is not None:
        return round(net_worth, precision)
    return net_worth


def calculate_unrealized_pnl(
    position_size: float,
    entry_price: float,
    current_price: float,
    precision: int = None
) -> float:
    """
    Calculate unrealized profit/loss for an open position.
    
    Simplified formula that works for both long and short positions:
    unrealized_pnl = position_size * (current_price - entry_price)
    
    For Long positions (pos > 0): profit when current > entry
    For Short positions (pos < 0): profit when current < entry
    
    Args:
        position_size: Size of position (positive for long, negative for short)
        entry_price: Price at which position was entered
        current_price: Current market price
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        
    Returns:
        Unrealized P&L value, optionally rounded to specified precision
    """
    position_size = clean_numeric(position_size, 0.0)
    entry_price = clean_numeric(entry_price, 0.0, warn_on_invalid=True)
    current_price = clean_numeric(current_price, 0.0, warn_on_invalid=True)
    
    # Return 0 if no position or invalid prices
    if position_size == 0 or entry_price <= 0 or current_price <= 0:
        return 0.0
    
    # Unified formula for both long and short positions
    unrealized_pnl = position_size * (current_price - entry_price)
    
    if precision is not None:
        return round(unrealized_pnl, precision)
    return unrealized_pnl


def calculate_realized_pnl(
    position_size: float,
    entry_price: float,
    exit_price: float,
    commission_or_rate: float = 0.0,
    commission_total: Optional[float] = None,
    precision: int = None,
    # New parameters for enhanced functionality
    use_commission_rate: bool = None
) -> float:
    """
    Calculate realized profit/loss when closing a position with backward compatibility.
    
    Gross P&L calculation (unified for long/short):
    gross_pnl = position_size * (exit_price - entry_price)
    
    Commission handling (backward compatible):
    - If commission_or_rate <= 0.01 (1%), interpret as commission rate
    - If commission_or_rate > 0.01, interpret as total commission amount
    - If commission_total is provided, use it directly
    - If use_commission_rate is explicitly set, use that mode
    
    Net P&L = Gross P&L - Total Commission
    
    Args:
        position_size: Size of position being closed (positive for long, negative for short)
        entry_price: Price at which position was entered
        exit_price: Price at which position is being closed
        commission_or_rate: Either commission amount (legacy) or commission rate (enhanced)
        commission_total: Pre-calculated total commission (entry + exit), overrides other calculations
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        use_commission_rate: Force interpretation of commission_or_rate as rate (True) or amount (False)
        
    Returns:
        Realized P&L value (net of commission), optionally rounded to specified precision
    """
    position_size = clean_numeric(position_size, 0.0)
    entry_price = clean_numeric(entry_price, 0.0, warn_on_invalid=True)
    exit_price = clean_numeric(exit_price, 0.0, warn_on_invalid=True)
    commission_or_rate = clean_numeric(commission_or_rate, 0.0)
    
    # Calculate gross P&L first
    if position_size == 0 or entry_price <= 0 or exit_price <= 0:
        # Handle invalid inputs - still apply commission if provided
        if commission_total is not None:
            net_pnl = -clean_numeric(commission_total, 0.0)
        elif use_commission_rate is False or (use_commission_rate is None and commission_or_rate > 0.01):
            # Treat as fixed commission amount
            net_pnl = -commission_or_rate
        else:
            net_pnl = 0.0  # No trade, no commission
        
        if precision is not None:
            return round(net_pnl, precision)
        return net_pnl
    
    # Calculate gross P&L using unified formula
    gross_pnl = position_size * (exit_price - entry_price)
    
    # Determine commission handling mode
    if commission_total is not None:
        # Explicit total commission provided
        total_commission = clean_numeric(commission_total, 0.0)
    elif use_commission_rate is True or (use_commission_rate is None and commission_or_rate <= 0.01):
        # Treat as commission rate - calculate entry + exit commission
        entry_value = abs(position_size) * entry_price
        exit_value = abs(position_size) * exit_price
        total_commission = (entry_value + exit_value) * commission_or_rate
    else:
        # Treat as fixed commission amount (backward compatibility)
        total_commission = commission_or_rate
    
    # Net P&L = Gross P&L - Total Commission
    net_pnl = gross_pnl - total_commission
    
    if precision is not None:
        return round(net_pnl, precision)
    return net_pnl


def calculate_pnl_percentage(
    net_pnl: float,
    position_value: float,
    precision: int = None
) -> float:
    """
    Calculate P&L as a percentage of position value with optional precision control.
    
    P&L Percentage = (Net P&L / Position Value) * 100
    
    Args:
        net_pnl: Net profit/loss amount
        position_value: Total value of the position
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        
    Returns:
        P&L percentage, optionally rounded to specified precision
    """
    net_pnl = clean_numeric(net_pnl, 0.0)
    position_value = clean_numeric(position_value, 0.0)
    
    if position_value <= 0:
        return 0.0
    
    pnl_percentage = (net_pnl / position_value) * 100
    
    if precision is not None:
        return round(pnl_percentage, precision)
    return pnl_percentage


def calculate_position_value(
    position_size: float,
    price: float,
    precision: int = None
) -> float:
    """
    Calculate total value of a position with optional precision control.
    
    Position Value = |position_size| * price
    
    Args:
        position_size: Size of position
        price: Current price
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        
    Returns:
        Position value, optionally rounded to specified precision
    """
    position_size = clean_numeric(position_size, 0.0)
    price = clean_numeric(price, 0.0, warn_on_invalid=True)
    
    if price <= 0:
        return 0.0
    
    position_value = abs(position_size) * price
    
    if precision is not None:
        return round(position_value, precision)
    return position_value


def calculate_commission(
    position_value: float,
    commission_rate: float,
    precision: int = None
) -> float:
    """
    Calculate commission fees for a trade with optional precision control.
    
    Commission = Position Value * Commission Rate
    
    Args:
        position_value: Total value of the position
        commission_rate: Commission rate (e.g., 0.0004 for 0.04%)
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        
    Returns:
        Commission amount, optionally rounded to specified precision
    """
    position_value = clean_numeric(position_value, 0.0)
    commission_rate = clean_numeric(commission_rate, 0.0)
    
    commission = position_value * commission_rate
    
    if precision is not None:
        return round(commission, precision)
    return commission


def calculate_total_pnl(
    realized_pnl: float,
    unrealized_pnl: float
) -> float:
    """
    Calculate total profit/loss (realized + unrealized).
    
    Total P&L = Realized P&L + Unrealized P&L
    
    Args:
        realized_pnl: Realized profit/loss from closed positions
        unrealized_pnl: Unrealized profit/loss from open positions
        
    Returns:
        Total P&L
    """
    realized_pnl = clean_numeric(realized_pnl, 0.0)
    unrealized_pnl = clean_numeric(unrealized_pnl, 0.0)
    
    return realized_pnl + unrealized_pnl


def calculate_cumulative_return(
    current_net_worth: float,
    starting_balance: float
) -> float:
    """
    Calculate cumulative return percentage.
    
    Cumulative Return = (Current Net Worth / Starting Balance - 1) * 100
    
    Args:
        current_net_worth: Current portfolio value
        starting_balance: Initial portfolio value
        
    Returns:
        Cumulative return percentage
    """
    current_net_worth = clean_numeric(current_net_worth, 0.0)
    starting_balance = clean_numeric(starting_balance, 1.0)  # Avoid division by zero
    
    if starting_balance <= 0:
        return 0.0
    
    return (current_net_worth / starting_balance - 1) * 100


def calculate_equity(
    balance: float,
    position_size: float,
    current_price: float
) -> float:
    """
    Calculate total equity (balance + position value).
    
    Equity = Cash Balance + |Position Size| * Current Price
    
    Args:
        balance: Current cash balance
        position_size: Size of current position
        current_price: Current market price
        
    Returns:
        Total equity value
    """
    balance = clean_numeric(balance, 0.0)
    position_value = calculate_position_value(position_size, current_price)
    
    return balance + position_value


def calculate_drawdown(
    current_equity: float,
    peak_equity: float
) -> float:
    """
    Calculate drawdown percentage from peak equity.
    
    Drawdown = (Peak Equity - Current Equity) / Peak Equity
    
    Args:
        current_equity: Current equity value
        peak_equity: Historical peak equity value
        
    Returns:
        Drawdown percentage (0.0 to 1.0)
    """
    current_equity = clean_numeric(current_equity, 0.0)
    peak_equity = clean_numeric(peak_equity, 1.0)  # Avoid division by zero
    
    if peak_equity <= 0:
        return 0.0
    
    drawdown = (peak_equity - current_equity) / peak_equity
    return max(0.0, drawdown)  # Drawdown cannot be negative


def calculate_risk_reward_ratio(
    net_pnl: float,
    risk_amount: float
) -> float:
    """
    Calculate risk/reward ratio for a trade.
    
    Risk/Reward Ratio = |Net P&L| / Risk Amount
    
    Args:
        net_pnl: Net profit/loss from the trade
        risk_amount: Amount of capital at risk
        
    Returns:
        Risk/reward ratio
    """
    net_pnl = clean_numeric(net_pnl, 0.0)
    risk_amount = clean_numeric(risk_amount, 1.0)  # Avoid division by zero
    
    if risk_amount <= 0:
        return 0.0
    
    return abs(net_pnl) / risk_amount


def classify_trade_outcome(net_pnl: float) -> str:
    """
    Classify trade outcome based on net P&L.
    
    Args:
        net_pnl: Net profit/loss from the trade
        
    Returns:
        Trade classification: "WIN", "LOSS", or "BREAKEVEN"
    """
    net_pnl = clean_numeric(net_pnl, 0.0)
    
    if net_pnl > 0:
        return "WIN"
    elif net_pnl < 0:
        return "LOSS"
    else:
        return "BREAKEVEN"


def calculate_holding_period_return(
    balance_before: float,
    balance_after: float,
    precision: int = None
) -> float:
    """
    Calculate holding period return percentage with optional precision control.
    
    Holding Period Return = (Balance After - Balance Before) / Balance Before * 100
    
    Args:
        balance_before: Balance before the trade
        balance_after: Balance after the trade
        precision: Number of decimal places for rounding (default: None for backward compatibility)
        
    Returns:
        Holding period return percentage, optionally rounded to specified precision
    """
    # Clean inputs first
    clean_before = clean_numeric(balance_before, 0.0)
    clean_after = clean_numeric(balance_after, 0.0)
    
    # If either input was NaN/inf, or balance_before is invalid, return 0
    if clean_before <= 0 or pd.isna(balance_before) or pd.isna(balance_after):
        return 0.0
    
    hpr = (clean_after - clean_before) / clean_before * 100
    
    if precision is not None:
        return round(hpr, precision)
    return hpr


# =============================================================================
# NEW ENHANCED FUNCTIONS
# =============================================================================

def calculate_position_size_from_balance(
    balance: float,
    price: float,
    commission_rate: float = 0.0004,
    risk_percentage: float = 1.0,
    precision: int = 8
) -> float:
    """
    Calculate position size based on available balance and risk management.
    
    Formula: 
    effective_balance = balance * risk_percentage * (1 - commission_rate)
    position_size = effective_balance / price
    
    Args:
        balance: Available cash balance
        price: Current market price
        commission_rate: Commission rate for entry (default: 0.0004 = 0.04%)
        risk_percentage: Percentage of balance to risk (default: 1.0 = 100%)
        precision: Number of decimal places for rounding (default: 8)
        
    Returns:
        Position size rounded to specified precision
    """
    balance = clean_numeric(balance, 0.0)
    price = clean_numeric(price, 0.0, warn_on_invalid=True)
    commission_rate = clean_numeric(commission_rate, 0.0)
    risk_percentage = clean_numeric(risk_percentage, 1.0)
    
    if balance <= 0 or price <= 0:
        return 0.0
    
    # Clamp risk percentage between 0 and 1
    risk_percentage = max(0.0, min(1.0, risk_percentage))
    
    # Calculate effective balance after commission and risk management
    effective_balance = balance * risk_percentage * (1 - commission_rate)
    position_size = effective_balance / price
    
    return round(position_size, precision)


def calculate_liquidation_price(
    entry_price: float,
    leverage: float,
    is_long: bool = True,
    maintenance_margin_rate: float = 0.004,
    precision: int = 2
) -> float:
    """
    Calculate liquidation price for leveraged positions.
    
    For Long: liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
    For Short: liquidation_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
    
    Args:
        entry_price: Entry price of the position
        leverage: Leverage multiplier (e.g., 10 for 10x)
        is_long: True for long position, False for short position
        maintenance_margin_rate: Maintenance margin rate (default: 0.004 = 0.4%)
        precision: Number of decimal places for rounding (default: 2)
        
    Returns:
        Liquidation price rounded to specified precision
    """
    entry_price = clean_numeric(entry_price, 0.0, warn_on_invalid=True)
    leverage = clean_numeric(leverage, 1.0)
    maintenance_margin_rate = clean_numeric(maintenance_margin_rate, 0.004)
    
    if entry_price <= 0 or leverage <= 0:
        return 0.0
    
    # Ensure leverage is at least 1
    leverage = max(1.0, leverage)
    
    if is_long:
        # Long liquidation: price falls below safe threshold
        liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
    else:
        # Short liquidation: price rises above safe threshold  
        liquidation_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
    
    return round(max(0.0, liquidation_price), precision)


def is_position_liquidated(
    entry_price: float,
    current_price: float,
    leverage: float,
    is_long: bool = True,
    maintenance_margin_rate: float = 0.004
) -> bool:
    """
    Check if a leveraged position would be liquidated at current price.
    
    Args:
        entry_price: Entry price of the position
        current_price: Current market price
        leverage: Leverage multiplier
        is_long: True for long position, False for short position
        maintenance_margin_rate: Maintenance margin rate (default: 0.004 = 0.4%)
        
    Returns:
        True if position would be liquidated, False otherwise
    """
    liquidation_price = calculate_liquidation_price(
        entry_price, leverage, is_long, maintenance_margin_rate
    )
    
    if liquidation_price <= 0:
        return False
    
    current_price = clean_numeric(current_price, 0.0)
    
    if is_long:
        # Long position liquidated if current price falls below liquidation price
        return current_price <= liquidation_price
    else:
        # Short position liquidated if current price rises above liquidation price
        return current_price >= liquidation_price


def calculate_partial_close_pnl(
    original_position_size: float,
    close_size: float,
    entry_price: float,
    exit_price: float,
    commission_rate: float = 0.0004,
    precision: int = 8
) -> Tuple[float, float]:
    """
    Calculate P&L for partial position close and remaining position size.
    
    Args:
        original_position_size: Original position size
        close_size: Size of position being closed (must be same sign as original)
        entry_price: Entry price of the position
        exit_price: Exit price for the partial close
        commission_rate: Commission rate (default: 0.0004 = 0.04%)
        precision: Number of decimal places for rounding (default: 8)
        
    Returns:
        Tuple of (realized_pnl, remaining_position_size)
    """
    original_position_size = clean_numeric(original_position_size, 0.0)
    close_size = clean_numeric(close_size, 0.0)
    entry_price = clean_numeric(entry_price, 0.0, warn_on_invalid=True)
    exit_price = clean_numeric(exit_price, 0.0, warn_on_invalid=True)
    commission_rate = clean_numeric(commission_rate, 0.0)
    
    # Validate inputs
    if abs(close_size) > abs(original_position_size):
        logger.warning(f"Close size ({close_size}) exceeds original position size ({original_position_size})")
        close_size = original_position_size  # Close entire position
    
    # Check if close_size has same sign as original position
    if original_position_size != 0 and np.sign(close_size) != np.sign(original_position_size):
        logger.warning(f"Close size sign ({np.sign(close_size)}) doesn't match position sign ({np.sign(original_position_size)})")
        return 0.0, original_position_size  # No close executed
    
    # Calculate realized P&L for the closed portion
    realized_pnl = calculate_realized_pnl(
        close_size, entry_price, exit_price, commission_rate, precision=precision
    )
    
    # Calculate remaining position size
    remaining_position_size = round(original_position_size - close_size, precision)
    
    return realized_pnl, remaining_position_size


def calculate_margin_requirement(
    position_size: float,
    price: float,
    leverage: float,
    precision: int = 8
) -> float:
    """
    Calculate margin requirement for a leveraged position.
    
    Margin Requirement = (|Position Size| * Price) / Leverage
    
    Args:
        position_size: Size of the position
        price: Current price
        leverage: Leverage multiplier
        precision: Number of decimal places for rounding (default: 8)
        
    Returns:
        Required margin amount
    """
    position_value = calculate_position_value(position_size, price)
    leverage = clean_numeric(leverage, 1.0)
    
    if leverage <= 0:
        leverage = 1.0
    
    margin_requirement = position_value / leverage
    return round(margin_requirement, precision)


def calculate_effective_leverage(
    position_value: float,
    account_balance: float,
    precision: int = 2
) -> float:
    """
    Calculate effective leverage based on position value and account balance.
    
    Effective Leverage = Position Value / Account Balance
    
    Args:
        position_value: Total value of positions
        account_balance: Total account balance
        precision: Number of decimal places for rounding (default: 2)
        
    Returns:
        Effective leverage ratio
    """
    position_value = clean_numeric(position_value, 0.0)
    account_balance = clean_numeric(account_balance, 1.0)  # Avoid division by zero
    
    if account_balance <= 0:
        return 0.0
    
    effective_leverage = position_value / account_balance
    return round(effective_leverage, precision)
