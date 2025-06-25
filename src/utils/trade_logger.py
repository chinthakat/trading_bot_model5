#!/usr/bin/env python3
"""
Trade Logger Module
Logs detailed trading activity including positions, P&L, and portfolio state
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import numpy as np

class TradeLogger:
    """
    Comprehensive trade logging system
    Tracks all trading activities with detailed metrics
    """
    def __init__(self, log_dir: str = "logs/trades", session_name: Optional[str] = None, 
                 enable_logging: bool = True, log_frequency: int = 10, enable_console_logging: bool = True):
        """
        Initialize trade logger
        
        Args:
            log_dir: Directory to store trade logs
            session_name: Optional session name for the log file            enable_logging: Whether to enable file logging
            log_frequency: How often to save to file (every N trades)
            enable_console_logging: Whether to log to console
        """
        self.enable_logging = enable_logging
        self.log_frequency = max(1, log_frequency)  # Ensure at least 1
        self.enable_console_logging = enable_console_logging
        
        self.log_dir = Path(log_dir)
        if self.enable_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use generic filenames based on session type
        # Extract session type from session_name if provided
        if session_name and ('training' in session_name.lower() or 'test' in session_name.lower()):
            if 'training' in session_name.lower():
                log_filename = "training_trades"
            else:
                log_filename = "testing_trades"
        else:
            # Default to generic trading log
            log_filename = "trades"
        
        self.session_name = session_name
        self.log_filename = log_filename
        
        if self.enable_logging:
            self.log_file = self.log_dir / f"{self.log_filename}.csv"
            self.summary_file = self.log_dir / f"{self.log_filename}_summary.json"
        else:
            self.log_file = None
            self.summary_file = None
        
        # Track if file has been initialized (check if CSV file exists and has header)
        self.file_initialized = self.log_file.exists() if self.enable_logging else False
          # Initialize trade log DataFrame
        self.trade_columns = [
            'episode',
            'batch',
            'timestamp',
            'datetime',
            'market_datetime',  # Add market datetime column
            'action',  # BUY, SELL, HOLD, CLOSE
            'symbol',
            'price',
            'size',
            'side',  # LONG, SHORT, FLAT
            'cash_balance',
            'btc_balance',
            'position_value',
            'net_worth',
            'unrealized_pnl',
            'realized_pnl',
            'total_pnl',
            'commission',
            'drawdown',
            'trade_reason',
            'confidence',
            'episode_step',
            'cumulative_return'
        ]
          # Initialize empty DataFrame
        self.trades_df = pd.DataFrame(columns=self.trade_columns)
        
        # Initialize trades list for save_session compatibility
        self.trades = []
        self.portfolio_history = []
          # Trade statistics
        self.trade_stats = {
            'total_logged_actions': 0,  # All logged actions including invalid ones
            'total_trades': 0,  # Only valid trades (BUY, SELL, CLOSE actions)
            'invalid_actions': 0,  # Count of invalid actions
            'winning_trades': 0,
            'losing_trades': 0,
            'total_commission': 0.0,
            'max_drawdown': 0.0,
            'max_profit': 0.0,
            'starting_balance': 0.0,
            'current_balance': 0.0,
            'session_start': datetime.now().isoformat(),
            'session_end': None
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized TradeLogger: {self.log_file}")
    def log_trade(self, 
                  action: str,
                  symbol: str,
                  price: float,
                  size: float,
                  cash_balance: float,
                  btc_balance: float,
                  position_value: float,
                  net_worth: float,
                  unrealized_pnl: float = 0.0,
                  realized_pnl: float = 0.0,
                  commission: float = 0.0,
                  drawdown: float = 0.0,
                  trade_reason: str = "",
                  confidence: float = 1.0,
                  episode_step: int = 0,
                  episode: int = 0,
                  batch: int = 0,
                  market_timestamp: Optional[pd.Timestamp] = None,
                  **kwargs) -> None:
        """
        Log a trading activity with proper trade classification
        
        Args:
            action: Trading action (BUY, SELL, HOLD, CLOSE)
            symbol: Trading symbol
            price: Execution price
            size: Trade size (positive for long, negative for short)
            cash_balance: Current cash balance
            btc_balance: Current BTC balance (or asset balance)
            position_value: Current position value
            net_worth: Total portfolio value
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            commission: Commission paid
            drawdown: Current drawdown
            trade_reason: Reason for the trade
            confidence: Trade confidence level
            episode_step: Current episode step
            market_timestamp: Market timestamp from trading data (if None, uses current time)
        """
        
        # Validate and clean input data
        def clean_numeric(value, default=0.0):
            """Clean numeric values, replacing NaN/inf with default"""
            if pd.isna(value) or np.isinf(value):
                return default
            return float(value)
        
        # Clean all numeric inputs
        price = clean_numeric(price, 0.0)
        size = clean_numeric(size, 0.0)
        cash_balance = clean_numeric(cash_balance, 0.0)
        btc_balance = clean_numeric(btc_balance, 0.0)
        position_value = clean_numeric(position_value, 0.0)
        net_worth = clean_numeric(net_worth, 0.0)
        unrealized_pnl = clean_numeric(unrealized_pnl, 0.0)
        realized_pnl = clean_numeric(realized_pnl, 0.0)
        commission = clean_numeric(commission, 0.0)
        drawdown = clean_numeric(drawdown, 0.0)
        confidence = clean_numeric(confidence, 1.0)
        
        # Skip logging if critical values are invalid
        if price <= 0:
            self.logger.warning(f"Skipping trade log due to invalid price: {price}")
            return
        
        # Handle market timestamp with validation to prevent future dates
        current_time = datetime.now()
        
        if market_timestamp is not None:
            if isinstance(market_timestamp, pd.Timestamp):
                trade_time = market_timestamp.to_pydatetime()
            else:
                trade_time = pd.to_datetime(market_timestamp).to_pydatetime()
            
            # Check if market timestamp is in the future
            if trade_time > current_time:
                self.logger.warning(f"Market timestamp {trade_time} is in future, adjusting to historical")
                # Shift to be historical - subtract the difference plus some buffer
                time_diff = (trade_time - current_time).total_seconds()
                historical_time = current_time - pd.Timedelta(days=30) - pd.Timedelta(seconds=time_diff)
                trade_time = historical_time
                self.logger.info(f"Adjusted timestamp to: {trade_time}")
        else:
            # Use current time if no market timestamp provided
            trade_time = current_time
        
        # Ensure trade_time is not in the future
        if trade_time > current_time:
            trade_time = current_time - pd.Timedelta(days=1)  # Default to yesterday
        
        # Determine position side
        if size > 0:
            side = "LONG"
        elif size < 0:
            side = "SHORT"
        else:
            side = "FLAT"
        
        # Calculate total P&L
        total_pnl = unrealized_pnl + realized_pnl
          # Calculate cumulative return with validation
        if self.trade_stats['starting_balance'] == 0:
            # Use cash balance for starting balance if this is first trade
            # This avoids using post-trade net worth as starting balance
            self.trade_stats['starting_balance'] = max(cash_balance, 1.0)
            cumulative_return = 0.0
        elif self.trade_stats['starting_balance'] > 0:
            cumulative_return = (net_worth / self.trade_stats['starting_balance'] - 1) * 100
        else:
            cumulative_return = 0.0
          # Update statistics with proper validation
        self.trade_stats['total_logged_actions'] += 1
        
        # Check if this is a valid action or invalid action
        is_invalid_action = any(invalid_type in action for invalid_type in ['_INVALID', 'INVALID_ACTION'])
        
        if is_invalid_action:
            self.trade_stats['invalid_actions'] += 1
        else:
            # Only count valid actions as trades
            self.trade_stats['total_trades'] += 1
            
            # Only add commission for valid actions
            self.trade_stats['total_commission'] = clean_numeric(
                self.trade_stats.get('total_commission', 0.0) + commission, 0.0
            )
        
        # Update max profit and drawdown
        if total_pnl > self.trade_stats.get('max_profit', 0.0):
            self.trade_stats['max_profit'] = total_pnl
        
        if drawdown > self.trade_stats.get('max_drawdown', 0.0):
            self.trade_stats['max_drawdown'] = drawdown
        
        # Classify trade as winning/losing if it involves a position change
        if action in ['SELL', 'CLOSE'] and realized_pnl != 0:
            if realized_pnl > 0:
                self.trade_stats['winning_trades'] = self.trade_stats.get('winning_trades', 0) + 1
            else:
                self.trade_stats['losing_trades'] = self.trade_stats.get('losing_trades', 0) + 1
          # Update current balance (use cash balance, not net worth which includes leveraged positions)
        self.trade_stats['current_balance'] = cash_balance
          # Create trade record with validated timestamp
        trade_record = {
            'episode': episode,
            'batch': batch,
            'timestamp': trade_time.timestamp(),
            'datetime': trade_time.strftime('%Y-%m-%d %H:%M:%S'),
            'market_datetime': trade_time.strftime('%Y-%m-%d %H:%M:%S'),  # Use same validated time
            'action': action,
            'symbol': symbol,
            'price': round(price, 8),
            'size': round(size, 8),
            'side': side,
            'cash_balance': round(cash_balance, 2),
            'btc_balance': round(btc_balance, 8),
            'position_value': round(position_value, 2),
            'net_worth': round(net_worth, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'realized_pnl': round(realized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'commission': round(commission, 4),
            'drawdown': round(drawdown, 4),
            'trade_reason': trade_reason,
            'confidence': round(confidence, 2),
            'episode_step': episode_step,
            'cumulative_return': round(cumulative_return, 4)
        }
          # Add any additional kwargs
        trade_record.update(kwargs)
          # Add to DataFrame and trades list
        new_row = pd.DataFrame([trade_record])
        self.trades_df = pd.concat([self.trades_df, new_row], ignore_index=True)
        self.trades.append(trade_record)
        
        # Save to file periodically based on configured frequency
        if self.enable_logging and len(self.trades_df) % self.log_frequency == 0:
            self._save_trade_log()
        
        # Log to console with timestamp info (if enabled)
        if self.enable_console_logging:
            self.logger.info(
                f"TRADE [{trade_time.strftime('%Y-%m-%d %H:%M:%S')}]: {action} {abs(size):.4f} {symbol} @ {price:.2f} | "
                f"Balance: ${cash_balance:.2f} | BTC: {btc_balance:.6f} | "
                f"Net Worth: ${net_worth:.2f} | P&L: ${total_pnl:.2f}"
            )
    
    def log_position_update(self,
                           symbol: str,
                           price: float,
                           size: float,
                           cash_balance: float,
                           btc_balance: float,
                           position_value: float,
                           net_worth: float,
                           unrealized_pnl: float = 0.0,
                           episode_step: int = 0) -> None:
        """
        Log position update without actual trade
        """
        self.log_trade(
            action="HOLD",
            symbol=symbol,
            price=price,
            size=size,
            cash_balance=cash_balance,
            btc_balance=btc_balance,
            position_value=position_value,
            net_worth=net_worth,
            unrealized_pnl=unrealized_pnl,
            trade_reason="Position Update",
            episode_step=episode_step
        )
    
    def _update_stats(self, trade_record: Dict[str, Any]) -> None:
        """Update trade statistics"""
        action = trade_record['action']
        
        if action in ['BUY', 'SELL']:
            self.trade_stats['total_trades'] += 1
            
            # Track realized P&L for win/loss counting
            realized_pnl = trade_record['realized_pnl']
            if realized_pnl > 0:
                self.trade_stats['winning_trades'] += 1
            elif realized_pnl < 0:
                self.trade_stats['losing_trades'] += 1
        
        # Update commission total
        self.trade_stats['total_commission'] += trade_record['commission']
        
        # Update max drawdown
        drawdown = abs(trade_record['drawdown'])
        if drawdown > self.trade_stats['max_drawdown']:
            self.trade_stats['max_drawdown'] = drawdown
        
        # Update max profit
        total_pnl = trade_record['total_pnl']
        if total_pnl > self.trade_stats['max_profit']:
            self.trade_stats['max_profit'] = total_pnl
        
        # Update current balance        self.trade_stats['current_balance'] = trade_record['net_worth']
    
    def _save_trade_log(self) -> None:
        """Save trade log to CSV file with data validation"""
        if not self.enable_logging or self.log_file is None:
            return  # Skip saving if logging is disabled
            
        try:
            if len(self.trades_df) > 0:                # Remove any rows with critical missing data before saving
                valid_trades = self.trades_df.dropna(subset=['price', 'net_worth'])
                
                if len(valid_trades) != len(self.trades_df):
                    dropped_count = len(self.trades_df) - len(valid_trades)
                    if self.enable_console_logging:
                        self.logger.warning(f"Dropped {dropped_count} invalid trade records before saving")
                
                if len(valid_trades) > 0:
                    # Always append mode to combine logs from all batches into one complete file
                    if not self.log_file.exists():
                        # First write - create file with header
                        valid_trades.to_csv(self.log_file, index=False, mode='w')
                        if self.enable_console_logging:
                            self.logger.info(f"Created new trade log file: {self.log_file}")
                    else:
                        # File exists - append without header to combine all batches
                        valid_trades.to_csv(self.log_file, index=False, mode='a', header=False)
                        if self.enable_console_logging:
                            self.logger.debug(f"Appended {len(valid_trades)} records to {self.log_file}")
                    
                    self.file_initialized = True
            
            # Save summary statistics if summary file is configured
            if self.summary_file:
                self.trade_stats['session_end'] = datetime.now().isoformat()
                self.trade_stats['total_records'] = len(self.trades_df)
                
                with open(self.summary_file, 'w') as f:
                    json.dump(self.trade_stats, f, indent=2)
                
        except Exception as e:
            if self.enable_console_logging:
                self.logger.error(f"Failed to save trade log: {e}")
            # Try alternative save with full dataframe
            try:
                alt_file = self.log_dir / f"backup_{self.session_name}_{datetime.now().strftime('%H%M%S')}.csv"
                self.trades_df.to_csv(alt_file, index=False)
                self.logger.info(f"Backup save completed: {alt_file}")
            except Exception as e2:
                self.logger.error(f"Backup save also failed: {e2}")

    def __del__(self):
        """Destructor to ensure data is saved when object is destroyed"""
        try:
            if hasattr(self, 'trades_df') and len(self.trades_df) > 0:
                self._save_trade_log()
                self.logger.info(f"Final save on destructor: {len(self.trades_df)} records")
        except:
            pass  # Ignore errors in destructor

    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive trade summary with proper calculations
        """
        if len(self.trades_df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_commission': 0.0,
                'max_drawdown': 0.0,
                'max_profit': 0.0,
                'total_return': 0.0,
                'starting_balance': 0.0,
                'current_balance': 0.0
            }
        
        # Calculate statistics from the actual trade data
        df = self.trades_df.copy()
        
        # Clean data
        df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce').fillna(0)
        df['commission'] = pd.to_numeric(df['commission'], errors='coerce').fillna(0)
        df['net_worth'] = pd.to_numeric(df['net_worth'], errors='coerce').fillna(0)
        
        # Count actual trades (not just position updates)
        trade_actions = df[df['action'].isin(['BUY', 'SELL', 'CLOSE'])]
        closing_trades = df[df['action'].isin(['SELL', 'CLOSE'])]
        
        # Calculate wins/losses from closing trades with realized P&L
        closing_with_pnl = closing_trades[closing_trades['realized_pnl'] != 0]
        winning_trades = len(closing_with_pnl[closing_with_pnl['realized_pnl'] > 0])
        losing_trades = len(closing_with_pnl[closing_with_pnl['realized_pnl'] < 0])
          # If no closing trades with P&L, classify based on overall performance
        if winning_trades == 0 and losing_trades == 0 and len(trade_actions) > 0:
            # Use net worth progression as proxy
            starting_balance = self.trade_stats.get('starting_balance', 10000)
            current_balance = df['net_worth'].iloc[-1] if len(df) > 0 else starting_balance
            
            if current_balance > starting_balance:
                winning_trades = 1  # Overall winning session
            elif current_balance < starting_balance:
                losing_trades = 1   # Overall losing session
          # Calculate metrics
        total_trades = max(len(trade_actions), winning_trades + losing_trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_commission = df['commission'].sum()
        max_drawdown = abs(df['drawdown'].min()) if 'drawdown' in df.columns else 0.0
        max_profit = df['total_pnl'].max() if 'total_pnl' in df.columns else 0.0
        starting_balance = self.trade_stats.get('starting_balance', 10000)
        # Use cash balance, not net worth (which includes leveraged positions)
        current_balance = df['cash_balance'].iloc[-1] if len(df) > 0 else starting_balance
        total_return = ((current_balance / starting_balance - 1) * 100) if starting_balance > 0 else 0.0
        
        return {
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': float(win_rate),
            'total_commission': float(total_commission),
            'max_drawdown': float(max_drawdown),
            'max_profit': float(max_profit),
            'total_return': float(total_return),
            'starting_balance': float(starting_balance),
            'current_balance': float(current_balance),
            'avg_trade_size': float(df['position_value'].mean()) if len(df) > 0 else 0.0,
            'total_volume': float(df['position_value'].sum()) if len(df) > 0 else 0.0
        }

    def print_summary(self) -> None:
        """Print trading summary to console"""
        summary = self.get_trade_summary()
        
        print("\n" + "="*60)
        print(f"TRADING SESSION SUMMARY: {self.session_name}")
        print("="*60)
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Losing Trades: {summary['losing_trades']}")
        print(f"Win Rate: {summary['win_rate']}%")
        print(f"Total Return: {summary['total_return']}%")
        print(f"Starting Balance: ${summary['starting_balance']:,.2f}")
        print(f"Ending Balance: ${summary['current_balance']:,.2f}")
        print(f"Total P&L: ${summary['current_balance'] - summary['starting_balance']:,.2f}")
        print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
        print(f"Max Profit: ${summary['max_profit']:,.2f}")
        print(f"Total Commission: ${summary['total_commission']:,.2f}")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
        print(f"Average Trade: ${summary['avg_trade']:,.2f}")
        print("="*60)
    
    def save_session(self, filename_prefix: str = "session"):
        """Save current session data"""
        try:
            # Sanitize filename_prefix to ensure it's a valid filename
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                filename_prefix = filename_prefix.replace(char, '_')
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.trades:
                # Create trades DataFrame with proper data cleaning
                trades_data = []
                for trade in self.trades:
                    # Clean numeric values
                    cleaned_trade = {}
                    for key, value in trade.items():
                        if isinstance(value, (int, float)):
                            # Handle inf and nan values
                            if np.isinf(value) or np.isnan(value):
                                cleaned_trade[key] = 0.0
                            else:
                                cleaned_trade[key] = float(value)
                        else:
                            cleaned_trade[key] = value
                    trades_data.append(cleaned_trade)
                
                trades_df = pd.DataFrame(trades_data)
                trades_file = self.log_dir / f"{filename_prefix}_trades_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False)
                self.logger.info(f"Saved {len(self.trades)} trades to {trades_file}")
            
            if self.portfolio_history:
                # Clean portfolio history data
                portfolio_data = []
                for entry in self.portfolio_history:
                    cleaned_entry = {}
                    for key, value in entry.items():
                        if isinstance(value, (int, float)):
                            if np.isinf(value) or np.isnan(value):
                                cleaned_entry[key] = 0.0
                            else:
                                cleaned_entry[key] = float(value)
                        else:
                            cleaned_entry[key] = value
                    portfolio_data.append(cleaned_entry)
                
                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_file = self.log_dir / f"{filename_prefix}_portfolio_{timestamp}.csv"
                portfolio_df.to_csv(portfolio_file, index=False)
                self.logger.info(f"Saved portfolio history to {portfolio_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")

    def initialize_session(self, initial_balance: float) -> None:
        """
        Initialize session with starting balance before any trades
        This ensures proper tracking of starting vs current balance
        """
        if self.trade_stats['starting_balance'] == 0:
            self.trade_stats['starting_balance'] = float(initial_balance)
            self.logger.info(f"Session initialized with starting balance: ${initial_balance:.2f}")

def main():
    """Example usage of TradeLogger"""
    # Initialize logger
    logger = TradeLogger(session_name="test_session")
      # Simulate some trades
    logger.log_trade(
        action="BUY",
        symbol="BINANCEFTS_PERP_BTC_USDT",
        price=45000.0,
        size=0.1,
        cash_balance=5500.0,
        btc_balance=0.1,
        position_value=4500.0,
        net_worth=10000.0,
        commission=2.25,
        trade_reason="Strong bullish signal"
    )
    
    logger.log_trade(
        action="SELL",
        symbol="BINANCEFTS_PERP_BTC_USDT",
        price=46000.0,
        size=0.0,
        cash_balance=10098.0,
        btc_balance=0.0,
        position_value=0.0,
        net_worth=10098.0,
        realized_pnl=98.0,
        commission=2.30,
        trade_reason="Target reached"
    )
    
    # Print summary
    logger.print_summary()
    
    # Save session
    logger.save_session()

if __name__ == "__main__":
    main()
