#!/usr/bin/env python3
"""
Enhanced Live Trade Visualizer
Real-time visualization of trading activity with trade connection lines
"""

import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import threading
import queue

class EnhancedLiveTradeVisualizer:
    """
    Real-time trade visualization that monitors trade traces and updates plots
    Enhanced to show lines connecting open and close trades
    """
    
    def __init__(self, trace_file: str, update_interval: float = 2.0, max_points: int = 1000, log_file: str = None, market_data_file: str = None):
        """
        Initialize the live visualizer
        
        Args:
            trace_file: Path to trade traces JSONL file
            update_interval: Update frequency in seconds
            max_points: Maximum number of data points to display
            log_file: Optional path to log file for trade details
            market_data_file: Optional path to market data CSV file for continuous price timeline
        """
        self.trace_file = Path(trace_file)
        self.update_interval = update_interval
        self.max_points = max_points
        
        # Set up logging
        self.setup_logging(log_file)
        
        # Data storage
        self.trades = []
        self.prices = []  # Sparse prices from trade events
        self.market_timeline = []  # Continuous market price timeline
        self.balance_history = []
        self.equity_history = []
        
        # Trade tracking for open/close pairs
        self.open_trades = {}  # trade_id -> open trade data
        self.closed_trades = {}  # trade_id -> (open_data, close_data)
        self.trade_lines = []  # List of trade connection lines
        
        # Market data file
        self.market_data_file = market_data_file or "data/BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv"
        
        # File monitoring
        self.last_position = 0
        self.running = False
        
        # Threading
        self.data_queue = queue.Queue()
        self.monitor_thread = None
        
        print(f"Initialized Enhanced LiveTradeVisualizer for: {self.trace_file}")
        self.logger.info(f"Enhanced Live Trade Visualizer started - monitoring: {self.trace_file}")
        
        # Load continuous market data
        self.load_market_data()
        
    def setup_logging(self, log_file: str = None):
        """Set up logging for trade details"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs/visualization")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default log file name if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"trade_visualization_{timestamp}.log"
        else:
            log_file = Path(log_file)
        
        # Configure logger
        self.logger = logging.getLogger('TradeVisualizer')
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture price updates
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log all levels to file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        self.logger.addHandler(console_handler)
        
        self.log_file_path = log_file
        print(f"Trade visualization logging to: {log_file}")
        
        # Log session start
        self.logger.info("="*80)
        self.logger.info("TRADE VISUALIZATION SESSION STARTED")
        self.logger.info(f"Trace file: {self.trace_file}")
        self.logger.info(f"Update interval: {self.update_interval}s")
        self.logger.info(f"Max data points: {self.max_points}")
        self.logger.info("="*80)
    
    def parse_trade_trace(self, line: str) -> Dict[str, Any]:
        """Parse a single trade trace line"""
        try:
            trace = json.loads(line.strip())
            return trace
        except json.JSONDecodeError:
            return None
    
    def parse_timestamp(self, timestamp_value: Any, context: str = "") -> pd.Timestamp:
        """
        Robust timestamp parsing that handles multiple formats
        
        Args:
            timestamp_value: The timestamp value (string, number, or other)
            context: Context string for error messages
            
        Returns:
            Parsed pandas Timestamp or current time as fallback
        """
        if timestamp_value is None:
            print(f"Warning: No timestamp found for {context}, using current time")
            return pd.Timestamp.now()
        
        try:
            # Handle string timestamps (ISO format, etc.)
            if isinstance(timestamp_value, str):
                return pd.to_datetime(timestamp_value)
            
            # Handle numeric timestamps (assume Unix timestamp)
            elif isinstance(timestamp_value, (int, float)):
                return pd.to_datetime(timestamp_value, unit='s')
            
            # Handle other types by converting to string first
            else:
                return pd.to_datetime(str(timestamp_value))
                
        except Exception as e:
            print(f"Warning: Failed to parse timestamp '{timestamp_value}' for {context}: {e}")
            return pd.Timestamp.now()
    
    def log_trade_event(self, trade_data: Dict[str, Any]):
        """Log individual trade events with detailed information"""
        timestamp = trade_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        trade_type = trade_data['type']
        action = trade_data.get('action', 'UNKNOWN')
        side = trade_data['side']
        price = trade_data['price']
        trade_id = trade_data['trade_id']
        market_price = trade_data.get('market_price', 0)
        
        if trade_type == 'OPEN':
            size = trade_data.get('size', 0)
            balance_before = trade_data.get('balance_before', 0)
            balance_after = trade_data.get('balance_after', 0)
            
            # Show both execution price and market price for comparison
            price_info = f"Exec: ${price:.2f}"
            if market_price and abs(market_price - price) > 0.01:
                price_info += f" | Market: ${market_price:.2f}"
            
            self.logger.info(f"TRADE_OPEN | {timestamp} | {side} | {action} | {price_info} | Size: {size:.6f} | ID: {trade_id} | Balance: ${balance_before:.2f} -> ${balance_after:.2f}")
            
        elif trade_type == 'CLOSE':
            pnl = trade_data.get('pnl', 0)
            win_loss = trade_data.get('win_loss', 'UNKNOWN')
            entry_price = trade_data.get('entry_price', 0)
            balance_before = trade_data.get('balance_before', 0)
            balance_after = trade_data.get('balance_after', 0)
            
            # Show both execution price and market price for comparison
            price_info = f"Exec: ${price:.2f}"
            if market_price and abs(market_price - price) > 0.01:
                price_info += f" | Market: ${market_price:.2f}"
            
            self.logger.info(f"TRADE_CLOSE | {timestamp} | {side} | CLOSE | {price_info} | Entry: ${entry_price:.2f} | P&L: ${pnl:+.2f} | {win_loss} | ID: {trade_id} | Balance: ${balance_before:.2f} -> ${balance_after:.2f}")
    
    def log_trade_pair_completion(self, open_data: Dict[str, Any], close_data: Dict[str, Any]):
        """Log completed trade pair with comprehensive details"""
        trade_id = close_data['trade_id']
        side = close_data['side']
        entry_time = open_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        exit_time = close_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        entry_price = open_data['price']
        exit_price = close_data['price']
        pnl = close_data.get('pnl', 0)
        win_loss = close_data.get('win_loss', 'UNKNOWN')
        
        # Include market prices if different from execution prices
        entry_market_price = open_data.get('market_price', 0)
        exit_market_price = close_data.get('market_price', 0)
        
        # Calculate duration
        duration = close_data['timestamp'] - open_data['timestamp']
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        # Calculate price change percentage
        price_change_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        if side == 'SHORT':
            price_change_pct = -price_change_pct  # Invert for short positions
        
        # Create price info strings
        entry_price_info = f"${entry_price:.2f}"
        if entry_market_price and abs(entry_market_price - entry_price) > 0.01:
            entry_price_info += f" (Market: ${entry_market_price:.2f})"
            
        exit_price_info = f"${exit_price:.2f}"
        if exit_market_price and abs(exit_market_price - exit_price) > 0.01:
            exit_price_info += f" (Market: ${exit_market_price:.2f})"
            
        self.logger.info(f"TRADE_PAIR | {trade_id} | {side} | Entry: {entry_time} @ {entry_price_info} | Exit: {exit_time} @ {exit_price_info} | Duration: {duration_str} | Price Change: {price_change_pct:+.2f}% | P&L: ${pnl:+.2f} | Result: {win_loss}")
    
    def log_price_update(self, price_data: Dict[str, Any], is_loading: bool = False):
        """Log price updates (less frequent to avoid spam, unless loading)"""
        # Only log every 10th price update during live monitoring to avoid log spam
        # But log every 5th during loading to capture more historical data
        if not hasattr(self, '_price_log_counter'):
            self._price_log_counter = 0
        
        self._price_log_counter += 1
        throttle_factor = 5 if is_loading else 10
        
        if self._price_log_counter % throttle_factor == 0:
            timestamp = price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            price = price_data['price']
            source = price_data.get('source', 'unknown')
            log_type = "PRICE_LOAD" if is_loading else "PRICE_UPDATE"
            self.logger.debug(f"{log_type} | {timestamp} | ${price:.2f} | Source: {source}")
    
    def log_balance_update(self, balance_data: Dict[str, Any]):
        """Log balance and equity updates"""
        timestamp = balance_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        balance = balance_data['balance']
        equity = balance_data['equity']
        
        self.logger.info(f"BALANCE_UPDATE | {timestamp} | Balance: ${balance:.2f} | Equity: ${equity:.2f}")
    
    def log_session_summary(self):
        """Log session summary statistics"""
        if not self.trade_lines:
            self.logger.info("SESSION_SUMMARY | No completed trades in this session")
            return
            
        wins = len([t for t in self.trade_lines if t['win_loss'] == 'WIN'])
        losses = len([t for t in self.trade_lines if t['win_loss'] == 'LOSS'])
        total_trades = len(self.trade_lines)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in self.trade_lines)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        open_trades_count = len(self.open_trades)
        
        self.logger.info("="*80)
        self.logger.info("SESSION_SUMMARY")
        self.logger.info(f"Total Completed Trades: {total_trades}")
        self.logger.info(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")
        self.logger.info(f"Total P&L: ${total_pnl:.2f} | Average P&L: ${avg_pnl:.2f}")
        self.logger.info(f"Currently Open Trades: {open_trades_count}")
        
        if self.balance_history:
            initial_balance = self.balance_history[0]['balance']
            current_balance = self.balance_history[-1]['balance']
            balance_change = current_balance - initial_balance
            balance_change_pct = (balance_change / initial_balance * 100) if initial_balance > 0 else 0
            
            self.logger.info(f"Balance Change: ${initial_balance:.2f} -> ${current_balance:.2f} ({balance_change_pct:+.2f}%)")
        
        self.logger.info("="*80)
    
    def extract_trade_data(self, trace: Dict[str, Any]) -> tuple:
        """Extract trade and price data from a trace"""
        event_data = trace.get('event_data', {})
        metadata = trace.get('trace_metadata', {})
        
        trade_data = None
        price_data = None
        balance_data = None
        market_data = None
        
        # Extract trade information
        if metadata.get('event_type') == 'TRADE_OPENED':
            # Parse timestamp from entry_timestamp
            timestamp_str = event_data.get('entry_timestamp')
            timestamp = self.parse_timestamp(timestamp_str, "TRADE_OPENED entry_timestamp")
            
            # Get market data at entry for the price timeline
            market_data_at_entry = event_data.get('market_data_at_entry', {})
            market_timestamp = market_data_at_entry.get('timestamp', 0)
            market_close_price = market_data_at_entry.get('Close', 0)
            entry_price = float(event_data.get('entry_price', 0))
            
            # Debug log for market data extraction
            self.logger.debug(f"MARKET_DATA_EXTRACT | OPEN | Trade ID: {event_data.get('trade_id', '')} | Entry Price: ${entry_price:.2f} | Market Close: ${market_close_price:.2f} | Market Time: {market_timestamp} | Trade Time: {timestamp_str}")
            
            trade_data = {
                'timestamp': timestamp,
                'price': entry_price,
                'action': event_data.get('entry_action', 'BUY'),
                'side': event_data.get('side', 'LONG'),
                'size': float(event_data.get('position_size', 0)),
                'trade_id': event_data.get('trade_id', ''),
                'balance_before': float(event_data.get('balance_before_entry', 0)),
                'balance_after': float(event_data.get('balance_after_entry', 0)),
                'type': 'OPEN',
                'market_price': float(market_close_price)  # Add market price for comparison
            }
            
            # Use market data for price timeline - convert Unix timestamp
            if market_timestamp and market_close_price:
                market_time = self.parse_timestamp(market_timestamp, "market_data_at_entry timestamp")
                price_data = {
                    'timestamp': market_time,
                    'price': float(market_close_price),
                    'source': 'market_data'
                }
            else:
                # Fallback to trade execution price
                price_data = {
                    'timestamp': timestamp,
                    'price': float(event_data.get('entry_price', 0)),
                    'source': 'trade_execution'
                }
            
            portfolio_data = event_data.get('observation_at_entry', {}).get('portfolio_overview', {})
            equity = float(portfolio_data.get('equity', event_data.get('balance_after_entry', 0)))
            
            balance_data = {
                'timestamp': timestamp,
                'balance': float(event_data.get('balance_after_entry', 0)),
                'equity': equity
            }
        
        elif metadata.get('event_type') == 'TRADE_CLOSED':
            # Parse timestamp from exit_timestamp
            timestamp_str = event_data.get('exit_timestamp')
            timestamp = self.parse_timestamp(timestamp_str, "TRADE_CLOSED exit_timestamp")
            
            # Parse entry timestamp for trade connection lines
            entry_timestamp_str = event_data.get('entry_timestamp')
            entry_timestamp = self.parse_timestamp(entry_timestamp_str, "TRADE_CLOSED entry_timestamp")
            
            # Get market data at exit for the price timeline
            market_data_at_exit = event_data.get('market_data_at_exit', {})
            market_timestamp = market_data_at_exit.get('timestamp', 0)
            market_close_price = market_data_at_exit.get('Close', 0)
            exit_price = float(event_data.get('exit_price', 0))
            
            # Debug log for market data extraction
            self.logger.debug(f"MARKET_DATA_EXTRACT | CLOSE | Trade ID: {event_data.get('trade_id', '')} | Exit Price: ${exit_price:.2f} | Market Close: ${market_close_price:.2f} | Market Time: {market_timestamp} | Trade Time: {timestamp_str}")
            
            trade_data = {
                'timestamp': timestamp,
                'price': exit_price,
                'action': 'CLOSE',
                'side': event_data.get('side', 'LONG'),
                'trade_id': event_data.get('trade_id', ''),
                'pnl': float(event_data.get('net_pnl', 0)),
                'balance_before': float(event_data.get('balance_before_exit', 0)),
                'balance_after': float(event_data.get('balance_after_exit', 0)),
                'type': 'CLOSE',
                'entry_price': float(event_data.get('entry_price', 0)),
                'exit_price': exit_price,
                'entry_timestamp': entry_timestamp,
                'win_loss': event_data.get('win_loss', 'UNKNOWN'),
                'market_price': float(market_close_price)  # Add market price for comparison
            }
            
            # Use market data for price timeline - convert Unix timestamp
            if market_timestamp and market_close_price:
                market_time = self.parse_timestamp(market_timestamp, "market_data_at_exit timestamp")
                price_data = {
                    'timestamp': market_time,
                    'price': float(market_close_price),
                    'source': 'market_data'
                }
            else:
                # Fallback to trade execution price
                price_data = {
                    'timestamp': timestamp,
                    'price': float(event_data.get('exit_price', 0)),
                    'source': 'trade_execution'
                }
            
            balance_data = {
                'timestamp': timestamp,
                'balance': float(event_data.get('balance_after_exit', 0)),
                'equity': float(event_data.get('balance_after_exit', 0))
            }
        
        return trade_data, price_data, balance_data
    
    def monitor_file(self):
        """Monitor the trade trace file for new entries"""
        print(f"Started monitoring {self.trace_file}")
        
        while self.running:
            try:
                if self.trace_file.exists():
                    with open(self.trace_file, 'r') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        
                        for line in new_lines:
                            if line.strip():
                                trace = self.parse_trade_trace(line)
                                if trace:
                                    self.data_queue.put(trace)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error monitoring file: {e}")
                time.sleep(self.update_interval)
    
    def process_queue(self):
        """Process queued trace data and track trade pairs"""
        new_data = False
        
        while not self.data_queue.empty():
            try:
                trace = self.data_queue.get_nowait()
                trade_data, price_data, balance_data = self.extract_trade_data(trace)
                
                if trade_data:
                    self.trades.append(trade_data)
                    trade_id = trade_data['trade_id']
                    
                    # Log trade data to file
                    self.log_trade_event(trade_data)
                    
                    # Track open/close pairs
                    if trade_data['type'] == 'OPEN':
                        self.open_trades[trade_id] = trade_data
                        side_emoji = "🟢" if trade_data['side'] == 'LONG' else "🔴"
                        print(f"{side_emoji} {trade_data['side']} trade opened: {trade_id} at ${trade_data['price']:.2f} | Time: {trade_data['timestamp']}")
                    
                    elif trade_data['type'] == 'CLOSE':
                        if trade_id in self.open_trades:
                            open_data = self.open_trades[trade_id]
                            self.closed_trades[trade_id] = (open_data, trade_data)
                            del self.open_trades[trade_id]
                            
                            # Log trade pair completion
                            self.log_trade_pair_completion(open_data, trade_data)
                            
                            # Create trade line data using proper timestamps
                            line_data = {
                                'trade_id': trade_id,
                                'open_time': open_data['timestamp'],  # From TRADE_OPENED entry_timestamp
                                'close_time': trade_data['timestamp'],  # From TRADE_CLOSED exit_timestamp
                                'open_price': open_data['price'],
                                'close_price': trade_data['price'],
                                'side': trade_data['side'],
                                'pnl': trade_data['pnl'],
                                'win_loss': trade_data['win_loss']
                            }
                            self.trade_lines.append(line_data)
                            
                            pnl_str = f"+${trade_data['pnl']:.2f}" if trade_data['pnl'] >= 0 else f"-${abs(trade_data['pnl']):.2f}"
                            side_emoji = "🟢" if trade_data['side'] == 'LONG' else "🔴"
                            close_emoji = "🟣"
                            print(f"{close_emoji} {trade_data['side']} trade closed: {trade_id} at ${trade_data['price']:.2f} | P&L: {pnl_str} | {trade_data['win_loss']} | Entry: {open_data['timestamp']} | Exit: {trade_data['timestamp']}")
                        else:
                            print(f"⚠️ Close without open: {trade_id} | Exit time: {trade_data['timestamp']}")
                            self.logger.warning(f"ORPHANED_CLOSE | Trade ID: {trade_id} | Price: ${trade_data['price']:.2f} | Time: {trade_data['timestamp']} | Side: {trade_data['side']} | P&L: ${trade_data.get('pnl', 0):.2f}")
                    
                    new_data = True
                
                if price_data:
                    self.prices.append(price_data)
                    # Log price data (throttled)
                    self.log_price_update(price_data, is_loading=False)
                
                if balance_data:
                    self.balance_history.append(balance_data)
                    self.equity_history.append(balance_data)
                    # Log balance update
                    self.log_balance_update(balance_data)
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing trace: {e}")
        
        # Limit data size
        if len(self.trades) > self.max_points:
            self.trades = self.trades[-self.max_points:]
        if len(self.prices) > self.max_points:
            self.prices = self.prices[-self.max_points:]
        if len(self.balance_history) > self.max_points:
            self.balance_history = self.balance_history[-self.max_points:]
        if len(self.trade_lines) > self.max_points // 2:
            self.trade_lines = self.trade_lines[-self.max_points // 2:]
        
        return new_data
    
    def load_existing_data(self):
        """Load existing data from the trace file"""
        print("Loading existing trade data...")
        self.logger.info("DATA_LOAD_START | Loading existing trade data from trace file")
        
        if not self.trace_file.exists():
            print(f"Trace file not found: {self.trace_file}")
            self.logger.warning(f"DATA_LOAD_WARNING | Trace file not found: {self.trace_file}")
            return
        
        try:
            with open(self.trace_file, 'r') as f:
                lines = f.readlines()
                self.last_position = f.tell()
            
            for line in lines:
                if line.strip():
                    trace = self.parse_trade_trace(line)
                    if trace:
                        trade_data, price_data, balance_data = self.extract_trade_data(trace)
                        
                        if trade_data:
                            self.trades.append(trade_data)
                            trade_id = trade_data['trade_id']
                            
                            # Log trade data during loading
                            self.log_trade_event(trade_data)
                            
                            # Track open/close pairs during loading
                            if trade_data['type'] == 'OPEN':
                                self.open_trades[trade_id] = trade_data
                            elif trade_data['type'] == 'CLOSE':
                                if trade_id in self.open_trades:
                                    open_data = self.open_trades[trade_id]
                                    self.closed_trades[trade_id] = (open_data, trade_data)
                                    del self.open_trades[trade_id]
                                    
                                    # Log trade pair completion during loading
                                    self.log_trade_pair_completion(open_data, trade_data)
                                    
                                    # Create trade line data
                                    line_data = {
                                        'trade_id': trade_id,
                                        'open_time': open_data['timestamp'],
                                        'close_time': trade_data['timestamp'],
                                        'open_price': open_data['price'],
                                        'close_price': trade_data['price'],
                                        'side': trade_data['side'],
                                        'pnl': trade_data['pnl'],
                                        'win_loss': trade_data['win_loss']
                                    }
                                    self.trade_lines.append(line_data)
                                else:
                                    # Log orphaned close during loading
                                    self.logger.warning(f"ORPHANED_CLOSE_LOADED | Trade ID: {trade_id} | Price: ${trade_data['price']:.2f} | Time: {trade_data['timestamp']} | Side: {trade_data['side']} | P&L: ${trade_data.get('pnl', 0):.2f}")
                        
                        if price_data:
                            self.prices.append(price_data)
                            # Log price data during loading (less throttled)
                            self.log_price_update(price_data, is_loading=True)
                            
                        if balance_data:
                            self.balance_history.append(balance_data)
                            self.equity_history.append(balance_data)
                            # Log balance update during loading
                            self.log_balance_update(balance_data)
            
            print(f"Loaded {len(self.trades)} trades, {len(self.trade_lines)} trade pairs, {len(self.prices)} price points")
            print(f"Currently open trades: {len(self.open_trades)}")
            
            # Log loading summary
            self.logger.info(f"DATA_LOAD_COMPLETE | Loaded {len(self.trades)} trades, {len(self.trade_lines)} completed pairs, {len(self.prices)} price points")
            self.logger.info(f"DATA_LOAD_COMPLETE | Currently open trades: {len(self.open_trades)}")
            
            if self.trade_lines:
                total_pnl = sum(t['pnl'] for t in self.trade_lines)
                wins = len([t for t in self.trade_lines if t['win_loss'] == 'WIN'])
                losses = len([t for t in self.trade_lines if t['win_loss'] == 'LOSS'])
                win_rate = (wins / len(self.trade_lines) * 100) if self.trade_lines else 0
                self.logger.info(f"DATA_LOAD_SUMMARY | Total P&L: ${total_pnl:.2f} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")
            
        except Exception as e:
            print(f"Error loading existing data: {e}")
            self.logger.error(f"DATA_LOAD_ERROR | {e}")
    
    def load_market_data(self):
        """Load continuous market data timeline for proper trade context"""
        market_file = Path(self.market_data_file)
        
        if not market_file.is_absolute():
            market_file = Path.cwd() / market_file
            
        if not market_file.exists():
            self.logger.warning(f"MARKET_DATA_WARNING | Market data file not found: {market_file}")
            print(f"Warning: Market data file not found: {market_file}")
            print("Will use sparse trade event prices only")
            return
            
        try:
            print(f"Loading continuous market data from: {market_file}")
            self.logger.info(f"MARKET_DATA_LOAD_START | Loading from: {market_file}")
            
            # Load the CSV file
            import pandas as pd
            df = pd.read_csv(market_file)
            
            # Ensure we have the required columns (handle both uppercase and lowercase)
            timestamp_col = 'timestamp'
            close_col = 'Close' if 'Close' in df.columns else 'close'
            
            if timestamp_col not in df.columns or close_col not in df.columns:
                self.logger.error(f"MARKET_DATA_ERROR | Missing required columns. Found: {list(df.columns)}. Need: timestamp and Close/close")
                return
                
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create market timeline
            self.market_timeline = []
            for _, row in df.iterrows():
                self.market_timeline.append({
                    'timestamp': row['datetime'],
                    'price': float(row[close_col]),
                    'source': 'market_data_continuous'
                })
            
            print(f"Loaded {len(self.market_timeline)} market data points")
            self.logger.info(f"MARKET_DATA_LOAD_COMPLETE | Loaded {len(self.market_timeline)} data points")
            
            if self.market_timeline:
                start_time = self.market_timeline[0]['timestamp']
                end_time = self.market_timeline[-1]['timestamp']
                self.logger.info(f"MARKET_DATA_RANGE | From: {start_time} | To: {end_time}")
                
        except Exception as e:
            self.logger.error(f"MARKET_DATA_LOAD_ERROR | {e}")
            print(f"Error loading market data: {e}")
    
    def update_plot(self, frame):
        """Update the plot with new data including trade connection lines"""
        # Process any new data
        new_data = self.process_queue()
        
        if not new_data and frame > 0:
            return  # No new data to plot
        
        # Clear the subplots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Price with trade markers and connection lines
        # Use continuous market data timeline if available, otherwise fall back to sparse trade prices
        if self.market_timeline:
            # Use continuous market data for the price line
            market_df = pd.DataFrame(self.market_timeline)
            
            # Filter to show relevant time range (around trades if any exist)
            if self.trades:
                # Get time range from trades with some padding
                trade_times = [t['timestamp'] for t in self.trades]
                min_trade_time = min(trade_times) - pd.Timedelta(hours=24)  # 24h before first trade
                max_trade_time = max(trade_times) + pd.Timedelta(hours=24)  # 24h after last trade
                
                # Filter market data to relevant time range
                mask = (market_df['timestamp'] >= min_trade_time) & (market_df['timestamp'] <= max_trade_time)
                filtered_df = market_df[mask]
                
                if not filtered_df.empty:
                    self.ax1.plot(filtered_df['timestamp'], filtered_df['price'], 'k-', linewidth=1, alpha=0.8, label='BTC Price (Market Data)')
                    print(f"Plotting {len(filtered_df)} market data points from {min_trade_time} to {max_trade_time}")
                else:
                    print("No market data found in trade time range, using full dataset")
                    self.ax1.plot(market_df['timestamp'], market_df['price'], 'k-', linewidth=1, alpha=0.8, label='BTC Price (Market Data)')
            else:
                # Show recent market data if no trades yet
                recent_df = market_df.tail(1000)  # Last 1000 points
                self.ax1.plot(recent_df['timestamp'], recent_df['price'], 'k-', linewidth=1, alpha=0.8, label='BTC Price (Market Data)')
                
        elif self.prices:
            # Fallback to sparse trade event prices
            price_df = pd.DataFrame(self.prices).drop_duplicates('timestamp').sort_values('timestamp')
            self.ax1.plot(price_df['timestamp'], price_df['price'], 'k-', linewidth=1, alpha=0.8, label='Price (Trade Events)')
            print(f"Using sparse trade event prices: {len(price_df)} points")

        # Draw trade connection lines with enhanced color coding (ALWAYS show trades regardless of price data source)
        if self.trade_lines:
            for line in self.trade_lines[-50:]:  # Last 50 trade pairs
                x_coords = [line['open_time'], line['close_time']]
                y_coords = [line['open_price'], line['close_price']]
                
                # Connection line colors: GREEN for LONG->CLOSE, RED for SHORT->CLOSE
                if line['side'] == 'LONG':
                    line_color = 'green'
                    entry_marker = '^'  # Upward triangle for LONG entry
                    entry_color = 'green'
                    entry_bg_color = 'lightgreen'
                else:  # SHORT
                    line_color = 'red'
                    entry_marker = 'v'  # Downward triangle for SHORT entry
                    entry_color = 'red'
                    entry_bg_color = 'lightcoral'
                
                line_alpha = 0.8
                line_width = 2.5
                
                # Draw the connection line
                self.ax1.plot(x_coords, y_coords, color=line_color, alpha=line_alpha, linewidth=line_width)
                
                # Entry marker - different shapes and colors for LONG vs SHORT
                self.ax1.scatter(line['open_time'], line['open_price'], 
                               marker=entry_marker, color=entry_color, s=100, alpha=0.9, zorder=6,
                               edgecolors='black', linewidth=1.5)
                
                # Exit marker - purple square for all CLOSE operations
                self.ax1.scatter(line['close_time'], line['close_price'], 
                               marker='s', color='purple', s=80, alpha=0.9, zorder=6,
                               edgecolors='black', linewidth=1.5)
                
                # Add detailed price labels for entry and exit
                # Entry price label with side indicator
                side_text = "LONG" if line['side'] == 'LONG' else "SHORT"
                self.ax1.annotate(f'{side_text} Entry: ${line["open_price"]:.2f}', 
                                xy=(line['open_time'], line['open_price']),
                                xytext=(5, 15), textcoords='offset points',
                                fontsize=8, color=entry_color, weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=entry_bg_color, alpha=0.9, edgecolor=entry_color))
                
                # Exit price label
                self.ax1.annotate(f'CLOSE Exit: ${line["close_price"]:.2f}', 
                                xy=(line['close_time'], line['close_price']),
                                xytext=(5, -20), textcoords='offset points',
                                fontsize=8, color='purple', weight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.9, edgecolor='purple'))
                
                # Add P&L label on the line
                mid_time = line['open_time'] + (line['close_time'] - line['open_time']) / 2
                mid_price = (line['open_price'] + line['close_price']) / 2
                pnl_color = 'darkgreen' if line['pnl'] >= 0 else 'darkred'
                pnl_text = f"P&L: ${line['pnl']:+.2f}"
                self.ax1.annotate(pnl_text, 
                                xy=(mid_time, mid_price),
                                xytext=(0, 25), textcoords='offset points',
                                fontsize=9, color=pnl_color, weight='bold', ha='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9, edgecolor=pnl_color))
        
        # Add trade markers for currently open trades (no close yet)
        if self.trades:
            for trade in self.trades[-50:]:  # Last 50 trades
                if trade['type'] == 'OPEN' and trade['trade_id'] in self.open_trades:
                    if trade['side'] == 'LONG':
                        # LONG entry - green upward triangle
                        self.ax1.scatter(trade['timestamp'], trade['price'], 
                                       marker='^', color='green', s=120, alpha=0.9, zorder=7,
                                       edgecolors='black', linewidth=2)
                        # Price label for open LONG entry
                        self.ax1.annotate(f'OPEN LONG: ${trade["price"]:.2f}', 
                                        xy=(trade['timestamp'], trade['price']),
                                        xytext=(5, 20), textcoords='offset points',
                                        fontsize=9, color='green', weight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9, edgecolor='green'))
                    elif trade['side'] == 'SHORT':
                        # SHORT entry - red downward triangle  
                        self.ax1.scatter(trade['timestamp'], trade['price'], 
                                       marker='v', color='red', s=120, alpha=0.9, zorder=7,
                                       edgecolors='black', linewidth=2)
                        # Price label for open SHORT entry
                        self.ax1.annotate(f'OPEN SHORT: ${trade["price"]:.2f}', 
                                        xy=(trade['timestamp'], trade['price']),
                                        xytext=(5, -25), textcoords='offset points',
                                        fontsize=9, color='red', weight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.9, edgecolor='red'))
        
        self.ax1.set_title('Live Trading Activity with Trade Connections')
        self.ax1.set_ylabel('Price (USD)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(['Price', 'LONG->CLOSE (Green Lines)', 'SHORT->CLOSE (Red Lines)', 'Open LONG Entries', 'Open SHORT Entries'], 
                        loc='upper left', fontsize=8)
        
        # Plot 2: Balance History
        if self.balance_history:
            balance_df = pd.DataFrame(self.balance_history).drop_duplicates('timestamp').sort_values('timestamp')
            self.ax2.plot(balance_df['timestamp'], balance_df['balance'], 'b-', linewidth=1, label='Balance')
            self.ax2.plot(balance_df['timestamp'], balance_df['equity'], 'g-', linewidth=1, label='Equity')
        
        self.ax2.set_ylabel('Balance (USD)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # Plot 3: Trade Performance Summary
        if self.trade_lines:
            wins = len([t for t in self.trade_lines if t['win_loss'] == 'WIN'])
            losses = len([t for t in self.trade_lines if t['win_loss'] == 'LOSS'])
            open_count = len(self.open_trades)
            
            categories = ['Wins', 'Losses', 'Open']
            counts = [wins, losses, open_count]
            colors = ['green', 'red', 'blue']
            
            bars = self.ax3.bar(categories, counts, color=colors, alpha=0.7)
            for bar, count in zip(bars, counts):
                if count > 0:
                    self.ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                str(count), ha='center', va='bottom')
            
            # Add win rate
            total_closed = wins + losses
            if total_closed > 0:
                win_rate = wins / total_closed * 100
                self.ax3.set_title(f'Trade Performance (Win Rate: {win_rate:.1f}%)')
            else:
                self.ax3.set_title('Trade Performance')
        else:
            self.ax3.set_title('Trade Performance')
        
        self.ax3.set_ylabel('Count')
        
        # Format time axis
        for ax in [self.ax1, self.ax2]:
            if len(self.prices) > 0:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add status text
        total_pnl = sum(t['pnl'] for t in self.trade_lines) if self.trade_lines else 0
        status_text = f"Trades: {len(self.trades)} | Closed: {len(self.trade_lines)} | Open: {len(self.open_trades)} | P&L: ${total_pnl:.2f} | Last Update: {datetime.now().strftime('%H:%M:%S')}"
        self.fig.suptitle(f"Enhanced Live Trade Monitor - {status_text}", fontsize=10)
        
        plt.tight_layout()
    
    def start_visualization(self):
        """Start the live visualization"""
        # Load existing data first
        self.load_existing_data()
        
        # Set up the plot
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # Start file monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_file)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
          # Start animation
        self.animation = FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=int(self.update_interval * 1000),
            blit=False,
            cache_frame_data=False
        )
        
        try:
            self.logger.info("Starting live visualization display...")
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping visualization...")
            self.logger.info("Visualization stopped by user (Ctrl+C)")
        finally:
            self.cleanup_and_log_summary()
    
    def cleanup_and_log_summary(self):
        """Clean up resources and log session summary"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        # Log final session summary
        self.log_session_summary()
        self.logger.info("TRADE VISUALIZATION SESSION ENDED")
        
        print(f"\n📊 Session complete! Detailed logs saved to: {self.log_file_path}")
        print(f"📈 Total trades processed: {len(self.trades)}")
        print(f"🔗 Trade pairs completed: {len(self.trade_lines)}")
        print(f"📖 Check the log file for detailed trade history and analysis.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced live trade visualization with connection lines and detailed logging")
    parser.add_argument("--file", "-f", 
                       default="logs/trade_traces/trade_traces.jsonl",
                       help="Path to trade traces JSONL file")
    parser.add_argument("--interval", "-i", 
                       type=float, default=2.0,
                       help="Update interval in seconds")
    parser.add_argument("--max-points", "-m", 
                       type=int, default=1000,
                       help="Maximum number of data points to display")
    parser.add_argument("--log-file", "-l",
                       help="Path to log file for trade details (auto-generated if not specified)")
    parser.add_argument("--market-data", "-md",
                       default="data/BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv",
                       help="Path to market data CSV file for continuous price timeline")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    trace_file = Path(args.file)
    if not trace_file.is_absolute():
        trace_file = Path.cwd() / trace_file
    
    print(f"🔴 Starting enhanced live trade visualization...")
    print(f"📁 Monitoring: {trace_file}")
    print(f"📈 Market data: {args.market_data}")
    print(f"⏱️  Update interval: {args.interval}s")
    print(f"📊 Max data points: {args.max_points}")
    print("🔗 Trade connections enabled")
    print("📝 Detailed logging enabled")
    if args.log_file:
        print(f"📄 Log file: {args.log_file}")
    else:
        print("📄 Log file: Auto-generated in logs/visualization/")
    print("Press Ctrl+C to stop")
    
    visualizer = EnhancedLiveTradeVisualizer(
        trace_file=str(trace_file),
        update_interval=args.interval,
        max_points=args.max_points,
        log_file=args.log_file,
        market_data_file=args.market_data
    )
    
    visualizer.start_visualization()

if __name__ == "__main__":
    main()
