#!/usr/bin/env python3
"""
Enhanced Live Trade Visualizer
Real-time visualization of trading activity with trade connection lines
"""

import json
import time
import argparse
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
    
    def __init__(self, trace_file: str, update_interval: float = 2.0, max_points: int = 1000):
        """
        Initialize the live visualizer
        
        Args:
            trace_file: Path to trade traces JSONL file
            update_interval: Update frequency in seconds
            max_points: Maximum number of data points to display
        """
        self.trace_file = Path(trace_file)
        self.update_interval = update_interval
        self.max_points = max_points
        
        # Data storage
        self.trades = []
        self.prices = []
        self.balance_history = []
        self.equity_history = []
        
        # Trade tracking for open/close pairs
        self.open_trades = {}  # trade_id -> open trade data
        self.closed_trades = {}  # trade_id -> (open_data, close_data)
        self.trade_lines = []  # List of trade connection lines
        
        # File monitoring
        self.last_position = 0
        self.running = False
        
        # Threading
        self.data_queue = queue.Queue()
        self.monitor_thread = None
        
        print(f"Initialized Enhanced LiveTradeVisualizer for: {self.trace_file}")
    
    def parse_trade_trace(self, line: str) -> Dict[str, Any]:
        """Parse a single trade trace line"""
        try:
            trace = json.loads(line.strip())
            return trace
        except json.JSONDecodeError:
            return None
    
    def extract_trade_data(self, trace: Dict[str, Any]) -> tuple:
        """Extract trade and price data from a trace"""
        event_data = trace.get('event_data', {})
        metadata = trace.get('trace_metadata', {})
        
        trade_data = None
        price_data = None
        balance_data = None
        
        # Extract trade information
        if metadata.get('event_type') == 'TRADE_OPENED':
            # Parse timestamp
            timestamp_str = event_data.get('entry_timestamp')
            if timestamp_str:
                try:
                    timestamp = pd.to_datetime(timestamp_str)
                except:
                    timestamp = pd.Timestamp.now()
            else:
                timestamp = pd.Timestamp.now()
            
            trade_data = {
                'timestamp': timestamp,
                'price': float(event_data.get('entry_price', 0)),
                'action': event_data.get('entry_action', 'BUY'),
                'side': event_data.get('side', 'LONG'),
                'size': float(event_data.get('position_size', 0)),
                'trade_id': event_data.get('trade_id', ''),
                'balance_before': float(event_data.get('balance_before_entry', 0)),
                'balance_after': float(event_data.get('balance_after_entry', 0)),
                'type': 'OPEN'
            }
            
            price_data = {
                'timestamp': timestamp,
                'price': float(event_data.get('entry_price', 0))
            }
            
            portfolio_data = event_data.get('observation_at_entry', {}).get('portfolio_overview', {})
            equity = float(portfolio_data.get('equity', event_data.get('balance_after_entry', 0)))
            
            balance_data = {
                'timestamp': timestamp,
                'balance': float(event_data.get('balance_after_entry', 0)),
                'equity': equity
            }
        
        elif metadata.get('event_type') == 'TRADE_CLOSED':
            timestamp_str = event_data.get('exit_timestamp')
            if timestamp_str:
                try:
                    timestamp = pd.to_datetime(timestamp_str)
                except:
                    timestamp = pd.Timestamp.now()
            else:
                timestamp = pd.Timestamp.now()
            
            trade_data = {
                'timestamp': timestamp,
                'price': float(event_data.get('exit_price', 0)),
                'action': 'CLOSE',
                'side': event_data.get('side', 'LONG'),
                'trade_id': event_data.get('trade_id', ''),
                'pnl': float(event_data.get('net_pnl', 0)),
                'balance_before': float(event_data.get('balance_before_exit', 0)),
                'balance_after': float(event_data.get('balance_after_exit', 0)),
                'type': 'CLOSE',
                'entry_price': float(event_data.get('entry_price', 0)),
                'exit_price': float(event_data.get('exit_price', 0)),
                'entry_timestamp': pd.to_datetime(event_data.get('entry_timestamp', timestamp)),
                'win_loss': event_data.get('win_loss', 'UNKNOWN')
            }
            
            price_data = {
                'timestamp': timestamp,
                'price': float(event_data.get('exit_price', 0))
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
                      # Track open/close pairs
                    if trade_data['type'] == 'OPEN':
                        self.open_trades[trade_id] = trade_data
                        side_emoji = "🟢" if trade_data['side'] == 'LONG' else "🔴"
                        print(f"{side_emoji} {trade_data['side']} trade opened: {trade_id} at ${trade_data['price']:.2f}")
                    
                    elif trade_data['type'] == 'CLOSE':
                        if trade_id in self.open_trades:
                            open_data = self.open_trades[trade_id]
                            self.closed_trades[trade_id] = (open_data, trade_data)
                            del self.open_trades[trade_id]
                            
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
                            
                            pnl_str = f"+${trade_data['pnl']:.2f}" if trade_data['pnl'] >= 0 else f"-${abs(trade_data['pnl']):.2f}"
                            side_emoji = "🟢" if trade_data['side'] == 'LONG' else "🔴"
                            close_emoji = "🟣"
                            print(f"{close_emoji} {trade_data['side']} trade closed: {trade_id} at ${trade_data['price']:.2f} | P&L: {pnl_str} | {trade_data['win_loss']}")
                        else:
                            print(f"⚠️ Close without open: {trade_id}")
                    
                    new_data = True
                
                if price_data:
                    self.prices.append(price_data)
                
                if balance_data:
                    self.balance_history.append(balance_data)
                    self.equity_history.append(balance_data)
                
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
        
        if not self.trace_file.exists():
            print(f"Trace file not found: {self.trace_file}")
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
                            
                            # Track open/close pairs during loading
                            if trade_data['type'] == 'OPEN':
                                self.open_trades[trade_id] = trade_data
                            elif trade_data['type'] == 'CLOSE':
                                if trade_id in self.open_trades:
                                    open_data = self.open_trades[trade_id]
                                    self.closed_trades[trade_id] = (open_data, trade_data)
                                    del self.open_trades[trade_id]
                                    
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
                        
                        if price_data:
                            self.prices.append(price_data)
                        if balance_data:
                            self.balance_history.append(balance_data)
                            self.equity_history.append(balance_data)
            
            print(f"Loaded {len(self.trades)} trades, {len(self.trade_lines)} trade pairs, {len(self.prices)} price points")
            print(f"Currently open trades: {len(self.open_trades)}")
            
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
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
        if self.prices:
            price_df = pd.DataFrame(self.prices).drop_duplicates('timestamp').sort_values('timestamp')
            self.ax1.plot(price_df['timestamp'], price_df['price'], 'k-', linewidth=1, alpha=0.8, label='Price')              # Draw trade connection lines with enhanced color coding
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
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9, edgecolor=pnl_color))              # Add trade markers for currently open trades (no close yet)
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
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping visualization...")
        finally:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced live trade visualization with connection lines")
    parser.add_argument("--file", "-f", 
                       default="logs/trade_traces/trade_traces.jsonl",
                       help="Path to trade traces JSONL file")
    parser.add_argument("--interval", "-i", 
                       type=float, default=2.0,
                       help="Update interval in seconds")
    parser.add_argument("--max-points", "-m", 
                       type=int, default=1000,
                       help="Maximum number of data points to display")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    trace_file = Path(args.file)
    if not trace_file.is_absolute():
        trace_file = Path.cwd() / trace_file
    
    print(f"🔴 Starting enhanced live trade visualization...")
    print(f"📁 Monitoring: {trace_file}")
    print(f"⏱️  Update interval: {args.interval}s")
    print(f"📊 Max data points: {args.max_points}")
    print("🔗 Trade connections enabled")
    print("Press Ctrl+C to stop")
    
    visualizer = EnhancedLiveTradeVisualizer(
        trace_file=str(trace_file),
        update_interval=args.interval,
        max_points=args.max_points
    )
    
    visualizer.start_visualization()

if __name__ == "__main__":
    main()
