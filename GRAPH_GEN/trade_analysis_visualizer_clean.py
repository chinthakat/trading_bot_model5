#!/usr/bin/env python3
"""
Trade Analysis Visualizer
Creates time/price graphs from trade traces using specific fields:
- event_type, episode, trade_id, status, BINANCEFTS_PERP_BTC_USDT, side
- entry_action, entry_price, entry_datetime, position_size, Close, Volume
- Draws lines between entry and exit with P&L markers
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any

class TradeAnalysisVisualizer:
    """Visualize trades using specific fields from trade traces"""
    
    def __init__(self, trace_file: str, market_data_file: str = None):
        """
        Initialize the visualizer
        
        Args:
            trace_file: Path to trade traces JSONL file
            market_data_file: Optional path to market data CSV for price timeline
        """
        self.trace_file = Path(trace_file)
        self.market_data_file = market_data_file
        
        # Data storage
        self.trades = []
        self.completed_trades = []
        self.market_timeline = []
        
        print(f"🔍 Trade Analysis Visualizer initialized")
        print(f"📁 Trace file: {self.trace_file}")
        if market_data_file:
            print(f"📈 Market data: {market_data_file}")
    
    def load_market_data(self):
        """Load continuous market data timeline if available"""
        if not self.market_data_file:
            print("ℹ️  No market data file specified, will use trade event prices only")
            return
            
        market_file = Path(self.market_data_file)
        if not market_file.is_absolute():
            market_file = Path.cwd() / market_file
            
        if not market_file.exists():
            print(f"⚠️  Market data file not found: {market_file}")
            return
            
        try:
            print(f"📊 Loading market data from: {market_file}")
            df = pd.read_csv(market_file)
            
            # Handle both uppercase and lowercase column names
            close_col = 'Close' if 'Close' in df.columns else 'close'
            volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
            
            if 'timestamp' not in df.columns or close_col not in df.columns:
                print(f"❌ Missing required columns. Found: {list(df.columns)}")
                return
                
            # Convert timestamp and create timeline
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            self.market_timeline = []
            for _, row in df.iterrows():
                self.market_timeline.append({
                    'datetime': row['datetime'],
                    'close': float(row[close_col]),
                    'volume': float(row[volume_col]) if volume_col in df.columns else 0.0
                })
            
            print(f"✅ Loaded {len(self.market_timeline)} market data points")
            
        except Exception as e:
            print(f"❌ Error loading market data: {e}")
    
    def parse_trade_traces(self):
        """Parse trade traces and extract relevant fields"""
        print(f"📖 Parsing trade traces from: {self.trace_file}")
        
        if not self.trace_file.exists():
            print(f"❌ Trace file not found: {self.trace_file}")
            return
            
        trades_by_id = {}  # Store trades by trade_id to match OPEN/CLOSE
        
        try:
            with open(self.trace_file, 'r') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Extract metadata
                    metadata = data.get('trace_metadata', {})
                    event_data = data.get('event_data', {})
                    
                    event_type = metadata.get('event_type', '')
                    episode = metadata.get('episode', 0)
                    trade_id = event_data.get('trade_id', '')
                    status = event_data.get('status', '')
                    symbol = event_data.get('symbol', '')
                    side = event_data.get('side', '')
                    
                    # Only process BTC trades
                    if 'BTC' not in symbol:
                        continue
                    
                    if event_type == 'TRADE_OPENED':
                        # Extract entry fields
                        entry_action = event_data.get('entry_action', '')
                        entry_price = event_data.get('entry_price', 0)
                        entry_datetime = event_data.get('entry_datetime', '')
                        position_size = event_data.get('position_size', 0)
                        
                        # Extract market data
                        market_entry = event_data.get('market_data_at_entry', {})
                        close_price = market_entry.get('Close', entry_price)
                        volume = market_entry.get('Volume', 0)
                        
                        trade_info = {
                            'trade_id': trade_id,
                            'episode': episode,
                            'symbol': symbol,
                            'side': side,
                            'entry_action': entry_action,
                            'entry_price': entry_price,
                            'entry_datetime': pd.to_datetime(entry_datetime),
                            'position_size': position_size,
                            'market_close_entry': close_price,
                            'market_volume_entry': volume,
                            'status': 'OPEN'
                        }
                        
                        trades_by_id[trade_id] = trade_info
                        
                    elif event_type == 'TRADE_CLOSED':
                        # Extract exit fields
                        exit_price = event_data.get('exit_price', 0)
                        exit_datetime = event_data.get('exit_datetime', '')
                        net_pnl = event_data.get('net_pnl', 0)
                        win_loss = event_data.get('win_loss', 'UNKNOWN')
                        
                        # Extract exit market data
                        market_exit = event_data.get('market_data_at_exit', {})
                        close_price_exit = market_exit.get('Close', exit_price)
                        volume_exit = market_exit.get('Volume', 0)
                        
                        if trade_id in trades_by_id:
                            # Complete the trade
                            trade = trades_by_id[trade_id]
                            trade.update({
                                'exit_price': exit_price,
                                'exit_datetime': pd.to_datetime(exit_datetime),
                                'net_pnl': net_pnl,
                                'win_loss': win_loss,
                                'market_close_exit': close_price_exit,
                                'market_volume_exit': volume_exit,
                                'status': 'CLOSED'
                            })
                            
                            self.completed_trades.append(trade)
                            del trades_by_id[trade_id]
                            
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON on line {line_num}")
                except Exception as e:
                    print(f"⚠️  Error processing line {line_num}: {e}")
            
            # Add any remaining open trades
            for trade in trades_by_id.values():
                self.trades.append(trade)
            
            print(f"✅ Parsed {len(self.completed_trades)} completed trades")
            print(f"✅ Found {len(self.trades)} open trades")
            
        except Exception as e:
            print(f"❌ Error parsing trade traces: {e}")
    
    def create_visualization(self, save_path: str = None, max_trades: int = 100):
        """Create the time/price visualization with trade markers and P&L lines"""
        
        if not self.completed_trades and not self.trades:
            print("❌ No trade data to visualize")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
        
        # Plot 1: Price timeline with trades
        self._plot_price_timeline(ax1, max_trades)
        
        # Plot 2: Trade statistics
        self._plot_trade_stats(ax2)
        
        # Format and save
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Chart saved to: {save_path}")
        
        plt.show()
    
    def _plot_price_timeline(self, ax, max_trades):
        """Plot the main price timeline with trade markers"""
        
        # Plot market data timeline if available
        if self.market_timeline and self.completed_trades:
            # Filter market data to trade time range
            all_trades = self.completed_trades + self.trades
            if all_trades:
                trade_times = []
                for trade in all_trades:
                    trade_times.append(trade['entry_datetime'])
                    if 'exit_datetime' in trade:
                        trade_times.append(trade['exit_datetime'])
                
                min_time = min(trade_times) - pd.Timedelta(hours=12)
                max_time = max(trade_times) + pd.Timedelta(hours=12)
                
                # Filter market data
                market_df = pd.DataFrame(self.market_timeline)
                filtered_market = market_df[
                    (market_df['datetime'] >= min_time) & 
                    (market_df['datetime'] <= max_time)
                ]
                
                if not filtered_market.empty:
                    ax.plot(filtered_market['datetime'], filtered_market['close'], 
                           'k-', linewidth=1, alpha=0.7, label='BTC Price')
                    print(f"📈 Plotting {len(filtered_market)} market data points")
        
        # Plot completed trades (limit to recent trades for clarity)
        recent_trades = self.completed_trades[-max_trades:] if len(self.completed_trades) > max_trades else self.completed_trades
        
        for trade in recent_trades:
            trade_id = trade['trade_id']
            side = trade['side']
            entry_time = trade['entry_datetime']
            exit_time = trade['exit_datetime']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            net_pnl = trade['net_pnl']
            win_loss = trade['win_loss']
            
            # Color coding: Green for LONG, Red for SHORT
            if side == 'LONG':
                line_color = 'green'
                entry_marker = '^'  # Up arrow
                entry_color = 'darkgreen'
            else:  # SHORT
                line_color = 'red'
                entry_marker = 'v'  # Down arrow
                entry_color = 'darkred'
            
            # P&L color: Green for profit, Red for loss
            pnl_color = 'darkgreen' if net_pnl >= 0 else 'darkred'
            
            # Draw connection line between entry and exit
            ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                   color=line_color, linewidth=2, alpha=0.8)
            
            # Entry marker
            ax.scatter(entry_time, entry_price, marker=entry_marker, 
                      color=entry_color, s=100, alpha=0.9, zorder=5, 
                      edgecolors='black', linewidth=1)
            
            # Exit marker
            ax.scatter(exit_time, exit_price, marker='s', 
                      color='purple', s=80, alpha=0.9, zorder=5,
                      edgecolors='black', linewidth=1)
            
            # P&L label at midpoint
            mid_time = entry_time + (exit_time - entry_time) / 2
            mid_price = (entry_price + exit_price) / 2
            
            ax.annotate(f'${net_pnl:+.2f}\\n{win_loss}', 
                       xy=(mid_time, mid_price),
                       xytext=(0, 20), textcoords='offset points',
                       fontsize=8, color=pnl_color, weight='bold', 
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.8, 
                                edgecolor=pnl_color))
            
            # Trade ID label (smaller, less prominent)
            ax.annotate(trade_id.replace('TRADE_', ''), 
                       xy=(entry_time, entry_price),
                       xytext=(5, -15), textcoords='offset points',
                       fontsize=6, color='gray', alpha=0.7)
        
        # Plot open trades
        for trade in self.trades:
            side = trade['side']
            entry_time = trade['entry_datetime']
            entry_price = trade['entry_price']
            
            if side == 'LONG':
                marker_color = 'lightgreen'
                marker = '^'
            else:
                marker_color = 'lightcoral' 
                marker = 'v'
            
            ax.scatter(entry_time, entry_price, marker=marker, 
                      color=marker_color, s=120, alpha=0.9, zorder=6,
                      edgecolors='black', linewidth=2)
            
            ax.annotate(f'OPEN {side}', 
                       xy=(entry_time, entry_price),
                       xytext=(5, 15), textcoords='offset points',
                       fontsize=8, color=marker_color, weight='bold')
        
        # Formatting
        ax.set_title(f'Trade Analysis: Entry/Exit Points with P&L\\n'
                    f'({len(recent_trades)} completed trades shown)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\\n%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=2, label='LONG Trades'),
            plt.Line2D([0], [0], color='red', lw=2, label='SHORT Trades'),
            plt.Line2D([0], [0], marker='^', color='darkgreen', lw=0, 
                      markersize=8, label='Entry (LONG)'),
            plt.Line2D([0], [0], marker='v', color='darkred', lw=0, 
                      markersize=8, label='Entry (SHORT)'),
            plt.Line2D([0], [0], marker='s', color='purple', lw=0, 
                      markersize=8, label='Exit')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    def _plot_trade_stats(self, ax):
        """Plot trade statistics in the bottom panel"""
        
        if not self.completed_trades:
            ax.text(0.5, 0.5, 'No completed trades to analyze', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_title('Trade Statistics')
            return
        
        # Calculate statistics
        total_trades = len(self.completed_trades)
        wins = len([t for t in self.completed_trades if t['win_loss'] == 'WIN'])
        losses = len([t for t in self.completed_trades if t['win_loss'] == 'LOSS'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t['net_pnl'] for t in self.completed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Long vs Short breakdown
        long_trades = [t for t in self.completed_trades if t['side'] == 'LONG']
        short_trades = [t for t in self.completed_trades if t['side'] == 'SHORT']
        
        # Create bar chart
        categories = ['Total', 'Wins', 'Losses', 'LONG', 'SHORT']
        values = [total_trades, wins, losses, len(long_trades), len(short_trades)]
        colors = ['blue', 'green', 'red', 'darkgreen', 'darkred']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(value), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Trade Statistics | Win Rate: {win_rate:.1f}% | '
                    f'Total P&L: ${total_pnl:.2f} | Avg P&L: ${avg_pnl:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add P&L distribution info
        if self.completed_trades:
            pnl_values = [t['net_pnl'] for t in self.completed_trades]
            best_trade = max(pnl_values)
            worst_trade = min(pnl_values)
            
            info_text = f'Best: ${best_trade:.2f} | Worst: ${worst_trade:.2f}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Trade Analysis Visualizer - Create time/price graphs with trade markers and P&L")
    parser.add_argument("--trace-file", "-t", 
                       default="logs/trade_traces/trade_traces.jsonl",
                       help="Path to trade traces JSONL file")
    parser.add_argument("--market-data", "-m",
                       default="data/BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv",
                       help="Path to market data CSV file for continuous price timeline")
    parser.add_argument("--save", "-s",
                       help="Path to save the chart (e.g., trade_analysis.png)")
    parser.add_argument("--max-trades", "-mt",
                       type=int, default=100,
                       help="Maximum number of recent trades to display")
    
    args = parser.parse_args()
    
    print("🚀 Starting Trade Analysis Visualization")
    print("📊 Using fields: event_type, episode, trade_id, status, symbol, side")
    print("📊 Also using: entry_action, entry_price, entry_datetime, position_size, Close, Volume")
    print("🔗 Drawing lines between entry/exit with P&L markers")
    print()
    
    # Create visualizer
    visualizer = TradeAnalysisVisualizer(
        trace_file=args.trace_file,
        market_data_file=args.market_data
    )
    
    # Load data
    visualizer.load_market_data()
    visualizer.parse_trade_traces()
    
    # Create visualization
    save_path = args.save or f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualizer.create_visualization(save_path=save_path, max_trades=args.max_trades)
    
    print("✅ Visualization complete!")

if __name__ == "__main__":
    main()
