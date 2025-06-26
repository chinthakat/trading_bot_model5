#!/usr/bin/env python3
"""
Trade Analysis Visualizer - Pure Trade Traces Version
Creates time/price graphs using ONLY data from trade_traces.jsonl
Uses embedded market data from trade events to show price context
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

class PureTradeAnalysisVisualizer:
    """Visualize trades using only trade traces data (no external market data CSV)"""
    
    def __init__(self, trace_file: str):
        """
        Initialize the visualizer
        
        Args:
            trace_file: Path to trade traces JSONL file
        """
        self.trace_file = Path(trace_file)
        
        # Data storage
        self.trades = []  # Open trades
        self.completed_trades = []  # Completed trade pairs
        self.market_snapshots = []  # Market data embedded in trades
        
        print(f"🔍 Pure Trade Analysis Visualizer initialized")
        print(f"📁 Trace file: {self.trace_file}")
        print(f"📊 Using ONLY embedded market data from trade traces")
    
    def parse_trade_traces(self):
        """Parse trade traces and extract all relevant fields"""
        print(f"📖 Parsing trade traces from: {self.trace_file}")
        
        if not self.trace_file.exists():
            print(f"❌ Trace file not found: {self.trace_file}")
            return
            
        trades_by_id = {}  # Store trades by trade_id to match OPEN/CLOSE
        
        try:
            with open(self.trace_file, 'r') as f:
                lines = f.readlines()
                
            print(f"📄 Processing {len(lines)} lines...")
                
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Extract metadata and event data
                    metadata = data.get('trace_metadata', {})
                    event_data = data.get('event_data', {})
                    
                    event_type = metadata.get('event_type', '')
                    episode = metadata.get('episode', 0)
                    trade_id = event_data.get('trade_id', '')
                    status = event_data.get('status', '')
                    symbol = event_data.get('symbol', '')
                    side = event_data.get('side', '')
                    
                    # Only process BTC trades
                    if 'BTC' not in symbol and 'BINANCEFTS_PERP_BTC_USDT' not in str(event_data):
                        continue
                    
                    if event_type == 'TRADE_OPENED':
                        # Extract all requested fields
                        entry_action = event_data.get('entry_action', '')
                        entry_price = float(event_data.get('entry_price', 0))
                        entry_datetime = event_data.get('entry_datetime', '')
                        position_size = float(event_data.get('position_size', 0))
                        
                        # Extract embedded market data at entry
                        market_entry = event_data.get('market_data_at_entry', {})
                        close_price = float(market_entry.get('Close', entry_price))
                        volume = float(market_entry.get('Volume', 0))
                        market_timestamp = market_entry.get('timestamp', 0)
                        
                        # Store market snapshot
                        if market_timestamp:
                            self.market_snapshots.append({
                                'datetime': pd.to_datetime(market_timestamp, unit='s'),
                                'close': close_price,
                                'volume': volume,
                                'source': 'entry'
                            })
                        
                        trade_info = {
                            'event_type': event_type,
                            'episode': episode,
                            'trade_id': trade_id,
                            'status': status,
                            'symbol': symbol,
                            'side': side,
                            'entry_action': entry_action,
                            'entry_price': entry_price,
                            'entry_datetime': pd.to_datetime(entry_datetime),
                            'position_size': position_size,
                            'market_close_entry': close_price,
                            'market_volume_entry': volume,
                            'trade_status': 'OPEN'
                        }
                        
                        trades_by_id[trade_id] = trade_info
                        
                    elif event_type == 'TRADE_CLOSED':
                        # Extract exit fields
                        exit_price = float(event_data.get('exit_price', 0))
                        exit_datetime = event_data.get('exit_datetime', '')
                        net_pnl = float(event_data.get('net_pnl', 0))
                        win_loss = event_data.get('win_loss', 'UNKNOWN')
                        
                        # Extract embedded market data at exit
                        market_exit = event_data.get('market_data_at_exit', {})
                        close_price_exit = float(market_exit.get('Close', exit_price))
                        volume_exit = float(market_exit.get('Volume', 0))
                        market_timestamp_exit = market_exit.get('timestamp', 0)
                        
                        # Store market snapshot
                        if market_timestamp_exit:
                            self.market_snapshots.append({
                                'datetime': pd.to_datetime(market_timestamp_exit, unit='s'),
                                'close': close_price_exit,
                                'volume': volume_exit,
                                'source': 'exit'
                            })
                        
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
                                'trade_status': 'CLOSED'
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
            
            # Sort market snapshots by time
            self.market_snapshots.sort(key=lambda x: x['datetime'])
            
            print(f"✅ Parsed {len(self.completed_trades)} completed trades")
            print(f"✅ Found {len(self.trades)} open trades")
            print(f"✅ Extracted {len(self.market_snapshots)} market data snapshots")
            
        except Exception as e:
            print(f"❌ Error parsing trade traces: {e}")
    
    def create_visualization(self, save_path: str = None, max_trades: int = 100):
        """Create the time/price visualization using only trade traces data"""
        
        if not self.completed_trades and not self.trades:
            print("❌ No trade data to visualize")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
        
        # Plot 1: Price timeline with trades (using embedded market data)
        self._plot_price_timeline_pure(ax1, max_trades)
        
        # Plot 2: Trade statistics
        self._plot_trade_stats(ax2)
        
        # Format and save
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Chart saved to: {save_path}")
        
        plt.show()
    
    def _plot_price_timeline_pure(self, ax, max_trades):
        """Plot price timeline using ONLY embedded market data from trades"""
        
        # Plot embedded market data as background price context
        if self.market_snapshots:
            market_df = pd.DataFrame(self.market_snapshots)
            market_df = market_df.drop_duplicates('datetime').sort_values('datetime')
            
            ax.plot(market_df['datetime'], market_df['close'], 
                   'gray', linewidth=1, alpha=0.6, label='BTC Price (from trades)', 
                   linestyle='--', marker='o', markersize=2)
            print(f"📈 Using {len(market_df)} embedded market data points")
        
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
                   color=line_color, linewidth=2.5, alpha=0.8, zorder=3)
            
            # Entry marker
            ax.scatter(entry_time, entry_price, marker=entry_marker, 
                      color=entry_color, s=120, alpha=0.9, zorder=5, 
                      edgecolors='black', linewidth=1.5)
            
            # Exit marker
            ax.scatter(exit_time, exit_price, marker='s', 
                      color='purple', s=100, alpha=0.9, zorder=5,
                      edgecolors='black', linewidth=1.5)
            
            # P&L label at midpoint
            mid_time = entry_time + (exit_time - entry_time) / 2
            mid_price = (entry_price + exit_price) / 2
            
            ax.annotate(f'${net_pnl:+.2f}\\n{win_loss}', 
                       xy=(mid_time, mid_price),
                       xytext=(0, 25), textcoords='offset points',
                       fontsize=9, color=pnl_color, weight='bold', 
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', alpha=0.9, 
                                edgecolor=pnl_color), zorder=4)
            
            # Trade ID label (smaller, less prominent)
            ax.annotate(trade_id.replace('TRADE_', ''), 
                       xy=(entry_time, entry_price),
                       xytext=(5, -20), textcoords='offset points',
                       fontsize=7, color='gray', alpha=0.8)
        
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
                      color=marker_color, s=150, alpha=0.9, zorder=6,
                      edgecolors='black', linewidth=2)
            
            ax.annotate(f'OPEN {side}', 
                       xy=(entry_time, entry_price),
                       xytext=(5, 15), textcoords='offset points',
                       fontsize=9, color=marker_color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='white', alpha=0.8))
        
        # Formatting
        data_source = "Embedded Market Data from Trade Traces"
        ax.set_title(f'Trade Analysis: Entry/Exit Points with P&L\\n'\
                    f'({len(recent_trades)} completed trades | Data Source: {data_source})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\\n%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='gray', lw=1, linestyle='--', 
                      label='BTC Price (from trades)'),
            plt.Line2D([0], [0], color='green', lw=2, label='LONG Trades'),
            plt.Line2D([0], [0], color='red', lw=2, label='SHORT Trades'),
            plt.Line2D([0], [0], marker='^', color='darkgreen', lw=0, 
                      markersize=10, label='Entry (LONG)'),
            plt.Line2D([0], [0], marker='v', color='darkred', lw=0, 
                      markersize=10, label='Entry (SHORT)'),
            plt.Line2D([0], [0], marker='s', color='purple', lw=0, 
                      markersize=10, label='Exit')
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
        
        ax.set_title(f'Trade Statistics | Win Rate: {win_rate:.1f}% | '\
                    f'Total P&L: ${total_pnl:.2f} | Avg P&L: ${avg_pnl:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add P&L distribution info
        if self.completed_trades:
            pnl_values = [t['net_pnl'] for t in self.completed_trades]
            best_trade = max(pnl_values)
            worst_trade = min(pnl_values)
            
            info_text = f'Best: ${best_trade:.2f} | Worst: ${worst_trade:.2f} | Market Snapshots: {len(self.market_snapshots)}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Pure Trade Analysis Visualizer - Uses ONLY trade_traces.jsonl data")
    parser.add_argument("--trace-file", "-t", 
                       default="logs/trade_traces/trade_traces.jsonl",
                       help="Path to trade traces JSONL file")
    parser.add_argument("--save", "-s",
                       help="Path to save the chart (e.g., trade_analysis_pure.png)")
    parser.add_argument("--max-trades", "-mt",
                       type=int, default=100,
                       help="Maximum number of recent trades to display")
    
    args = parser.parse_args()
    
    print("🚀 Starting PURE Trade Analysis Visualization")
    print("📊 Using ONLY data from trade_traces.jsonl")
    print("📊 Fields: event_type, episode, trade_id, status, symbol, side")
    print("📊 Fields: entry_action, entry_price, entry_datetime, position_size")
    print("📊 Fields: Close, Volume (from embedded market_data_at_entry/exit)")
    print("🔗 Drawing lines between entry/exit with P&L markers")
    print()
    
    # Create visualizer
    visualizer = PureTradeAnalysisVisualizer(trace_file=args.trace_file)
    
    # Load and parse data
    visualizer.parse_trade_traces()
    
    # Create visualization
    save_path = args.save or f"trade_analysis_pure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualizer.create_visualization(save_path=save_path, max_trades=args.max_trades)
    
    print("✅ Pure trade traces visualization complete!")
    print(f"💡 This uses ONLY embedded market data from trade events")
    print(f"💡 For smoother price curves, use the CSV version with continuous market data")

if __name__ == "__main__":
    main()
