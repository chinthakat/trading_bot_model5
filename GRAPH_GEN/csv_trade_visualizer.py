#!/usr/bin/env python3
"""
CSV-based Trade Visualizer
Creates time/price graphs using extracted CSV data from trade traces
This ensures proper time distribution and avoids cramped visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import argparse
from pathlib import Path

class CSVTradeVisualizer:
    """Visualize trades using extracted CSV data"""
    
    def __init__(self, trade_csv: str, market_csv: str = None, external_market_csv: str = None):
        """
        Initialize visualizer with CSV data
        
        Args:
            trade_csv: Path to extracted trades CSV
            market_csv: Path to extracted market data CSV (from trades)
            external_market_csv: Path to external continuous market data CSV
        """
        self.trade_csv = Path(trade_csv)
        self.market_csv = Path(market_csv) if market_csv else None
        self.external_market_csv = Path(external_market_csv) if external_market_csv else None
        
        self.trades_df = None
        self.market_df = None
        self.external_market_df = None
        
        print(f"🔍 CSV Trade Visualizer initialized")
        print(f"📄 Trades CSV: {self.trade_csv}")
        if self.market_csv:
            print(f"📈 Market CSV: {self.market_csv}")
        if self.external_market_csv:
            print(f"📊 External Market CSV: {self.external_market_csv}")
    
    def load_data(self):
        """Load all CSV data"""
        print("📖 Loading CSV data...")
        
        # Load trades
        if not self.trade_csv.exists():
            print(f"❌ Trade CSV not found: {self.trade_csv}")
            return False
            
        self.trades_df = pd.read_csv(self.trade_csv)
        self.trades_df['entry_datetime'] = pd.to_datetime(self.trades_df['entry_datetime'])
        self.trades_df['exit_datetime'] = pd.to_datetime(self.trades_df['exit_datetime'])
        
        print(f"✅ Loaded {len(self.trades_df)} trades")
        
        # Load market data from trades
        if self.market_csv and self.market_csv.exists():
            self.market_df = pd.read_csv(self.market_csv)
            self.market_df['datetime'] = pd.to_datetime(self.market_df['datetime'])
            self.market_df = self.market_df.sort_values('datetime')
            print(f"✅ Loaded {len(self.market_df)} market points from trades")
        
        # Load external market data
        if self.external_market_csv and self.external_market_csv.exists():
            self.external_market_df = pd.read_csv(self.external_market_csv)
            if 'timestamp' in self.external_market_df.columns:
                self.external_market_df['datetime'] = pd.to_datetime(self.external_market_df['timestamp'], unit='s')
            else:
                self.external_market_df['datetime'] = pd.to_datetime(self.external_market_df['datetime'])
            
            # Handle column names
            close_col = 'Close' if 'Close' in self.external_market_df.columns else 'close'
            volume_col = 'Volume' if 'Volume' in self.external_market_df.columns else 'volume'
            
            self.external_market_df = self.external_market_df.rename(columns={
                close_col: 'close',
                volume_col: 'volume'
            })
            
            self.external_market_df = self.external_market_df.sort_values('datetime')
            print(f"✅ Loaded {len(self.external_market_df)} external market points")
        
        return True
    
    def create_visualization(self, save_path: str = None, max_trades: int = 200, 
                           time_range_days: int = None, use_external_market: bool = True):
        """Create the visualization"""
        
        if self.trades_df is None:
            print("❌ No trade data loaded")
            return
            
        # Filter trades if time range specified
        trades_to_plot = self.trades_df.copy()
        if time_range_days:
            latest_date = self.trades_df['entry_datetime'].max()
            cutoff_date = latest_date - timedelta(days=time_range_days)
            trades_to_plot = trades_to_plot[trades_to_plot['entry_datetime'] >= cutoff_date]
            print(f"📅 Filtered to last {time_range_days} days: {len(trades_to_plot)} trades")
        
        # Limit number of trades for clarity
        if len(trades_to_plot) > max_trades:
            trades_to_plot = trades_to_plot.tail(max_trades)
            print(f"📊 Limited to last {max_trades} trades for clarity")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])
        
        # Plot price timeline
        self._plot_price_timeline(ax1, trades_to_plot, use_external_market)
        
        # Plot statistics
        self._plot_statistics(ax2, trades_to_plot)
        
        # Format and save
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Chart saved to: {save_path}")
        
        plt.show()
    
    def _plot_price_timeline(self, ax, trades_df, use_external_market):
        """Plot the price timeline with trades"""
        
        # Determine time range from trades
        min_time = trades_df['entry_datetime'].min() - timedelta(hours=12)
        max_time = trades_df['exit_datetime'].max() + timedelta(hours=12)
        
        print(f"📅 Plotting time range: {min_time} to {max_time}")
        
        # Plot background price data
        if use_external_market and self.external_market_df is not None:
            # Use external continuous market data
            market_filtered = self.external_market_df[
                (self.external_market_df['datetime'] >= min_time) & 
                (self.external_market_df['datetime'] <= max_time)
            ]
            
            if not market_filtered.empty:
                ax.plot(market_filtered['datetime'], market_filtered['close'], 
                       'gray', linewidth=1, alpha=0.7, label='BTC Price (Continuous)')
                print(f"📈 Plotted {len(market_filtered)} continuous market points")
            
        elif self.market_df is not None:
            # Use market data from trades
            market_filtered = self.market_df[
                (self.market_df['datetime'] >= min_time) & 
                (self.market_df['datetime'] <= max_time)
            ]
            
            if not market_filtered.empty:
                ax.plot(market_filtered['datetime'], market_filtered['close'], 
                       'gray', linewidth=1, alpha=0.6, label='BTC Price (from trades)', 
                       linestyle='--', marker='o', markersize=3)
                print(f"📈 Plotted {len(market_filtered)} trade-based market points")
        
        # Plot completed trades
        completed_trades = trades_df[trades_df['net_pnl'].notna()]
        
        for _, trade in completed_trades.iterrows():
            trade_id = trade['trade_id']
            side = trade['side']
            entry_time = trade['entry_datetime']
            exit_time = trade['exit_datetime']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            net_pnl = trade['net_pnl']
            win_loss = trade['win_loss']
            
            # Color coding
            if side == 'LONG':
                line_color = 'green'
                entry_marker = '^'
                entry_color = 'darkgreen'
            else:  # SHORT
                line_color = 'red'
                entry_marker = 'v'
                entry_color = 'darkred'
            
            pnl_color = 'darkgreen' if net_pnl >= 0 else 'darkred'
            
            # Draw connection line
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
            
            # P&L label (only for recent trades to avoid clutter)
            if len(completed_trades) <= 50:  # Only show P&L labels if not too many trades
                mid_time = entry_time + (exit_time - entry_time) / 2
                mid_price = (entry_price + exit_price) / 2
                
                ax.annotate(f'${net_pnl:+.2f}', 
                           xy=(mid_time, mid_price),
                           xytext=(0, 25), textcoords='offset points',
                           fontsize=8, color=pnl_color, weight='bold', 
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='yellow', alpha=0.8, 
                                    edgecolor=pnl_color), zorder=4)
        
        # Plot open trades (if any)
        open_trades = trades_df[trades_df['net_pnl'].isna()]
        for _, trade in open_trades.iterrows():
            side = trade['side']
            entry_time = trade['entry_datetime']
            entry_price = trade['entry_price']
            
            marker_color = 'lightgreen' if side == 'LONG' else 'lightcoral'
            marker = '^' if side == 'LONG' else 'v'
            
            ax.scatter(entry_time, entry_price, marker=marker, 
                      color=marker_color, s=150, alpha=0.9, zorder=6,
                      edgecolors='black', linewidth=2)
            
            ax.annotate(f'OPEN {side}', 
                       xy=(entry_time, entry_price),
                       xytext=(5, 15), textcoords='offset points',
                       fontsize=9, color=marker_color, weight='bold')
        
        # Formatting
        data_source = "External CSV" if use_external_market and self.external_market_df is not None else "Trade Events"
        ax.set_title(f'Trade Analysis: Entry/Exit Points with P&L\n'
                    f'({len(completed_trades)} trades | Price Source: {data_source} | '
                    f'Time Range: {min_time.strftime("%m/%d")} - {max_time.strftime("%m/%d")})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format time axis based on time range
        time_span = max_time - min_time
        if time_span.days > 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\\n%H:%M'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Legend
        legend_elements = [
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
    
    def _plot_statistics(self, ax, trades_df):
        """Plot trade statistics"""
        
        completed_trades = trades_df[trades_df['net_pnl'].notna()]
        
        if completed_trades.empty:
            ax.text(0.5, 0.5, 'No completed trades in selected range', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_title('Trade Statistics')
            return
        
        # Calculate statistics
        total_trades = len(completed_trades)
        wins = len(completed_trades[completed_trades['win_loss'] == 'WIN'])
        losses = len(completed_trades[completed_trades['win_loss'] == 'LOSS'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = completed_trades['net_pnl'].sum()
        avg_pnl = completed_trades['net_pnl'].mean()
        
        # Side breakdown
        long_trades = len(completed_trades[completed_trades['side'] == 'LONG'])
        short_trades = len(completed_trades[completed_trades['side'] == 'SHORT'])
        
        # Create bar chart
        categories = ['Total', 'Wins', 'Losses', 'LONG', 'SHORT']
        values = [total_trades, wins, losses, long_trades, short_trades]
        colors = ['blue', 'green', 'red', 'darkgreen', 'darkred']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(value), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Trade Statistics | Win Rate: {win_rate:.1f}% | '\
                    f'Total P&L: ${total_pnl:.2f} | Avg P&L: ${avg_pnl:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # P&L info
        best_trade = completed_trades['net_pnl'].max()
        worst_trade = completed_trades['net_pnl'].min()
        
        time_range = f"{completed_trades['entry_datetime'].min().strftime('%m/%d')} - {completed_trades['entry_datetime'].max().strftime('%m/%d')}"
        info_text = f'Best: ${best_trade:.2f} | Worst: ${worst_trade:.2f} | Period: {time_range}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="CSV-based Trade Visualizer")
    parser.add_argument("--trade-csv", "-t", 
                       default="extracted_trades.csv",
                       help="Path to extracted trades CSV file")
    parser.add_argument("--market-csv", "-m",
                       default="extracted_market_data.csv",
                       help="Path to extracted market data CSV file")
    parser.add_argument("--external-market-csv", "-e",
                       help="Path to external continuous market data CSV")
    parser.add_argument("--save", "-s",
                       help="Path to save the chart")
    parser.add_argument("--max-trades", "-mt",
                       type=int, default=200,
                       help="Maximum number of trades to display")
    parser.add_argument("--days", "-d",
                       type=int,
                       help="Number of recent days to show (default: all)")
    parser.add_argument("--no-external-market", action="store_true",
                       help="Don't use external market data even if available")
    
    args = parser.parse_args()
    
    print("🚀 Starting CSV-based Trade Visualization")
    print(f"📄 Using extracted CSV data for proper time distribution")
    print()
    
    # Create visualizer
    visualizer = CSVTradeVisualizer(
        trade_csv=args.trade_csv,
        market_csv=args.market_csv,
        external_market_csv=args.external_market_csv
    )
    
    # Load data
    if not visualizer.load_data():
        return
    
    # Create visualization
    save_path = args.save or f"trade_analysis_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualizer.create_visualization(
        save_path=save_path, 
        max_trades=args.max_trades,
        time_range_days=args.days,
        use_external_market=not args.no_external_market
    )
    
    print("✅ CSV-based visualization complete!")

if __name__ == "__main__":
    main()
