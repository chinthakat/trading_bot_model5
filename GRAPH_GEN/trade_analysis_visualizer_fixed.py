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
        if not market_file.exists():
            print(f"⚠️  Market data file not found: {market_file}")
            return
        
        try:
            print(f"📊 Loading market data from {market_file}")
            df = pd.read_csv(market_file)
            
            # Expected columns: timestamp, Close, Volume, etc.
            required_cols = ['timestamp', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Missing columns in market data: {missing_cols}")
                print(f"Available columns: {list(df.columns)}")
                return
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            # Store market timeline
            self.market_timeline = df[['datetime', 'Close', 'Volume']].copy()
            print(f"✅ Loaded {len(self.market_timeline)} market data points")
            print(f"📅 Date range: {self.market_timeline['datetime'].min()} to {self.market_timeline['datetime'].max()}")
            
        except Exception as e:
            print(f"❌ Error loading market data: {e}")
    
    def load_trade_traces(self):
        """Load trade traces from JSONL file using specific fields"""
        if not self.trace_file.exists():
            raise FileNotFoundError(f"Trade traces file not found: {self.trace_file}")
        
        print(f"📈 Loading trade traces from {self.trace_file}")
        
        trade_events = []
        market_data_points = []  # Embedded market data from traces
        
        try:
            with open(self.trace_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            # Required fields (prioritized list from user)
                            required_fields = [
                                'event_type', 'episode', 'trade_id', 'status', 
                                'BINANCEFTS_PERP_BTC_USDT', 'side', 'entry_action', 
                                'entry_price', 'entry_datetime', 'position_size', 
                                'Close', 'Volume'
                            ]
                            
                            # Check if this is a trade event
                            if data.get('event_type') == 'trade':
                                trade_events.append(data)
                                
                                # Extract embedded market data (Close, Volume, datetime)
                                if 'Close' in data and 'entry_datetime' in data:
                                    market_point = {
                                        'datetime': data['entry_datetime'],
                                        'Close': data['Close'],
                                        'Volume': data.get('Volume', 0)
                                    }
                                    market_data_points.append(market_point)
                            
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON decode error on line {line_num}: {e}")
                            continue
                        except Exception as e:
                            print(f"⚠️  Error processing line {line_num}: {e}")
                            continue
            
            print(f"✅ Loaded {len(trade_events)} trade events")
            print(f"✅ Extracted {len(market_data_points)} embedded market data points")
            
            # Process trades
            self.process_trades(trade_events)
            
            # Store embedded market data
            if market_data_points:
                df_market = pd.DataFrame(market_data_points)
                df_market['datetime'] = pd.to_datetime(df_market['datetime'])
                df_market = df_market.sort_values('datetime').drop_duplicates()
                
                # If no external market data, use embedded data
                if self.market_timeline is None or len(self.market_timeline) == 0:
                    self.market_timeline = df_market
                    print(f"📊 Using {len(self.market_timeline)} embedded market data points for timeline")
                else:
                    print(f"📊 External market data available, embedded data stored separately")
            
        except Exception as e:
            print(f"❌ Error loading trade traces: {e}")
            raise
    
    def process_trades(self, trade_events: List[Dict]):
        """Process trade events into complete trades"""
        trades_by_id = {}
        
        for event in trade_events:
            trade_id = event.get('trade_id')
            if not trade_id:
                continue
            
            if trade_id not in trades_by_id:
                trades_by_id[trade_id] = []
            
            trades_by_id[trade_id].append(event)
        
        print(f"📊 Processing {len(trades_by_id)} unique trades")
        
        # Create completed trades
        for trade_id, events in trades_by_id.items():
            try:
                # Sort by datetime
                events_sorted = sorted(events, key=lambda x: x.get('entry_datetime', ''))
                
                entry_event = None
                exit_event = None
                
                # Find entry and exit events
                for event in events_sorted:
                    if event.get('entry_action') in ['BUY', 'SELL'] and not entry_event:
                        entry_event = event
                    elif event.get('status') == 'CLOSED' and entry_event:
                        exit_event = event
                        break
                
                if entry_event and exit_event:
                    trade = self.create_trade_object(trade_id, entry_event, exit_event)
                    self.completed_trades.append(trade)
                elif entry_event:
                    # Open trade
                    trade = self.create_trade_object(trade_id, entry_event, None)
                    self.trades.append(trade)
                    
            except Exception as e:
                print(f"⚠️  Error processing trade {trade_id}: {e}")
                continue
        
        print(f"✅ Created {len(self.completed_trades)} completed trades")
        print(f"✅ Found {len(self.trades)} open trades")
    
    def create_trade_object(self, trade_id: str, entry_event: Dict, exit_event: Dict = None) -> Dict:
        """Create a standardized trade object"""
        trade = {
            'trade_id': trade_id,
            'episode': entry_event.get('episode'),
            'side': entry_event.get('side'),
            'entry_action': entry_event.get('entry_action'),
            'entry_price': float(entry_event.get('entry_price', 0)),
            'entry_datetime': pd.to_datetime(entry_event.get('entry_datetime')),
            'position_size': float(entry_event.get('position_size', 0)),
            'entry_close': float(entry_event.get('Close', 0)),
            'entry_volume': float(entry_event.get('Volume', 0)),
            'status': entry_event.get('status', 'OPEN')
        }
        
        if exit_event:
            trade.update({
                'exit_price': float(exit_event.get('entry_price', 0)),  # Exit price is in entry_price field of exit event
                'exit_datetime': pd.to_datetime(exit_event.get('entry_datetime')),
                'exit_close': float(exit_event.get('Close', 0)),
                'exit_volume': float(exit_event.get('Volume', 0)),
                'status': 'CLOSED'
            })
            
            # Calculate P&L
            if trade['side'] == 'LONG':
                trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['position_size']
            else:
                trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['position_size']
        
        return trade
    
    def create_visualization(self, output_file: str = "trade_analysis.png", show_plot: bool = True):
        """Create the main visualization"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot market data timeline if available
        if len(self.market_timeline) > 0:
            market_df = self.market_timeline.copy()
            ax.plot(market_df['datetime'], market_df['Close'], 
                   color='cyan', alpha=0.6, linewidth=0.8, label='Market Price')
            print(f"📈 Plotted {len(market_df)} market data points")
        
        # Plot completed trades (limit for clarity)
        recent_trades = self.completed_trades[-100:] if len(self.completed_trades) > 100 else self.completed_trades
        
        for trade in recent_trades:
            if trade['status'] == 'CLOSED':
                # Entry point
                color = 'lime' if trade['side'] == 'LONG' else 'red'
                ax.scatter(trade['entry_datetime'], trade['entry_price'], 
                          color=color, s=100, marker='^' if trade['side'] == 'LONG' else 'v',
                          alpha=0.8, zorder=5)
                
                # Exit point  
                exit_color = 'green' if trade.get('pnl', 0) > 0 else 'red'
                ax.scatter(trade['exit_datetime'], trade['exit_price'], 
                          color=exit_color, s=100, marker='X', alpha=0.8, zorder=5)
                
                # Connect entry to exit
                ax.plot([trade['entry_datetime'], trade['exit_datetime']], 
                       [trade['entry_price'], trade['exit_price']], 
                       color=exit_color, alpha=0.6, linewidth=2, zorder=4)
                
                # P&L label
                mid_time = trade['entry_datetime'] + (trade['exit_datetime'] - trade['entry_datetime']) / 2
                mid_price = (trade['entry_price'] + trade['exit_price']) / 2
                ax.annotate(f"${trade.get('pnl', 0):.2f}", 
                           xy=(mid_time, mid_price), xytext=(10, 10),
                           textcoords='offset points', fontsize=8, 
                           color='white', alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=exit_color, alpha=0.3))
        
        # Plot open trades
        for trade in self.trades:
            color = 'lime' if trade['side'] == 'LONG' else 'red'
            ax.scatter(trade['entry_datetime'], trade['entry_price'], 
                      color=color, s=100, marker='^' if trade['side'] == 'LONG' else 'v',
                      alpha=0.8, zorder=5)
            
            # Label open trades
            ax.annotate(f"OPEN", 
                       xy=(trade['entry_datetime'], trade['entry_price']), xytext=(10, 10),
                       textcoords='offset points', fontsize=8, 
                       color='yellow', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Time', fontsize=12, color='white')
        ax.set_ylabel('Price (USDT)', fontsize=12, color='white')
        ax.set_title(f'BTC/USDT Trade Analysis\n{len(self.completed_trades)} completed trades, {len(self.trades)} open trades', 
                    fontsize=14, color='white', pad=20)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        # Grid
        ax.grid(True, alpha=0.3, color='gray')
        
        # Legend
        ax.legend(loc='upper left', fontsize=10)
        
        # Add statistics
        if self.completed_trades:
            total_pnl = sum(trade.get('pnl', 0) for trade in self.completed_trades)
            profitable_trades = sum(1 for trade in self.completed_trades if trade.get('pnl', 0) > 0)
            win_rate = profitable_trades / len(self.completed_trades) * 100
            
            stats_text = f"Total P&L: ${total_pnl:.2f}\nWin Rate: {win_rate:.1f}%"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"💾 Saved visualization to {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()

def main():
    """Main function to run the visualizer"""
    parser = argparse.ArgumentParser(description='Visualize trade analysis from JSONL traces')
    parser.add_argument('trace_file', help='Path to trade traces JSONL file')
    parser.add_argument('--market-data', help='Path to market data CSV file (optional)')
    parser.add_argument('--output', default='trade_analysis.png', help='Output image file')
    parser.add_argument('--no-show', action='store_true', help='Don\'t show plot window')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TradeAnalysisVisualizer(args.trace_file, args.market_data)
    
    # Load data
    visualizer.load_market_data()
    visualizer.load_trade_traces()
    
    # Create visualization
    visualizer.create_visualization(args.output, not args.no_show)

if __name__ == '__main__':
    # For direct execution
    trace_file = r"c:\Projects\Model5\logs\trade_traces\trade_traces.jsonl"
    market_data_file = r"c:\Projects\Model5\data\BINANCEFTS_PERP_BTC_USDT_15m_2024-01-01_to_2025-04-01_consolidated.csv"
    
    print("🚀 Starting Trade Analysis Visualization")
    
    visualizer = TradeAnalysisVisualizer(trace_file, market_data_file)
    
    # Load data
    visualizer.load_market_data()
    visualizer.load_trade_traces()
    
    # Create visualization
    output_file = r"c:\Projects\Model5\GRAPH_GEN\trade_analysis_fixed.png"
    visualizer.create_visualization(output_file, True)
    
    print("✅ Visualization complete!")
