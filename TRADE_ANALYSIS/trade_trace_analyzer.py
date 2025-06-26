#!/usr/bin/env python3
"""
Trade Trace Analyzer
Comprehensive analysis of trade traces to extract:
1. Trade ID and time (entry/close)
2. Buy or Sell action
3. Entry, close price
4. PNL at close
5. Reward at entry
6. Reward at close
7. Entry/close net worth
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

class TradeTraceAnalyzer:
    """Analyze trade traces and extract comprehensive trade information"""
    
    def __init__(self, trace_file: str):
        """
        Initialize the analyzer
        
        Args:
            trace_file: Path to trade traces JSONL file
        """
        self.trace_file = Path(trace_file)
        self.trade_events = []
        self.analysis_data = []
        
        print(f"🔍 Trade Trace Analyzer initialized")
        print(f"📁 Trace file: {self.trace_file}")
    
    def load_trace_data(self):
        """Load and parse trade trace data"""
        if not self.trace_file.exists():
            raise FileNotFoundError(f"Trade traces file not found: {self.trace_file}")
        
        print(f"📖 Loading trade traces from {self.trace_file}")
        
        with open(self.trace_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        self.trade_events.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON decode error on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️  Error processing line {line_num}: {e}")
                        continue
        
        print(f"✅ Loaded {len(self.trade_events)} trace events")
    
    def extract_trade_metadata(self, event: Dict) -> Dict:
        """Extract metadata from trace event"""
        trace_metadata = event.get('trace_metadata', {})
        return {
            'event_type': trace_metadata.get('event_type', ''),
            'timestamp': trace_metadata.get('timestamp', ''),
            'episode': trace_metadata.get('episode', 0),
            'step': trace_metadata.get('step', 0)
        }
    
    def extract_event_data(self, event: Dict) -> Dict:
        """Extract event data from trace event"""
        event_data = event.get('event_data', {})
        
        # Extract portfolio data from observations
        observation_entry = event_data.get('observation_at_entry', {})
        observation_exit = event_data.get('observation_at_exit', {})
        portfolio_entry = observation_entry.get('portfolio_overview', {})
        portfolio_exit = observation_exit.get('portfolio_overview', {})
        
        # Extract reward analysis if available (for closed trades)
        trade_analysis = event_data.get('trade_analysis', {})
        reward_analysis = trade_analysis.get('reward_analysis', {})
        
        return {
            'trade_id': event_data.get('trade_id', ''),
            'status': event_data.get('status', ''),
            'symbol': event_data.get('symbol', ''),
            'side': event_data.get('side', ''),
            'entry_action': event_data.get('entry_action', ''),
            'entry_price': event_data.get('entry_price', 0),
            'entry_datetime': event_data.get('entry_datetime', ''),
            'position_size': event_data.get('position_size', 0),
            'exit_price': event_data.get('exit_price', 0),
            'exit_datetime': event_data.get('exit_datetime', ''),
            'net_pnl': event_data.get('net_pnl', 0),
            'win_loss': event_data.get('win_loss', ''),
            
            # Balance and equity data
            'balance_before_entry': event_data.get('balance_before_entry', 0),
            'balance_after_entry': event_data.get('balance_after_entry', 0),
            'balance_before_exit': event_data.get('balance_before_exit', 0),
            'balance_after_exit': event_data.get('balance_after_exit', 0),
            'equity_change': event_data.get('equity_change', 0),
            
            # Portfolio overview data
            'equity_entry': portfolio_entry.get('equity', 0),
            'equity_exit': portfolio_exit.get('equity', 0),
            'normalized_balance_entry': portfolio_entry.get('normalized_balance', 0),
            'normalized_balance_exit': portfolio_exit.get('normalized_balance', 0),
            
            # Reward data from trade analysis
            'entry_reward_points': reward_analysis.get('entry_reward_points', 0),
            'exit_reward_points': reward_analysis.get('exit_reward_points', 0),
            'total_reward_points': reward_analysis.get('total_reward_points', 0),
            'reward_efficiency': reward_analysis.get('reward_efficiency', 0),
            
            # Additional trade analysis
            'gross_pnl': event_data.get('gross_pnl', 0),
            'holding_period_return': event_data.get('holding_period_return', 0),
            'return_on_risk': event_data.get('return_on_risk', 0),
            'risk_reward_ratio': event_data.get('risk_reward_ratio', 0),
            'total_commission': event_data.get('total_commission', 0),
            'entry_commission': event_data.get('entry_commission', 0),
            'exit_commission': event_data.get('exit_commission', 0)
        }
    
    def extract_market_data(self, event: Dict, phase: str = 'entry') -> Dict:
        """Extract market data from trace event"""
        event_data = event.get('event_data', {})
        
        if phase == 'entry':
            market_data = event_data.get('market_data_at_entry', {})
        else:  # exit
            market_data = event_data.get('market_data_at_exit', {})
        
        return {
            f'{phase}_market_open': market_data.get('Open', 0),
            f'{phase}_market_high': market_data.get('High', 0),
            f'{phase}_market_low': market_data.get('Low', 0),
            f'{phase}_market_close': market_data.get('Close', 0),
            f'{phase}_market_volume': market_data.get('Volume', 0)
        }
    
    def analyze_trades(self):
        """Analyze trade events and extract comprehensive data"""
        print("📊 Analyzing trade events...")
        
        trades_by_id = {}  # Group events by trade_id
        
        # Group events by trade_id
        for event in self.trade_events:
            metadata = self.extract_trade_metadata(event)
            event_data = self.extract_event_data(event)
            
            trade_id = event_data['trade_id']
            if not trade_id:
                continue
            
            if trade_id not in trades_by_id:
                trades_by_id[trade_id] = []
            
            # Combine all data
            full_event = {**metadata, **event_data}
            
            # Add market data if available
            if metadata['event_type'] == 'TRADE_OPENED':
                market_entry = self.extract_market_data(event, 'entry')
                full_event.update(market_entry)
            elif metadata['event_type'] == 'TRADE_CLOSED':
                market_exit = self.extract_market_data(event, 'exit')
                full_event.update(market_exit)
            
            trades_by_id[trade_id].append(full_event)
        
        print(f"📈 Found {len(trades_by_id)} unique trades")
        
        # Process each trade
        for trade_id, events in trades_by_id.items():
            try:
                trade_analysis = self.process_single_trade(trade_id, events)
                if trade_analysis:
                    self.analysis_data.append(trade_analysis)
            except Exception as e:
                print(f"⚠️  Error processing trade {trade_id}: {e}")
                continue
        
        print(f"✅ Analyzed {len(self.analysis_data)} trades")
    
    def process_single_trade(self, trade_id: str, events: List[Dict]) -> Optional[Dict]:
        """Process a single trade's events"""
        # Sort events by timestamp
        events_sorted = sorted(events, key=lambda x: x.get('timestamp', ''))
        
        entry_event = None
        close_event = None
        
        # Find entry and close events
        for event in events_sorted:
            if event.get('event_type') == 'TRADE_OPENED' and not entry_event:
                entry_event = event
            elif event.get('event_type') == 'TRADE_CLOSED' and entry_event:
                close_event = event
                break
        
        if not entry_event:
            return None
        
        # Basic trade information
        analysis = {
            'trade_id': trade_id,
            'episode': entry_event.get('episode', 0),
            'step_entry': entry_event.get('step', 0),
            
            # Entry information
            'entry_datetime': pd.to_datetime(entry_event.get('entry_datetime', '')).strftime('%Y-%m-%d %H:%M:%S') if entry_event.get('entry_datetime') else '',
            'entry_timestamp': entry_event.get('timestamp', ''),
            'side': entry_event.get('side', ''),  # LONG or SHORT
            'entry_action': entry_event.get('entry_action', ''),  # BUY or SELL
            'entry_price': float(entry_event.get('entry_price', 0)),
            'position_size': float(entry_event.get('position_size', 0)),
            
            # Entry rewards and financial data
            'entry_reward': 0,  # Entry rewards are typically 0 (reward calculated at exit)
            'entry_net_worth': float(entry_event.get('equity_entry', 0)),
            'entry_balance': float(entry_event.get('balance_after_entry', 0)),
            'entry_equity': float(entry_event.get('equity_entry', 0)),
            'entry_balance_before': float(entry_event.get('balance_before_entry', 0)),
            'entry_normalized_balance': float(entry_event.get('normalized_balance_entry', 0)),
            
            # Entry market data
            'entry_market_open': float(entry_event.get('entry_market_open', 0)),
            'entry_market_high': float(entry_event.get('entry_market_high', 0)),
            'entry_market_low': float(entry_event.get('entry_market_low', 0)),
            'entry_market_close': float(entry_event.get('entry_market_close', 0)),
            'entry_market_volume': float(entry_event.get('entry_market_volume', 0)),
            
            # Commission data
            'entry_commission': float(entry_event.get('entry_commission', 0)),
        }
        
        # Close information (if available)
        if close_event:
            analysis.update({
                'status': 'CLOSED',
                'step_close': close_event.get('step', 0),
                
                # Close information
                'close_datetime': pd.to_datetime(close_event.get('exit_datetime', '')).strftime('%Y-%m-%d %H:%M:%S') if close_event.get('exit_datetime') else '',
                'close_timestamp': close_event.get('timestamp', ''),
                'close_price': float(close_event.get('exit_price', 0)),
                'net_pnl': float(close_event.get('net_pnl', 0)),
                'gross_pnl': float(close_event.get('gross_pnl', 0)),
                'win_loss': close_event.get('win_loss', ''),
                
                # Close rewards and financial data
                'close_reward': float(close_event.get('reward_efficiency', 0)),  # This is the actual reward (same as PnL)
                'total_reward_points': float(close_event.get('reward_efficiency', 0)),  # Same as close_reward
                'reward_efficiency': float(close_event.get('reward_efficiency', 0)),
                'close_net_worth': float(close_event.get('equity_exit', 0)),
                'close_balance': float(close_event.get('balance_after_exit', 0)),
                'close_equity': float(close_event.get('equity_exit', 0)),
                'close_balance_before': float(close_event.get('balance_before_exit', 0)),
                'equity_change': float(close_event.get('equity_change', 0)),
                'close_normalized_balance': float(close_event.get('normalized_balance_exit', 0)),
                
                # Trade performance metrics
                'holding_period_return': float(close_event.get('holding_period_return', 0)),
                'return_on_risk': float(close_event.get('return_on_risk', 0)),
                'risk_reward_ratio': float(close_event.get('risk_reward_ratio', 0)),
                
                # Commission data
                'exit_commission': float(close_event.get('exit_commission', 0)),
                'total_commission': float(close_event.get('total_commission', 0)),
                
                # Close market data
                'close_market_open': float(close_event.get('exit_market_open', 0)),
                'close_market_high': float(close_event.get('exit_market_high', 0)),
                'close_market_low': float(close_event.get('exit_market_low', 0)),
                'close_market_close': float(close_event.get('exit_market_close', 0)),
                'close_market_volume': float(close_event.get('exit_market_volume', 0)),
                
                # Duration
                'trade_duration_steps': close_event.get('step', 0) - entry_event.get('step', 0),
            })
            
            # Calculate time duration if both datetimes available
            if analysis['entry_datetime'] and analysis['close_datetime']:
                try:
                    entry_dt = pd.to_datetime(analysis['entry_datetime'])
                    close_dt = pd.to_datetime(analysis['close_datetime'])
                    duration = close_dt - entry_dt
                    analysis['trade_duration_minutes'] = duration.total_seconds() / 60
                    analysis['trade_duration_hours'] = duration.total_seconds() / 3600
                except:
                    analysis['trade_duration_minutes'] = 0
                    analysis['trade_duration_hours'] = 0
        else:
            # Open trade
            analysis.update({
                'status': 'OPEN',
                'step_close': None,
                'close_datetime': '',
                'close_timestamp': '',
                'close_price': 0,
                'net_pnl': 0,
                'gross_pnl': 0,
                'win_loss': '',
                'close_reward': 0,
                'total_reward_points': 0,
                'reward_efficiency': 0,
                'close_net_worth': 0,
                'close_balance': 0,
                'close_equity': 0,
                'close_balance_before': 0,
                'equity_change': 0,
                'close_normalized_balance': 0,
                'holding_period_return': 0,
                'return_on_risk': 0,
                'risk_reward_ratio': 0,
                'exit_commission': 0,
                'total_commission': float(entry_event.get('entry_commission', 0)),
                'close_market_open': 0,
                'close_market_high': 0,
                'close_market_low': 0,
                'close_market_close': 0,
                'close_market_volume': 0,
                'trade_duration_steps': 0,
                'trade_duration_minutes': 0,
                'trade_duration_hours': 0
            })
        
        return analysis
    
    def save_analysis(self, output_file: str, create_minimal: bool = True):
        """Save analysis results to CSV files"""
        if not self.analysis_data:
            print("❌ No analysis data to save")
            return
        
        df = pd.DataFrame(self.analysis_data)
        
        # Sort by entry datetime
        df = df.sort_values('entry_datetime')
        
        # Save detailed file
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        
        print(f"💾 Detailed analysis saved to: {output_path}")
        print(f"📊 Saved {len(df)} trade records with {len(df.columns)} columns")
        
        # Create minimal file if requested
        if create_minimal:
            minimal_path = self.create_minimal_file(df, output_path)
        
        # Print summary statistics
        self.print_summary(df)
        
        return output_path
    
    def create_minimal_file(self, df: pd.DataFrame, original_path: Path) -> Path:
        """Create a minimal CSV file with only essential trade data"""
        
        # Define essential columns for minimal output
        minimal_columns = [
            'trade_id',
            'entry_datetime',
            'close_datetime', 
            'side',
            'entry_action',
            'entry_price',
            'close_price',
            'net_pnl',
            'close_reward',
            'entry_net_worth',
            'close_net_worth',
            'trade_duration_hours',
            'status',
            'win_loss'
        ]
        
        # Filter to only existing columns
        available_columns = [col for col in minimal_columns if col in df.columns]
        df_minimal = df[available_columns].copy()
        
        # Round numeric columns for cleaner output
        numeric_columns = ['entry_price', 'close_price', 'net_pnl', 'close_reward', 
                          'entry_net_worth', 'close_net_worth', 'trade_duration_hours']
        for col in numeric_columns:
            if col in df_minimal.columns:
                df_minimal[col] = df_minimal[col].round(4)
        
        # Sort by entry datetime
        df_minimal = df_minimal.sort_values('entry_datetime')
        
        # Create minimal file path
        minimal_path = original_path.parent / f"{original_path.stem}_minimal{original_path.suffix}"
        df_minimal.to_csv(minimal_path, index=False)
        
        print(f"💾 Minimal analysis saved to: {minimal_path}")
        print(f"📊 Minimal file contains {len(df_minimal)} trades with {len(available_columns)} essential columns")
        
        return minimal_path

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print("\n📊 Trade Analysis Summary:")
        print(f"   Total trades: {len(df)}")
        
        closed_trades = df[df['status'] == 'CLOSED']
        open_trades = df[df['status'] == 'OPEN']
        
        print(f"   Closed trades: {len(closed_trades)}")
        print(f"   Open trades: {len(open_trades)}")
        
        if len(closed_trades) > 0:
            total_pnl = closed_trades['net_pnl'].sum()
            profitable_trades = len(closed_trades[closed_trades['net_pnl'] > 0])
            win_rate = profitable_trades / len(closed_trades) * 100
            avg_pnl = closed_trades['net_pnl'].mean()
            
            print(f"   Win rate: {win_rate:.1f}%")
            print(f"   Total PnL: ${total_pnl:.2f}")
            print(f"   Average PnL: ${avg_pnl:.2f}")
            print(f"   Best trade: ${closed_trades['net_pnl'].max():.2f}")
            print(f"   Worst trade: ${closed_trades['net_pnl'].min():.2f}")
            
            # Side distribution
            print(f"   LONG trades: {len(df[df['side'] == 'LONG'])}")
            print(f"   SHORT trades: {len(df[df['side'] == 'SHORT'])}")
            
            # Episode distribution
            print(f"   Episodes: {df['episode'].min()} to {df['episode'].max()}")
            print(f"   Date range: {df['entry_datetime'].min()} to {df['entry_datetime'].max()}")
            
            # Duration statistics for closed trades
            if 'trade_duration_hours' in closed_trades.columns:
                avg_duration = closed_trades['trade_duration_hours'].mean()
                print(f"   Average trade duration: {avg_duration:.1f} hours")
    
    def create_detailed_report(self, output_dir: str):
        """Create detailed analysis reports"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.analysis_data:
            print("❌ No analysis data to create reports")
            return
        
        df = pd.DataFrame(self.analysis_data)
        
        # 1. Main analysis file (detailed)
        main_file = output_path / "trade_analysis_detailed.csv"
        df.to_csv(main_file, index=False)
        print(f"💾 Detailed analysis saved to: {main_file}")
        
        # 2. Create minimal file
        minimal_path = self.create_minimal_file(df, main_file)
        
        # 3. Summary by episode
        if 'episode' in df.columns:
            episode_summary = df.groupby('episode').agg({
                'trade_id': 'count',
                'net_pnl': ['sum', 'mean', 'count'],
                'entry_net_worth': ['first', 'last'],
                'close_net_worth': ['first', 'last']
            }).round(2)
            
            episode_file = output_path / "episode_summary.csv"
            episode_summary.to_csv(episode_file)
            print(f"💾 Episode summary saved to: {episode_file}")
        
        # 4. Performance metrics
        closed_trades = df[df['status'] == 'CLOSED']
        if len(closed_trades) > 0:
            metrics = {
                'total_trades': len(df),
                'closed_trades': len(closed_trades),
                'open_trades': len(df[df['status'] == 'OPEN']),
                'win_rate': len(closed_trades[closed_trades['net_pnl'] > 0]) / len(closed_trades) * 100,
                'total_pnl': closed_trades['net_pnl'].sum(),
                'average_pnl': closed_trades['net_pnl'].mean(),
                'best_trade': closed_trades['net_pnl'].max(),
                'worst_trade': closed_trades['net_pnl'].min(),
                'long_trades': len(df[df['side'] == 'LONG']),
                'short_trades': len(df[df['side'] == 'SHORT']),
                'average_duration_hours': closed_trades['trade_duration_hours'].mean() if 'trade_duration_hours' in closed_trades.columns else 0
            }
            
            metrics_df = pd.DataFrame([metrics])
            metrics_file = output_path / "performance_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"💾 Performance metrics saved to: {metrics_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive Trade Trace Analysis")
    parser.add_argument("--trace-file", "-t", 
                       default=r"c:\Projects\Model5\logs\trade_traces\trade_traces.jsonl",
                       help="Path to trade traces JSONL file")
    parser.add_argument("--output", "-o",
                       default="trade_analysis.csv",
                       help="Output CSV file path")
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Create detailed reports in output directory")
    parser.add_argument("--output-dir", "-od",
                       default=".",
                       help="Output directory for detailed reports")
    
    args = parser.parse_args()
    
    print("🚀 Starting Trade Trace Analysis")
    print("📊 Extracting: Trade ID, Time, Buy/Sell, Prices, PnL, Rewards, Net Worth")
    print()
    
    # Create analyzer
    analyzer = TradeTraceAnalyzer(args.trace_file)
    
    # Load and analyze data
    analyzer.load_trace_data()
    analyzer.analyze_trades()
    
    # Save results
    if args.detailed:
        analyzer.create_detailed_report(args.output_dir)
    else:
        analyzer.save_analysis(args.output)
    
    print("✅ Trade analysis complete!")

if __name__ == "__main__":
    main()
