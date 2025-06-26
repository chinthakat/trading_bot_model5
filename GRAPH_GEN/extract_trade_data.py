#!/usr/bin/env python3
"""
Trade Data Extractor
Extracts relevant fields from trade_traces.jsonl into a clean CSV format
Fields: event_type, episode, trade_id, status, symbol, side, entry_action, 
        entry_price, entry_datetime, position_size, Close, Volume, net_pnl
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse

class TradeDataExtractor:
    """Extract trade data from traces into CSV format"""
    
    def __init__(self, trace_file: str):
        self.trace_file = Path(trace_file)
        self.trade_events = []
        self.market_data_points = []
        
    def extract_data(self):
        """Extract all relevant data from trade traces"""
        print(f"📖 Extracting data from: {self.trace_file}")
        
        if not self.trace_file.exists():
            print(f"❌ Trace file not found: {self.trace_file}")
            return
            
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
                        # Extract entry fields
                        entry_action = event_data.get('entry_action', '')
                        entry_price = float(event_data.get('entry_price', 0))
                        entry_datetime = event_data.get('entry_datetime', '')
                        position_size = float(event_data.get('position_size', 0))
                        
                        # Extract market data at entry
                        market_entry = event_data.get('market_data_at_entry', {})
                        close_price = float(market_entry.get('Close', entry_price))
                        volume = float(market_entry.get('Volume', 0))
                        market_timestamp = market_entry.get('timestamp', 0)
                        
                        # Store trade event
                        trade_event = {
                            'event_type': event_type,
                            'episode': episode,
                            'trade_id': trade_id,
                            'status': status,
                            'symbol': symbol,
                            'side': side,
                            'entry_action': entry_action,
                            'entry_price': entry_price,
                            'entry_datetime': entry_datetime,
                            'position_size': position_size,
                            'market_close': close_price,
                            'market_volume': volume,
                            'market_timestamp': market_timestamp,
                            'exit_price': None,
                            'exit_datetime': None,
                            'net_pnl': None,
                            'win_loss': None
                        }
                        
                        self.trade_events.append(trade_event)
                        
                        # Store market data point
                        if market_timestamp:
                            market_point = {
                                'timestamp': market_timestamp,
                                'datetime': pd.to_datetime(market_timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                                'close': close_price,
                                'volume': volume,
                                'source': 'entry',
                                'trade_id': trade_id
                            }
                            self.market_data_points.append(market_point)
                        
                    elif event_type == 'TRADE_CLOSED':
                        # Extract exit fields
                        exit_price = float(event_data.get('exit_price', 0))
                        exit_datetime = event_data.get('exit_datetime', '')
                        net_pnl = float(event_data.get('net_pnl', 0))
                        win_loss = event_data.get('win_loss', 'UNKNOWN')
                        
                        # Extract market data at exit
                        market_exit = event_data.get('market_data_at_exit', {})
                        close_price_exit = float(market_exit.get('Close', exit_price))
                        volume_exit = float(market_exit.get('Volume', 0))
                        market_timestamp_exit = market_exit.get('timestamp', 0)
                        
                        # Find corresponding OPEN event and update it
                        for trade_event in self.trade_events:
                            if trade_event['trade_id'] == trade_id and trade_event['event_type'] == 'TRADE_OPENED':
                                trade_event.update({
                                    'exit_price': exit_price,
                                    'exit_datetime': exit_datetime,
                                    'net_pnl': net_pnl,
                                    'win_loss': win_loss,
                                    'market_close_exit': close_price_exit,
                                    'market_volume_exit': volume_exit,
                                    'market_timestamp_exit': market_timestamp_exit
                                })
                                break
                        
                        # Store exit market data point
                        if market_timestamp_exit:
                            market_point = {
                                'timestamp': market_timestamp_exit,
                                'datetime': pd.to_datetime(market_timestamp_exit, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                                'close': close_price_exit,
                                'volume': volume_exit,
                                'source': 'exit',
                                'trade_id': trade_id
                            }
                            self.market_data_points.append(market_point)
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON on line {line_num}")
                except Exception as e:
                    print(f"⚠️  Error processing line {line_num}: {e}")
            
            print(f"✅ Extracted {len(self.trade_events)} trade events")
            print(f"✅ Extracted {len(self.market_data_points)} market data points")
            
        except Exception as e:
            print(f"❌ Error extracting data: {e}")
    
    def save_to_csv(self, trade_csv_path: str = "extracted_trades.csv", 
                    market_csv_path: str = "extracted_market_data.csv"):
        """Save extracted data to CSV files"""
        
        if not self.trade_events:
            print("❌ No trade data to save")
            return
            
        # Save trade events
        trade_df = pd.DataFrame(self.trade_events)
        trade_df.to_csv(trade_csv_path, index=False)
        print(f"💾 Trade data saved to: {trade_csv_path}")
        print(f"📊 Columns: {list(trade_df.columns)}")
        
        # Save market data points
        if self.market_data_points:
            market_df = pd.DataFrame(self.market_data_points)
            market_df = market_df.sort_values('timestamp').drop_duplicates()
            market_df.to_csv(market_csv_path, index=False)
            print(f"💾 Market data saved to: {market_csv_path}")
            print(f"📈 Market data points: {len(market_df)}")
        
        # Show sample data
        print("\\n📋 Sample trade data:")
        completed_trades = trade_df[trade_df['net_pnl'].notna()]
        if not completed_trades.empty:
            sample = completed_trades.head(3)[['trade_id', 'side', 'entry_price', 'exit_price', 'net_pnl', 'win_loss']]
            print(sample.to_string(index=False))
        
        print("\\n📋 Sample market data:")
        if self.market_data_points:
            market_sample = pd.DataFrame(self.market_data_points).head(3)
            print(market_sample[['datetime', 'close', 'volume', 'source']].to_string(index=False))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract trade data from traces to CSV")
    parser.add_argument("--trace-file", "-t", 
                       default="logs/trade_traces/trade_traces.jsonl",
                       help="Path to trade traces JSONL file")
    parser.add_argument("--trade-csv", "-tc",
                       default="extracted_trades.csv",
                       help="Output path for trade data CSV")
    parser.add_argument("--market-csv", "-mc",
                       default="extracted_market_data.csv", 
                       help="Output path for market data CSV")
    
    args = parser.parse_args()
    
    print("🚀 Starting Trade Data Extraction")
    print(f"📁 Input: {args.trace_file}")
    print(f"📄 Output trades: {args.trade_csv}")
    print(f"📈 Output market: {args.market_csv}")
    print()
    
    # Extract data
    extractor = TradeDataExtractor(args.trace_file)
    extractor.extract_data()
    extractor.save_to_csv(args.trade_csv, args.market_csv)
    
    print("\\n✅ Data extraction complete!")
    print("💡 You can now use these CSV files for visualization")

if __name__ == "__main__":
    main()
