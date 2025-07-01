#!/usr/bin/env python3
"""
Quick CSV Analysis Tool
Load and analyze the trade analysis CSV file
"""

import pandas as pd
import sys
from pathlib import Path

def analyze_csv(csv_file: str):
    """Analyze the trade analysis CSV file"""
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📊 Loading CSV file: {csv_path}")
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        print(f"✅ Loaded {len(df)} trade records")
        print(f"📅 Date range: {df['entry_datetime'].min()} to {df['entry_datetime'].max()}")
        print()
        
        # Show basic info
        print("📋 Column Information:")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Shape: {df.shape}")
        print()
        
        # Show sample data
        print("📖 Sample Data (first 5 trades):")
        sample_cols = ['trade_id', 'entry_datetime', 'side', 'entry_price', 'close_price', 'net_pnl', 'entry_reward', 'close_reward']
        if all(col in df.columns for col in sample_cols):
            print(df[sample_cols].head().to_string(index=False))
        else:
            print(df.head().to_string(index=False))
        print()
        
        # Basic statistics
        if 'net_pnl' in df.columns:
            print("💰 Performance Summary:")
            profitable = df[df['net_pnl'] > 0]
            print(f"   Total trades: {len(df)}")
            print(f"   Profitable trades: {len(profitable)}")
            print(f"   Win rate: {len(profitable)/len(df)*100:.1f}%")
            print(f"   Total PnL: ${df['net_pnl'].sum():.2f}")
            print(f"   Average PnL: ${df['net_pnl'].mean():.2f}")
            print(f"   Best trade: ${df['net_pnl'].max():.2f}")
            print(f"   Worst trade: ${df['net_pnl'].min():.2f}")
            print()
        
        # Reward analysis
        if 'close_reward' in df.columns:
            print("🎯 Reward Analysis:")
            print(f"   Reward range: {df['close_reward'].min():.3f} to {df['close_reward'].max():.3f}")
            print(f"   Average reward: {df['close_reward'].mean():.3f}")
            print(f"   Reward = PnL? {(df['close_reward'] == df['net_pnl']).all()}")
            print()
        
        # Net worth analysis
        if 'entry_net_worth' in df.columns and 'close_net_worth' in df.columns:
            print("💼 Net Worth Analysis:")
            print(f"   Starting net worth: ${df['entry_net_worth'].iloc[0]:.2f}")
            print(f"   Final net worth: ${df['close_net_worth'].iloc[-1]:.2f}")
            print(f"   Net worth change: ${df['close_net_worth'].iloc[-1] - df['entry_net_worth'].iloc[0]:.2f}")
            print()
        
        # Side distribution
        if 'side' in df.columns:
            print("📈 Trade Distribution:")
            side_counts = df['side'].value_counts()
            for side, count in side_counts.items():
                print(f"   {side}: {count} trades ({count/len(df)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Default to the corrected simplified file
        csv_file = r"TRADE_ANALYSIS\trade_analysis_corrected_simplified.csv"
    
    analyze_csv(csv_file)

if __name__ == "__main__":
    main()
