#!/usr/bin/env python3
"""
Trade Analysis Summary Generator
Creates readable summaries from the detailed trade analysis
"""

import pandas as pd
from pathlib import Path

def create_readable_summary():
    """Create readable summary of trade analysis"""
    
    # Load detailed analysis
    df = pd.read_csv('trade_analysis_detailed.csv')
    
    print("🔍 COMPREHENSIVE TRADE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Basic Statistics
    print(f"\n📊 OVERVIEW:")
    print(f"   Total Trades: {len(df):,}")
    print(f"   Closed Trades: {len(df[df['status'] == 'CLOSED']):,}")
    print(f"   Open Trades: {len(df[df['status'] == 'OPEN']):,}")
    
    # Time Range
    print(f"\n📅 TIME RANGE:")
    print(f"   Start: {df['entry_datetime'].min()}")
    print(f"   End: {df['entry_datetime'].max()}")
    print(f"   Episodes: {df['episode'].min()} to {df['episode'].max()}")
    
    # Performance
    closed_trades = df[df['status'] == 'CLOSED']
    if len(closed_trades) > 0:
        total_pnl = closed_trades['net_pnl'].sum()
        profitable_trades = len(closed_trades[closed_trades['net_pnl'] > 0])
        win_rate = profitable_trades / len(closed_trades) * 100
        
        print(f"\n💰 PERFORMANCE:")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Average PnL: ${closed_trades['net_pnl'].mean():.2f}")
        print(f"   Best Trade: ${closed_trades['net_pnl'].max():.2f}")
        print(f"   Worst Trade: ${closed_trades['net_pnl'].min():.2f}")
        
        # Trade Distribution
        print(f"\n📈 TRADE DISTRIBUTION:")
        print(f"   LONG Trades: {len(df[df['side'] == 'LONG']):,}")
        print(f"   SHORT Trades: {len(df[df['side'] == 'SHORT']):,}")
        print(f"   BUY Actions: {len(df[df['entry_action'] == 'BUY']):,}")
        print(f"   SELL Actions: {len(df[df['entry_action'] == 'SELL']):,}")
        
        # Duration
        if 'trade_duration_hours' in closed_trades.columns:
            avg_duration = closed_trades['trade_duration_hours'].mean()
            print(f"   Average Duration: {avg_duration:.1f} hours ({avg_duration*60:.0f} minutes)")
    
    # Sample trades
    print(f"\n📋 SAMPLE COMPLETED TRADES:")
    if len(closed_trades) > 0:
        sample_cols = ['trade_id', 'entry_datetime', 'side', 'entry_price', 'close_price', 'net_pnl', 'win_loss']
        sample = closed_trades[sample_cols].head(5)
        print(sample.to_string(index=False))
    
    # Top winning and losing trades
    if len(closed_trades) > 0:
        print(f"\n🏆 TOP 3 WINNING TRADES:")
        winners = closed_trades.nlargest(3, 'net_pnl')[['trade_id', 'entry_datetime', 'side', 'net_pnl']]
        print(winners.to_string(index=False))
        
        print(f"\n💸 TOP 3 LOSING TRADES:")
        losers = closed_trades.nsmallest(3, 'net_pnl')[['trade_id', 'entry_datetime', 'side', 'net_pnl']]
        print(losers.to_string(index=False))
    
    # Net Worth Analysis
    print(f"\n💼 NET WORTH ANALYSIS:")
    print(f"   Entry Net Worth Range: ${df['entry_net_worth'].min():.2f} to ${df['entry_net_worth'].max():.2f}")
    if 'close_net_worth' in df.columns:
        close_nw = df[df['close_net_worth'] != 0]['close_net_worth']
        if len(close_nw) > 0:
            print(f"   Close Net Worth Range: ${close_nw.min():.2f} to ${close_nw.max():.2f}")
    
    # Reward Analysis
    print(f"\n🎯 REWARD ANALYSIS:")
    print(f"   Entry Reward Range: {df['entry_reward'].min():.3f} to {df['entry_reward'].max():.3f}")
    if 'close_reward' in df.columns:
        close_rewards = df[df['close_reward'] != 0]['close_reward']
        if len(close_rewards) > 0:
            print(f"   Close Reward Range: {close_rewards.min():.3f} to {close_rewards.max():.3f}")
    
    print(f"\n📄 FILES CREATED:")
    print(f"   📊 trade_analysis_detailed.csv - Complete trade data")
    print(f"   📈 performance_metrics.csv - Key performance indicators")
    print(f"   📅 episode_summary.csv - Summary by episode")
    
    # Create a simplified view
    simplified_cols = [
        'trade_id', 'entry_datetime', 'close_datetime', 'side', 'entry_action',
        'entry_price', 'close_price', 'net_pnl', 'win_loss',
        'entry_reward', 'close_reward', 'entry_net_worth', 'close_net_worth'
    ]
    
    simplified_df = df[simplified_cols]
    simplified_df.to_csv('trade_analysis_simplified.csv', index=False)
    print(f"   📝 trade_analysis_simplified.csv - Key fields only")
    
    print(f"\n✅ Analysis complete! All requested data extracted:")
    print(f"   ✓ Trade ID and time (entry/close)")
    print(f"   ✓ Buy or Sell action")
    print(f"   ✓ Entry, close price") 
    print(f"   ✓ PNL at close")
    print(f"   ✓ Reward at entry")
    print(f"   ✓ Reward at close")
    print(f"   ✓ Entry/close net worth")

if __name__ == "__main__":
    create_readable_summary()
