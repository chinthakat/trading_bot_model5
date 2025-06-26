#!/usr/bin/env python3
"""
Test script to verify that the Enhanced Live Trade Visualizer logs data properly
"""

import sys
from pathlib import Path
sys.path.append('GRAPH_GEN')

from live_trade_visualizer_enhanced import EnhancedLiveTradeVisualizer

def test_logging():
    """Test that the visualizer logs data properly"""
    
    # Create a test visualizer instance
    print("🧪 Testing Enhanced Live Trade Visualizer logging...")
    
    visualizer = EnhancedLiveTradeVisualizer(
        trace_file="logs/trade_traces/trade_traces.jsonl",
        update_interval=1.0,
        max_points=100,
        log_file="logs/visualization/test_logging.log"
    )
    
    print(f"✅ Visualizer created successfully")
    print(f"📄 Log file: {visualizer.log_file_path}")
    
    # Test loading existing data (this should trigger logging)
    print("🔄 Loading existing data...")
    visualizer.load_existing_data()
    
    print("✅ Data loading complete")
    print(f"📊 Loaded {len(visualizer.trades)} trades")
    print(f"🔗 Created {len(visualizer.trade_lines)} trade pairs")
    print(f"📈 Loaded {len(visualizer.prices)} price points")
    print(f"💰 Loaded {len(visualizer.balance_history)} balance updates")
    
    # Log final summary
    visualizer.log_session_summary()
    
    print(f"\n📝 Check the log file for detailed trade information:")
    print(f"   {visualizer.log_file_path}")
    
    # Show sample of what's in the log
    if Path(visualizer.log_file_path).exists():
        print(f"\n📖 Sample log entries:")
        with open(visualizer.log_file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[-10:], 1):  # Show last 10 lines
                print(f"   {i:2d}: {line.strip()}")
    
    return visualizer

if __name__ == "__main__":
    test_logging()
