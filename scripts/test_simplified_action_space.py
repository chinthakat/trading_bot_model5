#!/usr/bin/env python3
"""
Test script to verify simplified action space works correctly
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from environment import TradingEnvironment

def create_test_data():
    """Create simple test data for environment testing"""
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)  # For reproducible results
    
    # Create upward trending data to make trades profitable
    base_price = 50000
    trend = np.linspace(0, 1000, 200)  # Upward trend
    noise = np.random.normal(0, 50, 200)
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'timestamp': [int(d.timestamp()) for d in dates],
        'Open': close_prices * (1 + np.random.normal(0, 0.001, 200)),
        'High': close_prices * (1 + np.abs(np.random.normal(0, 0.002, 200))),
        'Low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, 200))),
        'Close': close_prices,
        'Volume': np.random.uniform(100, 1000, 200)
    }, index=dates)
    
    return data

def test_simplified_action_space():
    """Test simplified action space functionality"""
    print("=" * 60)
    print("TESTING SIMPLIFIED ACTION SPACE")
    print("=" * 60)
    
    # Create test environment
    data = create_test_data()
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        use_discretized_actions=True,
        enable_trade_logging=False,  # Disable logging for cleaner output
        episode_num=0,
        batch_num=0
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment initialized. Action space: {env.action_space}")
    print(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")
    print()
    
    # Test sequence: Open 3 trades, then close all
    test_actions = [
        # Open 3 long trades
        (1, 2, 2.0),  # BUY with size_index=2, leverage=2.0
        (1, 1, 1.5),  # BUY with size_index=1, leverage=1.5  
        (1, 3, 3.0),  # BUY with size_index=3, leverage=3.0
        
        # Hold for a few steps to let prices move
        (0, 0, 1.0),  # HOLD
        (0, 0, 1.0),  # HOLD
        (0, 0, 1.0),  # HOLD
        
        # Close all trades
        (3, 0, 1.0),  # CLOSE_ALL (action_type=3)
        
        # Try invalid action types
        (4, 0, 1.0),  # Invalid action (should be rejected)
        (10, 0, 1.0), # Invalid action (should be rejected)
    ]
    
    for step, (action_type, size_index, leverage) in enumerate(test_actions):
        print(f"\nStep {step + 1}: Executing action_type={action_type}")
        action = np.array([action_type, size_index, leverage], dtype=np.float32)
        
        # Show current state before action
        print(f"  Before action: {len(env.open_trades)} open trades, balance=${env.balance:.2f}")
        if env.open_trades:
            for i, trade in enumerate(env.open_trades):
                print(f"    Trade {i+1}: {trade['side']} {trade['size_btc']:.6f} BTC, unrealized P&L=${trade['unrealized_pnl']:.2f}")
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Show result after action
        print(f"  After action: {len(env.open_trades)} open trades, balance=${env.balance:.2f}, reward={reward:.4f}")
        if env.open_trades:
            for i, trade in enumerate(env.open_trades):
                print(f"    Trade {i+1}: {trade['side']} {trade['size_btc']:.6f} BTC, unrealized P&L=${trade['unrealized_pnl']:.2f}")
        
        if terminated or truncated:
            print("Episode terminated!")
            break
    
    print("\n" + "=" * 60)
    print(f"FINAL STATE:")
    print(f"  Balance: ${env.balance:.2f}")
    print(f"  Equity: ${env.equity:.2f}")
    print(f"  Realized P&L: ${env.realized_pnl:.2f}")
    print(f"  Open trades: {len(env.open_trades)}")
    print("=" * 60)

if __name__ == "__main__":
    test_simplified_action_space()
