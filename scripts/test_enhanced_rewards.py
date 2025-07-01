#!/usr/bin/env python3
"""
Test script to verify the enhanced trade management reward system
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
    dates = pd.date_range('2024-01-01', periods=300, freq='15min')
    np.random.seed(42)  # For reproducible results
    
    # Create upward trending data to make trades profitable
    base_price = 50000
    trend = np.linspace(0, 1000, 300)  # Upward trend
    noise = np.random.normal(0, 50, 300)
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'timestamp': [int(d.timestamp()) for d in dates],
        'Open': close_prices * (1 + np.random.normal(0, 0.001, 300)),
        'High': close_prices * (1 + np.abs(np.random.normal(0, 0.002, 300))),
        'Low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, 300))),
        'Close': close_prices,
        'Volume': np.random.uniform(100, 1000, 300)
    }, index=dates)
    
    return data

def test_enhanced_reward_system():
    """Test enhanced trade management reward system"""
    print("=" * 60)
    print("TESTING ENHANCED TRADE MANAGEMENT REWARD SYSTEM")
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
    print(f"Environment initialized. Max trades: {env.MAX_OPEN_TRADES}")
    print()
    
    # Test sequence: Open many LONG trades to test reward penalties
    test_scenarios = [
        # Scenario 1: Open multiple LONG trades
        "Opening multiple LONG trades to test oversaturation penalties:",
        [(1, 2, 2.0)] * 8,  # 8 BUY actions
        
        # Scenario 2: Try to open more LONGs (should get rejected and penalized)
        "Trying to open more LONGs at capacity:",
        [(1, 2, 2.0)] * 3,  # 3 more BUY attempts (should be rejected)
        
        # Scenario 3: HOLD at max capacity
        "HOLD actions at max capacity:",
        [(0, 0, 1.0)] * 5,  # 5 HOLD actions
        
        # Scenario 4: Close all trades
        "Closing all trades:",
        [(3, 0, 1.0)],  # CLOSE_ALL
        
        # Scenario 5: Long inactivity
        "Long period of inactivity (100+ steps):",
        [(0, 0, 1.0)] * 105,  # 105 HOLD actions to test inactivity penalty
    ]
    
    step_counter = 0
    
    for scenario_desc, actions in zip(test_scenarios[::2], test_scenarios[1::2]):
        print(f"\n{scenario_desc}")
        print("-" * len(scenario_desc))
        
        for i, (action_type, size_index, leverage) in enumerate(actions):
            step_counter += 1
            action = np.array([action_type, size_index, leverage], dtype=np.float32)
            
            # Get before state
            before_trades = len(env.open_trades)
            before_long = sum(1 for trade in env.open_trades if trade['side'] == 'LONG')
            before_short = sum(1 for trade in env.open_trades if trade['side'] == 'SHORT')
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get after state
            after_trades = len(env.open_trades)
            after_long = sum(1 for trade in env.open_trades if trade['side'] == 'LONG')
            after_short = sum(1 for trade in env.open_trades if trade['side'] == 'SHORT')
            
            # Report significant changes or every 10th step for long sequences
            if (after_trades != before_trades or 
                abs(reward) > 0.01 or 
                i % 10 == 0 or 
                len(actions) <= 10):
                
                action_names = ["HOLD", "BUY", "SELL", "CLOSE_ALL"]
                print(f"  Step {step_counter}: {action_names[action_type]} -> "
                      f"Trades: {before_trades}→{after_trades} "
                      f"(L:{before_long}→{after_long}, S:{before_short}→{after_short}) "
                      f"Reward: {reward:.4f}")
                
                if hasattr(env, 'steps_since_last_position'):
                    print(f"    Steps since last position: {env.steps_since_last_position}")
            
            if terminated or truncated:
                print("Episode terminated!")
                return
        
        print(f"  ... completed {len(actions)} actions")
    
    print("\n" + "=" * 60)
    print(f"FINAL STATE:")
    print(f"  Balance: ${env.balance:.2f}")
    print(f"  Equity: ${env.equity:.2f}")
    print(f"  Realized P&L: ${env.realized_pnl:.2f}")
    print(f"  Open trades: {len(env.open_trades)}")
    print(f"  LONG positions: {sum(1 for trade in env.open_trades if trade['side'] == 'LONG')}")
    print(f"  SHORT positions: {sum(1 for trade in env.open_trades if trade['side'] == 'SHORT')}")
    if hasattr(env, 'steps_since_last_position'):
        print(f"  Steps since last position: {env.steps_since_last_position}")
    print("=" * 60)

if __name__ == "__main__":
    test_enhanced_reward_system()
