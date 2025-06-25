#!/usr/bin/env python3
"""
Test script to verify comprehensive logging in TradingEnvironment
"""

import sys
sys.path.append('.')
from environment import TradingEnvironment
from simple_reward_system import SimpleRewardSystem
from reward_config import SIMPLE_REWARD_CONFIG
import pandas as pd
import numpy as np
import logging

# Setup logging to see the output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_test_data(n_steps=50):
    """Create test market data"""
    dates = pd.date_range('2024-01-01', periods=n_steps, freq='15min')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price movements
    base_price = 50000
    price_changes = np.random.normal(0, 0.001, n_steps)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    test_data = pd.DataFrame(index=dates)
    test_data['Close'] = prices
    test_data['Open'] = test_data['Close'].shift(1).fillna(test_data['Close'].iloc[0])
    test_data['High'] = np.maximum(test_data['Open'], test_data['Close']) * np.random.uniform(1.0, 1.005, n_steps)
    test_data['Low'] = np.minimum(test_data['Open'], test_data['Close']) * np.random.uniform(0.995, 1.0, n_steps)
    test_data['Volume'] = np.random.uniform(1000, 10000, n_steps)
    
    return test_data

def test_comprehensive_logging():
    """Test the comprehensive logging functionality"""
    print("=== Testing Comprehensive Trading Execution Logging ===")
    
    # Create test data
    test_data = create_test_data(30)
    print(f"Created test data with {len(test_data)} rows")
    
    # Create environment
    env = TradingEnvironment(
        df=test_data,
        initial_balance=10000,
        lookback_window=5,
        reward_config=SIMPLE_REWARD_CONFIG,
        enable_trade_logging=True,
        use_discretized_actions=True
    )
    
    print(f"Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset. Starting balance: ${env.balance:.2f}")
    
    # Test different actions
    test_actions = [
        np.array([0, 0.0, 1.0]),  # HOLD
        np.array([1, 0.1, 2.0]),  # BUY with 10% position, 2x leverage
        np.array([0, 0.0, 1.0]),  # HOLD
        np.array([2, 0.05, 1.5]), # SELL with 5% position, 1.5x leverage  
        np.array([0, 0.0, 1.0]),  # HOLD
        np.array([3, 0.0, 1.0]),  # CLOSE first trade
    ]
    
    print("\n=== Running Test Actions ===")
    for i, action in enumerate(test_actions):
        print(f"\n--- Step {i+1}: Action {action} ---")
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Episode terminated")
            break
    
    print("\n=== Test Complete ===")
    print(f"Final balance: ${env.balance:.2f}")
    print(f"Open trades: {len(env.open_trades)}")
    print(f"Equity: ${env.equity:.2f}")

if __name__ == "__main__":
    test_comprehensive_logging()
