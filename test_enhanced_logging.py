#!/usr/bin/env python3
"""
Quick test script to verify enhanced logging with market timestamps and reward breakdown
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from environment import TradingEnvironment
from simple_reward_system import SimpleRewardSystem
from reward_config import SIMPLE_REWARD_CONFIG
import pandas as pd
import numpy as np
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_enhanced_logging():
    """Test the enhanced logging functionality"""
    
    # Create simple test data with timestamps
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    test_data = pd.DataFrame({
        'Open': 50000 + np.random.randn(100) * 100,
        'High': 50100 + np.random.randn(100) * 100,
        'Low': 49900 + np.random.randn(100) * 100,
        'Close': 50000 + np.random.randn(100) * 100,
        'Volume': 1000 + np.random.randn(100) * 100,
    }, index=dates)
    
    # Create environment with enhanced logging
    env = TradingEnvironment(
        df=test_data,
        initial_balance=10000,
        lookback_window=20,
        reward_config=SIMPLE_REWARD_CONFIG,
        trade_logger_session="test_session",
        enable_trade_logging=True,
        use_discretized_actions=True,
        episode_num=1,
        batch_num=1
    )
    
    print("Testing enhanced logging with market timestamps and reward breakdown...")
    print("=" * 80)
    
    # Reset environment
    obs = env.reset()
    
    # Test a few actions to see the enhanced logging
    actions = [
        np.array([0]),  # HOLD
        np.array([1]),  # BUY
        np.array([0]),  # HOLD
        np.array([3]),  # CLOSE
        np.array([0]),  # HOLD
    ]
    
    for i, action in enumerate(actions):
        print(f"\n--- Step {i+1} ---")
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 80)
    print("Enhanced logging test completed!")
    print("Check the log output above for:")
    print("1. Market timestamps from the data file")
    print("2. Detailed reward breakdown components")
    print("3. Action details for every step")

if __name__ == "__main__":
    test_enhanced_logging()
