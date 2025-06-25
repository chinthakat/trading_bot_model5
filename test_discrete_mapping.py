#!/usr/bin/env python3
"""
Test with proper discrete action space mapping
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environment import TradingEnvironment
from exploration_model import ExplorationTradingModel
from reward_config import SIMPLE_REWARD_CONFIG

def create_simple_data():
    """Create minimal test data"""
    data = pd.DataFrame({
        'Open': [30000, 30010, 30020],
        'High': [30050, 30060, 30070],
        'Low': [29950, 29960, 29970],
        'Close': [30000, 30010, 30020],
        'Volume': [1000, 1000, 1000]
    }, index=pd.date_range('2024-01-01', periods=3, freq='15min'))
    return data

def test_discrete_action_mapping():
    """Test proper discrete action mapping"""
    print("🎯 Testing Discrete Action Mapping")
    print("=" * 40)
    
    logging.basicConfig(level=logging.WARNING)
    data = create_simple_data()
    
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        lookback_window=2,
        reward_config=SIMPLE_REWARD_CONFIG,
        enable_trade_logging=False,
        use_discretized_actions=True
    )
    
    model = ExplorationTradingModel(
        env=env,
        model_name="discrete_test",
        verbose=0
    )
    
    obs, _ = env.reset()
    
    # Test manual action mapping
    print("🔧 Testing Manual Action Mapping:")
    print("-" * 30)
    
    # Test all action types manually
    test_actions = [
        np.array([0.0, 0, 1.0], dtype=np.float32),  # HOLD
        np.array([1.0, 0, 1.0], dtype=np.float32),  # BUY  
        np.array([2.0, 0, 1.0], dtype=np.float32),  # SELL
        np.array([3.0, 0, 1.0], dtype=np.float32),  # CLOSE_ALL
    ]
    
    action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE_ALL']
    
    for i, (action, name) in enumerate(zip(test_actions, action_names)):
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  {name:10}: ✅ Success (reward: {reward:.3f})")
            if terminated or truncated:
                obs, _ = env.reset()
        except Exception as e:
            print(f"  {name:10}: ❌ Failed ({e})")
    
    # Test action distribution with properly mapped actions
    print("\n🎲 Testing Action Distribution with Proper Mapping:")
    print("-" * 50)
    
    action_counts = [0, 0, 0, 0]
    
    for i in range(100):
        # Get continuous action from model
        continuous_action, _ = model.model.predict(obs, deterministic=False)
        
        # Map to discrete action
        action_type_continuous = continuous_action[0]
        
        # Map continuous [0, 3] to discrete [0, 1, 2, 3]
        # We'll discretize by rounding, but ensure all values are achievable
        action_type = int(round(action_type_continuous))
        action_type = np.clip(action_type, 0, 3)  # Ensure valid range
        
        action_counts[action_type] += 1
        
        # Create properly formatted action
        discrete_action = np.array([float(action_type), 0, 1.0], dtype=np.float32)
        
        try:
            obs, reward, terminated, truncated, info = env.step(discrete_action)
            if terminated or truncated:
                obs, _ = env.reset()
        except Exception as e:
            print(f"Error with action type {action_type}: {e}")
    
    print("Action distribution (100 samples):")
    for i, name in enumerate(action_names):
        count = action_counts[i]
        percentage = (count / 100) * 100
        print(f"  {name:10}: {count:3d} ({percentage:5.1f}%)")
    
    close_all_count = action_counts[3]
    if close_all_count > 0:
        print(f"\n🎉 SUCCESS: CLOSE_ALL used {close_all_count} times!")
    else:
        print(f"\n⚠️ ISSUE: CLOSE_ALL still not used")
        
        # Let's check the raw continuous values
        print("\n🔍 Analyzing raw continuous action values:")
        continuous_values = []
        for i in range(100):
            continuous_action, _ = model.model.predict(obs, deterministic=False)
            continuous_values.append(continuous_action[0])
        
        print(f"  Min value: {min(continuous_values):.3f}")
        print(f"  Max value: {max(continuous_values):.3f}")
        print(f"  Mean: {np.mean(continuous_values):.3f}")
        print(f"  Std: {np.std(continuous_values):.3f}")
        
        # Count how many would round to each action
        rounded_counts = [0, 0, 0, 0]
        for val in continuous_values:
            rounded = int(round(np.clip(val, 0, 3)))
            rounded_counts[rounded] += 1
        
        print("  Rounded distribution:")
        for i, name in enumerate(action_names):
            count = rounded_counts[i]
            print(f"    {name:10}: {count:3d} ({count}%)")

if __name__ == "__main__":
    test_discrete_action_mapping()
