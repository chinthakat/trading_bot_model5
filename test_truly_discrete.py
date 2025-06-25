#!/usr/bin/env python3
"""
Test with a truly discrete action environment wrapper
"""

import sys
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environment import TradingEnvironment
from exploration_model import ExplorationTradingModel
from reward_config import SIMPLE_REWARD_CONFIG

class TrulyDiscreteActionWrapper(gym.Wrapper):
    """Wrapper to convert continuous action space to truly discrete"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Create discrete action space: 4 actions × 8 sizes × 1 leverage = 32 total combinations
        # We'll simplify to just 4 action types for now
        self.action_space = spaces.Discrete(4)  # 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE_ALL
        
        # Keep observation space the same
        self.observation_space = env.observation_space
        
        # Store original environment's action mapping
        self.original_env = env.unwrapped
        
    def action(self, action):
        """Convert discrete action to continuous action for original environment"""
        # action is an integer from 0 to 3
        
        # Map to continuous action format [action_type, size_index, leverage]
        continuous_action = np.array([
            float(action),  # action type (0, 1, 2, or 3)
            0.0,           # size index (use smallest size)
            1.0            # leverage
        ], dtype=np.float32)
        
        return continuous_action
    
    def step(self, action):
        """Step with discrete action converted to continuous"""
        continuous_action = self.action(action)
        return self.env.step(continuous_action)

def create_simple_data():
    """Create test data"""
    data = pd.DataFrame({
        'Open': [30000, 30010, 30020, 30030, 30040],
        'High': [30050, 30060, 30070, 30080, 30090],
        'Low': [29950, 29960, 29970, 29980, 29990],
        'Close': [30000, 30010, 30020, 30030, 30040],
        'Volume': [1000, 1000, 1000, 1000, 1000]
    }, index=pd.date_range('2024-01-01', periods=5, freq='15min'))
    return data

def test_truly_discrete_actions():
    """Test with truly discrete action space"""
    print("🎯 Testing Truly Discrete Action Space")
    print("=" * 40)
    
    logging.basicConfig(level=logging.WARNING)
    data = create_simple_data()
    
    # Create base environment
    base_env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        lookback_window=2,
        reward_config=SIMPLE_REWARD_CONFIG,
        enable_trade_logging=False,
        use_discretized_actions=True
    )
    
    # Wrap with discrete action wrapper
    env = TrulyDiscreteActionWrapper(base_env)
    
    print(f"📊 Wrapped action space: {env.action_space}")
    print(f"📊 Wrapped observation space: {env.observation_space}")
    
    # Create model
    model = ExplorationTradingModel(
        env=env,
        model_name="truly_discrete_test",
        verbose=0
    )
    
    # Test manual discrete actions
    print("\n🔧 Testing Manual Discrete Actions:")
    print("-" * 30)
    
    action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE_ALL']
    obs, _ = env.reset()
    
    for action_id in range(4):
        try:
            obs, reward, terminated, truncated, info = env.step(action_id)
            print(f"  {action_names[action_id]:10}: ✅ Success (reward: {reward:.3f})")
            if terminated or truncated:
                obs, _ = env.reset()
        except Exception as e:
            print(f"  {action_names[action_id]:10}: ❌ Failed ({e})")
    
    # Test action distribution
    print("\n🎲 Testing Discrete Action Distribution:")
    print("-" * 40)
    
    action_counts = [0, 0, 0, 0]
    
    for i in range(100):
        # Get discrete action from model
        action, _ = model.model.predict(obs, deterministic=False)
        
        # action should now be a single integer
        if isinstance(action, np.ndarray):
            action = action.item()
        
        action = int(action)
        if 0 <= action < 4:
            action_counts[action] += 1
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        except Exception as e:
            print(f"Error with action {action}: {e}")
    
    print("Action distribution (100 samples):")
    for i, name in enumerate(action_names):
        count = action_counts[i]
        percentage = (count / 100) * 100
        print(f"  {name:10}: {count:3d} ({percentage:5.1f}%)")
    
    close_all_count = action_counts[3]
    if close_all_count > 0:
        print(f"\n🎉 SUCCESS: CLOSE_ALL used {close_all_count} times!")
        return True
    else:
        print(f"\n⚠️ ISSUE: CLOSE_ALL still not used")
        return False

if __name__ == "__main__":
    success = test_truly_discrete_actions()
    if success:
        print("\n✅ Truly discrete action test PASSED")
    else:
        print("\n❌ Truly discrete action test FAILED")
