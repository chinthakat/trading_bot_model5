#!/usr/bin/env python3
"""
Simple debug script to check action space configuration
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

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

def simple_debug():
    """Simple debugging of action space"""
    print("🔍 Simple Action Space Debug")
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
    
    print(f"📊 Environment action space: {env.action_space}")
    print(f"📊 Environment observation space: {env.observation_space}")
    
    model = ExplorationTradingModel(
        env=env,
        model_name="simple_debug",
        verbose=0
    )
    
    # Check action distribution
    obs, _ = env.reset()
    print(f"📊 Observation shape: {obs.shape}")
    
    # Sample some actions
    print("\n🎲 Sampling 20 actions:")
    action_samples = []
    for i in range(20):
        action, _ = model.model.predict(obs, deterministic=False)
        print(f"  Action {i+1}: {action}")
        action_samples.append(action)
        
        # Try to step with this action
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    → Step successful, reward: {reward:.3f}")
            if terminated or truncated:
                obs, _ = env.reset()
        except Exception as e:
            print(f"    → Step failed: {e}")
    
    print("\n📈 Action Summary:")
    print(f"   Actions sampled: {len(action_samples)}")
    if action_samples:
        sample_action = action_samples[0]
        print(f"   Sample action type: {type(sample_action)}")
        print(f"   Sample action shape: {getattr(sample_action, 'shape', 'N/A')}")

if __name__ == "__main__":
    simple_debug()
