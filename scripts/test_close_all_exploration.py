#!/usr/bin/env python3
"""
Specialized test to encourage CLOSE_ALL action exploration
Tests the model in scenarios where CLOSE_ALL would be beneficial
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

def create_close_all_scenario_data(n_samples=200):
    """Create market data designed to encourage CLOSE_ALL actions"""
    np.random.seed(42)
    
    # Create volatile market that benefits from closing all positions
    base_price = 30000
    
    # Create alternating bull/bear cycles to encourage position management
    cycle_length = 20
    volatility_phases = []
    prices = []
    
    for i in range(n_samples):
        cycle_pos = i % cycle_length
        
        # High volatility phases every 20 steps - good time to close all
        if cycle_pos < 5:  # First 5 steps: declining market
            price_change = -np.random.uniform(50, 150)
        elif cycle_pos < 10:  # Next 5 steps: volatile sideways
            price_change = np.random.uniform(-100, 100)
        elif cycle_pos < 15:  # Next 5 steps: rising market
            price_change = np.random.uniform(50, 150)
        else:  # Last 5 steps: very volatile - good time to close all
            price_change = np.random.uniform(-200, 200)
        
        if i == 0:
            prices.append(base_price)
        else:
            prices.append(prices[-1] + price_change)
    
    # Create timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='15min')
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price + np.random.uniform(0, 50)
        low = price - np.random.uniform(0, 50)
        volume = np.random.uniform(500, 1500)
        
        data.append({
            'Open': price + np.random.uniform(-20, 20),
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df

def force_positions_scenario(env, model, n_steps=50):
    """Force the model into a scenario with many open positions to encourage CLOSE_ALL"""
    action_counts = [0, 0, 0, 0]  # HOLD, BUY, SELL, CLOSE_ALL
    
    obs, _ = env.reset()
    
    # Force open several positions first
    print("🎯 Phase 1: Opening multiple positions...")
    for step in range(20):  # Open positions first
        # Force BUY actions to create positions
        action = np.array([1, 0, 1.0], dtype=np.float32)  # BUY
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
        
        # Check if we have enough positions
        if hasattr(env, 'open_trades'):
            total_positions = len(env.open_trades)
            if total_positions >= 8:  # Enough positions for CLOSE_ALL to be useful
                break
    
    print(f"✅ Opened {len(env.open_trades)} positions")
    
    # Now test if model will use CLOSE_ALL
    print("🎯 Phase 2: Testing CLOSE_ALL exploration...")
    close_all_attempts = 0
    for step in range(n_steps):
        # Get action from model
        action, _ = model.model.predict(obs, deterministic=False)
        
        # Handle action format
        if isinstance(action, np.ndarray) and len(action) > 0:
            if len(action) >= 3:
                action_type = int(round(action[0]))
            else:
                action_type = int(round(action[0]))
                action = np.array([action_type, 0, 1.0], dtype=np.float32)
        else:
            action_type = int(round(action))
            action = np.array([action_type, 0, 1.0], dtype=np.float32)
        
        if 0 <= action_type < 4:
            action_counts[action_type] += 1
            
        # Log CLOSE_ALL attempts
        if action_type == 3:
            close_all_attempts += 1
            print(f"   Step {step}: CLOSE_ALL attempted (attempt #{close_all_attempts})")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    print(f"📊 CLOSE_ALL attempts in Phase 2: {close_all_attempts}")
    
    return action_counts

def test_close_all_exploration():
    """Test the model's ability to explore CLOSE_ALL action"""
    print("🎯 Testing CLOSE_ALL Action Exploration")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce logging noise
    logger = logging.getLogger(__name__)
    
    # Create scenario data
    print("📊 Creating CLOSE_ALL scenario data...")
    data = create_close_all_scenario_data(200)
    print(f"✅ Created {len(data)} data points with volatile cycles")
    
    # Create environment
    print("🏗️ Creating trading environment...")
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        lookback_window=10,
        reward_config=SIMPLE_REWARD_CONFIG,
        enable_trade_logging=False,  # Disable logging for test
        use_discretized_actions=True
    )
    
    # Create exploration model with even higher exploration
    print("🤖 Creating high-exploration model...")
    model = ExplorationTradingModel(
        env=env,
        model_name="close_all_test_model",
        verbose=0
    )
    
    # Enhanced action bias initialization to strongly encourage CLOSE_ALL
    print("⚡ Enhancing CLOSE_ALL bias...")
    try:
        policy = model.model.policy
        for name, param in policy.named_parameters():
            if 'action_net' in name and 'bias' in name:
                with torch.no_grad():
                    if param.shape[0] == 4:  # 4 actions
                        param[0] = -0.5  # HOLD (strongly discourage)
                        param[1] = -0.2  # BUY (discourage)
                        param[2] = -0.1  # SELL (slightly discourage)
                        param[3] = 0.8   # CLOSE_ALL (strongly encourage)
                        print(f"✅ Enhanced action bias: CLOSE_ALL = {param[3].item():.2f}")
    except Exception as e:
        print(f"⚠️ Could not enhance bias: {e}")
    
    # Test 1: Normal exploration
    print("\n🧪 Test 1: Normal action exploration...")
    action_counts = [0, 0, 0, 0]
    obs, _ = env.reset()
    
    for step in range(100):
        action, _ = model.model.predict(obs, deterministic=False)
        
        # Handle action format
        if isinstance(action, np.ndarray) and len(action) > 0:
            if len(action) >= 3:
                action_type = int(round(action[0]))
            else:
                action_type = int(round(action[0]))
                action = np.array([action_type, 0, 1.0], dtype=np.float32)
        else:
            action_type = int(round(action))
            action = np.array([action_type, 0, 1.0], dtype=np.float32)
        
        if 0 <= action_type < 4:
            action_counts[action_type] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    print("📊 Normal exploration results:")
    action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE_ALL']
    for i, name in enumerate(action_names):
        count = action_counts[i]
        percentage = (count / 100) * 100
        print(f"   {name:10}: {count:3d} ({percentage:5.1f}%)")
    
    # Test 2: Forced positions scenario
    print("\n🧪 Test 2: Forced positions scenario...")
    forced_action_counts = force_positions_scenario(env, model, 50)
    
    print("📊 Forced positions scenario results:")
    for i, name in enumerate(action_names):
        count = forced_action_counts[i]
        percentage = (count / 50) * 100
        print(f"   {name:10}: {count:3d} ({percentage:5.1f}%)")
    
    # Overall results
    total_close_all = action_counts[3] + forced_action_counts[3]
    print(f"\n📈 CLOSE_ALL Usage Summary:")
    print(f"   Normal test: {action_counts[3]} times")
    print(f"   Forced test: {forced_action_counts[3]} times") 
    print(f"   Total: {total_close_all} times")
    
    if total_close_all > 0:
        print("🎉 SUCCESS: Model is exploring CLOSE_ALL action!")
        return True
    else:
        print("⚠️ ISSUE: Model still not exploring CLOSE_ALL action")
        return False

if __name__ == "__main__":
    import torch
    success = test_close_all_exploration()
    if success:
        print("\n✅ CLOSE_ALL exploration test PASSED")
    else:
        print("\n❌ CLOSE_ALL exploration test FAILED")
