#!/usr/bin/env python3
"""
Debug script to inspect the model's action distribution and policy outputs
"""

import sys
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environment import TradingEnvironment
from exploration_model import ExplorationTradingModel
from reward_config import SIMPLE_REWARD_CONFIG

def create_simple_data(n_samples=50):
    """Create simple market data for debugging"""
    np.random.seed(42)
    
    base_price = 30000
    prices = [base_price + i * 10 for i in range(n_samples)]  # Simple rising trend
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='15min')
    
    data = []
    for price in prices:
        data.append({
            'Open': price,
            'High': price + 20,
            'Low': price - 20,
            'Close': price,
            'Volume': 1000
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df

def debug_model_actions():
    """Debug the model's action selection mechanism"""
    print("🔍 Debugging Model Action Distribution")
    print("=" * 50)
    
    # Setup
    logging.basicConfig(level=logging.WARNING)
    data = create_simple_data(50)
    
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        lookback_window=10,
        reward_config=SIMPLE_REWARD_CONFIG,
        enable_trade_logging=False,
        use_discretized_actions=True
    )
    
    model = ExplorationTradingModel(
        env=env,
        model_name="debug_model",
        verbose=0
    )
    
    # Get initial observation
    obs, _ = env.reset()
    print(f"📊 Observation shape: {obs.shape}")
    
    # Test direct policy predictions
    print("\n🧠 Testing Direct Policy Outputs:")
    print("-" * 30)
    
    # Get policy reference
    policy = model.model.policy
    
    with torch.no_grad():
        # Convert observation to tensor and move to correct device
        device = next(policy.parameters()).device
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Get policy outputs directly
        features = policy.extract_features(obs_tensor)
        
        # Get action logits
        action_logits = policy.action_net(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        print(f"Device: {device}")
        print(f"Raw action logits: {action_logits.cpu().numpy()}")
        print(f"Action probabilities: {action_probs.cpu().numpy()}")
        
        action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE_ALL']
        for i, name in enumerate(action_names):
            prob = action_probs[0, i].item()
            logit = action_logits[0, i].item()
            print(f"  {name:10}: logit={logit:6.3f}, prob={prob:6.3f} ({prob*100:5.1f}%)")
    
    # Test model.predict() outputs
    print("\n🎯 Testing model.predict() outputs:")
    print("-" * 30)
    
    action_samples = []
    for i in range(100):
        action, _ = model.model.predict(obs, deterministic=False)
        
        # Extract action type
        if isinstance(action, np.ndarray) and len(action) > 0:
            action_type = int(round(action[0]))
        else:
            action_type = int(round(action))
        
        action_samples.append(action_type)
    
    # Count action distribution
    action_counts = [0, 0, 0, 0]
    for action_type in action_samples:
        if 0 <= action_type < 4:
            action_counts[action_type] += 1
    
    print("Action sampling distribution (100 samples):")
    for i, name in enumerate(action_names):
        count = action_counts[i]
        percentage = (count / 100) * 100
        print(f"  {name:10}: {count:3d} ({percentage:5.1f}%)")
    
    # Test with enhanced bias
    print("\n⚡ Testing with Enhanced CLOSE_ALL Bias:")
    print("-" * 40)
    
    # Manually enhance CLOSE_ALL bias
    try:
        for name, param in policy.named_parameters():
            if 'action_net' in name and 'bias' in name:
                with torch.no_grad():
                    if param.shape[0] == 4:
                        original_bias = param.clone()
                        print(f"Original bias: {original_bias.numpy()}")
                        
                        # Extreme bias towards CLOSE_ALL
                        param[0] = -2.0  # HOLD (strongly discourage)
                        param[1] = -2.0  # BUY (strongly discourage) 
                        param[2] = -2.0  # SELL (strongly discourage)
                        param[3] = 3.0   # CLOSE_ALL (strongly encourage)
                        
                        print(f"Modified bias: {param.numpy()}")
                        break
        
        # Test with enhanced bias
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            features = policy.extract_features(obs_tensor)
            action_logits = policy.action_net(features)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            print(f"Enhanced action logits: {action_logits.cpu().numpy()}")
            print(f"Enhanced action probs: {action_probs.cpu().numpy()}")
            
            for i, name in enumerate(action_names):
                prob = action_probs[0, i].item()
                logit = action_logits[0, i].item()
                print(f"  {name:10}: logit={logit:6.3f}, prob={prob:6.3f} ({prob*100:5.1f}%)")
        
        # Test sampling with enhanced bias
        enhanced_samples = []
        for i in range(100):
            action, _ = model.model.predict(obs, deterministic=False)
            
            if isinstance(action, np.ndarray) and len(action) > 0:
                action_type = int(round(action[0]))
            else:
                action_type = int(round(action))
            
            enhanced_samples.append(action_type)
        
        enhanced_counts = [0, 0, 0, 0]
        for action_type in enhanced_samples:
            if 0 <= action_type < 4:
                enhanced_counts[action_type] += 1
        
        print("\nEnhanced sampling distribution (100 samples):")
        for i, name in enumerate(action_names):
            count = enhanced_counts[i]
            percentage = (count / 100) * 100
            print(f"  {name:10}: {count:3d} ({percentage:5.1f}%)")
        
        if enhanced_counts[3] > 0:
            print("🎉 SUCCESS: CLOSE_ALL is now being sampled!")
        else:
            print("❌ ISSUE: CLOSE_ALL still not being sampled even with extreme bias")
            
    except Exception as e:
        print(f"❌ Error enhancing bias: {e}")
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    debug_model_actions()
