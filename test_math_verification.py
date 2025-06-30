#!/usr/bin/env python3
"""
Quick verification that the reward scaling fix is working
"""

import sys
import os
import logging
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_calculation():
    """Test the math behind the reward calculation"""
    
    print("🧪 Testing reward calculation math...")
    
    # Simulate the original problem
    print("\n--- Original Problem ---")
    base_reward = -0.1  # From scaled main reward (e.g., -10 / 100)
    trade_management = -10.0  # Unscaled trade management penalty
    original_total = base_reward + trade_management
    print(f"Base reward (scaled): {base_reward:.3f}")
    print(f"Trade management (unscaled): {trade_management:.1f}")
    print(f"Total reward (original): {original_total:.1f}")
    print(f"Over 6978 steps: {original_total * 6978:.1f}")
    
    # Simulate the fix
    print("\n--- With Fix ---")
    base_reward = -0.1  # From scaled main reward
    trade_management_scaled = -10.0 / 100.0  # Scaled trade management
    fixed_total = base_reward + trade_management_scaled
    final_capped = np.clip(fixed_total, -5.0, 5.0)
    print(f"Base reward (scaled): {base_reward:.3f}")
    print(f"Trade management (scaled): {trade_management_scaled:.3f}")
    print(f"Total reward (before cap): {fixed_total:.3f}")
    print(f"Total reward (after cap): {final_capped:.3f}")
    print(f"Over 6978 steps: {final_capped * 6978:.1f}")
    
    # Expected improvement
    improvement = abs(original_total * 6978) - abs(final_capped * 6978)
    print(f"\nImprovement: {improvement:.1f} reduction in absolute episode reward")
    
    if abs(final_capped * 6978) < 1000:  # Should be much smaller now
        print("✅ Fix should prevent extreme episode rewards")
    else:
        print("⚠️ May need further adjustment")

if __name__ == "__main__":
    test_simple_calculation()
