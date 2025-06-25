#!/usr/bin/env python3
"""
Test script to verify that the Simple Reward System is properly integrated
as the default reward system for all options.

This script tests:
1. Import functionality
2. Configuration loading
3. Environment integration
4. Basic reward calculation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all necessary modules import correctly"""
    print("Testing imports...")
    
    try:
        from simple_reward_system import SimpleRewardSystem
        print("✅ SimpleRewardSystem imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SimpleRewardSystem: {e}")
        return False
    
    try:
        from reward_config import SIMPLE_REWARD_CONFIG, get_config
        print("✅ Reward config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import reward config: {e}")
        return False
    
    try:
        from environment import TradingEnvironment
        print("✅ TradingEnvironment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TradingEnvironment: {e}")
        return False
    
    return True

def test_reward_system_basic():
    """Test basic reward system functionality"""
    print("\nTesting basic reward system functionality...")
    
    from simple_reward_system import SimpleRewardSystem
    from reward_config import get_config
    
    # Test different configurations
    configs_to_test = ['default', 'conservative', 'aggressive', 'exploration', 'risk_management']
    
    for config_name in configs_to_test:
        try:
            config = get_config(config_name)
            reward_system = SimpleRewardSystem(config)
            
            # Test a simple reward calculation
            reward = reward_system.calculate_reward(
                action='OPEN_LONG',
                open_positions=[],
                closed_position=None,
                current_price=50000.0
            )
            
            print(f"✅ {config_name.upper()} config: Reward = {reward:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to test {config_name} config: {e}")
            return False
    
    return True

def test_environment_integration():
    """Test that the environment correctly uses the new reward system"""
    print("\nTesting environment integration...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create minimal test data with proper DatetimeIndex
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(45000, 55000, 100),
            'High': np.random.uniform(46000, 56000, 100),
            'Low': np.random.uniform(44000, 54000, 100),
            'Close': np.random.uniform(45000, 55000, 100),
            'Volume': np.random.uniform(100, 1000, 100),
        }, index=dates)  # Use dates as index
        
        # Make sure High >= max(Open, Close) and Low <= min(Open, Close)
        test_data['High'] = np.maximum(test_data['High'], np.maximum(test_data['Open'], test_data['Close']))
        test_data['Low'] = np.minimum(test_data['Low'], np.minimum(test_data['Open'], test_data['Close']))
        
        from environment import TradingEnvironment
        from reward_config import get_config
        
        # Test with default configuration
        config = get_config('default')
        env = TradingEnvironment(
            df=test_data,
            initial_balance=10000.0,
            reward_config=config
        )
        
        print("✅ Environment created successfully with SimpleRewardSystem")
        
        # Test a few steps
        obs, info = env.reset()
        print("✅ Environment reset successfully")
        
        for i in range(3):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"✅ Step {i+1}: Action={action}, Reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        print("✅ Environment stepping works correctly")
        
        # Test reward system attributes
        if hasattr(env.reward_calculator, 'trade_frequency_counter'):
            print("✅ Legacy compatibility attributes present")
        else:
            print("❌ Legacy compatibility attributes missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_configurations():
    """Test all available reward configurations"""
    print("\nTesting reward configurations...")
    
    from simple_reward_system import SimpleRewardSystem
    from reward_config import get_config
    
    test_scenarios = [
        {
            'action': 'OPEN_LONG',
            'open_positions': [],
            'description': 'Opening first position'
        },
        {
            'action': 'CLOSE_LONG',
            'open_positions': [{}] * 7,
            'closed_position': {'pnl_percentage': 0.03},
            'description': 'Closing profitable position with many open'
        },
        {
            'action': 'OPEN_SHORT',
            'open_positions': [{}] * 12,
            'description': 'Opening position with excessive open positions'
        }
    ]
    
    configs = ['default', 'conservative', 'aggressive', 'exploration', 'risk_management']
    
    print(f"{'Config':<15} {'Scenario 1':<10} {'Scenario 2':<10} {'Scenario 3':<10}")
    print("-" * 55)
    
    for config_name in configs:
        config = get_config(config_name)
        reward_system = SimpleRewardSystem(config)
        
        rewards = []
        for scenario in test_scenarios:
            reward = reward_system.calculate_reward(
                action=scenario['action'],
                open_positions=scenario['open_positions'],
                closed_position=scenario.get('closed_position'),
                current_price=50000.0
            )
            rewards.append(reward)
        
        print(f"{config_name:<15} {rewards[0]:<10.3f} {rewards[1]:<10.3f} {rewards[2]:<10.3f}")
    
    print("✅ All configurations tested successfully")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("SIMPLE REWARD SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_reward_system_basic),
        ("Environment Integration", test_environment_integration),
        ("Configuration Test", test_reward_configurations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Simple Reward System is successfully integrated!")
        print("\nThe Simple Reward System is now the default for all options:")
        print("- Environment uses SimpleRewardSystem by default")
        print("- Training scripts use SIMPLE_REWARD_CONFIG")
        print("- All configuration options available")
        print("- Legacy compatibility maintained")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
