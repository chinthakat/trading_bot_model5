"""
Configuration file for Simple Reward System

This file contains configurable parameters for the reward system.
Modify these values to tune the behavior of the trading agent.

Author: AI Assistant
Date: June 25, 2025
"""

# Simple Reward System Configuration
SIMPLE_REWARD_CONFIG = {
    # === EXPLORATION SETTINGS ===
    'exploration_bonus': 0.1,          # Bonus for trying less-used actions
    'action_diversity_window': 20,      # Number of recent actions to consider for exploration
    
    # === POSITION MANAGEMENT ===
    'max_comfortable_positions': 5,     # Start encouraging closure after this many positions
    'max_positions_threshold': 10,      # Start heavy penalties after this many positions
    'position_penalty_multiplier': 0.5, # Penalty multiplier per excess position (exponential)
    
    # === PROFIT/LOSS REWARDS ===
    # Profit margins (as decimals: 0.01 = 1%)
    'profit_margins': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],  # 0.5%, 1%, 2%, 3%, 5%, 8%, 10%, 15%
    'profit_rewards': [2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0],     # SIGNIFICANTLY ENHANCED profitable close rewards
    
    # Loss margins (as decimals: -0.01 = -1%)
    'loss_margins': [-0.005, -0.01, -0.015, -0.02, -0.03, -0.05, -0.08, -0.10, -0.15],  # -0.5%, -1%, -1.5%, -2%, -3%, -5%, -8%, -10%, -15%
    'loss_penalties': [-1.0, -3.0, -5.0, -8.0, -12.0, -18.0, -25.0, -35.0, -50.0],    # SIGNIFICANTLY ENHANCED loss penalties (negative)
    
    # === POSITION CLOSURE INCENTIVES ===
    'closure_bonus_threshold': 5,       # When to start giving closure bonuses
    'closure_bonus': 0.5,               # Bonus for closing when >threshold positions
    'profitable_close_bonus': 1.0,      # EXTRA bonus for ANY profitable close (on top of profit rewards)
    'losing_close_penalty': -2.0,       # EXTRA penalty for ANY losing close (on top of loss penalties)
    
    # === BASE ACTION REWARDS ===
    'hold_reward': 0.0,                 # Reward for holding (neutral)
    'open_reward': 0.1,                 # Small reward for taking action (opening)
    'close_reward': 0.3,                # ENHANCED reward for closing (more decisive action)
}

# Alternative configurations for different trading styles

# Conservative Configuration (less risk-taking but still rewards profitable closes and heavily penalizes losses)
CONSERVATIVE_CONFIG = {
    **SIMPLE_REWARD_CONFIG,
    'max_comfortable_positions': 3,     # Fewer positions
    'max_positions_threshold': 6,       # Lower threshold
    'position_penalty_multiplier': 1.0, # Higher penalties for excess positions
    'close_reward': 0.4,                # Even higher closure reward for conservative approach
    'closure_bonus': 1.0,               # Higher closure bonus
    'profitable_close_bonus': 1.5,      # ENHANCED bonus for profitable closes in conservative mode
    'losing_close_penalty': -3.0,       # EVEN HIGHER penalty for losing closes in conservative mode
}

# Aggressive Configuration (more risk-taking)
AGGRESSIVE_CONFIG = {
    **SIMPLE_REWARD_CONFIG,
    'max_comfortable_positions': 8,     # More positions allowed
    'max_positions_threshold': 15,      # Higher threshold
    'position_penalty_multiplier': 0.3, # Lower penalties
    'exploration_bonus': 0.15,          # Higher exploration bonus
    'profit_rewards': [1.5, 3.0, 6.0, 12.0],  # Higher profit rewards
}

# Exploration-Focused Configuration
EXPLORATION_CONFIG = {
    **SIMPLE_REWARD_CONFIG,
    'exploration_bonus': 0.2,           # High exploration bonus
    'action_diversity_window': 30,      # Longer memory for exploration
    'hold_reward': -0.05,               # Small penalty for holding (encourage action)
    'open_reward': 0.15,                # Higher reward for opening
    'close_reward': 0.25,               # Higher reward for closing
}

# Risk Management Configuration
RISK_MANAGEMENT_CONFIG = {
    **SIMPLE_REWARD_CONFIG,
    'max_comfortable_positions': 4,     # Conservative position count
    'max_positions_threshold': 8,       # Lower threshold
    'position_penalty_multiplier': 0.8, # Higher penalties
    'loss_penalties': [-1.0, -2.5, -5.0, -10.0],  # Harsher loss penalties
    'closure_bonus_threshold': 4,       # Start closure bonus earlier
    'closure_bonus': 0.7,               # Higher closure bonus
}

# Configuration selector function
def get_config(config_name: str = 'default'):
    """
    Get configuration by name
    
    Args:
        config_name: One of 'default', 'conservative', 'aggressive', 'exploration', 'risk_management'
    
    Returns:
        dict: Configuration dictionary
    """
    configs = {
        'default': SIMPLE_REWARD_CONFIG,
        'conservative': CONSERVATIVE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
        'exploration': EXPLORATION_CONFIG,
        'risk_management': RISK_MANAGEMENT_CONFIG,
    }
    
    if config_name not in configs:
        print(f"Warning: Unknown config '{config_name}', using default")
        return SIMPLE_REWARD_CONFIG
    
    return configs[config_name]

# Quick test function
def test_config(config_name: str = 'default'):
    """Test a configuration with sample scenarios"""
    import sys
    import os
    
    # Add current directory to path to import simple_reward_system
    sys.path.append(os.path.dirname(__file__))
    
    from simple_reward_system import SimpleRewardSystem
    
    config = get_config(config_name)
    reward_system = SimpleRewardSystem(config)
    
    print(f"Testing {config_name.upper()} Configuration")
    print("=" * 50)
    
    # Test scenario: Opening position with 12 already open
    reward = reward_system.calculate_reward(
        action='OPEN_LONG',
        open_positions=[{}] * 12,
    )
    print(f"Opening with 12 positions: {reward:.3f}")
    
    # Test scenario: Closing profitable position with 7 open
    reward = reward_system.calculate_reward(
        action='CLOSE_LONG',
        open_positions=[{}] * 7,
        closed_position={'pnl_percentage': 0.03}  # 3% profit
    )
    print(f"Closing 3% profit with 7 positions: {reward:.3f}")
    
    # Test scenario: Large loss
    reward = reward_system.calculate_reward(
        action='CLOSE_SHORT',
        open_positions=[{}] * 3,
        closed_position={'pnl_percentage': -0.08}  # -8% loss
    )
    print(f"Closing -8% loss with 3 positions: {reward:.3f}")
    
    print()

if __name__ == "__main__":
    # Test all configurations
    configs = ['default', 'conservative', 'aggressive', 'exploration', 'risk_management']
    
    for config_name in configs:
        test_config(config_name)
