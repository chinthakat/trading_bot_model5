# Simple Reward System Documentation

## Overview

The Simple Reward System is a streamlined reinforcement learning reward mechanism designed for trading bots. It focuses on four key behavioral objectives:

1. **Action Exploration** - Encourages the agent to try different actions
2. **Position Management** - Penalizes excessive open positions (>10)
3. **Profit/Loss Optimization** - Multi-tier rewards based on profit margins
4. **Risk Control** - Encourages position closure when portfolio is crowded (>5 positions)

## Key Features

### 1. Action Exploration Rewards
- **Exploration Bonus**: Rewards trying actions that haven't been used recently
- **Diversity Window**: Tracks the last 20 actions to calculate exploration scores
- **Entropy-Based Scoring**: Higher rewards for balanced action distribution

### 2. Position Management System
```
Positions <= 5:  Normal operation
Positions 6-10:  Encourages closure, small penalty for opening
Positions > 10:  Heavy exponential penalties for excess positions
```

### 3. Multi-Tier Profit/Loss Rewards
**Profit Tiers:**
- 1%+ profit: +1.0 reward
- 3%+ profit: +2.5 reward  
- 5%+ profit: +5.0 reward
- 10%+ profit: +10.0 reward

**Loss Tiers:**
- -1% loss: -0.5 penalty
- -3% loss: -1.5 penalty
- -5% loss: -3.0 penalty
- -10% loss: -7.0 penalty

### 4. Position Closure Incentives
- **Closure Bonus**: +0.5 reward for closing positions when >5 are open
- **Adaptive Scaling**: Bonus increases with number of excess positions

## Usage

### Basic Usage
```python
from simple_reward_system import SimpleRewardSystem

# Initialize with default settings
reward_system = SimpleRewardSystem()

# Calculate reward for an action
reward = reward_system.calculate_reward(
    action='CLOSE_LONG',
    open_positions=[{}, {}, {}],  # 3 open positions
    closed_position={'pnl_percentage': 0.025},  # 2.5% profit
    current_price=50000.0
)
```

### Using Custom Configurations
```python
from reward_config import get_config
from simple_reward_system import SimpleRewardSystem

# Use a pre-defined configuration
config = get_config('conservative')
reward_system = SimpleRewardSystem(config)

# Or create custom configuration
custom_config = {
    'exploration_bonus': 0.15,
    'max_positions_threshold': 8,
    'profit_rewards': [2.0, 4.0, 8.0, 15.0]
}
reward_system = SimpleRewardSystem(custom_config)
```

## Available Configurations

### 1. Default Configuration
Balanced approach suitable for most trading scenarios.

### 2. Conservative Configuration
- Lower position limits (3 comfortable, 6 max)
- Higher penalties for excess positions
- Enhanced closure rewards

### 3. Aggressive Configuration  
- Higher position limits (8 comfortable, 15 max)
- Lower penalties for excess positions
- Higher profit rewards

### 4. Exploration Configuration
- Enhanced exploration bonuses
- Longer action diversity window
- Small penalty for holding (encourages action)

### 5. Risk Management Configuration
- Conservative position limits
- Harsh loss penalties
- Early closure incentives

## Reward Components Breakdown

### Base Action Rewards
```python
'hold_reward': 0.0,      # Neutral for holding
'open_reward': 0.1,      # Small reward for opening positions
'close_reward': 0.2,     # Higher reward for closing (decisive action)
```

### Position Management Penalties
```python
# For positions > max_positions_threshold (10)
penalty = -position_penalty_multiplier * (excess_positions ^ 2)

# Example: 12 positions with multiplier 0.5
# excess = 12 - 10 = 2
# penalty = -0.5 * (2^2) = -2.0
```

### Exploration Bonus Calculation
```python
# Based on action frequency in recent window
action_frequency = recent_action_count / total_recent_actions
exploration_bonus = exploration_bonus * (1.0 - action_frequency)

# Less frequent actions get higher bonuses
```

## Integration with Trading Systems

### Step 1: Initialize
```python
from simple_reward_system import SimpleRewardSystem
from reward_config import get_config

# Choose configuration based on trading strategy
config = get_config('risk_management')  # For conservative trading
reward_system = SimpleRewardSystem(config)
```

### Step 2: Calculate Rewards During Trading
```python
def step(action, market_state):
    # Execute action in trading environment
    new_state, closed_position = trading_env.execute_action(action)
    
    # Calculate reward
    reward = reward_system.calculate_reward(
        action=action,
        open_positions=new_state['positions'],
        closed_position=closed_position,
        current_price=new_state['current_price']
    )
    
    return new_state, reward
```

### Step 3: Monitor Performance
```python
# Get statistics
stats = reward_system.get_stats()
print(f"Exploration Score: {stats['exploration_score']:.3f}")
print(f"Action Distribution: {stats['action_distribution']}")

# Reset for new episode
reward_system.reset()
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `exploration_bonus` | Reward for trying diverse actions | 0.1 |
| `action_diversity_window` | Number of recent actions tracked | 20 |
| `max_comfortable_positions` | Position count before closure encouragement | 5 |
| `max_positions_threshold` | Position count before heavy penalties | 10 |
| `position_penalty_multiplier` | Penalty scaling for excess positions | 0.5 |
| `profit_margins` | Profit percentage thresholds | [0.01, 0.03, 0.05, 0.10] |
| `profit_rewards` | Rewards for profit tiers | [1.0, 2.5, 5.0, 10.0] |
| `loss_margins` | Loss percentage thresholds | [-0.01, -0.03, -0.05, -0.10] |
| `loss_penalties` | Penalties for loss tiers | [-0.5, -1.5, -3.0, -7.0] |
| `closure_bonus_threshold` | Position count for closure bonus | 5 |
| `closure_bonus` | Bonus for closing excess positions | 0.5 |

## Example Scenarios

### Scenario 1: Normal Trading
```python
# Opening first position
reward = reward_system.calculate_reward('OPEN_LONG', [])
# Result: +0.2 (base + exploration bonus)
```

### Scenario 2: Crowded Portfolio
```python
# Opening 8th position (crowded but not penalized)
reward = reward_system.calculate_reward('OPEN_LONG', [{}] * 7)
# Result: -0.05 (base reward - crowding penalty)
```

### Scenario 3: Excessive Positions
```python
# Opening 13th position (heavily penalized)
reward = reward_system.calculate_reward('OPEN_LONG', [{}] * 12)
# Result: -1.8 (base reward - heavy penalty)
```

### Scenario 4: Profitable Closure
```python
# Closing 5% profitable position with crowded portfolio
reward = reward_system.calculate_reward(
    'CLOSE_LONG', 
    [{}] * 7, 
    {'pnl_percentage': 0.05}
)
# Result: +5.7 (base + profit + closure bonus)
```

### Scenario 5: Loss Management
```python
# Closing losing position
reward = reward_system.calculate_reward(
    'CLOSE_SHORT', 
    [{}] * 3, 
    {'pnl_percentage': -0.02}
)
# Result: -1.3 (base + loss penalty)
```

## Best Practices

1. **Configuration Selection**: Choose configurations based on your risk tolerance and trading style
2. **Regular Monitoring**: Check exploration scores to ensure diverse action selection
3. **Parameter Tuning**: Adjust thresholds based on observed agent behavior
4. **Episode Reset**: Call `reset()` between training episodes to clear action history
5. **Statistics Tracking**: Use `get_stats()` to monitor reward system performance

## Advanced Customization

### Creating Custom Configurations
```python
custom_config = {
    # Copy default config and modify specific parameters
    **SIMPLE_REWARD_CONFIG,
    'max_positions_threshold': 8,  # Lower threshold
    'profit_rewards': [2.0, 5.0, 10.0, 20.0],  # Higher rewards
    'exploration_bonus': 0.2,  # More exploration
}

reward_system = SimpleRewardSystem(custom_config)
```

### Dynamic Configuration Updates
```python
# Update configuration during training
reward_system.config['exploration_bonus'] = 0.05  # Reduce exploration over time
reward_system.config['max_positions_threshold'] = 12  # Allow more positions as agent improves
```

## Performance Notes

- **Computational Efficiency**: O(1) reward calculation with minimal overhead
- **Memory Usage**: Tracks only recent actions for exploration (configurable window)
- **Scalability**: Suitable for high-frequency trading with thousands of decisions per episode

## Troubleshooting

### Low Exploration Scores
- Increase `exploration_bonus`
- Extend `action_diversity_window`
- Add small penalty for repetitive actions

### Excessive Position Opening
- Lower `max_positions_threshold`
- Increase `position_penalty_multiplier` 
- Raise `close_reward`

### Poor Risk Management
- Use 'risk_management' or 'conservative' configuration
- Increase loss penalties
- Lower closure bonus threshold

## Version History

- **v1.0**: Initial simple reward system implementation
- Focus on four core behavioral objectives
- Multiple pre-configured reward profiles
- Comprehensive documentation and examples
