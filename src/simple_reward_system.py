"""
Simple Reward System for RL Trading Bot

A simplified reward system that focuses on:
1. Encouraging exploration of all actions
2. Penalizing excessive open positions (>10)
3. Multi-level rewards/penalties for profit/loss based on margin
4. Encouraging position closure when >5 open positions

"""

import numpy as np
from typing import Dict, List, Optional


class SimpleRewardSystem:
    """
    Simplified reward system for trading reinforcement learning
    
    Key Features:
    - Action exploration encouragement
    - Position limit management
    - Profit margin-based rewards
    - Position closure incentives
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the simple reward system
        
        Args:
            config: Optional configuration dictionary
        """
        # Default configuration
        default_config = {
            # Exploration rewards
            'exploration_bonus': 0.1,
            'action_diversity_window': 20,  # Track last N actions
            
            # Position management
            'max_comfortable_positions': 5,  # Start encouraging closure after this
            'max_positions_threshold': 10,   # Start heavy penalties after this
            'position_penalty_multiplier': 0.5,  # Penalty per excess position
            
            # Profit/Loss rewards
            'profit_margins': [0.01, 0.03, 0.05, 0.10],  # 1%, 3%, 5%, 10%
            'profit_rewards': [1.0, 2.5, 5.0, 10.0],     # Corresponding rewards
            'loss_margins': [-0.01, -0.03, -0.05, -0.10], # -1%, -3%, -5%, -10%
            'loss_penalties': [-0.5, -1.5, -3.0, -7.0],   # Corresponding penalties
            
            # Position closure incentives
            'closure_bonus_threshold': 5,  # When to start giving closure bonuses
            'closure_bonus': 0.5,         # Bonus for closing when >5 positions
            
            # Base action rewards/penalties
            'hold_reward': 0.0,
            'open_reward': 0.1,    # Small reward for taking action
            'close_reward': 0.2,   # Slightly higher for closing (more decisive)
        }
        
        # Merge with user config
        self.config = {**default_config, **(config or {})}
        
        # Track action history for exploration
        self.action_history = []
        self.last_actions = []
        
        # Add compatibility attributes for legacy environment
        self.trade_frequency_counter = 0
        self.steps_since_last_trade = 0
        self.step_counter = 0
        
    def calculate_reward(self, 
                        action: str,
                        open_positions: List[Dict],
                        closed_position: Optional[Dict] = None,
                        current_price: float = None) -> float:
        """
        Calculate the total reward for the current action
        
        Args:
            action: The action taken ('HOLD', 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT')
            open_positions: List of currently open positions
            closed_position: Position that was just closed (if any)
            current_price: Current market price
            
        Returns:
            float: Total reward for the action
        """
        total_reward = 0.0
        
        # 1. Base action reward
        total_reward += self._get_base_action_reward(action)
        
        # 2. Exploration bonus
        total_reward += self._get_exploration_bonus(action)
        
        # 3. Position management penalty/reward
        total_reward += self._get_position_management_reward(action, len(open_positions))
        
        # 4. Profit/Loss reward for closed positions
        if closed_position and action.startswith('CLOSE'):
            total_reward += self._get_profit_loss_reward(closed_position)
        
        # 5. Position closure encouragement
        if action.startswith('CLOSE') and len(open_positions) > self.config['closure_bonus_threshold']:
            total_reward += self.config['closure_bonus']
        
        # Update action history
        self._update_action_history(action)
        
        # Update legacy compatibility counters
        self.step_counter += 1
        if action.startswith('OPEN') or action.startswith('CLOSE'):
            self.trade_frequency_counter += 1
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1
        
        return total_reward
    
    def _get_base_action_reward(self, action: str) -> float:
        """Get base reward for the action type"""
        if action == 'HOLD':
            return self.config['hold_reward']
        elif action.startswith('OPEN'):
            return self.config['open_reward']
        elif action.startswith('CLOSE'):
            return self.config['close_reward']
        else:
            return 0.0
    
    def _get_exploration_bonus(self, action: str) -> float:
        """
        Calculate exploration bonus to encourage trying different actions
        
        The bonus is higher when the agent tries actions it hasn't used recently
        """
        window_size = self.config['action_diversity_window']
        
        # Count recent action occurrences
        recent_actions = self.last_actions[-window_size:] if len(self.last_actions) >= window_size else self.last_actions
        
        if not recent_actions:
            return self.config['exploration_bonus']  # First action gets bonus
        
        # Calculate how rare this action is in recent history
        action_count = recent_actions.count(action)
        total_recent = len(recent_actions)
        
        # Bonus inversely proportional to how often action was used
        if total_recent > 0:
            action_frequency = action_count / total_recent
            exploration_bonus = self.config['exploration_bonus'] * (1.0 - action_frequency)
            return exploration_bonus
        
        return 0.0
    
    def _get_position_management_reward(self, action: str, num_open_positions: int) -> float:
        """
        Calculate reward/penalty based on position management
        
        Rules:
        - Penalty for having >10 open positions (increases with each additional position)
        - Encourage closing when >5 positions
        """
        reward = 0.0
        
        # Heavy penalty for exceeding position threshold
        if num_open_positions > self.config['max_positions_threshold']:
            excess_positions = num_open_positions - self.config['max_positions_threshold']
            # Exponentially increasing penalty
            penalty = -self.config['position_penalty_multiplier'] * (excess_positions ** 2)
            reward += penalty
        
        # Encourage closing when we have too many positions
        elif num_open_positions > self.config['max_comfortable_positions']:
            if action.startswith('CLOSE'):
                # Reward for closing when we have too many positions
                excess = num_open_positions - self.config['max_comfortable_positions']
                closure_reward = 0.2 * excess  # Increase reward with more excess positions
                reward += closure_reward
            elif action.startswith('OPEN'):
                # Small penalty for opening more when already crowded
                excess = num_open_positions - self.config['max_comfortable_positions']
                opening_penalty = -0.1 * excess
                reward += opening_penalty
        
        return reward
    
    def _get_profit_loss_reward(self, closed_position: Dict) -> float:
        """
        Calculate multi-level reward/penalty based on profit margin
        
        Args:
            closed_position: Dictionary containing position info including 'pnl_percentage'
        """
        if 'pnl_percentage' not in closed_position:
            return 0.0
        
        pnl_pct = closed_position['pnl_percentage']
        
        # Determine reward based on profit margin levels
        if pnl_pct > 0:  # Profit
            # Find the appropriate profit tier
            for i, margin in enumerate(self.config['profit_margins']):
                if pnl_pct >= margin:
                    continue
                else:
                    # Use the previous tier's reward
                    tier_index = max(0, i - 1)
                    return self.config['profit_rewards'][tier_index]
            
            # If profit exceeds all margins, use the highest reward
            return self.config['profit_rewards'][-1]
        
        else:  # Loss
            # Find the appropriate loss tier
            for i, margin in enumerate(self.config['loss_margins']):
                if pnl_pct <= margin:
                    continue
                else:
                    # Use the previous tier's penalty
                    tier_index = max(0, i - 1)
                    return self.config['loss_penalties'][tier_index]
            
            # If loss exceeds all margins, use the highest penalty
            return self.config['loss_penalties'][-1]
    
    def _update_action_history(self, action: str) -> None:
        """Update the action history for exploration tracking"""
        self.action_history.append(action)
        self.last_actions.append(action)
        
        # Keep only recent actions for exploration calculation
        max_history = self.config['action_diversity_window'] * 2
        if len(self.last_actions) > max_history:
            self.last_actions = self.last_actions[-max_history:]
    
    def get_action_distribution(self) -> Dict[str, float]:
        """Get the distribution of actions taken for analysis"""
        if not self.action_history:
            return {}
        
        total_actions = len(self.action_history)
        action_counts = {}
        
        for action in self.action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Convert to percentages
        action_distribution = {
            action: (count / total_actions) * 100 
            for action, count in action_counts.items()
        }
        
        return action_distribution
    
    def reset(self) -> None:
        """Reset the reward system for a new episode"""
        self.action_history = []
        self.last_actions = []
        # Reset legacy compatibility counters
        self.trade_frequency_counter = 0
        self.steps_since_last_trade = 0
        self.step_counter = 0
    
    def get_stats(self) -> Dict:
        """Get statistics about the reward system performance"""
        stats = {
            'total_actions': len(self.action_history),
            'action_distribution': self.get_action_distribution(),
            'exploration_score': self._calculate_exploration_score(),
            'config': self.config.copy()
        }
        return stats
    
    def _calculate_exploration_score(self) -> float:
        """Calculate how well the agent is exploring (0-1, higher is better)"""
        if not self.action_history:
            return 0.0
        
        # Count unique actions
        unique_actions = set(self.action_history)
        possible_actions = {'HOLD', 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'}
        
        # Calculate entropy of action distribution
        action_dist = self.get_action_distribution()
        if not action_dist:
            return 0.0
        
        # Normalize percentages to probabilities
        probs = [dist / 100.0 for dist in action_dist.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(possible_actions))
        
        # Normalize to 0-1
        exploration_score = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return exploration_score


# Example usage and testing
if __name__ == "__main__":
    # Initialize the reward system
    reward_system = SimpleRewardSystem()
    
    print("Simple Reward System Demo")
    print("=" * 40)
    
    # Test different scenarios
    scenarios = [
        {
            'action': 'OPEN_LONG',
            'open_positions': [],
            'description': 'Opening first position'
        },
        {
            'action': 'OPEN_SHORT', 
            'open_positions': [{}] * 3,
            'description': 'Opening position with 3 already open'
        },
        {
            'action': 'OPEN_LONG',
            'open_positions': [{}] * 7,
            'description': 'Opening position with 7 already open (should get penalty)'
        },
        {
            'action': 'CLOSE_LONG',
            'open_positions': [{}] * 12,
            'closed_position': {'pnl_percentage': 0.025},  # 2.5% profit
            'description': 'Closing profitable position with 12 open (should get bonus)'
        },
        {
            'action': 'CLOSE_SHORT',
            'open_positions': [{}] * 4,
            'closed_position': {'pnl_percentage': -0.04},  # -4% loss
            'description': 'Closing losing position'
        },
        {
            'action': 'HOLD',
            'open_positions': [{}] * 6,
            'description': 'Holding with 6 positions'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        reward = reward_system.calculate_reward(
            action=scenario['action'],
            open_positions=scenario['open_positions'],
            closed_position=scenario.get('closed_position'),
            current_price=50000.0
        )
        
        print(f"\nScenario {i}: {scenario['description']}")
        print(f"Action: {scenario['action']}")
        print(f"Open positions: {len(scenario['open_positions'])}")
        print(f"Reward: {reward:.3f}")
    
    print(f"\nFinal Stats:")
    stats = reward_system.get_stats()
    print(f"Total actions: {stats['total_actions']}")
    print(f"Action distribution: {stats['action_distribution']}")
    print(f"Exploration score: {stats['exploration_score']:.3f}")
