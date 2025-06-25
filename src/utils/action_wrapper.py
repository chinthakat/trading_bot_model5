"""
Action space wrapper for converting continuous action space to discrete action space
for better exploration in reinforcement learning models.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrulyDiscreteActionWrapper(gym.Wrapper):
    """
    Wrapper to convert continuous action space to truly discrete.
    
    This wrapper converts the environment's continuous action space [action_type, size_index, leverage]
    to a discrete action space with 4 actions: HOLD, BUY, SELL, CLOSE_ALL.
    
    This helps RL models explore all actions more evenly, especially important for
    action types like SELL and CLOSE_ALL that might be underexplored with continuous outputs.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Create discrete action space: 4 actions (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE_ALL)
        self.action_space = spaces.Discrete(4)
        
        # Keep observation space the same
        self.observation_space = env.observation_space
        
        # Store original environment for access
        self.original_env = env.unwrapped
        
    def action(self, action):
        """Convert discrete action to continuous action for original environment"""
        # action is an integer from 0 to 3
        
        # Map to continuous action format [action_type, size_index, leverage]
        continuous_action = np.array([
            float(action),  # action type (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE_ALL)
            0.0,           # size index (use smallest size for consistency)
            1.0            # leverage (keep at 1.0 for simplicity)
        ], dtype=np.float32)
        
        return continuous_action
    
    def step(self, action):
        """Take a step with the discrete action"""
        # Convert discrete action to continuous
        continuous_action = self.action(action)
        
        # Forward to original environment (Gymnasium API returns 5 values)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Add action mapping info to info dict
        if 'action_mapping' not in info:
            info['action_mapping'] = {}
        
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE_ALL'}
        info['action_mapping'] = {
            'discrete_action': action,
            'continuous_action': continuous_action.tolist(),
            'action_name': action_names.get(action, f'UNKNOWN_{action}')
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment"""
        return self.env.reset(**kwargs)
    
    def render(self, mode='human', **kwargs):
        """Render the environment"""
        return self.env.render(mode=mode, **kwargs)
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    def seed(self, seed=None):
        """Seed the environment"""
        return self.env.seed(seed)
    
    @property
    def unwrapped(self):
        """Get the unwrapped environment"""
        return self.env.unwrapped
