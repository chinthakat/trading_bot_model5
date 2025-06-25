#!/usr/bin/env python3
"""
Enhanced exploration model with proper action space scaling
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.distributions import DiagGaussianDistribution
import gymnasium as gym
from pathlib import Path

class DiscreteActionDistribution(DiagGaussianDistribution):
    """
    Custom action distribution that properly maps to discrete actions
    """
    
    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.action_dim = action_dim
    
    def sample(self) -> torch.Tensor:
        """Sample actions with proper scaling for discrete action types"""
        # Get normal samples
        normal_samples = super().sample()
        
        # For the first dimension (action type), we want uniform distribution across [0, 3]
        # Scale from normal distribution to uniform [0, 3]
        action_type = torch.sigmoid(normal_samples[:, 0]) * 3.0  # Maps to [0, 3]
        
        # Keep other dimensions as normal
        scaled_actions = normal_samples.clone()
        scaled_actions[:, 0] = action_type
        
        return scaled_actions
    
    def mode(self) -> torch.Tensor:
        """Return mode (mean) with proper scaling"""
        mode_actions = super().mode()
        
        # Scale action type dimension
        action_type = torch.sigmoid(mode_actions[:, 0]) * 3.0
        mode_actions[:, 0] = action_type
        
        return mode_actions

class ExplorationTradingFeatureExtractor(BaseFeaturesExtractor):
    """Enhanced feature extractor for better exploration"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        total_features = observation_space.shape[0]
        
        # Enhanced network for better feature extraction
        self.net = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure observations are properly shaped
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
        output = self.net(observations)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output

class EnhancedExplorationPolicy(ActorCriticPolicy):
    """Enhanced policy with better action distribution for exploration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=ExplorationTradingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        
        # Initialize for better exploration
        self._init_exploration_weights()
    
    def _init_exploration_weights(self):
        """Initialize weights to encourage exploration of all actions"""
        try:
            # Initialize action net with larger variance for exploration
            if hasattr(self.action_net, 'weight'):
                nn.init.normal_(self.action_net.weight, mean=0.0, std=0.5)  # Higher std
                
            if hasattr(self.action_net, 'bias') and self.action_net.bias is not None:
                with torch.no_grad():
                    # Initialize bias to encourage uniform action distribution
                    # For continuous actions, we want the mean to be in the middle of [0, 3]
                    if self.action_net.bias.shape[0] >= 1:
                        self.action_net.bias[0] = 1.5  # Middle of [0, 3] range
                        if self.action_net.bias.shape[0] >= 3:
                            self.action_net.bias[1] = 3.5  # Middle of size range
                            self.action_net.bias[2] = 2.0  # Middle of leverage range
                            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not initialize exploration weights: {e}")

class FixedExplorationTradingModel:
    """Trading model with fixed action space scaling for exploration"""
    
    def __init__(
        self,
        env: gym.Env,
        model_name: str = "fixed_exploration_ppo",
        tensorboard_log: str = "logs/tensorboard/",
        verbose: int = 1,
        logging_config: Optional[Dict] = None,
        device: str = "auto"
    ):
        """Initialize the fixed exploration model"""
        self.env = env
        self.model_name = model_name
        self.verbose = verbose
        self.device = device
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing FixedExplorationTradingModel: {model_name}")
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize model
        self.model = None
        self._create_fixed_exploration_model()
        
        self.logger.info(f"Successfully initialized FixedExplorationTradingModel: {model_name}")
    
    def _get_exploration_config(self) -> Dict[str, Any]:
        """Get PPO config optimized for discrete action exploration"""
        return {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.05,  # High entropy for exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            
            'policy_kwargs': {
                'net_arch': {
                    'pi': [256, 128, 64],
                    'vf': [256, 128, 64]
                },
                'activation_fn': torch.nn.ReLU,
                'optimizer_class': torch.optim.Adam,
                'optimizer_kwargs': {
                    'eps': 1e-8,
                },
                # Use larger log_std_init for more exploration
                'log_std_init': 0.0,  # exp(0) = 1.0 std deviation
            },
            
            'normalize_advantage': True,
            'use_sde': False,
            'target_kl': None,  # No KL constraint for exploration
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _create_fixed_exploration_model(self):
        """Create PPO model with fixed action space scaling"""
        try:
            config = self._get_exploration_config()
            
            self.model = PPO(
                policy=EnhancedExplorationPolicy,
                env=self.env,
                verbose=self.verbose,
                device=self.device,
                **config
            )
            
            self.logger.info(f"Created fixed exploration PPO model")
            
        except Exception as e:
            self.logger.error(f"Failed to create fixed exploration model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'status': 'initialized',
            'type': 'fixed_exploration_focused',
            'policy': str(type(self.model.policy)),
            'device': str(self.model.device) if hasattr(self.model, 'device') else self.device,
            'n_envs': self.model.n_envs if hasattr(self.model, 'n_envs') else 1,
            'learning_rate': self.model.learning_rate if hasattr(self.model, 'learning_rate') else 'N/A',
            'entropy_coef': self.model.ent_coef if hasattr(self.model, 'ent_coef') else 'N/A',
        }
