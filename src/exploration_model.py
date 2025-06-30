#!/usr/bin/env python3
"""
Enhanced PPO Model with Strong Exploration for RL Trading
Designed to encourage exploration of all actions: HOLD, BUY, SELL, CLOSE_ALL
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
import gymnasium as gym
from pathlib import Path

class ExplorationTradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Enhanced feature extractor designed for better exploration
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions
        total_features = observation_space.shape[0]
        
        # Assume last 5 features are account state, rest are market data
        self.market_features_dim = total_features - 5
        self.account_features_dim = 5
        
        # Enhanced market data processing with attention mechanism
        self.market_net = nn.Sequential(
            nn.Linear(self.market_features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Enhanced account state processing
        self.account_net = nn.Sequential(
            nn.Linear(self.account_features_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined processing with larger capacity
        self.combined_net = nn.Sequential(
            nn.Linear(64 + 16, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input validation and cleaning
        if torch.isnan(observations).any() or torch.isinf(observations).any():
            observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize observations to prevent extreme values
        observations = torch.clamp(observations, -5.0, 5.0)
        
        batch_size = observations.shape[0]
        
        # Split observations into market and account features
        if observations.shape[1] >= 5:
            market_data = observations[:, :-5]
            account_data = observations[:, -5:]
        else:
            market_data = observations
            account_data = torch.zeros(batch_size, 5, device=observations.device)
        
        # Process market data
        market_features = self.market_net(market_data)
        
        # Process account data
        account_features = self.account_net(account_data)
        
        # Combine features
        combined = torch.cat([market_features, account_features], dim=1)
        
        # Final processing
        output = self.combined_net(combined)
        
        # Final safety checks
        output = torch.clamp(output, -3.0, 3.0)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output

class ExplorationActorCriticPolicy(ActorCriticPolicy):
    """
    Enhanced Actor-Critic policy with exploration-focused initialization
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=ExplorationTradingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        
        # Initialize action distribution to be more uniform
        self._init_action_distribution()
    
    def _init_action_distribution(self):
        """Initialize the action distribution to encourage exploration"""
        try:
            # Find the action net (policy head)
            if hasattr(self.action_net, 'weight'):
                # Initialize with smaller weights to encourage uniform distribution
                nn.init.normal_(self.action_net.weight, mean=0.0, std=0.01)
                if hasattr(self.action_net, 'bias') and self.action_net.bias is not None:
                    # Initialize bias to slightly favor exploration actions
                    with torch.no_grad():
                        # Slightly bias towards less common actions (SELL=2, CLOSE_ALL=3)
                        if self.action_net.bias.shape[0] == 4:  # 4 actions
                            self.action_net.bias[0] = -0.1  # HOLD
                            self.action_net.bias[1] = 0.0   # BUY  
                            self.action_net.bias[2] = 0.2   # SELL (encourage)
                            self.action_net.bias[3] = 0.1   # CLOSE_ALL (encourage)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not initialize action distribution: {e}")

class ExplorationTradingModel:
    """
    Enhanced PPO model wrapper with strong exploration capabilities
    """
    
    def __init__(
        self,
        env: gym.Env,
        model_name: str = "exploration_ppo_trading",
        tensorboard_log: str = "logs/tensorboard/",
        verbose: int = 1,
        logging_config: Optional[Dict] = None,
        device: str = "auto"
    ):
        """
        Initialize enhanced trading model with exploration focus
        """
        self.env = env
        self.model_name = model_name
        self.verbose = verbose
        self.device = device
        
        # Apply logging configuration
        self.logging_config = logging_config or {}
        self.enable_tensorboard = self.logging_config.get('enable_tensorboard', True)
        
        # Set TensorBoard log directory only if enabled
        if self.enable_tensorboard:
            self.tensorboard_log = tensorboard_log
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
        else:
            self.tensorboard_log = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing ExplorationTradingModel: {model_name}")
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize model
        self.model = None
        self._create_exploration_model()
        
        self.logger.info(f"Successfully initialized ExplorationTradingModel: {model_name}")
    
    def _get_exploration_config(self) -> Dict[str, Any]:
        """
        Get PPO hyperparameters optimized for exploration
        """
        return {
            # High exploration parameters
            'learning_rate': 5e-4,      # Higher learning rate for faster learning
            'n_steps': 2048,            # Larger rollout buffer for diverse experiences
            'batch_size': 128,          # Larger batch size for stable gradients
            'n_epochs': 5,              # More epochs for better policy updates
            'gamma': 0.99,              # Standard discount factor
            'gae_lambda': 0.95,         # Standard GAE lambda
            'clip_range': 0.25,         # Higher clip range for exploration
            'ent_coef': 0.1,            # VERY HIGH entropy bonus for exploration
            'vf_coef': 0.5,             # Balanced value function weight
            'max_grad_norm': 0.5,       # Moderate gradient clipping
            
            # Enhanced network architecture for complex policies
            'policy_kwargs': {
                'net_arch': {
                    'pi': [512, 256, 128, 64],   # Deep actor network
                    'vf': [512, 256, 128, 64]    # Deep critic network
                },
                'activation_fn': torch.nn.ReLU,
                'optimizer_class': torch.optim.Adam,
                'optimizer_kwargs': {
                    'eps': 1e-8,
                    'weight_decay': 0.0  # No weight decay for exploration
                }
            },
            
            # Training stability with exploration focus
            'normalize_advantage': True,
            'use_sde': False,           # Disable SDE to avoid device issues
            'target_kl': 0.1,           # Higher KL tolerance for exploration
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _create_exploration_model(self):
        """Create PPO model with strong exploration capabilities"""
        try:
            config = self._get_exploration_config()
            
            # Prepare PPO arguments
            ppo_kwargs = {
                'policy': ExplorationActorCriticPolicy,
                'env': self.env,
                'verbose': self.verbose,
                'device': self.device,
                **config
            }

            # Only include tensorboard_log if TensorBoard logging is enabled
            if self.enable_tensorboard and self.tensorboard_log:
                ppo_kwargs['tensorboard_log'] = self.tensorboard_log
            
            self.model = PPO(**ppo_kwargs)
            
            # Log the actual device being used
            actual_device = str(self.model.device) if hasattr(self.model, 'device') else self.device
            self.logger.info(f"Created exploration PPO model on device: {actual_device}")
            
            # Enhanced parameter initialization for exploration
            self._initialize_exploration_parameters()
            
        except Exception as e:
            self.logger.error(f"Failed to create exploration PPO model: {e}")
            raise
    
    def _initialize_exploration_parameters(self):
        """Initialize model parameters to encourage exploration"""
        try:
            policy = self.model.policy
            
            # Initialize parameters with exploration-friendly values
            for name, param in policy.named_parameters():
                if 'action_net' in name and 'weight' in name:
                    # Initialize action network with very small weights for uniform distribution
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'action_net' in name and 'bias' in name:
                    # Initialize action bias to encourage less common actions
                    with torch.no_grad():
                        if param.shape[0] == 4:  # 4 actions: HOLD, BUY, SELL, CLOSE_ALL
                            param[0] = -0.2  # HOLD (discourage)
                            param[1] = 0.0   # BUY (neutral)
                            param[2] = 0.3   # SELL (encourage strongly)
                            param[3] = 0.2   # CLOSE_ALL (encourage)
                elif 'weight' in name and param.dim() > 1:
                    # Use Xavier initialization for other weights
                    torch.nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                elif 'log_std' in name:
                    # Initialize log std for higher exploration
                    torch.nn.init.constant_(param, -0.5)  # exp(-0.5) = ~0.6 std
            
            self.logger.info("Model parameters initialized for exploration")
            
        except Exception as e:
            self.logger.warning(f"Exploration parameter initialization failed: {e}")
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        save_freq: int = 10000,
        save_path: str = "models/"
    ) -> 'PPO':
        """
        Train the exploration-focused PPO model
        """
        self.logger.info(f"Starting exploration training for {total_timesteps} timesteps")
        
        # Setup callbacks
        callbacks = []
        
        try:
            # Custom trading callback with action monitoring
            trading_callback = ExplorationTrackingCallback(eval_freq=eval_freq)
            callbacks.append(trading_callback)
            
            # Checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix=self.model_name
            )
            callbacks.append(checkpoint_callback)
            
            # Evaluation callback if eval env provided
            if eval_env:
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=save_path,
                    log_path="logs/",
                    eval_freq=eval_freq,
                    deterministic=False,  # Use stochastic evaluation for exploration
                    render=False
                )
                callbacks.append(eval_callback)
            
            # Train model with exploration focus
            learn_kwargs = {
                'total_timesteps': total_timesteps,
                'callback': callbacks,
                'reset_num_timesteps': False
            }
            
            # Only include TensorBoard logging if enabled
            if self.enable_tensorboard:
                learn_kwargs['tb_log_name'] = self.model_name
            
            self.model.learn(**learn_kwargs)
            
            # Save final model
            final_model_path = Path(save_path) / f"{self.model_name}_final"
            self.model.save(final_model_path)
            self.logger.info(f"Exploration training completed. Model saved to {final_model_path}")
            
        except Exception as e:
            self.logger.error(f"Exploration training failed: {e}")
            raise
        
        return self.model
    
    def evaluate(
        self,
        eval_env: gym.Env,
        n_eval_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate trained model with step-level reward capping
        
        Args:
            eval_env: Environment for evaluation
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Use deterministic actions
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Evaluating exploration model for {n_eval_episodes} episodes")
        
        episode_rewards = []
        episode_stats = []
        
        try:
            for episode in range(n_eval_episodes):
                obs, info = eval_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    
                    # CRITICAL: Cap the step-level reward to prevent extreme values  
                    # Adjusted to match the new reward scaling in environment
                    reward = np.clip(reward, -5.0, 5.0)
                    episode_reward += reward
                    done = terminated or truncated

                episode_rewards.append(episode_reward)
                
                # Collect episode statistics if available
                if hasattr(eval_env, 'episode_stats'):
                    episode_stats.append(eval_env.episode_stats.copy())
            
            # Calculate evaluation metrics
            eval_metrics = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'n_episodes': n_eval_episodes
            }
            
            # Add trading-specific metrics if available
            if episode_stats:
                eval_metrics.update({
                    'mean_return': np.mean([stats['total_return'] for stats in episode_stats]),
                    'mean_sharpe': np.mean([stats['sharpe_ratio'] for stats in episode_stats]),
                    'mean_max_dd': np.mean([stats['max_drawdown'] for stats in episode_stats]),
                    'mean_win_rate': np.mean([stats['win_rate'] for stats in episode_stats]),
                    'mean_trades': np.mean([stats['total_trades'] for stats in episode_stats])
                })
            
            self.logger.info(f"Evaluation completed. Mean reward: {eval_metrics['mean_reward']:.2f}")
            return eval_metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get exploration model information"""
        if self.model is None:
            return {"status": "not_trained", "type": "exploration"}
        
        return {
            "status": "trained",
            "type": "exploration_focused",
            "policy": str(type(self.model.policy)),
            "device": str(self.model.device),
            "n_envs": self.model.n_envs,
            "learning_rate": self.model.learning_rate,
            "entropy_coef": getattr(self.model, 'ent_coef', 'unknown'),
            "use_sde": getattr(self.model, 'use_sde', 'unknown')
        }

class ExplorationTrackingCallback(BaseCallback):
    """
    Callback to track action exploration during training
    """
    
    def __init__(self, eval_freq: int = 1000, save_path: str = "logs/", verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Track action distributions
        self.action_counts = [0, 0, 0, 0]  # HOLD, BUY, SELL, CLOSE_ALL
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if hasattr(actions, '__iter__'):
                for action in actions:
                    if 0 <= action < 4:
                        self.action_counts[action] += 1
            else:
                if 0 <= actions < 4:
                    self.action_counts[actions] += 1
        
        # Log action distribution periodically
        if self.step_count % self.eval_freq == 0:
            total_actions = sum(self.action_counts)
            if total_actions > 0:
                action_probs = [count / total_actions for count in self.action_counts]
                self.logger.info(f"Step {self.step_count} - Action distribution:")
                self.logger.info(f"  HOLD: {action_probs[0]:.3f} ({self.action_counts[0]})")
                self.logger.info(f"  BUY:  {action_probs[1]:.3f} ({self.action_counts[1]})")
                self.logger.info(f"  SELL: {action_probs[2]:.3f} ({self.action_counts[2]})")
                self.logger.info(f"  CLOSE:{action_probs[3]:.3f} ({self.action_counts[3]})")
        
        # Check episode end info
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    if self.verbose >= 1:
                        print(f"Episode reward: {ep_reward:.2f}, Length: {ep_length}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Save training statistics"""
        # Save final action distribution
        total_actions = sum(self.action_counts)
        if total_actions > 0:
            action_stats = {
                'total_actions': total_actions,
                'action_counts': self.action_counts,
                'action_percentages': [count / total_actions for count in self.action_counts],
                'action_names': ['HOLD', 'BUY', 'SELL', 'CLOSE_ALL']
            }
            
            stats_file = self.save_path / "exploration_stats.json"
            import json
            with open(stats_file, 'w') as f:
                json.dump(action_stats, f, indent=2)
            
            print(f"Final action distribution:")
            for i, name in enumerate(action_stats['action_names']):
                print(f"  {name}: {action_stats['action_percentages'][i]:.3f}")
            
            print(f"Exploration statistics saved to {stats_file}")

def create_exploration_model_config() -> Dict:
    """
    Create exploration-focused model configuration
    """
    return {
        # Enhanced exploration parameters
        'learning_rate': 5e-4,
        'n_steps': 2048,
        'batch_size': 128,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.25,
        'ent_coef': 0.1,              # HIGH entropy for exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Network architecture
        'policy_kwargs': {
            'net_arch': {
                'pi': [512, 256, 128, 64],
                'vf': [512, 256, 128, 64]
            },
            'activation_fn': torch.nn.ReLU,
            'optimizer_class': torch.optim.Adam,
            'optimizer_kwargs': {
                'eps': 1e-8,
                'weight_decay': 0.0
            }
        },
        
        # Exploration features
        'normalize_advantage': True,
        'use_sde': False,             # Disable SDE for stability
        'target_kl': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

if __name__ == "__main__":
    print("Enhanced Exploration Trading Model")
    print("Designed to encourage exploration of all actions: HOLD, BUY, SELL, CLOSE_ALL")
