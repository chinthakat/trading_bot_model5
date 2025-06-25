#!/usr/bin/env python3
"""
PPO Model Configuration for RL Trading
Optimized PPO implementation using Stable Baselines3
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
from datetime import datetime

class TradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading data
    Processes market data and account features separately
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions
        total_features = observation_space.shape[0]
        
        # Assume last 5 features are account state, rest are market data
        self.market_features_dim = total_features - 5
        self.account_features_dim = 5
        
        # Market data processing (LSTM for sequential data)
        self.market_lstm = nn.LSTM(
            input_size=self.market_features_dim // 50,  # Assuming 50 timesteps
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Account state processing
        self.account_net = nn.Sequential(
            nn.Linear(self.account_features_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(128 + 16, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input validation and cleaning
        if torch.isnan(observations).any() or torch.isinf(observations).any():
            self.logger.warning("NaN/Inf detected in observations, cleaning...")
            observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp extreme values to prevent gradient explosion
        observations = torch.clamp(observations, -10.0, 10.0)
        
        batch_size = observations.shape[0]
        
        # Split observations into market and account features
        # Assuming last 5 features are account features
        if observations.shape[1] >= 5:
            market_data = observations[:, :-5]
            account_data = observations[:, -5:]
        else:
            # Fallback if observation shape is unexpected
            market_data = observations
            account_data = torch.zeros(batch_size, 5, device=observations.device)
        
        # Process market data with LSTM
        market_features = self._process_market_data(market_data)
        
        # Process account data with simple MLP
        account_features = self._process_account_data(account_data)
        
        # Combine features
        combined = torch.cat([market_features, account_features], dim=1)
        
        # Final processing
        output = self.combined_net(combined)
        
        # Final safety checks
        output = torch.clamp(output, -5.0, 5.0)  # Prevent extreme outputs
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output
    
    def _process_market_data(self, market_data: torch.Tensor) -> torch.Tensor:
        """Process market data through LSTM with robust error handling"""
        try:
            batch_size = market_data.shape[0]
            
            # Calculate sequence dimensions
            total_features = market_data.shape[1]
            seq_len = min(50, total_features // 5)  # Assume 5 features per timestep
            if seq_len == 0:
                seq_len = 1
            
            features_per_step = total_features // seq_len
            if features_per_step == 0:
                features_per_step = total_features
                seq_len = 1
            
            # Truncate to fit evenly
            used_features = seq_len * features_per_step
            market_data_truncated = market_data[:, :used_features]
            
            # Reshape for LSTM
            market_data_reshaped = market_data_truncated.view(batch_size, seq_len, features_per_step)
            
            # Additional safety check
            if torch.isnan(market_data_reshaped).any():
                market_data_reshaped = torch.nan_to_num(market_data_reshaped, nan=0.0)
            
            # Pass through LSTM
            lstm_out, (hidden, _) = self.market_lstm(market_data_reshaped)
            
            # Use last hidden state
            market_features = hidden[-1]  # Shape: (batch_size, hidden_size)
            
            # Ensure output size is correct (128)
            if market_features.shape[1] != 128:
                # Project to correct size
                if not hasattr(self, '_market_projection'):
                    self._market_projection = torch.nn.Linear(
                        market_features.shape[1], 128, device=market_features.device
                    )
                market_features = self._market_projection(market_features)
            
        except Exception as e:
            # Fallback: use mean pooling
            batch_size = market_data.shape[0]
            if market_data.shape[1] > 0:
                market_features = torch.mean(market_data.view(batch_size, -1, 5), dim=1)
                # Project to 128 dimensions
                if market_features.shape[1] != 128:
                    padding_size = max(0, 128 - market_features.shape[1])
                    if padding_size > 0:
                        padding = torch.zeros(batch_size, padding_size, device=market_data.device)
                        market_features = torch.cat([market_features, padding], dim=1)
                    else:
                        market_features = market_features[:, :128]
            else:
                market_features = torch.zeros(batch_size, 128, device=market_data.device)
        
        # Final safety check
        market_features = torch.nan_to_num(market_features, nan=0.0, posinf=1.0, neginf=-1.0)
        market_features = torch.clamp(market_features, -5.0, 5.0)
        
        return market_features
    
    def _process_account_data(self, account_data: torch.Tensor) -> torch.Tensor:
        """Process account data through MLP with robust error handling"""
        try:
            # Ensure input is clean
            account_data = torch.nan_to_num(account_data, nan=0.0, posinf=1.0, neginf=-1.0)
            account_data = torch.clamp(account_data, -10.0, 10.0)
            
            # Pass through account network
            account_features = self.account_net(account_data)
            
        except Exception as e:
            # Fallback: zeros
            batch_size = account_data.shape[0]
            account_features = torch.zeros(batch_size, 64, device=account_data.device)
        
        # Final safety check
        account_features = torch.nan_to_num(account_features, nan=0.0, posinf=1.0, neginf=-1.0)
        account_features = torch.clamp(account_features, -5.0, 5.0)
        
        return account_features

class TradingActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for trading
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TradingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

class TradingCallback(BaseCallback):
    """
    Custom callback for tracking training metrics
    """
    
    def __init__(self, eval_freq: int = 1000, save_path: str = "logs/", verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_stats = []
        
    def _on_step(self) -> bool:
        # Check if we have episode end info
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    if self.verbose >= 1:
                        print(f"Episode reward: {ep_reward:.2f}, Length: {ep_length}")
                
                # Log trading-specific metrics
                if 'total_return' in info:
                    self.episode_stats.append(info)
        
        return True
    
    def _on_training_end(self) -> None:
        """Save training statistics"""
        if self.episode_rewards:
            stats_file = self.save_path / "training_stats.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Average Episode Reward: {np.mean(self.episode_rewards):.2f}\n")
                f.write(f"Average Episode Length: {np.mean(self.episode_lengths):.2f}\n")
                f.write(f"Total Episodes: {len(self.episode_rewards)}\n")
            
            print(f"Training statistics saved to {stats_file}")


class TradingModel:
    """
    PPO model wrapper for trading
    Handles model creation, training, and evaluation
    """
    
    def __init__(
        self,
        env: gym.Env,
        model_name: str = "ppo_trading",
        tensorboard_log: str = "logs/tensorboard/",
        verbose: int = 1,
        logging_config: Optional[Dict] = None,
        device: str = "auto",
        config_overrides: Optional[Dict] = None
    ):
        """
        Initialize trading model
        
        Args:
            env: Trading environment
            model_name: Name for saving model
            tensorboard_log: TensorBoard log directory
            verbose: Verbosity level
            logging_config: Optional logging configuration dictionary
            device: Device to use for training ('auto', 'cpu', 'cuda')
            config_overrides: Optional config overrides for enhanced exploration
        """
        self.env = env
        self.model_name = model_name
        self.verbose = verbose
        self.device = device
        self.config_overrides = config_overrides or {}
        
        # Apply logging configuration
        self.logging_config = logging_config or {}
        self.enable_tensorboard = self.logging_config.get('enable_tensorboard', True)
        
        # Set TensorBoard log directory only if enabled
        if self.enable_tensorboard:
            self.tensorboard_log = tensorboard_log
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
        else:
            self.tensorboard_log = None
        
        # Setup logging FIRST before doing anything else
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing TradingModel: {model_name}")
        self.logger.info(f"TensorBoard logging: {'Enabled' if self.enable_tensorboard else 'Disabled'}")
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Model configuration
        self.model_config = self._get_optimized_config()
          # Initialize model
        self.model = None
        self._create_model()
        
        self.logger.info(f"Successfully initialized TradingModel: {model_name}")
    
    def _get_optimized_config(self) -> Dict[str, Any]:
        """
        Get optimized PPO hyperparameters for trading with enhanced exploration
        """
        return {
            # Enhanced learning parameters for better exploration
            'learning_rate': 3e-4,      # Higher learning rate for exploration
            'n_steps': 1024,            # Larger batch collection for diversity
            'batch_size': 64,           # Larger batch size for stable gradients
            'n_epochs': 4,              # Moderate epochs per update
            'gamma': 0.995,             # Higher gamma for longer-term rewards
            'gae_lambda': 0.98,         # Higher GAE lambda for better advantage estimation
            'clip_range': 0.3,          # Higher clip range for exploration
            'ent_coef': 0.05,           # MUCH higher entropy for exploration
            'vf_coef': 0.8,             # Higher value function weight
            'max_grad_norm': 1.0,       # Moderate gradient clipping
            
            # Enhanced network architecture for better learning
            'policy_kwargs': {
                'net_arch': {
                    'pi': [512, 256, 128],   # Larger actor network for complex policies
                    'vf': [512, 256, 128]    # Larger critic network for value estimation
                },                'activation_fn': torch.nn.ReLU,  # ReLU for faster learning
                'optimizer_class': torch.optim.Adam,  # Standard Adam optimizer
                'optimizer_kwargs': {
                    'eps': 1e-8,
                    'weight_decay': 1e-5    # Light L2 regularization
                }
            },
            
            # Training stability with exploration focus
            'normalize_advantage': True,
            'use_sde': False,
            'target_kl': 0.05,          # Higher KL tolerance for exploration
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _create_model(self):
        """Create PPO model with optimized configuration"""
        try:
            # Get the base configuration
            config = self._get_optimized_config()

            # Apply exploration-focused overrides
            config['ent_coef'] = 0.01
            config['learning_rate'] = 1e-4 # Use a more conservative learning rate
            
            # Prepare PPO arguments by spreading the config
            ppo_kwargs = {
                'policy': TradingActorCriticPolicy,
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
            self.logger.info(f"Created PPO model on device: {actual_device}")
            
            # Initialize parameters properly to avoid NaN
            self._initialize_parameters()
            
        except Exception as e:
            self.logger.error(f"Failed to create PPO model: {e}")
            raise
    
    def _initialize_parameters(self):
        """Initialize model parameters to prevent NaN values"""
        try:
            policy = self.model.policy
            
            # Initialize parameters with smaller values
            for name, param in policy.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    # Use smaller initialization
                    torch.nn.init.orthogonal_(param, gain=0.01)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
                elif 'log_std' in name:
                    # Initialize log std to very small values
                    torch.nn.init.constant_(param, -1.0)  # exp(-1) = ~0.37 std
            
            self.logger.info("Model parameters initialized with small values")
            
        except Exception as e:
            self.logger.warning(f"Parameter initialization failed: {e}")

    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        save_freq: int = 10000,
        save_path: str = "models/"
    ) -> 'PPO':
        """
        Train the PPO model
        
        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation
            eval_freq: Frequency of evaluation
            save_freq: Frequency of model saving
            save_path: Path to save models
            
        Returns:
            Trained PPO model
        """
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Setup callbacks
        callbacks = []
        
        try:
            # Custom trading callback
            trading_callback = TradingCallback(eval_freq=eval_freq)
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
                    deterministic=True,
                    render=False
                )
                callbacks.append(eval_callback)
              # Train model
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
            self.logger.info(f"Training completed. Model saved to {final_model_path}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        return self.model
    
    def evaluate(
        self,
        eval_env: gym.Env,
        n_eval_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate trained model
        
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
        
        self.logger.info(f"Evaluating model for {n_eval_episodes} episodes")
        
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
    
    def save_model(self, path: Union[str, Path]):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        try:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def archive_and_save_model(self, path: Union[str, Path], create_archive: bool = True):
        """
        Save model with optional archiving of previous models
        
        Args:
            path: Path to save the model
            create_archive: Whether to archive existing models first
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        try:
            save_path = Path(path)
            
            # Archive existing model if it exists and archiving is enabled
            if create_archive and save_path.exists():
                from utils.archiver import TrainingArchiver
                archiver = TrainingArchiver()
                
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{save_path.stem}_backup_{timestamp}{save_path.suffix}"
                backup_path = save_path.parent / backup_name
                
                # Move existing model to backup
                save_path.rename(backup_path)
                self.logger.info(f"Existing model backed up to: {backup_path}")
            
            # Save the new model
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model with archiving: {e}")
            raise

    def load_model(self, path: Union[str, Path], env: Optional[gym.Env] = None):
        """Load model from file"""
        if env is None:
            env = self.env
        
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        try:
            self.model = PPO.load(path, env=env)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def set_environment(self, env: gym.Env):
        """
        Set or update the environment for the model
        
        Args:
            env: New trading environment to use
        """
        try:
            self.env = env
            if self.model is not None:
                # Update the environment in the existing model
                self.model.set_env(env)
                self.logger.info("Model environment updated successfully")
            else:
                self.logger.warning("Model not initialized yet, environment will be used when model is created")
        except Exception as e:
            self.logger.error(f"Failed to set environment: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "policy": str(type(self.model.policy)),
            "device": str(self.model.device),
            "n_envs": self.model.n_envs,
            "learning_rate": self.model.learning_rate,
            "config": self.model_config
        }

def create_model_config(custom_config: Optional[Dict] = None) -> Dict:
    """
    Create model configuration with custom overrides
    
    Args:
        custom_config: Custom configuration to override defaults
        
    Returns:
        Complete model configuration
    """
    default_config = {
        # Data configuration
        'symbol': 'BINANCEFTS_PERP_BTC_USDT',
        'interval': '15m',
        
        # Date range options
        'use_date_range': False,  # Set to True to use specific date range instead of recent days
        'start_date': '2023-01-01',  # Start date for training data (YYYY-MM-DD)
        'end_date': '2023-12-31',    # End date for training data (YYYY-MM-DD)
        'data_days': 90,             # Number of recent days (used when use_date_range=False)
        
        # Data processing
        'use_recent_data': True,     # Keep for backward compatibility
        'train_ratio': 0.8,
        'lookback_window': 50,
        'use_funding_rates': True,
        
        # Conservative learning parameters
        'learning_rate': 1e-4,      # Lower learning rate
        'n_steps': 512,             # Smaller batch collection
        'batch_size': 32,           # Smaller batch size
        'n_epochs': 3,              # Fewer epochs per update
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,          # Adjusted clip range for stability
        'ent_coef': 0.001,          # Small entropy coefficient
        'vf_coef': 0.5,
        'max_grad_norm': 0.3,       # Strong gradient clipping
        
        # Network architecture
        'policy_kwargs': {
            'net_arch': {
                'pi': [128, 64],        # Smaller networks
                'vf': [128, 64]
            },
            'activation_fn': torch.nn.Tanh,  # Tanh for bounded outputs
            'optimizer_class': torch.optim.AdamW,  # More stable optimizer
            'optimizer_kwargs': {
                'eps': 1e-7,
                'weight_decay': 1e-4    # L2 regularization
            }
        },
        
        # Training stability
        'normalize_advantage': True,
        'use_sde': False,
        'target_kl': 0.03,          # Stop if KL divergence too high
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if custom_config:
        default_config.update(custom_config)
    
    return default_config

def main():
    """Example usage of TradingModel"""
    from environment import TradingEnvironment
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='15T')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    prices = 30000 + np.cumsum(np.random.randn(1000) * 10)
    
    sample_data = pd.DataFrame({
        'Open': prices + np.random.randn(1000) * 5,
        'High': prices + np.abs(np.random.randn(1000) * 10),
        'Low': prices - np.abs(np.random.randn(1000) * 10),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, 1000),
        'RSI': np.random.uniform(20, 80, 1000),
        'MACD': np.random.randn(1000),
    }, index=dates)
    
    # Create environment
    env = TradingEnvironment(sample_data, lookback_window=10)
    
    # Create model
    model = TradingModel(env, model_name="test_model")
    
    print("Model info:", model.get_model_info())
    print("Model created successfully!")

if __name__ == "__main__":
    main()
