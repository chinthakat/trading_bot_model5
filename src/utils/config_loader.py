#!/usr/bin/env python3
"""
Configuration loading utilities for the trading system
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any


def load_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load training configuration from JSON file
    
    Args:
        config_path: Path to config file, defaults to config/training_config.json
        
    Returns:
        Dictionary containing training configuration
    """
    if config_path is None:
        # Default to config/training_config.json relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "training_config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_path}: {e}")


def get_logging_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract logging configuration from training config
    
    Args:
        config: Training configuration dict, will load from file if None
        
    Returns:
        Dictionary containing logging configuration with defaults
    """
    if config is None:
        config = load_training_config()
    
    # Get logging config with sensible defaults
    logging_config = config.get('logging_config', {})
    
    # Apply defaults for missing values
    defaults = {
        'enable_trade_logging': True,
        'enable_trade_tracing': True,
        'trade_log_frequency': 10,
        'console_log_level': 'INFO',
        'file_log_level': 'DEBUG',
        'enable_tensorboard': True,
        'enable_detailed_rewards': False,
        'tensorboard_log_frequency': 100
    }
    
    # Merge with defaults
    for key, default_value in defaults.items():
        if key not in logging_config:
            logging_config[key] = default_value
    
    return logging_config


def setup_console_logging(log_level: str = 'INFO') -> None:
    """
    Setup console logging with specified level
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'DISABLED')
    """
    if log_level.upper() == 'DISABLED':
        logging.getLogger().setLevel(logging.CRITICAL + 1)  # Disable all logging
        return
    
    # Convert string to logging level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Get root logger and remove existing handlers to ensure basicConfig works
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_file_logging(log_file: str, log_level: str = 'DEBUG') -> None:
    """
    Setup file logging with specified level
    
    Args:
        log_file: Path to log file
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'DISABLED')
    """
    if log_level.upper() == 'DISABLED':
        return
    
    # Convert string to logging level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(log_level.upper(), logging.DEBUG)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(file_handler)


def print_logging_config(logging_config: Dict[str, Any]) -> None:
    """
    Print current logging configuration in a readable format
    
    Args:
        logging_config: Logging configuration dictionary
    """
    print("\n" + "="*50)
    print("LOGGING CONFIGURATION")
    print("="*50)
    
    print(f"Trade Logging: {'Enabled' if logging_config.get('enable_trade_logging', True) else 'Disabled'}")
    print(f"Trade Tracing: {'Enabled' if logging_config.get('enable_trade_tracing', True) else 'Disabled'}")
    print(f"Trade Log Frequency: {logging_config.get('trade_log_frequency', 10)} trades")
    print(f"Console Log Level: {logging_config.get('console_log_level', 'INFO')}")
    print(f"File Log Level: {logging_config.get('file_log_level', 'DEBUG')}")
    print(f"TensorBoard: {'Enabled' if logging_config.get('enable_tensorboard', True) else 'Disabled'}")
    print(f"Detailed Rewards: {'Enabled' if logging_config.get('enable_detailed_rewards', False) else 'Disabled'}")
    print(f"TensorBoard Log Frequency: {logging_config.get('tensorboard_log_frequency', 100)} steps")
    print("="*50 + "\n")
