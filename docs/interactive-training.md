# Interactive Training

> How `src/train_memory_efficient.py` behaves in interactive mode. Run the
> commands below from the repository root. Note that the training script does
> not currently run from a fresh clone - see the Status section of the
> [main README](../README.md) for why.

## Overview
The `train_memory_efficient.py` script has been enhanced with comprehensive interactive capabilities and detailed logging features.

## 🆕 New Features

### 1. **Interactive Data File Selection**
- Automatically scans multiple data directories for CSV files
- Shows file sizes and paths for easy identification
- Validates selected files to ensure they're readable
- Displays detected columns and basic file information

### 2. **Interactive Training Parameter Configuration**
- User-friendly prompts for:
  - Number of training episodes
  - Steps per episode
- Provides intelligent defaults (5 episodes, 100,000 steps)
- Shows estimated training time
- Calculates total training steps
- Final confirmation before starting

### 3. **Enhanced Logging for Every Step**
The trading execution log now includes comprehensive information for **every step**:
- **Action** (HOLD, BUY, SELL, CLOSE_ALL, etc.)
- **Closing price** (current market price)
- **Current net balance** (account balance)
- **Number of long open positions**
- **Number of short positions** 
- **Unrealized profit/loss**
- **Net worth** (balance + unrealized P&L)
- **Reward breakdown** (when significant)

### 4. **Multiple Usage Modes**
1. **🎮 Interactive Mode**: `--interactive`
   - Select data file from list
   - Configure parameters interactively
   - Perfect for experimentation

2. **🔧 Default Mode**: `--default` 
   - Predefined settings with real market data
   - Automatic archiving enabled
   - Good for production runs

3. **🎯 Training Mode**: `--training`
   - Uses synthetic balanced training data
   - Optimized for initial RL learning

4. **⚡ Auto Mode**: `--auto`
   - Fully automated with config file settings
   - No user prompts

### 5. **Comprehensive Help System**
- `--help-usage` flag shows detailed usage examples
- Clear explanations of each mode
- Visual formatting with emojis for easy reading

## 📊 Enhanced Logging Output

### Before:
```
2025-06-25 11:23:18,794 - environment - INFO - Action: HOLD
2025-06-25 11:23:18,795 - environment - INFO - Reward: position_management_points: -5.0, drawdown_penalty: -0.3, total_points: -3246.4, open_position_exp_penalty: -3241.0 | Total: -3246.4
```

### After:
```
2025-06-25 14:07:19,297 - environment - INFO - Step    1 | Action: HOLD         | Price: $50157.28 | Balance:  $  10000.00 | Long:  0 | Short:  0 | Unrealized P&L: $    0.00 | Net Worth: $  10000.00
2025-06-25 14:07:19,301 - environment - INFO - Step    2 | Action: BUY          | Price: $50195.77 | Balance:  $   9799.17 | Long:  1 | Short:  0 | Unrealized P&L: $    0.31 | Net Worth: $   9799.48
2025-06-25 14:07:19,302 - environment - INFO - Reward: 0.2
```

## 🚀 Usage Examples

### Interactive Mode (Recommended for new users):
```bash
python src/train_memory_efficient.py --interactive
```

### Quick Start with Help:
```bash
python src/train_memory_efficient.py --help-usage
```

### Production Training:
```bash
python src/train_memory_efficient.py --default
```

### Demo Mode:
```bash
python src/demo_interactive.py
```

## 🔧 Technical Improvements

1. **Data File Discovery**: Automatically searches common data directories
2. **File Validation**: Checks file readability before selection
3. **Path Handling**: Robust relative/absolute path management
4. **Error Handling**: Graceful handling of user cancellations and invalid inputs
5. **Time Estimation**: Provides rough training time estimates
6. **Progress Feedback**: Clear status messages throughout the process

## 🎯 Default Settings

- **Episodes**: 5 (configurable)
- **Steps per episode**: 100,000 (configurable) 
- **Reward system**: SimpleRewardSystem (default)
- **Action space**: Discretized for better learning
- **Archiving**: Enabled (unless `--no-archive`)

The script now provides a much more user-friendly experience while maintaining all the powerful features of the original memory-efficient training system.
