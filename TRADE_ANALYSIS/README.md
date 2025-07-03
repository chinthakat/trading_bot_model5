# Trade Analysis

Turns the JSONL trade traces written by `src/utils/trade_tracer.py` during a
training run into analysis CSVs.

> **What is actually in this directory.** Only the three scripts below are
> tracked in git. Every CSV named on this page - `trade_analysis_detailed.csv`,
> `trade_analysis_simplified.csv`, `performance_metrics.csv`,
> `episode_summary.csv` - is a generated output, is gitignored, and is not in
> the repository. Run `trade_trace_analyzer.py` against your own traces to
> produce them. The column layout of each file is described in
> [../docs/trade-analysis-outputs.md](../docs/trade-analysis-outputs.md).

## Generated Files

### 📊 Main Analysis Files

#### `trade_analysis_detailed.csv`
**Complete trade data with all fields**
- Trade ID and timestamps (entry/close)
- Buy/Sell actions and side (LONG/SHORT)
- Entry and close prices
- PnL at close
- Reward at entry and close
- Net worth at entry and close
- Market data (OHLCV) at entry and exit
- Trade duration in steps, minutes, and hours
- All additional metrics

#### `trade_analysis_simplified.csv`
**Key fields only for easy analysis**
Contains the core requested fields:
- `trade_id` - Unique trade identifier
- `entry_datetime` / `close_datetime` - Trade timing
- `side` - LONG or SHORT position
- `entry_action` - BUY or SELL
- `entry_price` / `close_price` - Entry and exit prices
- `net_pnl` - Profit/Loss at close
- `entry_reward` / `close_reward` - Rewards at entry and close
- `entry_net_worth` / `close_net_worth` - Net worth at entry and close

### 📈 Summary Files

#### `performance_metrics.csv`
Key performance indicators:
- Total trades, win rate
- Total and average PnL
- Best and worst trades
- Trade distribution (LONG vs SHORT)
- Average trade duration

#### `episode_summary.csv`
Summary statistics grouped by episode

## Key Statistics

These numbers describe one past training run, kept here as a record of what the
output looks like. They are not a benchmark and were not reproduced. The run
lost money.

- **Total Trades**: 5,609
- **Win Rate**: 37.8%
- **Total PnL**: -$977.18
- **Time Range**: 2024-01-01 to 2024-04-17
- **Average Duration**: 1.1 hours (66 minutes)
- **Trade Distribution**: 45% LONG, 55% SHORT

## Scripts

#### `trade_trace_analyzer.py`
Main analysis script that extracts all data from trade traces
- Parses JSONL trace files
- Extracts comprehensive trade information
- Creates detailed analysis files

Usage:
```bash
python trade_trace_analyzer.py --detailed --output-dir .
```

#### `create_summary.py`
Creates readable summary and simplified CSV file
```bash
python create_summary.py
```

#### `check_rewards.py`
A one-off script that prints reward statistics. It reads a hard-coded
`trade_analysis_detailed_minimal.csv` from the current directory and takes no
arguments.

## Data Fields Extracted

✅ **All Requested Fields Included:**

1. **Trade ID and Time**: `trade_id`, `entry_datetime`, `close_datetime`
2. **Buy or Sell**: `entry_action` (BUY/SELL), `side` (LONG/SHORT)
3. **Entry/Close Price**: `entry_price`, `close_price`
4. **PNL at Close**: `net_pnl`, `win_loss`
5. **Reward at Entry**: `entry_reward`
6. **Reward at Close**: `close_reward`
7. **Entry/Close Net Worth**: `entry_net_worth`, `close_net_worth`

## Additional Insights

From the same past run as the statistics above.

- **Best Trade**: $9.00 profit (TRADE_05431)
- **Worst Trade**: -$15.18 loss (TRADE_05430)
- **Most Common**: SHORT trades (55% of all trades)
- **Duration**: Most trades last about 1 hour

## Usage Examples

### Load simplified data in Python:
```python
import pandas as pd
df = pd.read_csv('trade_analysis_simplified.csv')

# Filter profitable trades
winners = df[df['net_pnl'] > 0]

# Analyze by side
long_trades = df[df['side'] == 'LONG']
short_trades = df[df['side'] == 'SHORT']
```

### Load detailed data for advanced analysis:
```python
detailed_df = pd.read_csv('trade_analysis_detailed.csv')

# Analysis with market data, durations, etc.
avg_duration = detailed_df['trade_duration_hours'].mean()
```
