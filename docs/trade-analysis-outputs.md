# Trade Trace Analysis - Output Files Summary

> Column reference for the CSVs that `TRADE_ANALYSIS/trade_trace_analyzer.py`
> writes. Run the commands below from inside `TRADE_ANALYSIS/`. None of the
> output files are tracked in git.
>
> `--trace-file` has no usable default: it falls back to the absolute path
> `c:\Projects\Model5\logs\trade_traces\trade_traces.jsonl` from the original
> author's machine, so always pass it explicitly.

## Overview
The trade trace analyzer now generates **two CSV output files** for comprehensive trade analysis:

1. **Detailed File**: Contains all available trade data (51 columns)
2. **Minimal File**: Contains only essential trade information (14 columns)

## Generated Files

### Standard Mode
```bash
python trade_trace_analyzer.py --trace-file ../logs/trade_traces/trade_traces.jsonl \
                               --output trade_analysis_standalone.csv
```
**Output:**
- `trade_analysis_standalone.csv` (detailed - 51 columns)
- `trade_analysis_standalone_minimal.csv` (minimal - 14 columns)

### Detailed Report Mode
```bash
python trade_trace_analyzer.py --trace-file ../logs/trade_traces/trade_traces.jsonl \
                               --detailed --output-dir .
```
**Output:**
- `trade_analysis_detailed.csv` (detailed - 51 columns) 
- `trade_analysis_detailed_minimal.csv` (minimal - 14 columns)
- `episode_summary.csv` (summary by episode)
- `performance_metrics.csv` (overall performance stats)

## File Structures

### Detailed File (51 columns)
Contains comprehensive trade data including:
- Trade identification and timing
- Entry/exit details (prices, actions, timestamps)
- Financial metrics (PnL, commissions, returns)
- **Rewards** (extracted from `reward_efficiency`)
- **Net worth** tracking (entry and close values)
- Market data (OHLCV at entry/exit)
- Risk metrics and performance ratios

### Minimal File (14 essential columns)
```
trade_id, entry_datetime, close_datetime, side, entry_action, 
entry_price, close_price, net_pnl, close_reward, entry_net_worth, 
close_net_worth, trade_duration_hours, status, win_loss
```

## Key Improvements

### ✅ Reward Extraction Fixed
- **Previous issue**: Rewards were showing as 0.0
- **Solution**: Now extracts rewards from `reward_efficiency` field in trade analysis
- **Result**: All 5,609 trades have correct non-zero reward values
- **Verification**: Rewards match PnL values 100% (reward_efficiency = net_pnl)

### ✅ Net Worth Tracking
- **Entry net worth**: From `portfolio_overview.net_worth` at trade entry
- **Close net worth**: From `portfolio_overview.net_worth` at trade close
- **Range**: $9,999.71 to $10,000.22 (starting from $10,000)

### ✅ Data Quality
- **Total trades**: 5,609 successfully extracted
- **Date range**: 2024-01-01 to 2024-04-17
- **Win rate**: 37.8% (2,121 winning trades)
- **Trade duration**: Average 1.1 hours

## Trade Performance Summary
- **Total PnL**: -$977.18
- **Average PnL**: -$0.17 per trade
- **Best trade**: +$9.00
- **Worst trade**: -$15.18
- **LONG trades**: 2,532 (45.1%)
- **SHORT trades**: 3,077 (54.9%)

## Usage Examples

### Quick Analysis
```python
import pandas as pd

# Load minimal file for quick analysis
df = pd.read_csv('trade_analysis_detailed_minimal.csv')
print(f"Total trades: {len(df)}")
print(f"Win rate: {(df['win_loss'] == 'WIN').sum() / len(df) * 100:.1f}%")
print(f"Total PnL: ${df['net_pnl'].sum():.2f}")
```

### Detailed Analysis
```python
# Load detailed file for comprehensive analysis
df_detailed = pd.read_csv('trade_analysis_detailed.csv')
print(f"Available columns: {len(df_detailed.columns)}")
print(df_detailed.columns.tolist())
```

## Files Purpose
- **Minimal CSV**: Quick analysis, reporting, dashboard creation
- **Detailed CSV**: Deep analysis, research, model evaluation
- **Episode/Performance CSV**: Training episode analysis and overall metrics

All files are sorted by entry datetime and contain the same trade records with different levels of detail.
