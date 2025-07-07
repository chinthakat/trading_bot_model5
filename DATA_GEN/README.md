# Bitcoin Synthetic Data Generator

A Python script to generate synthetic Bitcoin price data in Binance CSV format
under different market conditions. Run every command below from the repository
root; output paths are relative to wherever you run from.
The `CUSTOM_1` market type is documented separately in
[../docs/custom-1-dataset.md](../docs/custom-1-dataset.md).

## Features

- **Multiple Market Types**: 
  - `UPTREND`: Bullish market with consistent upward price movement
  - `DOWNTREND`: Bearish market with consistent downward price movement  
  - `SWING`: Sideways/ranging market with oscillating prices
  - `MIXED`: Combination of all patterns with changing market conditions
  - `CUSTOM_1`: Scripted twelve-month, eight-phase curriculum dataset

- **Multiple Timeframes**: 1m, 5m, 15m intervals
- **Realistic OHLCV Data**: Generates Open, High, Low, Close, Volume, and Timestamp data
- **Configurable Parameters**: Customizable volatility, trend strength, and volume patterns

## Installation

```bash
pip install -r DATA_GEN/requirements.txt
```

That covers this tool only (pandas and numpy). The repository-wide
`requirements.txt` at the root is a superset.

## Usage

### Command Line Interface

```bash
python DATA_GEN/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-02-01 --interval 15m --market-type UPTREND --output data/uptrend_data.csv
```

#### Parameters:
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format  
- `--interval`: Time interval (1m, 5m, 15m)
- `--market-type`: Market condition (UPTREND, DOWNTREND, SWING, MIXED, CUSTOM_1)
- `--initial-price`: Starting BTC price (default: 50000.0)
- `--output`: Output CSV file path (optional)

### Programmatic Usage

```python
from DATA_GEN.btc_data_generator import BTCDataGenerator

generator = BTCDataGenerator(initial_price=45000.0)

# Generate uptrend data
data = generator.generate_timeframe_data(
    start_date='2024-01-01',
    end_date='2024-01-31',
    interval='15m',
    market_type='UPTREND',
    output_path='data/my_data.csv'
)
```

## Examples

### Generate 1 week of bullish 15m data:
```bash
python DATA_GEN/btc_data_generator.py --start-date 2024-06-01 --end-date 2024-06-08 --interval 15m --market-type UPTREND
```

### Generate 1 month of mixed market 5m data:
```bash
python DATA_GEN/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-02-01 --interval 5m --market-type MIXED
```

### Run example script:

This one is the exception to the "run from the repository root" rule above.
`example_generator.py` hard-codes its four output paths as `../data/...`, so it
has to be run from inside `DATA_GEN/` for them to land in the repository's
`data/` directory. Run from the root, it writes to the parent of the repository.

```bash
cd DATA_GEN
python example_generator.py
```

## Output Format

The generated CSV files match Binance format:
```csv
,open,high,low,close,volume,timestamp
0,45000.0,45123.5,44987.2,45098.1,2150.45,1704067200
1,45098.1,45234.7,45067.8,45201.3,1876.23,1704068100
...
```

## Market Type Characteristics

- **UPTREND**: Strong positive trend with moderate volatility
- **DOWNTREND**: Strong negative trend with higher volatility  
- **SWING**: No clear trend, oscillating around a range
- **MIXED**: Alternating segments of different market conditions

## Data Analysis

Use the included analyzer to examine generated datasets:

```bash
# Analyze a single file
python DATA_GEN/data_analyzer.py data/your_data.csv

# Compare multiple files
python DATA_GEN/data_analyzer.py data/file1.csv data/file2.csv

# Quick comparison summary
python DATA_GEN/data_analyzer.py data/*.csv --summary
```

## Batch Generation

`generate_datasets.bat` and `generate_datasets.ps1` generate a spread of
datasets in one go. Both were written to be run from the repository root and
both invoke `python src/btc_data_generator.py`, which is an empty file in this
repository - so they do not work as committed. Point them at
`DATA_GEN/btc_data_generator.py` first, or copy the individual commands out of
them.

## Data Quality

The generator creates realistic market data with:
- Proper OHLC relationships (High ≥ max(Open,Close), Low ≤ min(Open,Close))
- Volume correlation with price movements
- Realistic price gaps and continuity between candles
- Configurable noise and volatility levels

## Files Structure

```
DATA_GEN/
├── btc_data_generator.py    # Main generator script
├── data_analyzer.py         # Data analysis tool
├── example_generator.py     # Usage examples
├── analyze_custom1.py       # One-off check on a generated CUSTOM_1 dataset
├── generate_datasets.bat    # Windows batch script
├── generate_datasets.ps1    # PowerShell script
├── requirements.txt         # Dependencies for this tool only
└── README.md                # This file
```

Generated CSVs are written to `../data/`, which is gitignored and not part of
the repository.
