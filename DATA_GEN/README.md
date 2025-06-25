# Bitcoin Synthetic Data Generator

A Python script to generate synthetic Bitcoin price data similar to Binance format with different market conditions.

## Features

- **Multiple Market Types**: 
  - `UPTREND`: Bullish market with consistent upward price movement
  - `DOWNTREND`: Bearish market with consistent downward price movement  
  - `SWING`: Sideways/ranging market with oscillating prices
  - `MIXED`: Combination of all patterns with changing market conditions

- **Multiple Timeframes**: 1m, 5m, 15m intervals
- **Realistic OHLCV Data**: Generates Open, High, Low, Close, Volume, and Timestamp data
- **Configurable Parameters**: Customizable volatility, trend strength, and volume patterns

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python src/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-02-01 --interval 15m --market-type UPTREND --output data/uptrend_data.csv
```

#### Parameters:
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format  
- `--interval`: Time interval (1m, 5m, 15m)
- `--market-type`: Market condition (UPTREND, DOWNTREND, SWING, MIXED)
- `--initial-price`: Starting BTC price (default: 50000.0)
- `--output`: Output CSV file path (optional)

### Programmatic Usage

```python
from src.btc_data_generator import BTCDataGenerator

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
python src/btc_data_generator.py --start-date 2024-06-01 --end-date 2024-06-08 --interval 15m --market-type UPTREND
```

### Generate 1 month of mixed market 5m data:
```bash
python src/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-02-01 --interval 5m --market-type MIXED
```

### Run example script:
```bash
python src/example_generator.py
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
python src/data_analyzer.py data/your_data.csv

# Compare multiple files
python src/data_analyzer.py data/file1.csv data/file2.csv

# Quick comparison summary
python src/data_analyzer.py data/*.csv --summary
```

## Batch Generation

For Windows users, use the included batch scripts:

```bash
# Generate multiple datasets at once
.\generate_datasets.bat
# or
.\generate_datasets.ps1
```

## Data Quality

The generator creates realistic market data with:
- Proper OHLC relationships (High ≥ max(Open,Close), Low ≤ min(Open,Close))
- Volume correlation with price movements
- Realistic price gaps and continuity between candles
- Configurable noise and volatility levels

## Files Structure

```
c:\Projects\Model5\
├── src/
│   ├── btc_data_generator.py    # Main generator script
│   ├── data_analyzer.py         # Data analysis tool
│   └── example_generator.py     # Usage examples
├── data/                        # Generated CSV files
├── generate_datasets.bat        # Windows batch script
├── generate_datasets.ps1        # PowerShell script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
