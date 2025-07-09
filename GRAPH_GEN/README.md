# Bitcoin Price Graph Generator

Tools to visualize Bitcoin price data from CSV files, and to draw trade entries,
exits and P&L on top of a price series. Run the commands below from inside
`GRAPH_GEN/`.

> **Which file to run.** `btc_graph_generator.py` is an empty file in this
> repository - it was committed as a placeholder and never filled in, so every
> command that names it, and the helpers `launcher.py`, `example_usage.py`,
> `debug_import.py`, `test_interactive.py` and `summary.py` that import
> `BTCGraphGenerator` from it, will not work. Use `btc_graph_generator_fixed.py`
> (the full generator) or `btc_graph_simple.py` (price and volume only)
> instead. Both take the same `--file` / `--interactive` flags described below.

## Features

- **Interactive File Selection**: Browse and select data files from the `/data` folder
- **Multiple Chart Types**:
  - Single file visualization with price and volume
  - Multi-file comparison charts (normalized)
  - Advanced analysis with OHLC, moving averages, and statistical distribution
- **Smart File Detection**: Automatically detects file types and extracts metadata
- **Export Options**: Save graphs as high-quality PNG files
- **Statistical Overlays**: Price statistics and performance metrics

## Installation

```bash
cd GRAPH_GEN
pip install -r requirements.txt
```

That covers this tool only. The repository-wide `requirements.txt` at the root
is a superset.

## Usage

### Interactive Mode

```bash
python btc_graph_generator_fixed.py --interactive
```

`launch_graph_generator.bat` and `launcher.py` both target the empty
`btc_graph_generator.py` and do not work as committed.

### Command Line Mode

```bash
# Single file graph
python btc_graph_generator_fixed.py --file "your_data.csv"

# Advanced analysis
python btc_graph_generator_fixed.py --file "your_data.csv" --type advanced

# Save graph
python btc_graph_generator_fixed.py --file "your_data.csv" --save "output.png"

# Price and volume only, no technical panels
python btc_graph_simple.py --file "your_data.csv"
```

`--data-folder` defaults to `../data`, so `--file` is resolved relative to the
repository's gitignored `data/` directory unless you pass an explicit path.

### Trade visualizers

These read the JSONL trade traces written during training, or the CSVs produced
by `TRADE_ANALYSIS/`, and draw entry/exit markers with P&L labels.

```bash
# entry/exit markers over a price series
python trade_analysis_visualizer_clean.py --trace-file ../logs/trade_traces/trade_traces.jsonl \
                                          --save trade_analysis.png

# same idea, driven from extracted CSVs instead of traces
# (run extract_trade_data.py below first - it writes extracted_trades.csv)
python csv_trade_visualizer.py --trade-csv extracted_trades.csv \
                               --external-market-csv ../data/your_data.csv

# chart that follows a run as it happens
python live_trade_visualizer_enhanced.py --file ../logs/trade_traces/trade_traces.jsonl \
                                         --interval 5

# pull trade and market rows out of a trace file
python extract_trade_data.py --trace-file ../logs/trade_traces/trade_traces.jsonl
```

Note the path and flag details:

- `--trace-file` must be passed explicitly. These scripts default it to the
  relative `logs/trade_traces/trade_traces.jsonl`, which from inside
  `GRAPH_GEN/` points at `GRAPH_GEN/logs/...`; training writes to `logs/` at the
  repository root.
- `csv_trade_visualizer.py` reads the CSVs from `extract_trade_data.py`, not the
  ones from `TRADE_ANALYSIS/`. It requires an `exit_datetime` column;
  `trade_analysis_detailed.csv` names that column `close_datetime`, so passing it
  fails with `KeyError: 'exit_datetime'`. Run `extract_trade_data.py` first and
  pass the `extracted_trades.csv` it writes.
- `csv_trade_visualizer.py` takes market data through either of two flags, and
  they expect different files. `--market-csv` is for `extracted_market_data.csv`
  produced by `extract_trade_data.py`, which carries a `datetime` column.
  A raw OHLCV file - anything out of `DATA_GEN/`, or a downloaded Binance
  export - has `timestamp` instead and must be passed to
  `--external-market-csv`, which converts the Unix seconds itself. The wrong
  pairing fails with `KeyError: 'datetime'`.
- `live_trade_visualizer_enhanced.py --file` wants the **JSONL trace file**,
  not a CSV. Every line is parsed with `json.loads`, so a CSV produces a JSON
  decode failure on the first row.

There are four near-identical trade visualizers: `trade_analysis_visualizer.py`
and its `_clean`, `_fixed` and `_pure` variants. Use `_clean`. The unsuffixed
`trade_analysis_visualizer.py` is corrupted - from line 245 onward the file is a
single line of literal `\n` escape sequences - and raises
`SyntaxError` when imported or run.

## Interactive Menu Options

1. **Individual Graphs**: Generate separate charts for each selected file
2. **Comparison Graph**: Overlay multiple datasets on a single normalized chart
3. **Advanced Analysis**: Comprehensive analysis with multiple chart panels

## Chart Types

### Single File Chart
- Time vs Closing Price line chart
- Volume bar chart below price
- Statistics overlay box
- Date formatting on x-axis

### Comparison Chart
- Multiple datasets normalized to base 100
- Different colors for each dataset
- Legend with file identification
- Synchronized time axis

### Advanced Analysis (4-Panel)
- **OHLC Chart**: Candlestick-style price action
- **Moving Averages**: Price with 20 and 50-period moving averages
- **Volume Analysis**: Trading volume over time
- **Price Distribution**: Histogram of price changes

## File Support

Supports CSV files with the following columns:
- `timestamp` - Unix timestamp
- `open` - Opening price
- `high` - Highest price
- `low` - Lowest price
- `close` - Closing price
- `volume` - Trading volume

## Output

- **Screen Display**: All charts displayed in matplotlib windows
- **File Export**: Optional PNG export with 300 DPI quality
- **Graphs Folder**: Saved files organized in `graphs/` subdirectory

No generated PNGs are tracked in git - `graphs/`, `example_graphs/` and loose
`trade_analysis_*.png` files are all gitignored. Regenerate them with the
commands above.

## Examples

```bash
# Launch interactive mode
python btc_graph_generator_fixed.py --interactive

# Quick single file
python btc_graph_generator_fixed.py --file "BTC_SYNTHETIC_UPTREND_15m_2024-06-01_to_2024-06-03.csv"

# Advanced analysis with save
python btc_graph_generator_fixed.py --file "my_data.csv" --type advanced --save "analysis.png"
```

## Dependencies

- `matplotlib` - Chart generation and display
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `seaborn` - Enhanced plot styling

## Tips

- Use comparison mode to analyze different market conditions side by side
- Advanced analysis is best for detailed technical analysis
- Large datasets are automatically sampled for better performance
- All dates are automatically formatted based on data range
