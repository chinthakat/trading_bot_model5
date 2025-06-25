# Bitcoin Price Graph Generator

Interactive tool to visualize Bitcoin price data from CSV files with multiple chart types and analysis options.

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

## Usage

### Interactive Mode (Recommended)

#### Windows:
```bash
launch_graph_generator.bat
```

#### Command Line:
```bash
python btc_graph_generator.py --interactive
```

#### Simple Launcher:
```bash
python launcher.py
```

### Command Line Mode

```bash
# Single file graph
python btc_graph_generator.py --file "your_data.csv"

# Advanced analysis
python btc_graph_generator.py --file "your_data.csv" --type advanced

# Save graph
python btc_graph_generator.py --file "your_data.csv" --save "output.png"
```

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

## Examples

```bash
# Launch interactive mode
python btc_graph_generator.py --interactive

# Quick single file
python btc_graph_generator.py --file "BTC_SYNTHETIC_UPTREND_15m_2024-06-01_to_2024-06-03.csv"

# Advanced analysis with save
python btc_graph_generator.py --file "my_data.csv" --type advanced --save "analysis.png"
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
