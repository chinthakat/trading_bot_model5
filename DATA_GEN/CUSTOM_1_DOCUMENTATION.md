# CUSTOM_1 Dataset Documentation

## Overview
The CUSTOM_1 dataset generates a comprehensive 12-month Bitcoin price dataset specifically designed for reinforcement learning model training. It provides diverse market conditions with clear patterns for optimal learning.

## Dataset Characteristics

### Price Range & Progression
- **Starting Price**: $40,000
- **Final Price**: $104,753+ (achieves 100K+ target)
- **Price Range**: $39,356 - $108,316
- **Total Appreciation**: 161.9% over 12 months

### Market Phases (8 Distinct Phases)

#### Phase 1: Initial Consolidation (Month 1)
- **Type**: SWING
- **Duration**: ~8% of dataset (2,803 candles)
- **Target**: $42,500
- **Purpose**: Model learns sideways trading patterns

#### Phase 2: First Major Uptrend (Months 2-3)
- **Type**: UPTREND  
- **Duration**: ~17% of dataset (5,956 candles)
- **Target**: $65,000
- **Purpose**: Model learns bull market entry/exit strategies

#### Phase 3: Correction Consolidation (Months 3-4)
- **Type**: SWING
- **Duration**: ~8% of dataset (2,803 candles)
- **Target**: $65,000
- **Purpose**: Model learns to handle post-rally consolidation

#### Phase 4: Major Correction (Months 4-5)
- **Type**: DOWNTREND
- **Duration**: ~17% of dataset (5,956 candles)
- **Target**: $45,000
- **Purpose**: Model learns bear market and risk management

#### Phase 5: Bottom Formation (Month 6)
- **Type**: SWING
- **Duration**: ~8% of dataset (2,803 candles)
- **Target**: $45,000
- **Purpose**: Model learns bottom fishing and trend reversal

#### Phase 6: Recovery Rally (Months 7-8)
- **Type**: UPTREND
- **Duration**: ~17% of dataset (5,956 candles)
- **Target**: $80,000
- **Purpose**: Model learns recovery trading and momentum

#### Phase 7: High Volatility Range (Month 9)
- **Type**: SWING (High Volatility)
- **Duration**: ~8% of dataset (2,803 candles)
- **Target**: $80,000
- **Purpose**: Model learns high-volatility trading

#### Phase 8: Final Bull Run (Months 10-12)
- **Type**: UPTREND
- **Duration**: ~17% of dataset (5,960 candles)
- **Target**: $105,000
- **Purpose**: Model learns parabolic moves and profit-taking

## Technical Specifications

### Data Quality
- **Total Candles**: 35,040 (15-minute intervals)
- **Duration**: Exactly 365 days
- **OHLCV Structure**: Fully valid and realistic
- **No Missing Data**: Complete dataset
- **Positive Prices**: All prices > $1,000

### Volatility Profile
- **Average Volatility**: 0.61% per 15-minute candle
- **Max Single Gain**: 2.60%
- **Max Single Loss**: -3.13%
- **Realistic Range**: Suitable for trading algorithm training

### Volume Characteristics
- **Dynamic Volume**: Responds to price movements
- **Higher Volume**: During trend phases
- **Range**: 508 - 14,090 per candle
- **Total Volume**: 170M+ units

## Learning Opportunities

### Market Condition Distribution
- **Uptrend Phases**: 3 major bull markets
- **Downtrend Phases**: 1 major bear market
- **Swing Phases**: Multiple consolidation periods
- **Volatility Regimes**: Low, medium, and high volatility periods

### Trading Strategy Training
1. **Trend Following**: Clear uptrends and downtrends
2. **Mean Reversion**: Multiple swing phases
3. **Breakout Trading**: Phase transitions
4. **Risk Management**: Drawdown periods
5. **Profit Taking**: Multiple rally peaks

### Model Learning Benefits
- **Pattern Recognition**: Clear trend changes
- **Risk Assessment**: Multiple market conditions
- **Position Sizing**: Variable volatility environments
- **Entry/Exit Timing**: Diverse signal environments
- **Portfolio Management**: Long-term performance tracking

## Usage Examples

### Generate Dataset
```bash
python btc_data_generator.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --interval 15m \
    --market-type CUSTOM_1 \
    --initial-price 40000 \
    --output data/BTC_CUSTOM_1_15m_2024_training.csv
```

### Load in Trading Environment
```python
import pandas as pd

# Load CUSTOM_1 dataset
df = pd.read_csv('data/BTC_CUSTOM_1_15m_2024_training.csv')

# Verify data quality
print(f"Dataset contains {len(df)} candles")
print(f"Price range: ${df['low'].min():.0f} - ${df['high'].max():.0f}")
print(f"Complete year: {len(df) * 15 / (60 * 24):.1f} days")
```

## Advantages for RL Training

1. **Comprehensive Coverage**: All major market conditions represented
2. **Realistic Progression**: Natural price evolution over 12 months
3. **Clear Patterns**: Distinct phases for pattern learning
4. **Balanced Dataset**: Equal representation of different market types
5. **Scalable Generation**: Can generate multiple years with different parameters
6. **High Quality**: Proper OHLCV structure with realistic volume

## Recommended Use Cases

- **Primary Training**: Main dataset for RL model development
- **Backtesting**: Comprehensive strategy validation
- **Benchmarking**: Standard dataset for model comparison
- **Research**: Academic and professional trading algorithm research
- **Multi-timeframe**: Can be aggregated to higher timeframes (1H, 4H, 1D)

The CUSTOM_1 dataset provides an ideal foundation for training robust trading algorithms that can handle diverse market conditions effectively.
