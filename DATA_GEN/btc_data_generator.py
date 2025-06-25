"""
Bitcoin Price Data Generator

Generates synthetic Bitcoin price data similar to Binance format with different market conditions:
1. UPTREND - Bullish market with upward price movement
2. DOWNTREND - Bearish market with downward price movement  
3. SWING - Sideways/ranging market with oscillating prices
4. MIXED - Combination of all three patterns

Output format matches Binance CSV: index,open,high,low,close,volume,timestamp
"""

import pandas as pd
import numpy as np
import datetime
import argparse
import os
from typing import Literal, Tuple
from dataclasses import dataclass

@dataclass
class MarketConfig:
    """Configuration for different market conditions"""
    trend_strength: float  # How strong the trend is (0.0 to 1.0)
    volatility: float     # Price volatility (0.0 to 1.0)
    noise_factor: float   # Random noise level (0.0 to 1.0)
    volume_base: float    # Base volume level
    volume_variation: float # Volume variation factor

# Market condition configurations
MARKET_CONFIGS = {
    'UPTREND': MarketConfig(
        trend_strength=0.7,
        volatility=0.3,
        noise_factor=0.2,
        volume_base=2000.0,
        volume_variation=0.5
    ),
    'DOWNTREND': MarketConfig(
        trend_strength=-0.7,
        volatility=0.4,
        noise_factor=0.25,
        volume_base=2500.0,
        volume_variation=0.6
    ),
    'SWING': MarketConfig(
        trend_strength=0.0,
        volatility=0.5,
        noise_factor=0.3,
        volume_base=1800.0,
        volume_variation=0.4
    ),
    'MIXED': MarketConfig(
        trend_strength=0.1,
        volatility=0.4,
        noise_factor=0.35,
        volume_base=2200.0,
        volume_variation=0.7
    )
}

class BTCDataGenerator:
    def __init__(self, initial_price: float = 50000.0):
        self.initial_price = initial_price
        np.random.seed(None)  # Use current time as seed for randomness
    
    def generate_timeframe_data(
        self,
        start_date: str,
        end_date: str,
        interval: Literal['1m', '5m', '15m'],
        market_type: Literal['UPTREND', 'DOWNTREND', 'SWING', 'MIXED'],
        output_path: str = None
    ) -> pd.DataFrame:
        """Generate Bitcoin price data for specified timeframe and market condition"""
        
        # Convert interval to minutes
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15}[interval]
        interval_seconds = interval_minutes * 60
        
        # Parse dates
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate timestamp array
        timestamps = []
        current_dt = start_dt
        while current_dt < end_dt:
            timestamps.append(int(current_dt.timestamp()))
            current_dt += datetime.timedelta(minutes=interval_minutes)
        
        num_candles = len(timestamps)
        print(f"Generating {num_candles} candles for {market_type} market ({interval} interval)")
        
        # Get market configuration
        config = MARKET_CONFIGS[market_type]
        
        # Generate price data
        if market_type == 'MIXED':
            data = self._generate_mixed_market(timestamps, config, interval_minutes)
        else:
            data = self._generate_single_market(timestamps, config, interval_minutes)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        df.rename(columns={'index': ''}, inplace=True)
        
        # Save to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Data saved to: {output_path}")
        
        return df
    
    def _generate_single_market(self, timestamps, config: MarketConfig, interval_minutes: int):
        """Generate data for a single market condition"""
        num_candles = len(timestamps)
        
        # Initialize arrays
        opens = np.zeros(num_candles)
        highs = np.zeros(num_candles)
        lows = np.zeros(num_candles)
        closes = np.zeros(num_candles)
        volumes = np.zeros(num_candles)
        
        # Set initial price
        current_price = self.initial_price
        opens[0] = current_price
        
        for i in range(num_candles):
            # Calculate trend component
            trend_factor = config.trend_strength * (interval_minutes / 15.0)  # Scale by interval
            trend_change = np.random.normal(trend_factor, abs(trend_factor) * 0.3)
            
            # Calculate volatility component
            volatility_change = np.random.normal(0, config.volatility) * current_price * 0.01
            
            # Calculate noise component
            noise_change = np.random.normal(0, config.noise_factor) * current_price * 0.005
            
            # Total price change
            total_change = trend_change + volatility_change + noise_change
            
            # Generate OHLC for this candle
            if i > 0:
                opens[i] = closes[i-1]
            
            # Generate close price
            closes[i] = max(opens[i] + total_change, 1.0)  # Ensure positive price
            
            # Generate high and low around open and close
            candle_range = abs(closes[i] - opens[i]) * (1 + np.random.uniform(0.2, 0.8))
            high_extra = np.random.uniform(0, candle_range * 0.3)
            low_extra = np.random.uniform(0, candle_range * 0.3)
            
            highs[i] = max(opens[i], closes[i]) + high_extra
            lows[i] = min(opens[i], closes[i]) - low_extra
            lows[i] = max(lows[i], 1.0)  # Ensure positive price
            
            # Generate volume
            base_volume = config.volume_base
            volume_variation = np.random.uniform(1 - config.volume_variation, 1 + config.volume_variation)
            price_impact = abs(total_change) / current_price * 10  # Higher volume for bigger moves
            volumes[i] = base_volume * volume_variation * (1 + price_impact)
            
            current_price = closes[i]
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'timestamp': timestamps
        }
    
    def _generate_mixed_market(self, timestamps, config: MarketConfig, interval_minutes: int):
        """Generate mixed market data with changing conditions"""
        num_candles = len(timestamps)
        
        # Divide data into segments with different market conditions
        segment_length = max(100, num_candles // 6)  # At least 100 candles per segment
        segments = []
        
        current_pos = 0
        market_types = ['UPTREND', 'DOWNTREND', 'SWING', 'UPTREND', 'SWING', 'DOWNTREND']
        
        while current_pos < num_candles:
            segment_end = min(current_pos + segment_length, num_candles)
            segment_timestamps = timestamps[current_pos:segment_end]
            
            # Choose market type for this segment
            market_type = np.random.choice(market_types)
            segment_config = MARKET_CONFIGS[market_type]
            
            # Generate segment data
            segment_data = self._generate_single_market(
                segment_timestamps, segment_config, interval_minutes
            )
            
            # Adjust prices to continue from previous segment
            if segments:
                price_offset = segments[-1]['close'][-1] - segment_data['open'][0]
                segment_data['open'] += price_offset
                segment_data['high'] += price_offset
                segment_data['low'] += price_offset
                segment_data['close'] += price_offset
            
            segments.append(segment_data)
            current_pos = segment_end
        
        # Combine all segments
        combined_data = {
            'open': np.concatenate([seg['open'] for seg in segments]),
            'high': np.concatenate([seg['high'] for seg in segments]),
            'low': np.concatenate([seg['low'] for seg in segments]),
            'close': np.concatenate([seg['close'] for seg in segments]),
            'volume': np.concatenate([seg['volume'] for seg in segments]),
            'timestamp': np.concatenate([seg['timestamp'] for seg in segments])
        }
        
        return combined_data

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic Bitcoin price data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', choices=['1m', '5m', '15m'], required=True, help='Time interval')
    parser.add_argument('--market-type', choices=['UPTREND', 'DOWNTREND', 'SWING', 'MIXED'], 
                       required=True, help='Market condition type')
    parser.add_argument('--initial-price', type=float, default=50000.0, help='Initial BTC price')
    parser.add_argument('--output', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        args.output = f'data/BTC_SYNTHETIC_{args.market_type}_{args.interval}_{args.start_date}_to_{args.end_date}.csv'
    
    # Create generator and generate data
    generator = BTCDataGenerator(args.initial_price)
    df = generator.generate_timeframe_data(
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        market_type=args.market_type,
        output_path=args.output
    )
    
    print(f"\nGenerated {len(df)} candles")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Final price: ${df['close'].iloc[-1]:.2f}")
    print(f"Total volume: {df['volume'].sum():.2f}")

if __name__ == "__main__":
    main()
