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
    ),
    'CUSTOM_1': MarketConfig(
        trend_strength=0.3,  # Moderate trend strength for realistic movements
        volatility=0.45,     # Higher volatility for learning opportunities
        noise_factor=0.25,   # Moderate noise
        volume_base=2500.0,  # Good volume for realistic trading
        volume_variation=0.8  # High volume variation for diverse conditions
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
        elif market_type == 'CUSTOM_1':
            data = self._generate_custom_1_market(timestamps, config, interval_minutes)
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
    
    def _generate_custom_1_market(self, timestamps, config: MarketConfig, interval_minutes: int):
        """
        Generate CUSTOM_1: 12-month comprehensive training data with clear patterns
        Price range: 40,000 - 100,000+ with distinct market phases for optimal learning
        """
        num_candles = len(timestamps)
        print(f"Generating CUSTOM_1: 12-month comprehensive training dataset ({num_candles} candles)")
        
        # Define market phases for 12 months (each phase ~1-2 months)
        phases = [
            # Phase 1: Initial consolidation (40k-45k) - Month 1
            {'type': 'SWING', 'duration_ratio': 0.08, 'price_target': 42500, 'volatility_mult': 0.8},
            
            # Phase 2: First major uptrend (45k-65k) - Month 2-3
            {'type': 'UPTREND', 'duration_ratio': 0.17, 'price_target': 65000, 'volatility_mult': 1.2},
            
            # Phase 3: Correction swing (60k-70k) - Month 3-4
            {'type': 'SWING', 'duration_ratio': 0.08, 'price_target': 65000, 'volatility_mult': 1.0},
            
            # Phase 4: Major downtrend (70k-45k) - Month 4-5
            {'type': 'DOWNTREND', 'duration_ratio': 0.17, 'price_target': 45000, 'volatility_mult': 1.4},
            
            # Phase 5: Bottom consolidation (40k-50k) - Month 6
            {'type': 'SWING', 'duration_ratio': 0.08, 'price_target': 45000, 'volatility_mult': 0.9},
            
            # Phase 6: Recovery uptrend (45k-80k) - Month 7-8
            {'type': 'UPTREND', 'duration_ratio': 0.17, 'price_target': 80000, 'volatility_mult': 1.3},
            
            # Phase 7: High volatility swing (75k-85k) - Month 9
            {'type': 'SWING', 'duration_ratio': 0.08, 'price_target': 80000, 'volatility_mult': 1.5},
            
            # Phase 8: Final rally to 100k+ - Month 10-11
            {'type': 'UPTREND', 'duration_ratio': 0.17, 'price_target': 105000, 'volatility_mult': 1.6}
        ]
        
        # Initialize arrays
        opens = np.zeros(num_candles)
        highs = np.zeros(num_candles)
        lows = np.zeros(num_candles)
        closes = np.zeros(num_candles)
        volumes = np.zeros(num_candles)
        
        # Start at 40k
        current_price = 40000.0
        opens[0] = current_price
        
        # Generate data phase by phase
        current_index = 0
        
        for phase_num, phase in enumerate(phases):
            phase_length = int(num_candles * phase['duration_ratio'])
            if phase_num == len(phases) - 1:  # Last phase gets remaining candles
                phase_length = num_candles - current_index
            
            if phase_length <= 0:
                continue
                
            print(f"  Phase {phase_num + 1}: {phase['type']} for {phase_length} candles (target: ${phase['price_target']:,.0f})")
            
            # Calculate price progression for this phase
            start_price = current_price
            target_price = phase['price_target']
            price_diff = target_price - start_price
            
            # Generate phase-specific market configuration
            phase_config = MarketConfig(
                trend_strength=self._get_phase_trend_strength(phase['type'], price_diff, phase_length),
                volatility=config.volatility * phase['volatility_mult'],
                noise_factor=config.noise_factor,
                volume_base=config.volume_base * (1 + abs(price_diff) / start_price),  # Higher volume for bigger moves
                volume_variation=config.volume_variation
            )
            
            # Generate this phase
            for i in range(phase_length):
                candle_index = current_index + i
                if candle_index >= num_candles:
                    break
                
                # Progress through phase (0.0 to 1.0)
                phase_progress = i / max(phase_length - 1, 1)
                
                # Calculate target price for this candle
                target_for_candle = start_price + (price_diff * phase_progress)
                
                # Generate price movement
                if phase['type'] == 'UPTREND':
                    trend_component = self._calculate_uptrend_movement(phase_progress, phase_config, interval_minutes)
                elif phase['type'] == 'DOWNTREND':
                    trend_component = self._calculate_downtrend_movement(phase_progress, phase_config, interval_minutes)
                else:  # SWING
                    trend_component = self._calculate_swing_movement(phase_progress, phase_config, interval_minutes)
                
                # Add volatility and noise
                volatility_component = np.random.normal(0, phase_config.volatility) * current_price * 0.01
                noise_component = np.random.normal(0, phase_config.noise_factor) * current_price * 0.005
                
                # Total price change with trend guidance
                trend_guidance = (target_for_candle - current_price) * 0.1  # Gentle guidance toward target
                total_change = trend_component + volatility_component + noise_component + trend_guidance
                
                # Generate OHLC
                if candle_index > 0:
                    opens[candle_index] = closes[candle_index - 1]
                
                closes[candle_index] = max(opens[candle_index] + total_change, 1000.0)  # Min price 1k
                
                # Generate realistic high/low
                candle_range = abs(closes[candle_index] - opens[candle_index])
                wick_factor = np.random.uniform(0.3, 1.2)  # Variable wick sizes
                
                high_wick = np.random.uniform(0, candle_range * wick_factor * 0.5)
                low_wick = np.random.uniform(0, candle_range * wick_factor * 0.5)
                
                highs[candle_index] = max(opens[candle_index], closes[candle_index]) + high_wick
                lows[candle_index] = min(opens[candle_index], closes[candle_index]) - low_wick
                lows[candle_index] = max(lows[candle_index], 1000.0)  # Min price 1k
                
                # Generate volume based on price movement and phase
                price_change_pct = abs(total_change) / current_price
                volume_multiplier = 1 + (price_change_pct * 10)  # Higher volume for big moves
                phase_volume_mult = 1.5 if phase['type'] in ['UPTREND', 'DOWNTREND'] else 1.0
                
                base_volume = phase_config.volume_base * phase_volume_mult
                volume_variation = np.random.uniform(1 - phase_config.volume_variation, 1 + phase_config.volume_variation)
                volumes[candle_index] = base_volume * volume_variation * volume_multiplier
                
                current_price = closes[candle_index]
            
            current_index += phase_length
            print(f"    Phase {phase_num + 1} completed. Price: ${current_price:,.2f}")
        
        print(f"CUSTOM_1 generation complete. Final price: ${current_price:,.2f}")
        print(f"Price range achieved: ${min(lows):,.2f} - ${max(highs):,.2f}")
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'timestamp': timestamps
        }
    
    def _get_phase_trend_strength(self, phase_type: str, price_diff: float, phase_length: int) -> float:
        """Calculate appropriate trend strength for phase"""
        if phase_type == 'UPTREND':
            return abs(price_diff) / phase_length * 0.001  # Positive trend
        elif phase_type == 'DOWNTREND':
            return -abs(price_diff) / phase_length * 0.001  # Negative trend
        else:  # SWING
            return 0.0
    
    def _calculate_uptrend_movement(self, progress: float, config: MarketConfig, interval_minutes: int) -> float:
        """Calculate uptrend movement with realistic patterns"""
        # Stronger moves at beginning and end, consolidation in middle
        strength_curve = 1.0 - abs(0.5 - progress) * 0.8
        base_move = config.trend_strength * strength_curve * (interval_minutes / 15.0)
        # Add some pullbacks (20% chance of temporary down move)
        if np.random.random() < 0.2:
            base_move *= -0.3
        return np.random.normal(base_move, abs(base_move) * 0.4)
    
    def _calculate_downtrend_movement(self, progress: float, config: MarketConfig, interval_minutes: int) -> float:
        """Calculate downtrend movement with realistic patterns"""
        # Accelerating downtrends
        strength_curve = 0.5 + progress * 0.8
        base_move = config.trend_strength * strength_curve * (interval_minutes / 15.0)
        # Add some relief rallies (15% chance of temporary up move)
        if np.random.random() < 0.15:
            base_move *= -0.4
        return np.random.normal(base_move, abs(base_move) * 0.5)
    
    def _calculate_swing_movement(self, progress: float, config: MarketConfig, interval_minutes: int) -> float:
        """Calculate swing/ranging movement"""
        # Oscillating movement
        oscillation = np.sin(progress * np.pi * 4) * config.volatility * 0.5
        return np.random.normal(oscillation, config.volatility * 0.3) * (interval_minutes / 15.0)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic Bitcoin price data')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', choices=['1m', '5m', '15m'], required=True, help='Time interval')
    parser.add_argument('--market-type', choices=['UPTREND', 'DOWNTREND', 'SWING', 'MIXED', 'CUSTOM_1'], 
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
