"""
Example usage of the BTC Data Generator

This script demonstrates how to generate synthetic Bitcoin data for different scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from btc_data_generator import BTCDataGenerator
import pandas as pd

def generate_sample_data():
    """Generate sample data for different market conditions"""
    
    generator = BTCDataGenerator(initial_price=45000.0)
    
    # Example 1: Generate 1 week of 15m uptrend data
    print("Generating 1 week of 15m UPTREND data...")
    uptrend_data = generator.generate_timeframe_data(
        start_date='2024-06-01',
        end_date='2024-06-08',
        interval='15m',
        market_type='UPTREND',
        output_path='../data/sample_uptrend_15m.csv'
    )
    
    # Example 2: Generate 3 days of 5m downtrend data
    print("\nGenerating 3 days of 5m DOWNTREND data...")
    downtrend_data = generator.generate_timeframe_data(
        start_date='2024-06-10',
        end_date='2024-06-13',
        interval='5m',
        market_type='DOWNTREND',
        output_path='../data/sample_downtrend_5m.csv'
    )
    
    # Example 3: Generate 2 days of 1m swing data
    print("\nGenerating 2 days of 1m SWING data...")
    swing_data = generator.generate_timeframe_data(
        start_date='2024-06-15',
        end_date='2024-06-17',
        interval='1m',
        market_type='SWING',
        output_path='../data/sample_swing_1m.csv'
    )
    
    # Example 4: Generate 1 month of mixed market data
    print("\nGenerating 1 month of 15m MIXED data...")
    mixed_data = generator.generate_timeframe_data(
        start_date='2024-07-01',
        end_date='2024-08-01',
        interval='15m',
        market_type='MIXED',
        output_path='../data/sample_mixed_15m.csv'
    )
    
    print("\n=== Generation Summary ===")
    print(f"UPTREND (15m): {len(uptrend_data)} candles, Final price: ${uptrend_data['close'].iloc[-1]:.2f}")
    print(f"DOWNTREND (5m): {len(downtrend_data)} candles, Final price: ${downtrend_data['close'].iloc[-1]:.2f}")
    print(f"SWING (1m): {len(swing_data)} candles, Final price: ${swing_data['close'].iloc[-1]:.2f}")
    print(f"MIXED (15m): {len(mixed_data)} candles, Final price: ${mixed_data['close'].iloc[-1]:.2f}")

if __name__ == "__main__":
    generate_sample_data()
