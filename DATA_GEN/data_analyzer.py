"""
Bitcoin Data Analyzer

Analyzes and provides statistics for generated Bitcoin price data.
"""

import pandas as pd
import numpy as np
import argparse
import os
from typing import Dict, Any

class BTCDataAnalyzer:
    def __init__(self):
        pass
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single CSV file and return statistics"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Basic statistics
        stats = {
            'file_name': os.path.basename(file_path),
            'total_candles': len(df),
            'time_period': {
                'start': pd.to_datetime(df['timestamp'].min(), unit='s'),
                'end': pd.to_datetime(df['timestamp'].max(), unit='s'),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()) / (24 * 3600)
            },
            'price_stats': {
                'initial_price': df['open'].iloc[0],
                'final_price': df['close'].iloc[-1],
                'highest_price': df['high'].max(),
                'lowest_price': df['low'].min(),
                'price_change': df['close'].iloc[-1] - df['open'].iloc[0],
                'price_change_percent': ((df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]) * 100,
                'volatility': df['close'].pct_change().std() * 100
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'average_volume': df['volume'].mean(),
                'max_volume': df['volume'].max(),
                'min_volume': df['volume'].min()
            },
            'candle_analysis': {
                'green_candles': len(df[df['close'] > df['open']]),
                'red_candles': len(df[df['close'] < df['open']]),
                'doji_candles': len(df[df['close'] == df['open']]),
                'average_body_size': abs(df['close'] - df['open']).mean(),
                'average_wick_size': ((df['high'] - df[['open', 'close']].max(axis=1)) + 
                                    (df[['open', 'close']].min(axis=1) - df['low'])).mean()
            }
        }
        
        # Determine market trend
        stats['market_trend'] = self._analyze_trend(df)
        
        return stats
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the overall market trend"""
        prices = df['close'].values
        
        # Simple trend analysis using linear regression
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate trend strength
        correlation = np.corrcoef(x, prices)[0, 1]
        
        if slope > 0 and correlation > 0.3:
            trend_type = "UPTREND"
        elif slope < 0 and correlation < -0.3:
            trend_type = "DOWNTREND"
        else:
            trend_type = "SIDEWAYS"
        
        return {
            'type': trend_type,
            'slope': slope,
            'correlation': correlation,
            'strength': abs(correlation)
        }
    
    def print_analysis(self, stats: Dict[str, Any]):
        """Print formatted analysis results"""
        print(f"\n{'='*60}")
        print(f"ANALYSIS: {stats['file_name']}")
        print(f"{'='*60}")
        
        # Time period
        print(f"\n📅 TIME PERIOD:")
        print(f"   Duration: {stats['time_period']['duration_days']:.1f} days")
        print(f"   From: {stats['time_period']['start']}")
        print(f"   To: {stats['time_period']['end']}")
        print(f"   Total Candles: {stats['total_candles']:,}")
        
        # Price statistics
        print(f"\n💰 PRICE STATISTICS:")
        print(f"   Initial Price: ${stats['price_stats']['initial_price']:,.2f}")
        print(f"   Final Price: ${stats['price_stats']['final_price']:,.2f}")
        print(f"   Highest Price: ${stats['price_stats']['highest_price']:,.2f}")
        print(f"   Lowest Price: ${stats['price_stats']['lowest_price']:,.2f}")
        print(f"   Price Change: ${stats['price_stats']['price_change']:,.2f} ({stats['price_stats']['price_change_percent']:+.2f}%)")
        print(f"   Volatility: {stats['price_stats']['volatility']:.2f}%")
        
        # Volume statistics
        print(f"\n📊 VOLUME STATISTICS:")
        print(f"   Total Volume: {stats['volume_stats']['total_volume']:,.2f}")
        print(f"   Average Volume: {stats['volume_stats']['average_volume']:,.2f}")
        print(f"   Max Volume: {stats['volume_stats']['max_volume']:,.2f}")
        print(f"   Min Volume: {stats['volume_stats']['min_volume']:,.2f}")
        
        # Candle analysis
        print(f"\n🕯️ CANDLE ANALYSIS:")
        total_candles = stats['total_candles']
        green_pct = (stats['candle_analysis']['green_candles'] / total_candles) * 100
        red_pct = (stats['candle_analysis']['red_candles'] / total_candles) * 100
        print(f"   Green Candles: {stats['candle_analysis']['green_candles']} ({green_pct:.1f}%)")
        print(f"   Red Candles: {stats['candle_analysis']['red_candles']} ({red_pct:.1f}%)")
        print(f"   Doji Candles: {stats['candle_analysis']['doji_candles']}")
        print(f"   Avg Body Size: ${stats['candle_analysis']['average_body_size']:.2f}")
        print(f"   Avg Wick Size: ${stats['candle_analysis']['average_wick_size']:.2f}")
        
        # Market trend
        print(f"\n📈 MARKET TREND ANALYSIS:")
        print(f"   Trend Type: {stats['market_trend']['type']}")
        print(f"   Trend Strength: {stats['market_trend']['strength']:.3f}")
        print(f"   Correlation: {stats['market_trend']['correlation']:.3f}")
        print(f"   Slope: {stats['market_trend']['slope']:.4f}")
    
    def analyze_multiple_files(self, file_paths: list):
        """Analyze multiple files and provide comparison"""
        all_stats = []
        
        for file_path in file_paths:
            try:
                stats = self.analyze_file(file_path)
                all_stats.append(stats)
                self.print_analysis(stats)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        if len(all_stats) > 1:
            self._print_comparison(all_stats)
    
    def _print_comparison(self, all_stats: list):
        """Print comparison between multiple datasets"""
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        for stats in all_stats:
            name = stats['file_name']
            change_pct = stats['price_stats']['price_change_percent']
            trend = stats['market_trend']['type']
            volatility = stats['price_stats']['volatility']
            print(f"   {name[:40]:<40} | {change_pct:+7.2f}% | {trend:>9} | Vol: {volatility:5.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze Bitcoin synthetic data files')
    parser.add_argument('files', nargs='+', help='CSV files to analyze')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    analyzer = BTCDataAnalyzer()
    
    if args.summary and len(args.files) > 1:
        # Quick summary for multiple files
        all_stats = []
        for file_path in args.files:
            try:
                stats = analyzer.analyze_file(file_path)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error: {e}")
        
        if all_stats:
            analyzer._print_comparison(all_stats)
    else:
        # Detailed analysis
        analyzer.analyze_multiple_files(args.files)

if __name__ == "__main__":
    main()
