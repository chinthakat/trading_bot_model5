"""
Bitcoin Data Generator Suite - Usage Summary

This script provides a summary of all available tools and usage examples.
"""

import os
import sys

def print_header():
    print("🚀 Bitcoin Synthetic Data Generator Suite")
    print("=" * 50)
    print()

def print_tools():
    print("📋 Available Tools:")
    print("1. btc_data_generator.py  - Main data generation script")
    print("2. data_analyzer.py       - Analyze generated datasets")
    print("3. example_generator.py   - Run example generations")
    print("4. generate_datasets.bat  - Batch generate multiple datasets (Windows)")
    print("5. generate_datasets.ps1  - PowerShell batch generation")
    print()

def print_quick_examples():
    print("⚡ Quick Examples:")
    print()
    
    print("Generate 1 week uptrend data:")
    print("  python src/btc_data_generator.py --start-date 2024-06-01 --end-date 2024-06-08 --interval 15m --market-type UPTREND")
    print()
    
    print("Generate 1 month mixed market data:")
    print("  python src/btc_data_generator.py --start-date 2024-01-01 --end-date 2024-02-01 --interval 5m --market-type MIXED")
    print()
    
    print("Analyze generated data:")
    print("  python src/data_analyzer.py data/your_file.csv")
    print()
    
    print("Compare multiple datasets:")
    print("  python src/data_analyzer.py data/*.csv --summary")
    print()

def print_market_types():
    print("📈 Market Types Available:")
    print("• UPTREND   - Bullish market with consistent upward movement")
    print("• DOWNTREND - Bearish market with consistent downward movement")
    print("• SWING     - Sideways/ranging market with oscillations")
    print("• MIXED     - Combination of all patterns with changing conditions")
    print()

def print_intervals():
    print("⏰ Supported Intervals:")
    print("• 1m  - 1 minute candles")
    print("• 5m  - 5 minute candles") 
    print("• 15m - 15 minute candles")
    print()

def check_files():
    print("📁 Current Data Files:")
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if files:
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file}")
        else:
            print("  No CSV files found in data directory")
    else:
        print("  Data directory not found")
    print()

def print_batch_commands():
    print("🔄 Batch Generation:")
    print("Windows Batch File:")
    print("  .\\generate_datasets.bat")
    print()
    print("PowerShell:")
    print("  .\\generate_datasets.ps1")
    print()
    print("Python Example Script:")
    print("  python src/example_generator.py")
    print()

def main():
    print_header()
    print_tools()
    print_market_types()
    print_intervals()
    print_quick_examples()
    print_batch_commands()
    check_files()
    
    print("💡 Tips:")
    print("• Use larger time periods for better trend visibility")
    print("• MIXED market type creates realistic changing conditions")
    print("• Analyze your data with data_analyzer.py to verify characteristics")
    print("• All generated data follows Binance CSV format")
    print()
    
    print("🆘 Need help? Check README.md for detailed documentation")

if __name__ == "__main__":
    main()
