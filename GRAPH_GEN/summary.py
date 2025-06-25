"""
GRAPH_GEN Summary and Quick Start Guide

This script provides an overview of the graph generation capabilities.
"""

import os

def print_header():
    print("📊 Bitcoin Graph Generator Suite")
    print("=" * 50)
    print()

def print_features():
    print("✨ Features:")
    print("• Interactive numbered menu system (1-10 options)")
    print("• Smart file selection from /data folder with metadata")
    print("• 10 Different chart types:")
    print("  1. Single File Chart (Price + Volume)")
    print("  2. Comparison Chart (Multiple Files Normalized)")
    print("  3. Advanced Analysis (4-Panel Technical)")
    print("  4. OHLC Candlestick Chart")
    print("  5. Price with Moving Averages")
    print("  6. Volume Analysis Only")
    print("  7. Price Distribution Histogram")
    print("  8. Price Change Analysis")
    print("  9. Multi-timeframe Analysis")
    print("  10. Exit")
    print("• Improved time axis with multiple time points")
    print("• Export to high-quality PNG files")
    print("• Smart file detection and metadata extraction")
    print()

def print_quick_start():
    print("🚀 Quick Start:")
    print()
    
    print("1. Interactive Mode (Recommended):")
    print("   launch_graph_generator.bat")
    print("   # or")
    print("   python btc_graph_generator.py --interactive")
    print()
    
    print("2. Command Line Mode:")
    print("   python btc_graph_generator.py --file \"your_data.csv\"")
    print("   python btc_graph_generator.py --file \"data.csv\" --type advanced")
    print()
    
    print("3. Examples:")
    print("   python example_usage.py")
    print()

def print_file_structure():
    print("📁 GRAPH_GEN Folder Structure:")
    files = [
        "btc_graph_generator.py  - Main interactive graph generator",
        "launcher.py             - Simple Python launcher",
        "example_usage.py        - Usage examples and demos",
        "launch_graph_generator.bat - Windows batch launcher",
        "requirements.txt        - Python dependencies",
        "README.md              - Detailed documentation"
    ]
    
    for file in files:
        print(f"  {file}")
    print()

def check_requirements():
    print("🔧 Requirements Check:")
    
    required_packages = ["matplotlib", "pandas", "numpy", "seaborn"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All requirements satisfied!")
    print()

def check_data_availability():
    print("📊 Data Availability:")
    
    data_folder = "../data"
    if not os.path.exists(data_folder):
        print(f"  ❌ Data folder '{data_folder}' not found")
        print("     Please generate data using DATA_GEN tools first")
        return
    
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"  ❌ No CSV files found in {data_folder}")
        print("     Please generate data using DATA_GEN tools first")
    else:
        print(f"  ✅ Found {len(csv_files)} data file(s):")
        for i, file in enumerate(csv_files[:5], 1):  # Show first 5
            print(f"     {i}. {file}")
        if len(csv_files) > 5:
            print(f"     ... and {len(csv_files) - 5} more")
    print()

def print_usage_examples():
    print("💡 Interactive Usage Flow:")
    print()
    
    print("Step 1: Launch Interactive Mode")
    print("  launch_graph_generator.bat")
    print("  # or python btc_graph_generator.py --interactive")
    print()
    
    print("Step 2: Select Graph Type (1-10)")
    print("  📈 Graph Type Options:")
    print("  1. Single File Chart (Price + Volume)")
    print("  2. Comparison Chart (Multiple Files)")
    print("  3. Advanced Analysis (4-Panel)")
    print("  4. OHLC Candlestick Chart")
    print("  5. Price with Moving Averages")
    print("  ... and more")
    print()
    
    print("Step 3: Select Data File(s)")
    print("  📁 Available Data Files:")
    print("  1. BTC_SYNTHETIC_UPTREND_15m_2024-06-01.csv")
    print("  2. BINANCE_REAL_DATA.csv")
    print("  ... (numbered list with metadata)")
    print()
    
    print("Step 4: Choose Export Option")
    print("  Save graph to file? (y/n)")
    print()
    
    print("🚀 Command Line Examples:")
    print("  python btc_graph_generator.py --file \"data.csv\" --type candlestick")
    print("  python btc_graph_generator.py --file \"data.csv\" --type ma --save \"chart.png\"")
    print()

def main():
    print_header()
    print_features()
    print_quick_start()
    print_file_structure()
    check_requirements()
    check_data_availability()
    print_usage_examples()
    
    print("🎯 Ready to Generate Graphs!")
    print("Run 'launch_graph_generator.bat' to get started!")

if __name__ == "__main__":
    main()
