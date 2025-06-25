"""
Quick Launcher for BTC Graph Generator

Simple script to launch the graph generator in interactive mode.
"""

import os
import sys

def main():
    print("🚀 Bitcoin Graph Generator Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("btc_graph_generator.py"):
        print("Error: btc_graph_generator.py not found in current directory")
        print("Please run this script from the GRAPH_GEN folder")
        return
    
    # Check if data folder exists
    data_folder = "../data"
    if not os.path.exists(data_folder):
        print(f"Warning: Data folder '{data_folder}' not found")
        print("Make sure you have generated some data first using the DATA_GEN tools")
        return
    
    # Import and run the generator
    try:
        from btc_graph_generator import BTCGraphGenerator
        generator = BTCGraphGenerator(data_folder)
        generator.run_interactive_mode()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
