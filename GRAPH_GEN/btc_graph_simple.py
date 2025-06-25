#!/usr/bin/env python3
"""
Bitcoin Price Graph Generator - Simplified Version

Interactive tool to visualize Bitcoin price data from CSV files.
"""

print("Module starting...")

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

print("Imports completed, defining class...")

class BTCGraphGenerator:
    def __init__(self, data_folder: str = "../data"):
        self.data_folder = data_folder
        self.figure_size = (15, 8)
        print(f"BTCGraphGenerator initialized with data_folder: {data_folder}")
        
    def get_available_files(self) -> List[str]:
        """Get list of CSV files in the data folder"""
        pattern = os.path.join(self.data_folder, "*.csv")
        files = glob.glob(pattern)
        return [os.path.basename(f) for f in files if os.path.isfile(f)]
    
    def get_graph_type_selection(self) -> int:
        """Get user's graph type selection"""
        print("\n📈 Graph Type Options:")
        print("=" * 40)
        print("1. Single File Chart (Price + Volume)")
        print("2. Comparison Chart (Multiple Files Normalized)")
        print("3. Advanced Analysis (4-Panel Technical)")
        print("4. OHLC Candlestick Chart")
        print("5. Price with Moving Averages")
        print("6. Volume Analysis Only")
        print("7. Price Distribution Histogram")
        print("8. Price Change Analysis")
        print("9. Multi-timeframe Analysis")
        print("10. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect graph type (1-10): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= 10:
                    return choice_num
                else:
                    print("Invalid selection. Please enter a number between 1-10.")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                return 10

    def get_file_selection(self, allow_multiple: bool = False) -> List[str]:
        """Get user's file selection"""
        files = self.get_available_files()
        
        if not files:
            print(f"No CSV files found in {self.data_folder}")
            return []
        
        print("\n📁 Available Data Files:")
        print("=" * 50)
        for i, file in enumerate(files, 1):
            print(f"{i:2d}. {file}")
        
        if allow_multiple:
            print("\nEnter file numbers separated by commas (e.g., 1,3,5)")
            selection = input("Select files: ").strip()
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                return [files[i] for i in indices if 0 <= i < len(files)]
            except:
                print("Invalid selection")
                return []
        else:
            while True:
                try:
                    choice = int(input("Select file number: ").strip())
                    if 1 <= choice <= len(files):
                        return [files[choice - 1]]
                    else:
                        print(f"Please enter a number between 1 and {len(files)}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    return []

    def create_simple_chart(self, filename: str, save_path: str = None):
        """Create a simple price chart"""
        try:
            filepath = os.path.join(self.data_folder, filename)
            df = pd.read_csv(filepath)
            
            # Assume standard format with timestamp and close price
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                plt.figure(figsize=self.figure_size)
                plt.plot(df['timestamp'], df['close'], linewidth=1)
                plt.title(f'BTC Price - {filename}')
                plt.xlabel('Time')
                plt.ylabel('Price (USDT)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Chart saved to: {save_path}")
                else:
                    plt.savefig(f"chart_{filename.replace('.csv', '.png')}", dpi=300, bbox_inches='tight')
                    print(f"Chart saved to: chart_{filename.replace('.csv', '.png')}")
                
                plt.close()
            else:
                print("No timestamp column found in the data")
                
        except Exception as e:
            print(f"Error creating chart: {e}")

    def run_interactive_mode(self):
        """Run the interactive graph generation mode"""
        print("🚀 Bitcoin Price Graph Generator")
        print("=" * 40)
        
        while True:
            try:
                # Get graph type selection
                graph_type = self.get_graph_type_selection()
                
                if graph_type == 10:  # Exit
                    print("Goodbye!")
                    break
                
                # Get file selection
                selected_files = self.get_file_selection(allow_multiple=False)
                
                if not selected_files:
                    print("No files selected.")
                    continue
                
                filename = selected_files[0]
                print(f"\nProcessing: {filename}")
                
                # Ask if user wants to save
                save_graphs = input("Save graph to file? (y/n): ").lower().startswith('y')
                save_path = None
                
                if save_graphs:
                    os.makedirs("graphs", exist_ok=True)
                    base_name = filename.replace('.csv', '')
                    save_path = f"graphs/{base_name}_chart.png"
                
                # For now, just create a simple chart regardless of type
                self.create_simple_chart(filename, save_path)
                
                # Ask if user wants to continue
                if input("\nGenerate more graphs? (y/n): ").lower().startswith('n'):
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    print("Main function starting...")
    parser = argparse.ArgumentParser(description='Generate Bitcoin price graphs')
    parser.add_argument('--data-folder', default='../data', help='Path to data folder')
    parser.add_argument('--file', help='Specific file to graph')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    print(f"Args: interactive={args.interactive}, file={args.file}")
    
    generator = BTCGraphGenerator(args.data_folder)
    
    if args.interactive or not args.file:
        print("Starting interactive mode...")
        generator.run_interactive_mode()
    else:
        print(f"Processing file: {args.file}")
        generator.create_simple_chart(args.file)

print("Class defined, setting up main execution...")

if __name__ == "__main__":
    print("Running as main module...")
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

print("Module execution completed.")
