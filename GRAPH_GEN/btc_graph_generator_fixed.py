#!/usr/bin/env python3
"""
Bitcoin Price Graph Generator

Interactive tool to visualize Bitcoin price data from CSV files.
Allows users to select files from the data folder and generate time vs closing price graphs.
"""

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

# Try to set a nice style, but don't fail if it doesn't work
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

class BTCGraphGenerator:
    def __init__(self, data_folder: str = "../data"):
        self.data_folder = data_folder
        self.figure_size = (15, 8)
        
    def get_available_files(self) -> List[str]:
        """Get list of CSV files in the data folder"""
        pattern = os.path.join(self.data_folder, "*.csv")
        files = glob.glob(pattern)
        return [os.path.basename(f) for f in files if os.path.isfile(f)]
    
    def display_file_menu(self) -> List[str]:
        """Display interactive menu for file selection"""
        files = self.get_available_files()
        
        if not files:
            print(f"No CSV files found in {self.data_folder}")
            return []
        
        print("\n📁 Available Data Files:")
        print("=" * 50)
        for i, file in enumerate(files, 1):
            # Extract info from filename
            file_info = self._extract_file_info(file)
            print(f"{i:2d}. {file}")
            if file_info:
                print(f"     {file_info}")
        
        return files
    
    def _extract_file_info(self, filename: str) -> str:
        """Extract information from filename"""
        try:
            if "SYNTHETIC" in filename:
                parts = filename.replace("BTC_SYNTHETIC_", "").replace(".csv", "").split("_")
                if len(parts) >= 4:
                    market_type = parts[0]
                    interval = parts[1]
                    start_date = parts[2]
                    return f"Type: {market_type}, Interval: {interval}, Start: {start_date}"
            elif "BINANCE" in filename:
                return "Real Binance Data"
            return ""
        except:
            return ""
    
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
        files = self.display_file_menu()
        
        if not files:
            return []
        
        if allow_multiple:
            print(f"\nEnter file numbers separated by commas (1-{len(files)})")
            print("Example: 1,3,5")
            
            while True:
                try:
                    selection = input("Select files: ").strip()
                    if not selection:
                        return []
                    
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_files = []
                    
                    for idx in indices:
                        if 0 <= idx < len(files):
                            selected_files.append(files[idx])
                        else:
                            print(f"Invalid file number: {idx + 1}")
                            break
                    else:
                        return selected_files
                        
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    return []
        else:
            while True:
                try:
                    choice = input(f"Select file number (1-{len(files)}): ").strip()
                    if not choice:
                        return []
                    
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(files):
                        return [files[choice_num - 1]]
                    else:
                        print(f"Invalid selection. Please enter a number between 1-{len(files)}.")
                        
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    return []

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load and prepare data from CSV file"""
        filepath = os.path.join(self.data_folder, filename)
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            
        df = df.sort_values('timestamp')
        return df

    def create_single_file_graph(self, filename: str, save_path: str = None):
        """Create a single file price and volume chart"""
        try:
            df = self.load_data(filename)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Price chart
            ax1.plot(df['timestamp'], df['close'], linewidth=1, color='blue', alpha=0.8)
            ax1.set_title(f'Bitcoin Price - {filename}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis for better time display
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // 10)))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # Volume chart
            if 'volume' in df.columns:
                ax2.bar(df['timestamp'], df['volume'], width=0.8, alpha=0.7, color='green')
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.set_xlabel('Time', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                # Format x-axis
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df) // 10)))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add statistics
            stats_text = f"Min: ${df['close'].min():.2f}  Max: ${df['close'].max():.2f}  Avg: ${df['close'].mean():.2f}"
            fig.suptitle(stats_text, y=0.02, fontsize=10)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Graph saved to: {save_path}")
            else:
                plt.savefig(f"graph_{filename.replace('.csv', '.png')}", dpi=300, bbox_inches='tight')
                print(f"Graph saved to: graph_{filename.replace('.csv', '.png')}")
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating graph: {e}")

    def create_comparison_graph(self, filenames: List[str], save_path: str = None):
        """Create a comparison chart with multiple files"""
        try:
            plt.figure(figsize=self.figure_size)
            
            for i, filename in enumerate(filenames):
                df = self.load_data(filename)
                
                # Normalize prices to start at 100
                df['normalized_price'] = (df['close'] / df['close'].iloc[0]) * 100
                
                label = filename.replace('.csv', '').replace('BTC_SYNTHETIC_', '')
                plt.plot(df['timestamp'], df['normalized_price'], 
                        linewidth=2, alpha=0.8, label=label)
            
            plt.title('Bitcoin Price Comparison (Normalized to 100)', fontsize=14, fontweight='bold')
            plt.ylabel('Normalized Price', fontsize=12)
            plt.xlabel('Time', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Comparison graph saved to: {save_path}")
            else:
                plt.savefig("comparison_graph.png", dpi=300, bbox_inches='tight')
                print("Comparison graph saved to: comparison_graph.png")
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating comparison graph: {e}")

    def create_advanced_analysis(self, filename: str, save_path: str = None):
        """Create advanced 4-panel technical analysis"""
        try:
            df = self.load_data(filename)
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
            
            # Price with moving averages
            ax1.plot(df['timestamp'], df['close'], linewidth=1, label='Close Price', alpha=0.8)
            ax1.plot(df['timestamp'], df['sma_20'], linewidth=2, label='SMA 20', alpha=0.7)
            ax1.plot(df['timestamp'], df['sma_50'], linewidth=2, label='SMA 50', alpha=0.7)
            ax1.set_title('Price with Moving Averages')
            ax1.set_ylabel('Price (USDT)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume
            if 'volume' in df.columns:
                ax2.bar(df['timestamp'], df['volume'], alpha=0.7, color='green')
                ax2.set_title('Volume')
                ax2.set_ylabel('Volume')
                ax2.grid(True, alpha=0.3)
            
            # Price changes
            ax3.plot(df['timestamp'], df['price_change'], linewidth=1, alpha=0.8, color='red')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Price Changes (%)')
            ax3.set_ylabel('Change (%)')
            ax3.grid(True, alpha=0.3)
            
            # Volatility
            ax4.plot(df['timestamp'], df['volatility'], linewidth=2, alpha=0.8, color='orange')
            ax4.set_title('Volatility (20-period)')
            ax4.set_ylabel('Volatility')
            ax4.set_xlabel('Time')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Advanced analysis saved to: {save_path}")
            else:
                plt.savefig(f"advanced_{filename.replace('.csv', '.png')}", dpi=300, bbox_inches='tight')
                print(f"Advanced analysis saved to: advanced_{filename.replace('.csv', '.png')}")
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating advanced analysis: {e}")

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
                
                # Determine if multiple files are allowed
                allow_multiple = graph_type == 2  # Only comparison allows multiple files
                
                # Get file selection
                selected_files = self.get_file_selection(allow_multiple)
                
                if not selected_files:
                    print("No files selected.")
                    continue
                
                print(f"\nSelected {len(selected_files)} file(s)")
                
                # Ask if user wants to save
                save_graphs = input("Save graph to file? (y/n): ").lower().startswith('y')
                
                # Generate the appropriate graph
                filename = selected_files[0]  # Use first file for single-file graphs
                save_path = None
                
                if save_graphs:
                    os.makedirs("graphs", exist_ok=True)
                    base_name = filename.replace('.csv', '')
                    
                if graph_type == 1:  # Single File Chart
                    if save_graphs:
                        save_path = f"graphs/{base_name}_single_chart.png"
                    self.create_single_file_graph(filename, save_path)
                    
                elif graph_type == 2:  # Comparison Chart
                    if save_graphs:
                        save_path = "graphs/comparison_chart.png"
                    self.create_comparison_graph(selected_files, save_path)
                    
                elif graph_type == 3:  # Advanced Analysis
                    if save_graphs:
                        save_path = f"graphs/{base_name}_advanced.png"
                    self.create_advanced_analysis(filename, save_path)
                    
                else:
                    # For other types, default to single file chart
                    if save_graphs:
                        save_path = f"graphs/{base_name}_chart.png"
                    self.create_single_file_graph(filename, save_path)
                
                # Ask if user wants to continue
                if input("\nGenerate more graphs? (y/n): ").lower().startswith('n'):
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate Bitcoin price graphs')
    parser.add_argument('--data-folder', default='../data', help='Path to data folder')
    parser.add_argument('--file', help='Specific file to graph')
    parser.add_argument('--type', choices=['single', 'advanced', 'comparison'], 
                       default='single', help='Graph type for specific file')
    parser.add_argument('--save', help='Save path for the graph')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    generator = BTCGraphGenerator(args.data_folder)
    
    if args.interactive or not args.file:
        generator.run_interactive_mode()
    else:
        # Command line mode
        if args.type == 'advanced':
            generator.create_advanced_analysis(args.file, args.save)
        else:
            generator.create_single_file_graph(args.file, args.save)

if __name__ == "__main__":
    main()
