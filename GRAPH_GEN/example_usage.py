"""
Example Usage of BTC Graph Generator

This script demonstrates different ways to use the graph generator programmatically.
"""

import os
import sys
from btc_graph_generator import BTCGraphGenerator

def demo_single_graph():
    """Demo: Generate single file graph"""
    print("📈 Demo: Single File Graph")
    print("-" * 30)
    
    generator = BTCGraphGenerator("../data")
    files = generator.get_available_files()
    
    if not files:
        print("No data files found. Please generate some data first.")
        return
    
    # Use the first available file
    filename = files[0]
    print(f"Generating graph for: {filename}")
    
    # Create graphs folder
    os.makedirs("example_graphs", exist_ok=True)
    
    # Generate the graph
    save_path = f"example_graphs/{filename.replace('.csv', '_demo.png')}"
    generator.create_single_file_graph(filename, save_path)
    print(f"Demo graph saved to: {save_path}")

def demo_comparison_graph():
    """Demo: Generate comparison graph"""
    print("\n📊 Demo: Comparison Graph")
    print("-" * 30)
    
    generator = BTCGraphGenerator("../data")
    files = generator.get_available_files()
    
    if len(files) < 2:
        print("Need at least 2 files for comparison. Generating single file instead.")
        demo_single_graph()
        return
    
    # Use first 3 files or all files if less than 3
    selected_files = files[:min(3, len(files))]
    print(f"Comparing {len(selected_files)} files:")
    for f in selected_files:
        print(f"  - {f}")
    
    # Create graphs folder
    os.makedirs("example_graphs", exist_ok=True)
    
    # Generate comparison graph
    save_path = "example_graphs/comparison_demo.png"
    generator.create_comparison_graph(selected_files, save_path)
    print(f"Comparison graph saved to: {save_path}")

def demo_advanced_analysis():
    """Demo: Generate advanced analysis"""
    print("\n🔬 Demo: Advanced Analysis")
    print("-" * 30)
    
    generator = BTCGraphGenerator("../data")
    files = generator.get_available_files()
    
    if not files:
        print("No data files found. Please generate some data first.")
        return
    
    # Use the first available file
    filename = files[0]
    print(f"Advanced analysis for: {filename}")
    
    # Create graphs folder
    os.makedirs("example_graphs", exist_ok=True)
    
    # Generate advanced analysis
    save_path = f"example_graphs/{filename.replace('.csv', '_advanced_demo.png')}"
    generator.create_advanced_analysis(filename, save_path)
    print(f"Advanced analysis saved to: {save_path}")

def list_available_files():
    """List all available data files"""
    print("📁 Available Data Files:")
    print("-" * 30)
    
    generator = BTCGraphGenerator("../data")
    files = generator.get_available_files()
    
    if not files:
        print("No CSV files found in ../data")
        print("Please generate some data using the DATA_GEN tools first.")
        return files
    
    for i, file in enumerate(files, 1):
        file_info = generator._extract_file_info(file)
        print(f"{i:2d}. {file}")
        if file_info:
            print(f"     {file_info}")
    
    return files

def main():
    print("🚀 Bitcoin Graph Generator - Examples")
    print("=" * 50)
    
    # List available files
    files = list_available_files()
    
    if not files:
        return
    
    print(f"\nFound {len(files)} data file(s)")
    
    try:
        # Demo single graph
        demo_single_graph()
        
        # Demo comparison (if multiple files)
        if len(files) > 1:
            demo_comparison_graph()
        
        # Demo advanced analysis
        demo_advanced_analysis()
        
        print("\n✅ All demos completed!")
        print("Check the 'example_graphs' folder for generated charts.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")

if __name__ == "__main__":
    main()
