#!/usr/bin/env python3
"""
Demo script to show the interactive data selection and parameter configuration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import just the interactive functions we created
from train_memory_efficient import list_available_data_files, get_training_parameters, print_usage_help

def demo_interactive_mode():
    """Demo the interactive mode functionality"""
    print("🎮 Interactive Training Mode Demo")
    print("=" * 50)
    
    # Show usage help
    print("\n📖 First, let's see the usage help:")
    print_usage_help()
    
    # Demo data file selection
    print("\n🔍 Let's see the data file selection process:")
    try:
        # This will show the available files but not actually select one
        selected_file = list_available_data_files()
        if selected_file:
            print(f"\n✅ You selected: {Path(selected_file).name}")
            
            # Demo parameter configuration
            print("\n⚙️ Now let's configure training parameters:")
            episodes, steps = get_training_parameters()
            if episodes and steps:
                print(f"\n🎯 Final Configuration:")
                print(f"   📁 Data file: {Path(selected_file).name}")
                print(f"   📊 Episodes: {episodes}")
                print(f"   🏃 Steps per episode: {steps:,}")
                print(f"   🔢 Total training steps: {episodes * steps:,}")
                print(f"\n✅ Ready to start training with these settings!")
            else:
                print("\n❌ Parameter configuration cancelled")
        else:
            print("\n❌ Data file selection cancelled")
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled by user")

if __name__ == "__main__":
    demo_interactive_mode()
