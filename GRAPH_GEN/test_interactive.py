#!/usr/bin/env python3
print("Test script starting...")

try:
    print("Importing modules...")
    import sys
    print(f"Python version: {sys.version}")
    
    import matplotlib
    matplotlib.use('Agg')
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    
    from btc_graph_generator import BTCGraphGenerator
    print("BTCGraphGenerator imported successfully")
    
    print("Creating generator...")
    generator = BTCGraphGenerator()
    
    print("Running interactive mode...")
    generator.run_interactive_mode()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test script finished.")
