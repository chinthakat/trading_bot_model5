#!/usr/bin/env python3
print("Starting import test...")

try:
    print("About to import btc_graph_generator...")
    import btc_graph_generator
    print("Module imported successfully")
    print(f"Module contents: {dir(btc_graph_generator)}")
    
    if hasattr(btc_graph_generator, 'BTCGraphGenerator'):
        print("BTCGraphGenerator class found!")
    else:
        print("BTCGraphGenerator class NOT found!")
        
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
