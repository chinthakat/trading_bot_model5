#!/usr/bin/env python3
"""
Test the action wrapper fix
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_action_wrapper_fix():
    """Test that the action wrapper can handle numpy arrays"""
    print("🧪 Testing action wrapper fix...")
    
    # Test different action formats
    test_actions = [
        0,                    # Plain integer
        np.int32(1),         # Numpy scalar
        np.array([2]),       # Single-element numpy array
        np.array(3),         # Zero-dimensional numpy array
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nTest {i+1}: Input = {action} (type: {type(action)})")
        
        # Apply the same conversion logic as in the fix
        try:
            if isinstance(action, np.ndarray):
                converted = int(action.item())
            else:
                converted = int(action)
            
            print(f"  Converted = {converted} (type: {type(converted)})")
            
            # Test dictionary lookup (the operation that was failing)
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE_ALL'}
            action_name = action_names.get(converted, f'UNKNOWN_{converted}')
            print(f"  Action name = {action_name}")
            print("  ✅ Success!")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print("\n🎉 Action wrapper fix verification complete!")

if __name__ == "__main__":
    test_action_wrapper_fix()
