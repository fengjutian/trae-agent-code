#!/usr/bin/env python3
"""
Test script for hello_world.py
Verifies that the hello world script works correctly.
"""

import subprocess
import sys

def test_hello_world():
    """Test that hello_world.py prints 'Hello, World!'"""
    try:
        # Run the hello_world.py script
        result = subprocess.run(
            [sys.executable, "hello_world.py"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check the output
        output = result.stdout.strip()
        expected = "Hello, World!"
        
        if output == expected:
            print(f"Test passed: Output '{output}' matches expected '{expected}'")
            return True
        else:
            print(f"Test failed: Output '{output}' does not match expected '{expected}'")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed: {e}")
        print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_hello_world()
    sys.exit(0 if success else 1)