#!/usr/bin/env python3
"""
Test script for PufferDrive raylib rendering functionality.
"""

import os
import subprocess
import sys
import numpy as np


def test_drive_render():
    """Test that PufferDrive can generate GIFs using the raylib renderer."""
    print("Testing PufferDrive rendering...")

    # Check if drive binary exists
    if not os.path.exists("./drive"):
        print("Drive binary not found, attempting to build...")
        try:
            result = subprocess.run(
                ["bash", "scripts/build_ocean.sh", "drive", "local"], capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0 or not os.path.exists("./drive"):
                print(f"Build failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Build error: {e}")
            return False

    # Create dummy weights file
    os.makedirs("resources/drive", exist_ok=True)
    weights_path = "resources/drive/puffer_drive_weights.bin"
    dummy_weights = np.random.randn(10000).astype(np.float32)
    dummy_weights.tofile(weights_path)

    try:
        # Run the renderer with xvfb
        print("Running renderer (this takes 60+ seconds)...")
        result = subprocess.run(
            ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./drive"], capture_output=True, text=True, timeout=600
        )

        print(f"Renderer exit code: {result.returncode}")
        if result.stderr:
            print(f"stderr: {result.stderr}")

        # Check for output GIFs
        output_files = ["resources/drive/output_topdown.gif", "resources/drive/output_agent.gif"]

        success = True
        for output_file in output_files:
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"Generated {output_file} ({size} bytes)")
            else:
                print(f"Missing {output_file}")
                success = False

        return success

    except subprocess.TimeoutExpired:
        print("Renderer timed out")
        return False
    except Exception as e:
        print(f"Render test failed: {e}")
        return False
    finally:
        # Cleanup
        for path in [weights_path, "resources/drive/output_topdown.gif", "resources/drive/output_agent.gif"]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    if test_drive_render():
        print("Render test passed!")
        sys.exit(0)
    else:
        print("Render test failed")
        sys.exit(1)
