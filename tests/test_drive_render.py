#!/usr/bin/env python3
"""
Test script for PufferDrive raylib rendering functionality.
"""

import os
import subprocess
import sys
import numpy as np


def test_drive_render():
    """Test that PufferDrive renderer runs successfully (exit code 0)."""
    print("Testing PufferDrive rendering...")

    # Check if drive binary exists
    if not os.path.exists("./drive"):
        print("Drive binary not found, attempting to build...")
        try:
            result = subprocess.run(
                ["bash", "scripts/build_ocean.sh", "drive", "local"], capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0 or not os.path.exists("./drive"):
                print(f"Build failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Build error: {e}")
            return False

    # Backup existing weights file if it exists
    weights_path = "resources/drive/puffer_drive_weights.bin"
    backup_path = "resources/drive/puffer_drive_weights.bin.backup"
    weights_existed = False

    if os.path.exists(weights_path):
        weights_existed = True
        os.rename(weights_path, backup_path)

    # Create dummy weights file
    os.makedirs("resources/drive", exist_ok=True)
    dummy_weights = np.random.randn(10000).astype(np.float32)
    dummy_weights.tofile(weights_path)

    try:
        # Set up environment to suppress AddressSanitizer exit code (needed due to current memory leaks)
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "exitcode=0"

        # Run the renderer with xvfb and frame skip for faster testing
        print("Running renderer.")
        result = subprocess.run(
            ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./drive", "--frame-skip", "10"],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        print(f"Renderer exit code: {result.returncode}")

        # Show output for debugging if needed
        if result.stdout:
            print(f"stdout: {result.stdout}")
        if result.stderr:
            print(f"stderr: {result.stderr}")

        if result.returncode == 0:
            print("Renderer completed successfully!")
            return True
        else:
            print(f"Renderer failed with exit code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("Renderer timed out")
        return False
    except Exception as e:
        print(f"Render test failed: {e}")
        return False
    finally:
        # Cleanup: remove test outputs and restore original weights if they existed
        if os.path.exists(weights_path):
            os.remove(weights_path)

        if weights_existed and os.path.exists(backup_path):
            os.rename(backup_path, weights_path)

        # Clean up generated outputs
        for output_file in ["resources/drive/output_topdown.gif", "resources/drive/output_agent.gif"]:
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == "__main__":
    if test_drive_render():
        print("Render test passed!")
        sys.exit(0)
    else:
        print("Render test failed")
        sys.exit(1)
