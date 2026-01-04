#!/usr/bin/env python3
"""
Test that map loading works with arbitrary .bin filenames,
not just the map_XXX.bin convention.
"""

import os
import sys
import shutil
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_env(map_dir, num_maps=1):
    """Create a Drive environment with proper config loading."""
    from pufferlib.pufferl import load_config, load_env

    with patch("sys.argv", ["test"]):
        args = load_config("puffer_drive")

    args["env"].update(
        {
            "num_agents": 4,
            "num_maps": num_maps,
            "map_dir": map_dir,
            "action_type": "discrete",
            "init_mode": "create_all_valid",
            "control_mode": "control_agents",
        }
    )
    args["vec"].update(
        {
            "num_workers": 1,
            "num_envs": 1,
            "batch_size": 1,
        }
    )

    return load_env("puffer_drive", args)


def test_standard_map_naming():
    """Test that standard map_000.bin naming still works (backward compatibility)."""
    print("Testing standard map_000.bin naming...")

    map_dir = "resources/drive/binaries"

    try:
        env = create_env(map_dir, num_maps=1)
        env.reset()

        # Run a few steps
        for _ in range(10):
            actions = env.action_space.sample()
            env.step(actions)

        env.close()
        print("  Standard naming test passed!")
        return True

    except Exception as e:
        print(f"  Standard naming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_arbitrary_map_naming():
    """Test that arbitrarily named .bin files work."""
    print("Testing arbitrary map naming...")

    # Create a temporary directory with arbitrarily named maps
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy an existing map with a different name
        src_map = "resources/drive/binaries/map_000.bin"
        if not os.path.exists(src_map):
            print(f"  Skipping: source map {src_map} not found")
            return True

        # Copy with arbitrary names
        shutil.copy(src_map, os.path.join(tmpdir, "highway_scene.bin"))
        shutil.copy(src_map, os.path.join(tmpdir, "downtown.bin"))
        shutil.copy(src_map, os.path.join(tmpdir, "my_custom_map.bin"))

        try:
            env = create_env(tmpdir, num_maps=3)
            env.reset()

            # Run a few steps
            for _ in range(10):
                actions = env.action_space.sample()
                env.step(actions)

            env.close()
            print("  Arbitrary naming test passed!")
            return True

        except Exception as e:
            print(f"  Arbitrary naming test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_single_arbitrary_map():
    """Test with a single arbitrarily named map."""
    print("Testing single arbitrary map...")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_map = "resources/drive/binaries/map_000.bin"
        if not os.path.exists(src_map):
            print(f"  Skipping: source map {src_map} not found")
            return True

        # Copy with a single arbitrary name
        shutil.copy(src_map, os.path.join(tmpdir, "only_map.bin"))

        try:
            env = create_env(tmpdir, num_maps=1)
            env.reset()

            for _ in range(10):
                actions = env.action_space.sample()
                env.step(actions)

            env.close()
            print("  Single arbitrary map test passed!")
            return True

        except Exception as e:
            print(f"  Single arbitrary map test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_empty_directory_error():
    """Test that an appropriate error is raised for empty directories."""
    print("Testing empty directory error handling...")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            env = create_env(tmpdir, num_maps=1)
            print("  Empty directory test failed: should have raised an error!")
            return False

        except FileNotFoundError as e:
            if ".bin" in str(e):
                print("  Empty directory test passed (correct error raised)!")
                return True
            else:
                print(f"  Empty directory test failed: wrong error message: {e}")
                return False

        except Exception as e:
            # The error might be wrapped, check if it's about .bin files
            if ".bin" in str(e):
                print("  Empty directory test passed (correct error raised)!")
                return True
            print(f"  Empty directory test failed with unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_nonexistent_directory_error():
    """Test that an appropriate error is raised for non-existent directories."""
    print("Testing non-existent directory error handling...")

    try:
        env = create_env("/nonexistent/path/to/maps", num_maps=1)
        print("  Non-existent directory test failed: should have raised an error!")
        return False

    except FileNotFoundError as e:
        print("  Non-existent directory test passed (correct error raised)!")
        return True

    except Exception as e:
        # The error might be wrapped
        if "not found" in str(e).lower() or "nonexistent" in str(e).lower():
            print("  Non-existent directory test passed (correct error raised)!")
            return True
        print(f"  Non-existent directory test failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sanity_subdirectory_creation():
    """Test that sanity check creates proper per-map subdirectories."""
    print("Testing sanity subdirectory creation logic...")

    from pathlib import Path
    from pufferlib.ocean.drive.drive import load_map

    with tempfile.TemporaryDirectory() as tmpdir:
        binary_dir = Path(tmpdir)

        # Simulate what pufferl.py now does for each sanity map
        src_json = Path("pufferlib/resources/drive/sanity/sanity_jsons/forward_goal_in_front.json")
        if not src_json.exists():
            print("  Skipping: sanity JSON files not found")
            return True

        try:
            # Create subdirectory for this map (as the new code does)
            map_name = "forward_goal_in_front"
            map_subdir = binary_dir / map_name
            map_subdir.mkdir(parents=True, exist_ok=True)
            output_path = map_subdir / f"{map_name}.bin"

            # Convert JSON to binary
            load_map(str(src_json), 0, str(output_path))

            # Verify the structure
            assert map_subdir.exists(), "Subdirectory should exist"
            assert output_path.exists(), "Binary file should exist"

            # Verify the binary can be loaded by creating an env
            env = create_env(str(map_subdir), num_maps=1)
            env.reset()
            env.close()

            print("  Sanity subdirectory creation test passed!")
            return True

        except Exception as e:
            print(f"  Sanity subdirectory creation test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def run_all_tests():
    """Run all map filename flexibility tests."""
    print("=" * 60)
    print("Running map filename flexibility tests")
    print("=" * 60)

    results = []
    results.append(("Standard naming", test_standard_map_naming()))
    results.append(("Arbitrary naming", test_arbitrary_map_naming()))
    results.append(("Single arbitrary map", test_single_arbitrary_map()))
    results.append(("Empty directory error", test_empty_directory_error()))
    results.append(("Non-existent directory error", test_nonexistent_directory_error()))
    results.append(("Sanity subdirectory creation", test_sanity_subdirectory_creation()))

    print("=" * 60)
    print("Test Results:")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
