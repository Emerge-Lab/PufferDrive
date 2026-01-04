#!/usr/bin/env python3
"""
Tests for the map_sources feature that allows mixing maps from multiple directories.

Running the test: python -m pytest tests/test_map_sources.py -v
"""

import os
import tempfile
import unittest

from pufferlib.ocean.drive.drive import create_mixed_map_directory


class TestMapSources(unittest.TestCase):
    def setUp(self):
        """Create temporary directories with fake map files for testing."""
        self.temp_dirs = []

        # Create first source directory with 5 maps
        self.source1 = tempfile.mkdtemp(prefix="test_maps_source1_")
        self.temp_dirs.append(self.source1)
        for i in range(5):
            open(os.path.join(self.source1, f"map_{i:03d}.bin"), "w").close()

        # Create second source directory with 3 maps
        self.source2 = tempfile.mkdtemp(prefix="test_maps_source2_")
        self.temp_dirs.append(self.source2)
        for i in range(3):
            open(os.path.join(self.source2, f"map_{i:03d}.bin"), "w").close()

    def tearDown(self):
        """Clean up temporary directories."""
        import shutil

        for d in self.temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_creates_correct_number_of_symlinks(self):
        """Test that the correct total number of map symlinks are created."""
        map_sources = f"{self.source1}:0.5,{self.source2}:0.5"
        result_dir = create_mixed_map_directory(map_sources, num_maps=4)
        self.temp_dirs.append(result_dir)

        bin_files = [f for f in os.listdir(result_dir) if f.endswith(".bin")]
        self.assertEqual(len(bin_files), 4)

    def test_symlinks_point_to_real_files(self):
        """Test that created symlinks point to actual files."""
        map_sources = f"{self.source1}:1.0"
        result_dir = create_mixed_map_directory(map_sources, num_maps=3)
        self.temp_dirs.append(result_dir)

        for f in os.listdir(result_dir):
            path = os.path.join(result_dir, f)
            self.assertTrue(os.path.islink(path))
            self.assertTrue(os.path.exists(path))  # Symlink target exists

    def test_unique_sampling_when_possible(self):
        """Test that maps are unique when count <= available."""
        map_sources = f"{self.source1}:1.0"
        result_dir = create_mixed_map_directory(map_sources, num_maps=5)
        self.temp_dirs.append(result_dir)

        # Get all symlink targets
        targets = set()
        for f in os.listdir(result_dir):
            path = os.path.join(result_dir, f)
            targets.add(os.path.realpath(path))

        # All 5 should be unique
        self.assertEqual(len(targets), 5)

    def test_resampling_when_count_exceeds_available(self):
        """Test that resampling occurs when requesting more maps than available."""
        map_sources = f"{self.source2}:1.0"  # Only 3 maps available
        result_dir = create_mixed_map_directory(map_sources, num_maps=6)
        self.temp_dirs.append(result_dir)

        bin_files = [f for f in os.listdir(result_dir) if f.endswith(".bin")]
        self.assertEqual(len(bin_files), 6)

        # Get all symlink targets - should have duplicates
        targets = []
        for f in os.listdir(result_dir):
            path = os.path.join(result_dir, f)
            targets.append(os.path.realpath(path))

        # Should have duplicates since we requested 6 from 3 available
        self.assertEqual(len(targets), 6)
        self.assertLess(len(set(targets)), 6)

    def test_weighted_distribution(self):
        """Test that weights are respected in map distribution."""
        # 80% from source1, 20% from source2
        map_sources = f"{self.source1}:0.8,{self.source2}:0.2"
        result_dir = create_mixed_map_directory(map_sources, num_maps=10)
        self.temp_dirs.append(result_dir)

        source1_count = 0
        source2_count = 0
        for f in os.listdir(result_dir):
            path = os.path.join(result_dir, f)
            target = os.path.realpath(path)
            if self.source1 in target:
                source1_count += 1
            elif self.source2 in target:
                source2_count += 1

        # Should be approximately 8 from source1, 2 from source2
        self.assertEqual(source1_count, 8)
        self.assertEqual(source2_count, 2)

    def test_invalid_format_raises_error(self):
        """Test that invalid map_sources format raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_mixed_map_directory("invalid_format", num_maps=5)
        self.assertIn("Invalid map_sources format", str(ctx.exception))

    def test_empty_directory_raises_error(self):
        """Test that an empty source directory raises ValueError."""
        empty_dir = tempfile.mkdtemp(prefix="test_maps_empty_")
        self.temp_dirs.append(empty_dir)

        with self.assertRaises(ValueError) as ctx:
            create_mixed_map_directory(f"{empty_dir}:1.0", num_maps=5)
        self.assertIn("No .bin files found", str(ctx.exception))

    def test_symlink_naming_format(self):
        """Test that symlinks are named correctly (map_000.bin, map_001.bin, etc.)."""
        map_sources = f"{self.source1}:1.0"
        result_dir = create_mixed_map_directory(map_sources, num_maps=3)
        self.temp_dirs.append(result_dir)

        expected_names = {"map_000.bin", "map_001.bin", "map_002.bin"}
        actual_names = set(os.listdir(result_dir))
        self.assertEqual(actual_names, expected_names)


if __name__ == "__main__":
    unittest.main()
