"""Unit tests for the code-isolation step of scripts/submit_cluster.py."""

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("submitit")

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("submit_cluster", REPO_ROOT / "scripts" / "submit_cluster.py")
submit_cluster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(submit_cluster)


def make_project(root: Path) -> Path:
    """Minimal project tree with the entries isolate_code treats specially."""
    project = root / "project"
    (project / "pufferlib" / "ocean").mkdir(parents=True)
    (project / "pufferlib" / "__init__.py").write_text("")
    (project / "pufferlib" / "resources" / "drive").mkdir(parents=True)
    (project / "pufferlib" / "resources" / "drive" / "map.bin").write_text("data")
    (project / "data").mkdir()
    (project / "scripts").mkdir()
    (project / "setup.py").write_text("")
    return project


def test_symlinks_top_level_and_hard_copies_pufferlib(tmp_path):
    project = make_project(tmp_path)
    save_dir = tmp_path / "runs" / "run1"
    save_dir.mkdir(parents=True)

    isolated = Path(submit_cluster.isolate_code(str(project), str(save_dir)))

    assert isolated == save_dir / "code"
    # Top-level entries are symlinks into the real tree
    assert (isolated / "data").is_symlink()
    assert (isolated / "setup.py").is_symlink()
    assert (isolated / "data").resolve() == (project / "data").resolve()
    # pufferlib is a hard copy, not a link
    assert (isolated / "pufferlib").is_dir()
    assert not (isolated / "pufferlib").is_symlink()
    assert (isolated / "pufferlib" / "__init__.py").is_file()
    assert not (isolated / "pufferlib" / "__init__.py").is_symlink()
    # ...except resources/, which is symlinked back to the shared data
    assert (isolated / "pufferlib" / "resources").is_symlink()
    assert (isolated / "pufferlib" / "resources" / "drive" / "map.bin").read_text() == "data"


def test_save_dir_inside_project_creates_no_symlink_cycle(tmp_path):
    """Regression test: --save_dir inside the repo used to link experiments/
    into the snapshot, creating experiments/code/experiments -> experiments.
    setuptools package discovery follows symlinks, so that cycle made every
    later build walk an effectively infinite tree."""
    project = make_project(tmp_path)
    save_dir = project / "experiments" / "run1"
    save_dir.mkdir(parents=True)

    isolated = Path(submit_cluster.isolate_code(str(project), str(save_dir)))

    # The ancestor entry is skipped entirely
    assert not (isolated / "experiments").exists()
    assert not (isolated / "experiments").is_symlink()
    # No symlink in the snapshot points at any ancestor of the snapshot
    isolated_real = isolated.resolve()
    for entry in isolated.iterdir():
        if entry.is_symlink():
            assert not isolated_real.is_relative_to(entry.resolve())
    # Sibling entries are still linked
    assert (isolated / "data").is_symlink()
    # A symlink-following walk over the snapshot terminates
    dir_count = 0
    for _root, _dirs, _files in os.walk(isolated, followlinks=True):
        dir_count += 1
        assert dir_count < 1000, "walk exploded: symlink cycle in snapshot"


def test_second_snapshot_gets_versioned_dir(tmp_path):
    project = make_project(tmp_path)
    save_dir = tmp_path / "runs" / "run1"
    save_dir.mkdir(parents=True)

    first = Path(submit_cluster.isolate_code(str(project), str(save_dir)))
    second = Path(submit_cluster.isolate_code(str(project), str(save_dir)))

    assert first == save_dir / "code"
    assert second == save_dir / "code_v1"
    assert (second / "pufferlib" / "__init__.py").is_file()
