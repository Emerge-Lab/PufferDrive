"""A nonexistent map_dir must fail fast with an actionable message: when the
path's basename is a dataset registered in data_utils/datasets.yaml, the
error names the exact fetch command.

Paths are absolute under tmp_path so the tests are independent of the CWD
and of whether the real dataset has been fetched into <repo>/data/.
"""

import pytest

from pufferlib.ocean.drive.drive import Drive


def test_registered_dataset_error_names_fetch_command(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"fetch_data\.py nuplan_mini_val"):
        Drive(map_dir=str(tmp_path / "nuplan_mini_val"))


def test_unregistered_path_error_is_plain(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"does not exist") as excinfo:
        Drive(map_dir=str(tmp_path / "no_such_dataset_xyz"))
    assert "fetch_data.py" not in str(excinfo.value)
