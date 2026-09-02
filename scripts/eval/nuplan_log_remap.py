"""Load nuPlan simulation logs produced on another machine (e.g. the cluster) locally.

Simulation logs pickle the NuPlanScenario with the data/map roots baked in and the planner object
with its classes. Importing this module patches the scenario and map-db constructors to remap
CLUSTER_NUPLAN_ROOT -> NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT (val logs are looked up in splits/trainval
when splits/val is absent) and provides `load_log`, which unpickles with placeholder classes for any
module that cannot be imported here.

Env: CLUSTER_NUPLAN_ROOT (default /home/shared/data/nuplan), NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT.
"""

import io
import lzma
import os
import pickle
from pathlib import Path

import msgpack

from nuplan.database.maps_db import gpkg_mapsdb as _gpkg
from nuplan.planning.scenario_builder.nuplan_db import nuplan_scenario as _scenario

CLUSTER_ROOT = os.environ.get("CLUSTER_NUPLAN_ROOT", "/home/shared/data/nuplan")
LOCAL_DATA_ROOT = os.environ.get("NUPLAN_DATA_ROOT", CLUSTER_ROOT)
LOCAL_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT", CLUSTER_ROOT + "/maps")


def remap_path(path):
    if not isinstance(path, str) or not path.startswith(CLUSTER_ROOT):
        return path
    if path.startswith(CLUSTER_ROOT + "/maps"):
        return LOCAL_MAPS_ROOT + path[len(CLUSTER_ROOT + "/maps") :]
    local = LOCAL_DATA_ROOT + path[len(CLUSTER_ROOT) :]
    if "/splits/val/" in local and not Path(local).exists():
        local = local.replace("/splits/val/", "/splits/trainval/")
    return local


_orig_scenario_init = _scenario.NuPlanScenario.__init__
_orig_gpkg_init = _gpkg.GPKGMapsDB.__init__


def _scenario_init(self, data_root, log_file_load_path, initial_lidar_token, initial_lidar_timestamp, scenario_type, map_root, *args, **kwargs):
    _orig_scenario_init(
        self,
        remap_path(data_root),
        remap_path(log_file_load_path),
        initial_lidar_token,
        initial_lidar_timestamp,
        scenario_type,
        remap_path(map_root),
        *args,
        **kwargs,
    )


def _gpkg_init(self, map_version, map_root, *args, **kwargs):
    _orig_gpkg_init(self, map_version, remap_path(map_root), *args, **kwargs)


_scenario.NuPlanScenario.__init__ = _scenario_init
_gpkg.GPKGMapsDB.__init__ = _gpkg_init


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_Placeholder,), {})


def load_log(path):
    """SimulationLog.load_data with the remap + placeholder classes (msgpack.xz and pkl.xz)."""
    path = Path(path)
    with lzma.open(str(path), "rb") as f:
        raw = f.read()
    if path.suffixes[-2] == ".msgpack":
        raw = msgpack.unpackb(raw)
    return _LenientUnpickler(io.BytesIO(raw)).load()
