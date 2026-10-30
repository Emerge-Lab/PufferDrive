"""Which ego_is_comfortable sub-metric fails how often in nuPlan closed-loop eval runs of the PufferDrive planner.
usage: python scripts/eval/nuplan_comfort_breakdown.py <group_dir> [<group_dir> ...]
Each group dir is a run_simulation.py `group` directory; the per-scenario metric parquets under
simulation/<challenge>/<ts>/metrics are read. nuPlan's longitudinal acceleration is taken at the vehicle center,
which the devkit computes as rear-axle accel + d*yaw_rate^2 + d*yaw_accel (d = rear axle to center), so a yaw
acceleration while accelerating reads as longitudinal acceleration.
"""

import glob
import sys

import pandas as pd

SUBMETRICS = (
    "ego_lon_acceleration",
    "ego_lat_acceleration",
    "ego_jerk",
    "ego_lon_jerk",
    "ego_yaw_rate",
    "ego_yaw_acceleration",
)
LON_ACCEL_MAX_MPS2 = 2.40  # ego_lon_acceleration_statistics.yaml


def main() -> None:
    for group in sys.argv[1:]:
        metrics_dirs = glob.glob(f"{group}/simulation/*/20*/metrics")
        if not metrics_dirs:
            print(f"{group}: no metrics directory")
            continue
        metrics_dir = metrics_dirs[0]
        comfortable = pd.read_parquet(f"{metrics_dir}/ego_is_comfortable.parquet")
        failed = int((comfortable["metric_score"] < 1).sum())
        print(f"{group.rstrip('/').split('/')[-1]}")
        print(f"  ego_is_comfortable fails {failed}/{len(comfortable)}")
        for name in SUBMETRICS:
            df = pd.read_parquet(f"{metrics_dir}/{name}.parquet")
            within = df[f"abs_{name}_within_bounds_stat_value"].astype(bool)
            fails = df[~within]
            detail = ""
            if name == "ego_lon_acceleration":
                positive = int((fails[f"max_{name}_stat_value"] > LON_ACCEL_MAX_MPS2).sum())
                detail = f" ({positive} above +{LON_ACCEL_MAX_MPS2}, {len(fails) - positive} below the braking bound)"
            print(f"  {name:22s} {len(fails):4d} fail{detail}")


if __name__ == "__main__":
    main()
