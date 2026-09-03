import time
import torch
import numpy as np
from pathlib import Path

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.adapter import export_drive_scenarios
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig, optimize_frozen_ego_scenario
from pufferlib.ocean.regents.rollout import replay_optimized_scenario_in_c

REPO_ROOT = Path(__file__).resolve().parents[1]
NUPLAN_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/nuplan"


def main():
    map_idx = 8
    seed = 50
    horizon = 100

    print(f"Initializing Drive for Scenario {map_idx}, Seed {seed} with non-reactive 'replay' SDC...")
    drive = Drive(
        map_dir=str(NUPLAN_MAP_DIR),
        num_maps=map_idx + 1,
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[map_idx],
        eval_scenario_seeds=[seed],
        seed=42,
        simulation_mode="replay",
        eval_mode=True,
        control_mode="control_sdc_only",
        sdc_controller="replay",  # Non-intelligent / non-reactive logged replay SDC!
        non_sdc_controller="replay",
        non_vehicle_controller="replay",
        action_type="continuous",
        dynamics_model="classic",
        dt=0.1,
        scenario_length=200,
        resample_frequency=200,
        init_step=0,
        init_step_spread=False,
        reward_conditioning=False,
        reward_randomization=False,
        use_neighbor_cache=0,
    )

    try:
        drive.reset(seed=seed)
        print("Exporting drive scenario to Torch representation...")
        scenario = export_drive_scenarios(drive, raster_resolution_meters=5.0)

        print("Running ReGentS optimization against the static SDC logged trajectory (500 iterations)...")
        start_time = time.perf_counter()
        optimization = optimize_frozen_ego_scenario(
            scenario,
            config=ReGentSOptimizationConfig(iteration_count=500, learning_rate=1e-3),
            deterministic_seed=seed,
            horizon_transition_count=horizon,
        )
        elapsed = time.perf_counter() - start_time
        print(f"Optimization completed in {elapsed:.2f} seconds.")

        print("Replaying optimized scenario in C with replay SDC...")
        replay = replay_optimized_scenario_in_c(drive, scenario, optimization, seed=seed)

        metrics = replay.metrics
        print("\n" + "=" * 40)
        print("           REPLAY METRICS")
        print("=" * 40)
        print(f"Replay Success:             {replay.success}")
        print(f"Ego Collision:              {metrics.ego_collision}")
        print(f"Actionable Ego Collision:   {metrics.actionable_collision}")
        print(f"Background Collision:       {metrics.background_collision}")
        print(f"Offroad Infraction:         {metrics.offroad}")
        print(f"First Collision Timestep:   {metrics.first_collision_timestep}")
        print(f"C/Torch Trajectory Error:    {metrics.maximum_trajectory_error:.6f}")
        print(f"Failure Reason:             {replay.failure_reason}")
        print("=" * 40)

    finally:
        drive.close()
        print("Drive environment closed.")


if __name__ == "__main__":
    main()
