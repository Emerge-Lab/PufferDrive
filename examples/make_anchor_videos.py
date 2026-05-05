"""
Load a trained BC anchor policy and record N rollout videos.

Edit the globals below, then run:
    python record_anchor_videos.py
"""

import glob
import os
import shutil

import torch

from pufferlib.pufferl import load_env

from train_bc_policy import (
    build_env_args,
    get_output_sizes,
    load_bc_policy,
)

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "models/anchors/bc_delta_local_12000maps_djotwyvn.pt"
DYNAMICS_MODEL = "delta_local"  # "delta_local" or "classic"
NUM_VIDEOS = 10
OUTPUT_DIR = "anchor_videos_12k_maps_wo_constraints"
MAP_DIR = "resources/drive/binaries/training"

# Must match the values used during training
INPUT_SIZE = 128
HIDDEN_SIZE = 512


# ---------------------------------------------------------------------------
def record_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build the rollout env. We need it before loading the policy because
    # the policy constructor depends on env-derived dimensions.
    args = build_env_args(DYNAMICS_MODEL, num_maps=NUM_VIDEOS)
    args["env"]["map_dir"] = MAP_DIR
    args["env"]["control_mode"] = "control_sdc_only"
    args["env"]["termination_mode"] = 1
    args["env"]["obs_partner_noise_speed"] = 0.0
    args["env"]["obs_partner_noise_pos"] = 0.0
    args["env"]["async_resets"] = False
    args["env"]["resample_frequency"] = 0
    args["env"]["lambda_value"] = 0.0
    args["env"]["fix_lambdas"] = True
    args["env"]["fix_rewards"] = True
    args["env"]["num_agents"] = 128
    args["vec"] = dict(backend="PufferEnv", num_envs=1)

    env = load_env("puffer_drive", args)
    driver_env = env.driver_env

    output_sizes = get_output_sizes(DYNAMICS_MODEL)
    policy, _metrics = load_bc_policy(
        checkpoint_path=CHECKPOINT_PATH,
        obs_dim=driver_env.num_obs,
        input_size=INPUT_SIZE,
        max_partner_objects=driver_env.max_partner_objects,
        partner_features=driver_env.partner_features,
        max_road_objects=driver_env.max_road_objects,
        road_features=driver_env.road_features,
        ego_dim=driver_env.ego_features,
        hidden_size=HIDDEN_SIZE,
        output_sizes=output_sizes,
        device=str(device),
    )

    episode_length = driver_env.episode_length or 91
    logged = 0

    for map_idx in range(NUM_VIDEOS):
        obs, _info = env.reset()
        print(f"Rendering map {map_idx + 1}/{NUM_VIDEOS}")
        before = set(glob.glob("*.mp4"))

        for _t in range(episode_length):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = policy(obs_tensor, deterministic=True)

            action_np = actions.cpu().numpy()
            if action_np.ndim == 1:
                action_np = action_np[:, None]

            obs, _, terminated, truncated, _ = env.step(action_np)

            if not terminated[map_idx]:
                driver_env.render(env_idx=map_idx)

            if truncated.all() or terminated.all():
                break

        # Flush ffmpeg so the file appears on disk
        driver_env.stop_recorder(map_idx)

        new_files = sorted(set(glob.glob("*.mp4")) - before)
        if not new_files:
            print(f"  Warning: no video produced for map_idx={map_idx}")
            continue

        for mp4_path in new_files:
            dest = os.path.join(OUTPUT_DIR, os.path.basename(mp4_path))
            shutil.move(mp4_path, dest)
            print(f"  Saved: {dest}")
            logged += 1

    env.close()
    print(f"\nDone. Saved {logged} videos to {OUTPUT_DIR}/")


if __name__ == "__main__":
    record_videos()
