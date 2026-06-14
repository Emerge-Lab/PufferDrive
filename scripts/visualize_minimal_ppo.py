"""Render a trained minimal PPO checkpoint with PufferDrive's native 3D view."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.drive.drive import Drive, RenderView
from scripts.minimal_ppo_train import ActorCritic


def start_xvfb_if_needed():
    if os.environ.get("DISPLAY"):
        return None

    for display_num in range(99, 120):
        display = f":{display_num}"
        lock_path = Path(f"/tmp/.X{display_num}-lock")
        socket_path = Path(f"/tmp/.X11-unix/X{display_num}")
        if lock_path.exists() or socket_path.exists():
            continue

        error_log = tempfile.NamedTemporaryFile(prefix="pufferdrive_eval_xvfb_", suffix=".log", delete=False)
        error_log_path = Path(error_log.name)
        proc = subprocess.Popen(
            [
                "Xvfb",
                display,
                "-screen",
                "0",
                "1280x720x24",
                "+extension",
                "GLX",
                "-ac",
                "-noreset",
            ],
            stdout=subprocess.DEVNULL,
            stderr=error_log,
        )
        error_log.close()
        os.environ["DISPLAY"] = display

        for _ in range(40):
            if proc.poll() is not None:
                break
            if lock_path.exists() or socket_path.exists():
                time.sleep(0.5)
                error_log_path.unlink(missing_ok=True)
                return proc
            time.sleep(0.1)

        proc.terminate()
        proc.wait(timeout=2)
        os.environ.pop("DISPLAY", None)
        detail = error_log_path.read_text(errors="replace").strip()
        error_log_path.unlink(missing_ok=True)
        if detail:
            print(detail)

    raise RuntimeError("Could not start Xvfb")


def stop_xvfb(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    os.environ.pop("DISPLAY", None)


def probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,duration,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "video_frames": int(stream.get("nb_read_frames", 0)),
        "video_duration_seconds": float(stream.get("duration", 0.0)),
        "video_frame_rate": stream.get("r_frame_rate"),
    }


def render_checkpoint(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_config = checkpoint.get("config", {})

    goal_behavior = args.goal_behavior
    if goal_behavior is None:
        goal_behavior = int(train_config.get("goal_behavior", 1))

    goal_target_distance = args.goal_target_distance
    if goal_target_distance is None:
        goal_target_distance = float(train_config.get("goal_target_distance", 30.0))

    env = Drive(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        num_agents=args.num_envs * args.controlled_agents_per_env,
        control_mode="control_mixed_play",
        init_mode="create_all_valid",
        goal_behavior=goal_behavior,
        goal_target_distance=goal_target_distance,
        action_type="continuous",
        episode_length=args.episode_length,
        termination_mode=0,
        resample_frequency=0,
        render_mode=1,
        human_agent_idx=args.agent_index,
        max_controlled_agents=args.controlled_agents_per_env,
    )

    observation, _ = env.reset(seed=args.seed)
    observation_dim = observation.shape[1]
    action_dim = env.single_action_space.shape[0]
    hidden_size = int(train_config.get("hidden_size", 256))
    model = ActorCritic(observation_dim, action_dim, hidden_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scenario_id = env.scenario_ids[args.env_id]
    native_output = Path(f"{scenario_id}.mp4")
    native_output.unlink(missing_ok=True)
    returns = np.zeros(observation.shape[0], dtype=np.float32)
    reward_history = []
    action_history = []
    xvfb_proc = start_xvfb_if_needed()

    try:
        env.render(
            view_mode=RenderView.AGENT_PERSP,
            draw_traces=args.draw_traces,
            env_id=args.env_id,
        )
        for step in range(args.episode_length):
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, values = model.deterministic_action(obs_tensor)
            actions_np = actions.cpu().numpy().astype(np.float32)
            observation, rewards, terminals, truncations, infos = env.step(actions_np)
            env.render(
                view_mode=RenderView.AGENT_PERSP,
                draw_traces=args.draw_traces,
                env_id=args.env_id,
            )
            returns += rewards
            reward_history.append(float(np.mean(rewards)))
            action_history.append(actions_np[args.agent_index].tolist())
            if step % 10 == 0:
                print(
                    f"step={step:02d}/{args.episode_length} "
                    f"mean_reward={np.mean(rewards):+.4f} "
                    f"ego_action={actions_np[args.agent_index]}"
                )
    finally:
        env.close()
        stop_xvfb(xvfb_proc)

    if not native_output.exists():
        raise FileNotFoundError(f"Native renderer did not produce {native_output}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"{scenario_id}_ppo_step_{checkpoint.get('global_step', 0)}.mp4"
    shutil.move(native_output, video_path)
    video_probe = probe_video(video_path)
    if video_probe["video_frames"] < args.episode_length:
        raise RuntimeError(
            f"Native renderer produced only {video_probe['video_frames']} frames; "
            f"expected at least {args.episode_length}"
        )

    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "scenario_id": scenario_id,
        "episode_length": args.episode_length,
        "goal_behavior": goal_behavior,
        "goal_target_distance": goal_target_distance,
        "mean_reward_per_step": float(np.mean(reward_history)),
        "mean_agent_return": float(np.mean(returns)),
        "ego_return": float(returns[args.agent_index]),
        "mean_absolute_ego_action": np.mean(np.abs(action_history), axis=0).tolist(),
        "video": str(video_path),
        **video_probe,
    }
    metrics_path = video_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return video_path, metrics_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/minimal_ppo/ppo_final.pt"),
    )
    parser.add_argument("--map-dir", default="resources/drive/binaries/waymo_native")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--controlled-agents-per-env", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=91)
    parser.add_argument("--goal-behavior", type=int, choices=[0, 1, 2])
    parser.add_argument("--goal-target-distance", type=float)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--agent-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--draw-traces", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("training_visualizations"))
    return parser.parse_args()


if __name__ == "__main__":
    render_checkpoint(parse_args())
