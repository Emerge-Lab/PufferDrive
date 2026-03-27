#!/usr/bin/env python3
"""Run a trained PufferDrive policy once per controllable agent across a map dataset."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import random
import re
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pufferlib
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive, RenderView
from pufferlib.pufferl import load_config, load_policy


RENDER_VIEW_MAP = {
    "full_sim_state": RenderView.FULL_SIM_STATE,
    "bev_agent_obs": RenderView.BEV_AGENT_OBS,
    "agent_persp": RenderView.AGENT_PERSP,
}
OBJECT_TYPES = {1, 2, 3}
SDC_TRACK_INDEX_OFFSET = 16
LOG_METRIC_KEYS = (
    "n",
    "score",
    "offroad_rate",
    "collision_rate",
    "collision_rate_valid",
    "episode_length",
    "episode_return",
    "dnf_rate",
    "completion_rate",
    "lane_alignment_rate",
    "perc_controlled",
    "perc_other",
    "offroad_per_agent",
    "collisions_per_agent",
    "goals_sampled_this_episode",
    "goals_reached_this_episode",
    "speed_at_goal",
)
DEFAULT_EVAL_MODES = ("human_replay", "self_play")


@dataclass(frozen=True)
class MapRecord:
    map_id: int
    map_name: str
    source_path: Path


def main():
    args = parse_args()
    args.policy_path = args.policy_path.resolve()
    args.map_dir = args.map_dir.resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()

    if not args.policy_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_path}")
    if not args.map_dir.is_dir():
        raise FileNotFoundError(f"Map directory not found: {args.map_dir}")

    maps = list_maps(args.map_dir)
    if not maps:
        raise FileNotFoundError(f"No map_XXX.bin files found in {args.map_dir}")

    output_dir = ensure_output_dir(args.output_dir, args.policy_path)
    base_args = load_base_args(args)
    seed_everything(args.seed)

    # Find the controllable agents on each map.
    discovery_cases, skipped_maps = discover_cases(base_args, maps)
    if not discovery_cases:
        raise RuntimeError("No controllable agents were discovered in the provided dataset.")

    # Build eval cases and load the policy once against a matching env.
    eval_cases = build_eval_cases(discovery_cases, args)
    if not eval_cases:
        raise RuntimeError("No evaluation cases were built for the selected eval modes.")
    with open_case_env(base_args, eval_cases[0]) as env:
        policy = create_policy(base_args, env)

    # Run evaluation, then optionally rerender a sample of rollouts.
    df = run_evaluations(base_args, policy, eval_cases, output_dir, args)
    df, random_rendered, collision_rendered = maybe_render_videos(
        base_args, policy, df, output_dir, args
    )

    summary = build_summary(df, maps, skipped_maps, output_dir, args)
    summary["random_videos_rendered"] = random_rendered
    summary["collision_videos_rendered"] = collision_rendered
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(df)} evaluation rows to {output_dir / 'results.csv'}")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument

    add("--policy-path", type=Path, required=True, help="Path to the trained policy .pt checkpoint")
    add("--map-dir", type=Path, required=True, help="Directory containing map_XXX.bin files")
    add("--output-dir", type=Path, default=None, help="Directory to save dataframe artifacts and optional videos")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used for inference")
    add(
        "--override",
        action="append",
        default=[],
        help="Config override in key=value or section.key=value form. Can be passed multiple times.",
    )
    add("--seed", type=int, default=42, help="Base seed for rollouts and random video sampling")
    add(
        "--eval-modes",
        nargs="+",
        choices=sorted(DEFAULT_EVAL_MODES),
        default=list(DEFAULT_EVAL_MODES),
        help="Which evaluation modes to run.",
    )
    add("--deterministic", action="store_true", help="Use argmax/mean actions instead of sampling")
    add("--checkpoint-every", type=int, default=100, help="Write a partial CSV every N evaluated runs")
    add("--num-random-videos", type=int, default=10, help="How many random rollout videos to render")
    add("--num-collision-videos", type=int, default=10, help="How many colliding rollout videos to render if available")
    add("--no-render-videos", action="store_true", help="Disable post-hoc video rendering")
    add("--render-view", choices=sorted(RENDER_VIEW_MAP), default="full_sim_state")
    add("--draw-traces", action="store_true", help="Draw logged replay traces in rerendered videos")
    return parser.parse_args()


def load_base_args(cli_args):
    argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = argv

    args["wandb"] = False
    args["neptune"] = False
    args["load_model_path"] = str(cli_args.policy_path)
    args["train"]["device"] = cli_args.device
    args["train"]["compile"] = False
    args["env"]["render_mode"] = 1
    args["env"]["init_mode"] = "create_all_valid"

    for override in cli_args.override:
        apply_override(args, override)

    args["train"]["use_rnn"] = args["rnn_name"] is not None
    return args


def apply_override(config: dict, override: str):
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value or section.key=value.")

    path, raw = override.split("=", 1)
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        value = raw

    target = config
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def list_maps(map_dir: Path) -> list[MapRecord]:
    pattern = re.compile(r"map_(\d+)\.bin$")
    maps = []
    for path in sorted(map_dir.glob("map_*.bin")):
        if match := pattern.match(path.name):
            maps.append(MapRecord(int(match.group(1)), path.name, path.resolve()))
    return maps


def ensure_output_dir(base_dir: Path | None, policy_path: Path) -> Path:
    if base_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path("artifacts") / "policy_dataset_eval" / f"{policy_path.stem}_{stamp}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir.resolve()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@contextlib.contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def read_object_indices(source_path: Path) -> dict[int, int]:
    data = source_path.read_bytes()
    view = memoryview(data)
    offset = SDC_TRACK_INDEX_OFFSET + 4

    (num_tracks_to_predict,) = struct.unpack_from("i", view, offset)
    offset += 4 + 4 * num_tracks_to_predict
    num_objects, num_roads = struct.unpack_from("ii", view, offset)
    offset += 8

    object_id_to_index = {}
    for entity_idx in range(num_objects + num_roads):
        _, entity_type, entity_id, array_size = struct.unpack_from("iiii", view, offset)
        offset += 16

        if entity_idx < num_objects:
            if entity_id in object_id_to_index:
                raise ValueError(f"Duplicate object id {entity_id} in {source_path}")
            object_id_to_index[entity_id] = entity_idx

        offset += 12 * array_size
        if entity_type in OBJECT_TYPES:
            offset += 12 * array_size
            offset += 4 * array_size
            offset += 4 * array_size
            offset += 20 * array_size
        offset += 24
        offset += 4

    if offset != len(data):
        raise ValueError(f"Unexpected binary layout while parsing {source_path}")
    return object_id_to_index


def patch_sdc_track_index(source_path: Path, output_path: Path, track_index: int):
    data = bytearray(source_path.read_bytes())
    struct.pack_into("i", data, SDC_TRACK_INDEX_OFFSET, int(track_index))
    output_path.write_bytes(data)


def symlink_or_copy(source_path: Path, output_path: Path):
    try:
        output_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, output_path)


@contextlib.contextmanager
def staged_map_dir(source_path: Path, target_track_index: int | None = None):
    with tempfile.TemporaryDirectory(prefix="drive_policy_eval_") as tmp:
        staged_path = Path(tmp) / "map_000.bin"
        if target_track_index is None:
            symlink_or_copy(source_path, staged_path)
        else:
            patch_sdc_track_index(source_path, staged_path, target_track_index)
        yield Path(tmp)


def env_kwargs(base_args: dict, *, map_dir: Path, num_agents: int, control_mode: str, max_controlled_agents: int):
    kwargs = dict(base_args["env"])
    kwargs.update(
        map_dir=str(map_dir),
        num_maps=1,
        num_agents=int(num_agents),
        init_mode="create_all_valid",
        control_mode=control_mode,
        max_controlled_agents=int(max_controlled_agents),
        report_interval=1,
        render_mode=1,
    )
    return kwargs


@contextlib.contextmanager
def open_env(base_args: dict, *, map_dir: Path, num_agents: int, control_mode: str, max_controlled_agents: int):
    with pushd(REPO_ROOT):
        env = Drive(
            **env_kwargs(
                base_args,
                map_dir=map_dir,
                num_agents=num_agents,
                control_mode=control_mode,
                max_controlled_agents=max_controlled_agents,
            )
        )
        try:
            yield env
        finally:
            env.close()


def case_env_spec(case: dict) -> dict:
    if case["eval_mode"] == "human_replay":
        return {
            "target_track_index": int(case["target_track_index"]),
            "num_agents": 1,
            "control_mode": "control_sdc_only",
            "max_controlled_agents": 1,
        }
    return {
        "target_track_index": None,
        "num_agents": int(case["num_policy_agents"]),
        "control_mode": "control_agents",
        "max_controlled_agents": int(case["num_policy_agents"]),
    }


@contextlib.contextmanager
def open_case_env(base_args: dict, case: dict):
    spec = case_env_spec(case)
    with staged_map_dir(Path(case["source_map_path"]), spec["target_track_index"]) as map_dir:
        with open_env(
            base_args,
            map_dir=map_dir,
            num_agents=spec["num_agents"],
            control_mode=spec["control_mode"],
            max_controlled_agents=spec["max_controlled_agents"],
        ) as env:
            yield env


def create_policy(base_args: dict, env: Drive):
    wrapper = SimpleNamespace(driver_env=env)
    try:
        policy = load_policy(base_args, wrapper, "puffer_drive")
    except RuntimeError as exc:
        if base_args.get("rnn_name") is None and "lstm.weight_ih_l0" in str(exc):
            print("Checkpoint appears recurrent; retrying with rnn_name=Recurrent.")
            base_args["rnn_name"] = "Recurrent"
            base_args["train"]["use_rnn"] = True
            policy = load_policy(base_args, wrapper, "puffer_drive")
        else:
            raise
    return policy.eval()


def init_rnn_state(base_args: dict, env: Drive, policy):
    if not base_args["train"]["use_rnn"]:
        return {}
    hidden_size = policy.hidden_size
    device = base_args["train"]["device"]
    return {
        "lstm_h": torch.zeros(env.num_agents, hidden_size, device=device),
        "lstm_c": torch.zeros(env.num_agents, hidden_size, device=device),
    }


def select_action(policy, obs, env: Drive, state: dict, device: str, deterministic: bool):
    with torch.no_grad():
        obs = torch.as_tensor(obs, device=device)
        logits, _ = policy.forward_eval(obs, state)
        if deterministic:
            if isinstance(logits, torch.distributions.Normal):
                action = logits.mean
            else:
                action = torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)
        else:
            action, _, _ = pufferlib.pytorch.sample_logits(logits)

    action = action.detach().cpu().numpy().reshape(env.action_space.shape)
    if isinstance(logits, torch.distributions.Normal):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    return action


def validate_controlled_agents(env: Drive, expected_agent_ids: list[int]) -> list[int]:
    controlled_ids = [int(agent_id) for agent_id in env.get_global_agent_state()["id"].tolist()]
    if sorted(controlled_ids) != sorted(int(agent_id) for agent_id in expected_agent_ids):
        raise RuntimeError(
            f"Expected controlled agents {expected_agent_ids}, got {controlled_ids} "
            f"for scenario {env.scenario_ids[0]}"
        )
    return controlled_ids


def rollout_policy(
    env: Drive,
    policy,
    base_args: dict,
    *,
    seed: int,
    deterministic: bool,
    expected_agent_ids: list[int],
    render_view=None,
    draw_traces=False,
):
    obs, _ = env.reset(seed=seed)
    controlled_ids = validate_controlled_agents(env, expected_agent_ids)
    state = init_rnn_state(base_args, env, policy)
    final_info = None

    for _ in range(env.episode_length):
        if render_view is not None:
            env.render(view_mode=render_view, draw_traces=draw_traces, env_id=0)

        action = select_action(policy, obs, env, state, base_args["train"]["device"], deterministic)
        obs, _, _, truncations, infos = env.step(action)
        if infos:
            final_info = infos[0]
        if np.asarray(truncations).all():
            break

    if final_info is None:
        raise RuntimeError(f"Rollout for scenario {env.scenario_ids[0]} finished without logs.")
    return final_info, controlled_ids


def discover_cases(base_args: dict, maps: list[MapRecord]):
    discovery_cases = []
    skipped_maps = []

    for map_record in tqdm(maps, desc="Discovering maps", colour="cyan"):
        try:
            discovery_cases.extend(discover_controllable_agents(base_args, map_record))
        except ValueError as exc:
            skipped_maps.append(map_record.map_id)
            print(f"Skipping {map_record.map_name}: {exc}")

    return discovery_cases, skipped_maps


def discover_controllable_agents(base_args: dict, map_record: MapRecord):
    object_id_to_index = read_object_indices(map_record.source_path)

    with staged_map_dir(map_record.source_path) as map_dir:
        with open_env(
            base_args,
            map_dir=map_dir,
            num_agents=binding.MAX_AGENTS,
            control_mode="control_agents",
            max_controlled_agents=binding.MAX_AGENTS,
        ) as env:
            env.reset(seed=0)
            if env.num_envs < 1:
                return []

            state = env.get_global_agent_state()
            first_env_count = env.agent_offsets[1] - env.agent_offsets[0]
            scenario_id = env.scenario_ids[0]
            cases = []

            for rank in range(first_env_count):
                agent_id = int(state["id"][rank])
                if agent_id not in object_id_to_index:
                    raise ValueError(
                        f"Agent id {agent_id} from {map_record.map_name} was not found in the binary map"
                    )

                cases.append(
                    {
                        "map_id": map_record.map_id,
                        "map_name": map_record.map_name,
                        "source_map_path": str(map_record.source_path),
                        "scenario_id": scenario_id,
                        "policy_agent_id": agent_id,
                        "target_track_index": int(object_id_to_index[agent_id]),
                        "controllable_agent_rank": rank,
                        "agent_length": float(state["length"][rank]),
                        "agent_width": float(state["width"][rank]),
                    }
                )

            return cases


def build_self_play_cases(discovery_cases: list[dict]) -> list[dict]:
    grouped = {}
    for case in discovery_cases:
        key = (int(case["map_id"]), str(case["source_map_path"]))
        grouped.setdefault(key, []).append(case)

    self_play_cases = []
    for cases in grouped.values():
        cases = sorted(cases, key=lambda case: int(case["controllable_agent_rank"]))
        expected_agent_ids = [int(case["policy_agent_id"]) for case in cases]
        self_play_cases.append(
            {
                "eval_mode": "self_play",
                "map_id": int(cases[0]["map_id"]),
                "map_name": cases[0]["map_name"],
                "source_map_path": cases[0]["source_map_path"],
                "scenario_id": cases[0]["scenario_id"],
                "expected_agent_ids": expected_agent_ids,
                "num_policy_agents": len(expected_agent_ids),
            }
        )

    return sorted(self_play_cases, key=lambda case: int(case["map_id"]))


def build_eval_cases(discovery_cases: list[dict], cli_args) -> list[dict]:
    cases = []

    if "human_replay" in cli_args.eval_modes:
        cases.extend(
            {
                **case,
                "eval_mode": "human_replay",
                "expected_agent_ids": [int(case["policy_agent_id"])],
                "deterministic": cli_args.deterministic,
            }
            for case in discovery_cases
        )

    if "self_play" in cli_args.eval_modes:
        cases.extend(
            {**case, "deterministic": cli_args.deterministic}
            for case in build_self_play_cases(discovery_cases)
        )

    order = {mode: i for i, mode in enumerate(DEFAULT_EVAL_MODES)}
    return sorted(
        cases,
        key=lambda case: (
            int(case["map_id"]),
            order.get(case["eval_mode"], len(order)),
            int(case.get("controllable_agent_rank", -1)),
        ),
    )


def evaluate_case(base_args: dict, policy, case: dict, run_seed: int):
    with open_case_env(base_args, case) as env:
        final_info, controlled_ids = rollout_policy(
            env,
            policy,
            base_args,
            seed=run_seed,
            deterministic=case["deterministic"],
            expected_agent_ids=list(case["expected_agent_ids"]),
        )
        scenario_id = env.scenario_ids[0]

    row = {k: v for k, v in case.items() if k != "expected_agent_ids"}
    row.update({key: float(final_info[key]) for key in LOG_METRIC_KEYS if key in final_info})
    row["scenario_id"] = scenario_id
    row["expected_agent_ids_json"] = json.dumps(list(case["expected_agent_ids"]))
    row["controlled_agent_ids"] = json.dumps(controlled_ids)
    row["num_policy_agents"] = len(controlled_ids)
    row["had_collision"] = bool(
        row.get("collision_rate", 0.0) > 0.0 or row.get("collisions_per_agent", 0.0) > 0.0
    )
    row["had_valid_collision"] = bool(row.get("collision_rate_valid", 0.0) > 0.0)
    row["seed"] = run_seed
    return row


def run_evaluations(base_args: dict, policy, eval_cases: list[dict], output_dir: Path, cli_args):
    rows = []

    for run_idx, case in enumerate(tqdm(eval_cases, desc="Evaluating agents", colour="green"), start=1):
        run_seed = cli_args.seed + run_idx
        seed_everything(run_seed)
        rows.append(evaluate_case(base_args, policy, case, run_seed))

        if cli_args.checkpoint_every > 0 and run_idx % cli_args.checkpoint_every == 0:
            write_results(pd.DataFrame(rows), output_dir, partial=True)

    df = pd.DataFrame(rows)
    sort_rank = df["controllable_agent_rank"].fillna(-1) if "controllable_agent_rank" in df else -1
    df = df.assign(_sort_rank=sort_rank)
    df = df.sort_values(["map_id", "eval_mode", "_sort_rank"]).drop(columns="_sort_rank").reset_index(drop=True)
    df["random_video_path"] = None
    df["collision_video_path"] = None

    write_results(df, output_dir)
    cleanup_partial_results(output_dir)
    return df


def make_case_mask(df: pd.DataFrame, case: dict):
    mask = (df["map_id"] == int(case["map_id"])) & (df["eval_mode"] == str(case["eval_mode"]))
    if case["eval_mode"] == "human_replay":
        mask &= df["policy_agent_id"] == int(case["policy_agent_id"])
    return mask


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_") or "unknown"


def build_render_filename(case: dict, idx: int):
    name = (
        f"{idx:02d}_mode_{case['eval_mode']}"
        f"__map_{int(case['map_id']):03d}"
        f"__scenario_{sanitize_name(str(case['scenario_id']))}"
    )
    if case["eval_mode"] == "human_replay":
        name += f"__agent_{int(case['policy_agent_id'])}"
    else:
        name += f"__agents_{int(case['num_policy_agents'])}"
    return f"{name}.mp4"


def render_case(base_args: dict, policy, case: dict, output_path: Path, render_view, draw_traces: bool):
    render_file = REPO_ROOT / f"{case['scenario_id']}.mp4"
    if render_file.exists():
        render_file.unlink()

    try:
        seed_everything(int(case["seed"]))
        with open_case_env(base_args, case) as env:
            rollout_policy(
                env,
                policy,
                base_args,
                seed=int(case["seed"]),
                deterministic=bool(case["deterministic"]),
                expected_agent_ids=json.loads(case["expected_agent_ids_json"]),
                render_view=render_view,
                draw_traces=draw_traces,
            )

        if not render_file.exists():
            if case["eval_mode"] == "human_replay":
                detail = f"agent {case['policy_agent_id']}"
            else:
                detail = f"{int(case['num_policy_agents'])} self-play agents"
            raise RuntimeError(f"No mp4 was produced while rendering {case['scenario_id']} ({detail})")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(render_file), output_path)
        return output_path
    finally:
        if render_file.exists():
            render_file.unlink()


def maybe_render_videos(base_args: dict, policy, df: pd.DataFrame, output_dir: Path, cli_args):
    if cli_args.no_render_videos:
        return df, 0, 0
    if shutil.which("ffmpeg") is None:
        print("Skipping video rendering because ffmpeg is not installed.")
        return df, 0, 0

    render_view = RENDER_VIEW_MAP[cli_args.render_view]
    random_cases, collision_cases = choose_render_cases(df, cli_args)
    random_rendered = 0
    collision_rendered = 0

    for category, cases in [("random", random_cases), ("collision", collision_cases)]:
        for idx, case in enumerate(tqdm(cases, desc=f"Rendering {category} videos", colour="magenta"), start=1):
            output_path = output_dir / "videos" / case["eval_mode"] / category / build_render_filename(case, idx)
            render_case(base_args, policy, case, output_path, render_view, cli_args.draw_traces)

            column = "random_video_path" if category == "random" else "collision_video_path"
            df.loc[make_case_mask(df, case), column] = str(output_path.relative_to(output_dir))
            if category == "random":
                random_rendered += 1
            else:
                collision_rendered += 1

    write_results(df, output_dir)
    return df, random_rendered, collision_rendered


def choose_render_cases(df: pd.DataFrame, cli_args):
    rng = random.Random(cli_args.seed)
    random_cases = rng.sample(df.to_dict("records"), min(cli_args.num_random_videos, len(df)))

    collision_cases = df[df["had_collision"]].to_dict("records")
    collision_cases = rng.sample(collision_cases, min(cli_args.num_collision_videos, len(collision_cases)))
    return random_cases, collision_cases


def write_results(df: pd.DataFrame, output_dir: Path, partial: bool = False):
    stem = "results.partial" if partial else "results"
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    df.to_pickle(output_dir / f"{stem}.pkl")

    if not partial and "eval_mode" in df:
        for eval_mode in sorted(df["eval_mode"].dropna().unique()):
            mode_df = df[df["eval_mode"] == eval_mode].reset_index(drop=True)
            mode_df.to_csv(output_dir / f"{eval_mode}_results.csv", index=False)
            mode_df.to_pickle(output_dir / f"{eval_mode}_results.pkl")


def cleanup_partial_results(output_dir: Path):
    for path in [output_dir / "results.partial.csv", output_dir / "results.partial.pkl"]:
        if path.exists():
            path.unlink()


def metric_summary(df: pd.DataFrame) -> dict:
    summary = {
        "num_runs": int(len(df)),
        "num_collisions": int(df["had_collision"].sum()) if "had_collision" in df else 0,
        "num_valid_collisions": int(df["had_valid_collision"].sum()) if "had_valid_collision" in df else 0,
    }
    for column in [*LOG_METRIC_KEYS, "had_collision", "had_valid_collision"]:
        if column in df:
            summary[f"mean/{column}"] = float(df[column].mean())
    return summary


def build_summary(df: pd.DataFrame, maps: list[MapRecord], skipped_maps: list[int], output_dir: Path, cli_args):
    summary = {
        "policy_path": str(cli_args.policy_path),
        "map_dir": str(cli_args.map_dir),
        "output_dir": str(output_dir),
        "eval_modes": list(cli_args.eval_modes),
        "num_maps_requested": len(maps),
        "num_maps_skipped": len(skipped_maps),
        "skipped_map_ids": skipped_maps,
        **metric_summary(df),
        "modes": {},
    }

    if "eval_mode" in df:
        for eval_mode in sorted(df["eval_mode"].dropna().unique()):
            summary["modes"][eval_mode] = metric_summary(df[df["eval_mode"] == eval_mode])

    return summary


if __name__ == "__main__":
    main()
