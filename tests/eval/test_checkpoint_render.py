import base64
import copy
import dataclasses
import inspect
import json
import random
import re
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import pufferlib
from pufferlib import pufferl, replay_format, replay_video
from pufferlib.config_schema import normalize_puffer_drive_config, validate_puffer_drive_config
from pufferlib.ocean.evaluation_utils import eval_replay as drive_eval_replay
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from test_eval import (
    CARLA_MAP_DIR,
    REPLAY_MAP_DIR,
    SEED,
    TRAIN_AGENTS_PER_ENV,
    TRAIN_ENV_COUNT,
    TRAIN_EPOCH_COUNT,
    TRAIN_HORIZON,
    RecordingLogger,
    ZeroPolicy,
    _assert_nested_exact,
    _load_config,
    _seed_training,
    _set_small_observation_config,
    _training_args,
)


RENDER_BENCHMARK = "carla_render"
GIGAFLOW_BENCHMARK = "carla_gigaflow"
REPLAY_BENCHMARK = "replay_single"
MISSING_MAPS_BENCHMARK = "missing_maps"
TRAINING_EVAL_BENCHMARK = "training_eval"
RENDER_VIEWS = ["world", "bev", "agent"]
RENDER_SCENARIO_COUNT = 2
VALIDATOR_SCENARIO_LENGTH = 32
VALIDATOR_EVAL_AGENT_COUNT = 16
TRAINING_MAX_AGENTS_PER_ENV = 8
GIGAFLOW_MAX_AGENTS_PER_ENV = 6
EVAL_READY_MINIBATCH_SIZE = 2048
E2E_SCENARIO_LENGTH = 32
E2E_EPOCH = 5
E2E_GLOBAL_STEP = 1234
E2E_LOG_STEP = 4321
TRAIN_RENDER_SCENARIO_LENGTH = 16
TRIGGER_INTERVAL_CHECKPOINTS = 2
EXPECTED_TRIGGER_EPOCHS = [2, 4, 6, 7]
EXACT_STATE_INTERVAL_CHECKPOINTS = 4
EXPECTED_EXACT_STATE_RENDER_EPOCHS = [4, 7]
STEPS_PER_EPOCH = TRAIN_ENV_COUNT * TRAIN_AGENTS_PER_ENV * TRAIN_HORIZON
DISCRETE_CLASSIC_ACTION_COUNT = 63
VIDEO_WAIT_SECONDS = 900
KILLED_RETURN_CODE = -9
FFMPEG_ENCODERS_WITH_X264 = (
    "Encoders:\n V..... = Video\n ------\n V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC (codec h264)\n"
    " V....D libx264rgb           libx264 H.264 / AVC / MPEG-4 AVC RGB (codec h264)\n"
)
FFMPEG_ENCODERS_WITHOUT_X264 = (
    "Encoders:\n V..... = Video\n ------\n V....D libx264rgb           H.264 RGB encoder (codec h264)\n"
    " V....D mpeg4                MPEG-4 part 2\n"
)
PAYLOAD_CHUNK_PATTERN = re.compile(r'<script type="application/octet-stream" class="payload-chunk">([^<]*)</script>')

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)


def _write_render_benchmarks(path):
    path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "eval_mode": 1,
                    "compute_eval_metrics": True,
                    "termination_mode": False,
                    "obs_dropout_lane": 0.0,
                    "obs_dropout_boundary": 0.0,
                },
                "benchmarks": [
                    {
                        "name": TRAINING_EVAL_BENCHMARK,
                        "seed": SEED,
                        "num_scenarios": 2,
                        "env": {
                            "simulation_mode": "gigaflow",
                            "map_dir": str(CARLA_MAP_DIR),
                            "num_maps": 2,
                            "scenario_length": TRAIN_HORIZON,
                            "max_agents_per_env": TRAIN_AGENTS_PER_ENV,
                            "control_mode": "control_vehicles",
                            "use_neighbor_cache": True,
                        },
                    },
                    {
                        "name": RENDER_BENCHMARK,
                        "seed": SEED,
                        "num_scenarios": 8,
                        "env": {
                            "simulation_mode": "gigaflow",
                            "map_dir": str(CARLA_MAP_DIR),
                            "num_maps": 2,
                            "eval_training_render": True,
                        },
                    },
                    {
                        "name": GIGAFLOW_BENCHMARK,
                        "seed": SEED,
                        "num_scenarios": 4,
                        "env": {
                            "simulation_mode": "gigaflow",
                            "map_dir": str(CARLA_MAP_DIR),
                            "num_maps": 2,
                            "scenario_length": 64,
                            "min_agents_per_env": 1,
                            "max_agents_per_env": GIGAFLOW_MAX_AGENTS_PER_ENV,
                            "control_mode": "control_vehicles",
                            "use_neighbor_cache": True,
                            "max_scenarios_per_batch": None,
                        },
                    },
                    {
                        "name": REPLAY_BENCHMARK,
                        "seed": SEED,
                        "num_scenarios": 2,
                        "env": {
                            "simulation_mode": "replay",
                            "map_dir": str(REPLAY_MAP_DIR),
                            "num_maps": 1,
                            "scenario_length": 64,
                            "control_mode": "control_sdc_only",
                            "max_scenarios_per_batch": 16,
                        },
                    },
                    {
                        "name": MISSING_MAPS_BENCHMARK,
                        "seed": SEED,
                        "num_scenarios": 2,
                        "env": {
                            "simulation_mode": "gigaflow",
                            "map_dir": str(path.parent / "no_such_maps"),
                            "num_maps": 2,
                            "scenario_length": 64,
                            "max_agents_per_env": TRAINING_MAX_AGENTS_PER_ENV,
                            "control_mode": "control_vehicles",
                        },
                    },
                ],
            },
            sort_keys=False,
        )
    )
    return path


def _render_section(**overrides):
    section = {
        "interval_checkpoints": TRIGGER_INTERVAL_CHECKPOINTS,
        "views": list(RENDER_VIEWS),
        "benchmark": RENDER_BENCHMARK,
        "num_scenarios": RENDER_SCENARIO_COUNT,
        "scenario_length": VALIDATOR_SCENARIO_LENGTH,
    }
    section.update(overrides)
    return section


def _eval_ready_args(benchmark_config_path, eval_agent_count):
    args = _load_config()
    args["train"].update(
        {
            "seed": SEED,
            "device": "cpu",
            "compile": False,
            "torch_deterministic": True,
            "minibatch_size": EVAL_READY_MINIBATCH_SIZE,
            "max_minibatch_size": EVAL_READY_MINIBATCH_SIZE,
        }
    )
    args["vec"].update({"seed": SEED, "num_envs": 2, "num_workers": 2})
    args["env"].update(
        {
            "num_agents": TRAINING_MAX_AGENTS_PER_ENV,
            "min_agents_per_env": TRAINING_MAX_AGENTS_PER_ENV,
            "max_agents_per_env": TRAINING_MAX_AGENTS_PER_ENV,
            "num_maps": 2,
            "map_dir": str(CARLA_MAP_DIR),
            "use_map_cache": True,
            "scenario_length": 64,
            "resample_frequency": 64,
            "action_type": "discrete",
            "dynamics_model": "classic",
        }
    )
    _set_small_observation_config(args)
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": RENDER_BENCHMARK,
            "num_agents": eval_agent_count,
            "render_scenarios": False,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": False,
            "keep_zlib_replays": False,
            "output_name": None,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["tb"] = False
    return args


def _validator_args(tmp_path, **render_overrides):
    benchmark_config_path = _write_render_benchmarks(tmp_path / "render_benchmarks.yaml")
    args = _eval_ready_args(benchmark_config_path, VALIDATOR_EVAL_AGENT_COUNT)
    args["render"] = _render_section(**render_overrides)
    return args


def _normalize_and_validate_render(args):
    normalized = normalize_puffer_drive_config(copy.deepcopy(args), "training")
    validate_puffer_drive_config(normalized, "training")
    return drive_benchmark.validate_checkpoint_render_config(normalized)


def _install_ffmpeg_probe(monkeypatch, stdout=FFMPEG_ENCODERS_WITH_X264, returncode=0, missing=False):
    real_run = subprocess.run
    probe_commands = []

    def fake_run(command, *args, **kwargs):
        if isinstance(command, str) or list(command)[:1] != ["ffmpeg"]:
            return real_run(command, *args, **kwargs)
        probe_commands.append(list(command))
        if missing:
            raise FileNotFoundError(2, "No such file or directory", "ffmpeg")
        text_mode = bool(kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding"))
        output = stdout if text_mode else stdout.encode()
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, command, output=output, stderr=output[:0])
        return subprocess.CompletedProcess(command, returncode, stdout=output, stderr=output[:0])

    monkeypatch.setattr(drive_benchmark.subprocess, "run", fake_run)
    return probe_commands


def _view_names(views):
    return [getattr(view, "value", view) for view in views]


def test_default_yaml_render_section_is_disabled():
    args = _load_config()
    assert args["render"] == {
        "interval_checkpoints": None,
        "views": ["world", "bev", "agent"],
        "benchmark": "carla_render",
        "num_scenarios": 4,
        "scenario_length": 300,
    }
    normalized = normalize_puffer_drive_config(args, "training")
    assert normalized["render"]["interval_checkpoints"] is None
    assert _view_names(normalized["render"]["views"]) == ["world", "bev", "agent"]


def test_schema_accepts_a_null_or_absent_render_section():
    args = _load_config()
    args["render"] = None
    normalize_puffer_drive_config(copy.deepcopy(args), "training")
    del args["render"]
    normalize_puffer_drive_config(args, "training")


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("interval_checkpoints", 0),
        ("interval_checkpoints", -2),
        ("interval_checkpoints", True),
        ("num_scenarios", 0),
        ("num_scenarios", False),
        ("scenario_length", 0),
        ("views", ["world", "top"]),
    ],
)
def test_schema_rejects_bad_render_fields(field_name, bad_value):
    args = _load_config()
    args["render"][field_name] = bad_value
    with pytest.raises(pufferlib.APIUsageError):
        normalized = normalize_puffer_drive_config(args, "training")
        validate_puffer_drive_config(normalized, "training")


def test_enabled_render_requires_an_eval_section():
    args = _load_config()
    args["render"]["interval_checkpoints"] = 1
    args["eval"] = None
    with pytest.raises(pufferlib.APIUsageError):
        validate_puffer_drive_config(normalize_puffer_drive_config(args, "training"), "training")


def test_validator_is_disabled_without_a_render_interval(tmp_path, monkeypatch):
    probe_commands = _install_ffmpeg_probe(monkeypatch)
    args = _validator_args(tmp_path)

    args["render"] = None
    assert drive_benchmark.validate_checkpoint_render_config(args) is False
    args["render"] = _render_section(interval_checkpoints=None)
    assert drive_benchmark.validate_checkpoint_render_config(args) is False
    assert _normalize_and_validate_render(args) is False
    del args["render"]
    assert drive_benchmark.validate_checkpoint_render_config(args) is False
    assert probe_commands == []


@pytest.mark.parametrize(
    "benchmark_name, scenario_length",
    [
        (RENDER_BENCHMARK, VALIDATOR_SCENARIO_LENGTH),
        (GIGAFLOW_BENCHMARK, None),
        (GIGAFLOW_BENCHMARK, 48),
        (REPLAY_BENCHMARK, None),
    ],
)
def test_validator_accepts_valid_render_configs(tmp_path, monkeypatch, benchmark_name, scenario_length):
    probe_commands = _install_ffmpeg_probe(monkeypatch)
    args = _validator_args(tmp_path, benchmark=benchmark_name, scenario_length=scenario_length)

    assert _normalize_and_validate_render(args) is True
    assert probe_commands == [["ffmpeg", "-hide_banner", "-encoders"]]


INVALID_RENDER_CASES = {
    "empty_views": (dict(views=[]), None),
    "duplicate_views": (dict(views=["world", "bev", "world"]), None),
    "comma_separated_benchmarks": (dict(benchmark=f"{RENDER_BENCHMARK},{GIGAFLOW_BENCHMARK}"), None),
    "unknown_benchmark": (dict(benchmark="no_such_benchmark"), None),
    "training_render_without_length": (dict(benchmark=RENDER_BENCHMARK, scenario_length=None), "scenario_length"),
    "replay_with_length": (dict(benchmark=REPLAY_BENCHMARK, scenario_length=32), r"scenario_length[\s\S]*null"),
    "unresolvable_benchmark_maps": (dict(benchmark=MISSING_MAPS_BENCHMARK, scenario_length=None), None),
}


@pytest.mark.parametrize("case_name", sorted(INVALID_RENDER_CASES))
def test_validator_rejects_invalid_render_configs(tmp_path, monkeypatch, case_name):
    _install_ffmpeg_probe(monkeypatch)
    render_overrides, message_pattern = INVALID_RENDER_CASES[case_name]
    args = _validator_args(tmp_path, **render_overrides)

    with pytest.raises(pufferlib.APIUsageError) as raised:
        _normalize_and_validate_render(args)
    if message_pattern is not None:
        assert re.search(message_pattern, str(raised.value)), str(raised.value)


@pytest.mark.parametrize(
    "probe_kwargs",
    [dict(missing=True), dict(returncode=1), dict(stdout=FFMPEG_ENCODERS_WITHOUT_X264)],
    ids=["ffmpeg_missing", "ffmpeg_fails", "no_libx264_token"],
)
def test_validator_requires_ffmpeg_with_libx264(tmp_path, monkeypatch, probe_kwargs):
    probe_commands = _install_ffmpeg_probe(monkeypatch, **probe_kwargs)
    args = _validator_args(tmp_path)

    with pytest.raises(pufferlib.APIUsageError):
        _normalize_and_validate_render(args)
    assert probe_commands == [["ffmpeg", "-hide_banner", "-encoders"]]


def test_checkpoint_render_args_force_capture_and_drop_the_checkpoint(tmp_path):
    args = _validator_args(tmp_path)
    args["load_model_path"] = "experiments/old/models/model_puffer_drive_000100.pt"
    args["eval"].update(
        {
            "render_filter": "all_infractions",
            "failure_replay_csv": "failures.csv",
            "output_name": "named",
            "keep_zlib_replays": False,
            "capture_observations": False,
            "render_scenarios": False,
        }
    )
    original = copy.deepcopy(args)

    render_args = drive_benchmark.checkpoint_render_args(args)

    assert args == original
    assert render_args["eval"]["render_scenarios"] is True
    assert render_args["eval"]["capture_observations"] is True
    assert render_args["eval"]["keep_zlib_replays"] is True
    assert render_args["eval"]["render_filter"] is None
    assert render_args["eval"]["failure_replay_csv"] is None
    assert render_args["eval"]["output_name"] is None
    assert render_args["load_model_path"] is None
    changed_eval_keys = {
        "render_scenarios",
        "capture_observations",
        "keep_zlib_replays",
        "render_filter",
        "failure_replay_csv",
        "output_name",
    }
    assert {k: v for k, v in render_args["eval"].items() if k not in changed_eval_keys} == {
        k: v for k, v in original["eval"].items() if k not in changed_eval_keys
    }
    assert {k: v for k, v in render_args.items() if k not in ("eval", "load_model_path")} == {
        k: v for k, v in original.items() if k not in ("eval", "load_model_path")
    }
    render_args["env"]["num_agents"] = -1
    render_args["render"]["views"].append("extra")
    assert args == original


def test_training_benchmark_base_args_copies_args_and_carries_dropout(tmp_path):
    args = _validator_args(tmp_path)
    args["env"]["obs_dropout_lane"] = 0.25
    args["env"]["obs_dropout_boundary"] = 0.5
    environment_config, _ = drive_benchmark.load_benchmark_config(args["eval"]["benchmark_config"], RENDER_BENCHMARK)

    base_args = drive_benchmark.training_benchmark_base_args(args, environment_config)

    assert base_args == args
    assert base_args is not args and base_args["env"] is not args["env"]
    assert environment_config["obs_dropout_lane"] == 0.25
    assert environment_config["obs_dropout_boundary"] == 0.5


@pytest.mark.parametrize(
    "benchmark_name, scenario_length, extra_overrides",
    [
        (
            RENDER_BENCHMARK,
            VALIDATOR_SCENARIO_LENGTH,
            [
                f"env.scenario_length={VALIDATOR_SCENARIO_LENGTH}",
                f"env.resample_frequency={VALIDATOR_SCENARIO_LENGTH}",
                f"eval.num_agents={TRAINING_MAX_AGENTS_PER_ENV}",
            ],
        ),
        (GIGAFLOW_BENCHMARK, None, [f"eval.num_agents={GIGAFLOW_MAX_AGENTS_PER_ENV}"]),
        (
            GIGAFLOW_BENCHMARK,
            48,
            ["env.scenario_length=48", "env.resample_frequency=48", f"eval.num_agents={GIGAFLOW_MAX_AGENTS_PER_ENV}"],
        ),
        (REPLAY_BENCHMARK, None, []),
    ],
)
def test_checkpoint_render_overrides_compose_the_render_run(tmp_path, benchmark_name, scenario_length, extra_overrides):
    args = _validator_args(tmp_path, benchmark=benchmark_name, scenario_length=scenario_length)
    environment_config, benchmarks = drive_benchmark.load_benchmark_config(
        args["eval"]["benchmark_config"], benchmark_name
    )

    overrides = drive_benchmark.checkpoint_render_overrides(args, benchmarks[0], environment_config)

    expected = [
        f"num_scenarios={RENDER_SCENARIO_COUNT}",
        "env.termination_mode=false",
        "train.compile=false",
        *extra_overrides,
    ]
    assert sorted(overrides) == sorted(expected)


def test_render_api_surface():
    assert pufferl.RENDER_WAIT_TIMEOUT_SECONDS == 480
    assert [field.name for field in dataclasses.fields(pufferl.CheckpointRender)] == [
        "process",
        "video_dir",
        "log_path",
        "epoch",
    ]
    assert inspect.signature(pufferl.eval).parameters["overrides"].default is None
    assert list(inspect.signature(pufferl.run_checkpoint_render).parameters) == [
        "env_name",
        "args",
        "policy",
        "epoch",
        "global_step",
        "run_dir",
    ]


class _FakeRenderProcess:
    def __init__(self, events, epoch, behaviour):
        self.events = events
        self.epoch = epoch
        self.behaviour = behaviour
        self.killed = False
        self.returncode = 0 if behaviour == "exits_immediately" else None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.events.append(("wait", self.epoch, timeout, self.killed))
        if self.behaviour == "hangs" and not self.killed:
            raise subprocess.TimeoutExpired(cmd="pufferlib.replay_video", timeout=timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.events.append(("kill", self.epoch))
        self.killed = True
        self.returncode = KILLED_RETURN_CODE


class _EventLogger(RecordingLogger):
    def __init__(self, run_id, events):
        super().__init__(run_id)
        self.events = events

    def close(self, model_path, early_stop):
        self.events.append(("close",))
        super().close(model_path, early_stop)


def _train_with_render(tmp_path, benchmark_config_path, run_id, render_section, logger):
    _seed_training()
    args = _training_args(tmp_path, benchmark_config_path, evaluation_enabled=False)
    args["train"]["checkpoint_interval"] = 1
    run_dir = tmp_path / run_id
    args["train"]["data_dir"] = str(run_dir)
    args["run_name"] = run_id
    args["render"] = render_section
    pufferl.train("puffer_drive", args=args, logger=logger)
    trainer_state = torch.load(run_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    return run_dir, trainer_state


def _event_index(events, kind, epoch):
    return next(idx for idx, event in enumerate(events) if event[0] == kind and event[1] == epoch)


@pytest.mark.parametrize("behaviour", ["exits_immediately", "runs_until_waited", "hangs"])
def test_training_renders_every_n_checkpoints_and_logs_before_close(tmp_path, monkeypatch, behaviour):
    _install_ffmpeg_probe(monkeypatch)
    benchmark_config_path = _write_render_benchmarks(tmp_path / "render_benchmarks.yaml")
    events = []
    failing_epochs = {4} if behaviour == "exits_immediately" else set()

    def fake_run_checkpoint_render(env_name, args, policy, epoch, global_step, run_dir):
        events.append(("render", epoch, global_step, run_dir))
        assert policy is not None
        if epoch in failing_epochs:
            return None
        return pufferl.CheckpointRender(
            process=_FakeRenderProcess(events, epoch, behaviour),
            video_dir=str(Path(run_dir) / f"videos_{epoch}"),
            log_path=str(Path(run_dir) / f"render_{epoch}.log"),
            epoch=epoch,
        )

    def fake_log_checkpoint_render(render, logger, global_step):
        events.append(("log", render.epoch, global_step, render.process.returncode))

    monkeypatch.setattr(pufferl, "run_checkpoint_render", fake_run_checkpoint_render)
    monkeypatch.setattr(pufferl, "log_checkpoint_render", fake_log_checkpoint_render)
    previous_thread_count = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run_dir, trainer_state = _train_with_render(
            tmp_path,
            benchmark_config_path,
            f"trigger_{behaviour}",
            _render_section(scenario_length=TRAIN_RENDER_SCENARIO_LENGTH),
            _EventLogger(f"trigger_{behaviour}", events),
        )
    finally:
        torch.set_num_threads(previous_thread_count)

    assert trainer_state["epoch"] == TRAIN_EPOCH_COUNT
    render_events = [event for event in events if event[0] == "render"]
    assert [event[1] for event in render_events] == EXPECTED_TRIGGER_EPOCHS
    for _, epoch, global_step, render_run_dir in render_events:
        assert global_step == epoch * STEPS_PER_EPOCH
        assert Path(render_run_dir) == run_dir
    logged_epochs = [event[1] for event in events if event[0] == "log"]
    expected_logged_epochs = [epoch for epoch in EXPECTED_TRIGGER_EPOCHS if epoch not in failing_epochs]
    assert logged_epochs == expected_logged_epochs
    close_index = next(idx for idx, event in enumerate(events) if event[0] == "close")
    expected_return_code = KILLED_RETURN_CODE if behaviour == "hangs" else 0
    for epoch in expected_logged_epochs:
        log_index = _event_index(events, "log", epoch)
        assert _event_index(events, "render", epoch) < log_index < close_index
        later_renders = [later for later in EXPECTED_TRIGGER_EPOCHS if later > epoch]
        if later_renders:
            assert log_index < _event_index(events, "render", later_renders[0])
        log_event = events[log_index]
        assert log_event[2] >= epoch * STEPS_PER_EPOCH
        assert log_event[3] == expected_return_code
    unkilled_waits = [event for event in events if event[0] == "wait" and not event[3]]
    assert all(event[2] == pufferl.RENDER_WAIT_TIMEOUT_SECONDS for event in unkilled_waits)
    killed_epochs = [event[1] for event in events if event[0] == "kill"]
    if behaviour == "hangs":
        assert killed_epochs == expected_logged_epochs
    else:
        assert killed_epochs == []
    if behaviour == "runs_until_waited":
        assert sorted({event[1] for event in unkilled_waits}) == expected_logged_epochs


def _training_render_directory(epoch):
    return f"epoch_{epoch:06d}_step_{epoch * STEPS_PER_EPOCH}"


def _replay_from_html(html_path):
    payload = "".join(PAYLOAD_CHUNK_PATTERN.findall(Path(html_path).read_text()))
    return replay_format.decode_interactive_replay(base64.b64decode(payload))


def _ffprobe_frame_count(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _rendered_pages(render_output_dir):
    return sorted(path for path in (render_output_dir / "rendered_replays").glob("*.html") if path.name != "index.html")


def _assert_videos_match_replays(render_output_dir, views):
    manifest = json.loads((render_output_dir / "videos" / "manifest.json").read_text())
    assert manifest["failed"] == []
    pages = _rendered_pages(render_output_dir)
    entries = {(video["stem"], video["view"]): video for video in manifest["videos"]}
    assert set(entries) == {(page.stem, view) for page in pages for view in views}
    for page in pages:
        header, chunks = _replay_from_html(page)
        tracked_frame_count = replay_video.select_tracked_agent(chunks["agent_i32"])[2]
        assert tracked_frame_count <= header["frames"]
        expected_frames = {"world": header["frames"], "bev": tracked_frame_count, "agent": tracked_frame_count}
        for view in views:
            video = entries[(page.stem, view)]
            expected_path = render_output_dir / "videos" / f"{page.stem}__{view}.mp4"
            assert Path(video["path"]).resolve() == expected_path.resolve()
            assert video["frame_count"] == expected_frames[view]
            assert _ffprobe_frame_count(video["path"]) == expected_frames[view]
    return manifest


@requires_ffmpeg
def test_checkpoint_renders_leave_the_training_state_unchanged(tmp_path):
    benchmark_config_path = _write_render_benchmarks(tmp_path / "render_benchmarks.yaml")
    previous_thread_count = torch.get_num_threads()
    previous_deterministic_setting = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        _, baseline_state = _train_with_render(
            tmp_path, benchmark_config_path, "without_render", None, RecordingLogger("without_render")
        )
        rendered_dir, rendered_state = _train_with_render(
            tmp_path,
            benchmark_config_path,
            "with_render",
            _render_section(
                interval_checkpoints=EXACT_STATE_INTERVAL_CHECKPOINTS, scenario_length=TRAIN_RENDER_SCENARIO_LENGTH
            ),
            RecordingLogger("with_render"),
        )
    finally:
        torch.set_num_threads(previous_thread_count)
        torch.use_deterministic_algorithms(previous_deterministic_setting, warn_only=True)

    for state_key in (
        "policy_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "global_step",
        "agent_step",
        "epoch",
        "update",
        "rng_state",
    ):
        _assert_nested_exact(rendered_state[state_key], baseline_state[state_key])
    render_root = rendered_dir / "render" / RENDER_BENCHMARK
    expected_directories = [_training_render_directory(epoch) for epoch in EXPECTED_EXACT_STATE_RENDER_EPOCHS]
    assert sorted(path.name for path in render_root.iterdir() if path.is_dir()) == expected_directories
    for directory_name in expected_directories:
        render_output_dir = render_root / directory_name
        assert (render_output_dir / "rendered_replays" / "index.html").is_file()
        assert len(_rendered_pages(render_output_dir)) == RENDER_SCENARIO_COUNT
        _assert_videos_match_replays(render_output_dir, RENDER_VIEWS)
        assert not (render_output_dir / drive_eval_replay.ZLIB_REPLAY_DIR_NAME).exists()


def _e2e_args(benchmark_config_path):
    args = _eval_ready_args(benchmark_config_path, TRAINING_MAX_AGENTS_PER_ENV)
    args["render"] = _render_section(interval_checkpoints=1, scenario_length=E2E_SCENARIO_LENGTH)
    return args


@pytest.fixture(scope="module")
def checkpoint_render(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoint_render")
    benchmark_config_path = _write_render_benchmarks(root / "render_benchmarks.yaml")
    args = _e2e_args(benchmark_config_path)
    run_dir = root / "run"
    run_dir.mkdir()
    policy = ZeroPolicy(action_count=DISCRETE_CLASSIC_ACTION_COUNT)
    policy.train(True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng_before = pufferl.capture_rng_state()

    render = pufferl.run_checkpoint_render(
        env_name="puffer_drive",
        args=args,
        policy=policy,
        epoch=E2E_EPOCH,
        global_step=E2E_GLOBAL_STEP,
        run_dir=str(run_dir),
    )

    rng_after = pufferl.capture_rng_state()
    policy_training_after = policy.training
    return_code = None
    if render is not None:
        try:
            return_code = render.process.wait(timeout=VIDEO_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            render.process.kill()
            return_code = render.process.wait()
    return {
        "args": args,
        "run_dir": run_dir,
        "render": render,
        "return_code": return_code,
        "rng_before": rng_before,
        "rng_after": rng_after,
        "policy_training_after": policy_training_after,
        "output_dir": run_dir / "render" / RENDER_BENCHMARK / f"epoch_{E2E_EPOCH:06d}_step_{E2E_GLOBAL_STEP}",
    }


def _render_log(checkpoint_render):
    log_path = Path(checkpoint_render["render"].log_path)
    return log_path.read_text() if log_path.is_file() else "<no render.log>"


def test_checkpoint_render_writes_html_pages_under_the_run_dir(checkpoint_render):
    assert checkpoint_render["render"] is not None
    output_dir = checkpoint_render["output_dir"]
    assert (output_dir / "rendered_replays" / "index.html").is_file()
    pages = _rendered_pages(output_dir)
    assert len(pages) == RENDER_SCENARIO_COUNT
    for page in pages:
        header, chunks = _replay_from_html(page)
        assert header["replay_format_version"] == replay_format.REPLAY_FORMAT_VERSION
        assert header["frames"] == E2E_SCENARIO_LENGTH
        assert "obs" in chunks
        assert chunks["obs"].dtype == np.float16
        assert header["active_count"] > 1


def test_checkpoint_render_restores_rng_and_policy_mode(checkpoint_render):
    _assert_nested_exact(checkpoint_render["rng_after"], checkpoint_render["rng_before"])
    assert checkpoint_render["policy_training_after"] is True


def test_checkpoint_render_returns_the_background_video_job(checkpoint_render):
    render = checkpoint_render["render"]
    assert isinstance(render, pufferl.CheckpointRender)
    assert render.epoch == E2E_EPOCH
    assert Path(render.video_dir).resolve() == (checkpoint_render["output_dir"] / "videos").resolve()
    assert isinstance(render.process, subprocess.Popen)
    assert Path(render.log_path).is_file()


@requires_ffmpeg
def test_checkpoint_render_videos_cover_every_scenario_and_view(checkpoint_render):
    assert checkpoint_render["return_code"] == 0, _render_log(checkpoint_render)
    output_dir = checkpoint_render["output_dir"]
    manifest = _assert_videos_match_replays(output_dir, RENDER_VIEWS)
    assert len(manifest["videos"]) == RENDER_SCENARIO_COUNT * len(RENDER_VIEWS)
    assert not (output_dir / drive_eval_replay.ZLIB_REPLAY_DIR_NAME).exists()


class _FakeVideo:
    def __init__(self, data_or_path=None, caption=None, fps=None, format=None, **kwargs):
        self.path = data_or_path
        self.caption = caption
        self.format = format


class _WandbStubLogger:
    def __init__(self):
        self.wandb = types.SimpleNamespace(Video=_FakeVideo)
        self.calls = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))


@requires_ffmpeg
def test_log_checkpoint_render_sends_every_view_to_wandb(checkpoint_render):
    assert checkpoint_render["return_code"] == 0, _render_log(checkpoint_render)
    manifest = json.loads((checkpoint_render["output_dir"] / "videos" / "manifest.json").read_text())
    logger = _WandbStubLogger()

    pufferl.log_checkpoint_render(checkpoint_render["render"], logger, E2E_LOG_STEP)

    assert len(logger.calls) == 1
    metrics, step = logger.calls[0]
    assert step == E2E_LOG_STEP
    assert set(metrics) == {f"render/{view}" for view in RENDER_VIEWS} | {"render/epoch"}
    assert metrics["render/epoch"] == E2E_EPOCH
    for view in RENDER_VIEWS:
        videos = metrics[f"render/{view}"]
        expected_paths = {Path(video["path"]).resolve() for video in manifest["videos"] if video["view"] == view}
        assert {Path(video.path).resolve() for video in videos} == expected_paths
        stems = {video["stem"] for video in manifest["videos"] if video["view"] == view}
        for video in videos:
            assert isinstance(video, _FakeVideo)
            assert video.format == "mp4"
            assert f"epoch {E2E_EPOCH}" in video.caption
            assert any(stem in video.caption for stem in stems)


def test_log_checkpoint_render_without_wandb_does_not_log(checkpoint_render):
    logger = RecordingLogger("no_wandb")
    pufferl.log_checkpoint_render(checkpoint_render["render"], logger, E2E_LOG_STEP)
    assert logger.calls == []


class _FinishedProcess:
    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def test_log_checkpoint_render_never_raises_on_a_failed_job(tmp_path):
    broken = pufferl.CheckpointRender(
        process=_FinishedProcess(returncode=1),
        video_dir=str(tmp_path / "missing_videos"),
        log_path=str(tmp_path / "missing_render.log"),
        epoch=3,
    )
    corrupt_dir = tmp_path / "corrupt_videos"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("{not json")
    corrupt = pufferl.CheckpointRender(
        process=_FinishedProcess(returncode=0),
        video_dir=str(corrupt_dir),
        log_path=str(tmp_path / "render.log"),
        epoch=4,
    )
    logger = _WandbStubLogger()

    pufferl.log_checkpoint_render(broken, logger, 10)
    pufferl.log_checkpoint_render(corrupt, logger, 11)

    logged_video_keys = [
        key for metrics, _ in logger.calls for key in metrics if key.startswith("render/") and key != "render/epoch"
    ]
    assert logged_video_keys == []


def test_run_checkpoint_render_skips_an_existing_output(checkpoint_render):
    run_dir = checkpoint_render["run_dir"]
    files_before = sorted(str(path) for path in run_dir.rglob("*"))

    second = pufferl.run_checkpoint_render(
        env_name="puffer_drive",
        args=checkpoint_render["args"],
        policy=ZeroPolicy(action_count=DISCRETE_CLASSIC_ACTION_COUNT),
        epoch=E2E_EPOCH,
        global_step=E2E_GLOBAL_STEP,
        run_dir=str(run_dir),
    )

    assert second is None
    assert sorted(str(path) for path in run_dir.rglob("*")) == files_before


def test_run_checkpoint_render_returns_none_and_restores_state_on_failure(tmp_path):
    args = _e2e_args(tmp_path / "missing_benchmarks.yaml")
    policy = ZeroPolicy(action_count=DISCRETE_CLASSIC_ACTION_COUNT)
    policy.train(True)
    rng_before = pufferl.capture_rng_state()

    result = pufferl.run_checkpoint_render(
        env_name="puffer_drive",
        args=args,
        policy=policy,
        epoch=1,
        global_step=STEPS_PER_EPOCH,
        run_dir=str(tmp_path / "run"),
    )

    assert result is None
    _assert_nested_exact(pufferl.capture_rng_state(), rng_before)
    assert policy.training is True


@requires_ffmpeg
def test_log_checkpoint_render_writes_wandb_offline_media(checkpoint_render, tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    assert checkpoint_render["return_code"] == 0, _render_log(checkpoint_render)
    for variable_name, subdirectory in (
        ("WANDB_DIR", "wandb_dir"),
        ("WANDB_CACHE_DIR", "wandb_cache"),
        ("WANDB_CONFIG_DIR", "wandb_config"),
        ("WANDB_DATA_DIR", "wandb_data"),
    ):
        (tmp_path / subdirectory).mkdir()
        monkeypatch.setenv(variable_name, str(tmp_path / subdirectory))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    wandb.init(project="checkpoint-render-test", dir=str(tmp_path / "wandb_dir"), mode="offline")
    logger = types.SimpleNamespace(wandb=wandb, log=lambda logs, step: wandb.log(logs, step=step))
    try:
        pufferl.log_checkpoint_render(checkpoint_render["render"], logger, E2E_LOG_STEP)
    finally:
        wandb.finish()

    media_names = [path.name for path in (tmp_path / "wandb_dir").rglob("media/videos/render/*.mp4")]
    for view in RENDER_VIEWS:
        assert sum(name.startswith(f"{view}_") for name in media_names) == RENDER_SCENARIO_COUNT, media_names
