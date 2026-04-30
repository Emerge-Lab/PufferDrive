import glob
import json
import os
import subprocess
import sys



class BaseEvaluatorSubprocess:
    """Base class for in-training evaluations launched as subprocesses.

    Subclasses implement run() and optionally override render().

    Override configs registered via add_override_configs() are forwarded
    as --section.key value flags to every subprocess invocation.  This
    mirrors the pufferl.py load_config convention so callers never have
    to hand-build CLI flag lists. Subprocess calls puffer eval puffer_drive with corresponding flags set based on override configs.

    Typical usage::

        evaluator = MyEvaluator(config, logger, global_step, run_id)
        evaluator.add_override_configs("env", {"simulation_mode": SIMULATION_REPLAY,
                                                "control_mode": CONTROL_SDC_ONLY})
        evaluator.add_override_configs("eval", {"human_replay_eval": True})
        evaluator.run()
    """

    def __init__(self, config: dict, logger, global_step: int, run_id: str):
        """
        Args:
            config: Full args dict (must include checkpoint_dir, env_name, env, eval).
            logger: wandb run object or None.
            global_step: Current training step (used for wandb logging).
            run_id: Unique training run identifier used to locate checkpoints.
        """
        self.config = config
        self.logger = logger
        self.global_step = global_step
        self.run_id = run_id
        self._overrides: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Override config management
    # ------------------------------------------------------------------

    def add_override_configs(self, section: str, overrides: dict) -> None:
        """Register CLI overrides forwarded to subprocess as --section.key value.

        Keys use underscores; they are converted to hyphens to match the
        pufferl argparse convention (e.g. simulation_mode -> --env.simulation-mode).
        """
        self._overrides.setdefault(section, {}).update(overrides)

    def _override_flags(self) -> list[str]:
        """Flatten stored overrides into a CLI flag list."""
        flags = []
        for section, kvs in self._overrides.items():
            for key, value in kvs.items():
                flags.extend([f"--{section}.{key.replace('_', '-')}", str(value)])
        return flags

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _env_name(self) -> str:
        return self.config.get("env_name") or self.config.get("env", "")

    def _latest_model_path(self) -> str | None:
        """Return the most recently saved checkpoint for this run."""
        checkpoint_dir = os.path.join(
            self.config["checkpoint_dir"], self._env_name(), self.run_id
        )
        candidates = glob.glob(os.path.join(checkpoint_dir, "*.bin"))
        if not candidates:
            return None
        return max(candidates, key=os.path.getctime)

    def _run_subprocess(
        self,
        model_path: str,
        extra_flags: list[str] | None = None,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess:
        """Run pufferl eval in a subprocess with all stored override flags."""
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            self._env_name(),
            "--load-model-path",
            model_path,
        ] + self._override_flags() + (extra_flags or [])
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=os.getcwd()
        )

    def _parse_metrics(
        self, stdout: str, start_marker: str, end_marker: str
    ) -> dict | None:
        """Extract JSON metrics from subprocess stdout between named markers."""
        if start_marker not in stdout or end_marker not in stdout:
            return None
        start = stdout.find(start_marker) + len(start_marker)
        end = stdout.find(end_marker)
        return json.loads(stdout[start:end].strip())

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        map_dir: str,
        scenario_length: int,
        output_dir: str,
        simulation_mode: str = "replay",
        view: str = "topdown_sim",
        **kwargs,
    ) -> None:
        """Render a scenario video via render_scenario.py.

        Applies the eval render overrides common to all evaluators (e.g.
        simulation mode, scenario length) plus any extra kwargs forwarded
        as --key value flags to render_scenario.py.

        Args:
            map_dir: Path to the directory of .bin scenario files.
            scenario_length: Number of simulation steps to render.
            output_dir: Directory where mp4 files are written.
            simulation_mode: "replay" or "gigaflow" (render_scenario.py string form).
            view: Camera view mode passed to render_scenario.py.
            **kwargs: Additional flags forwarded verbatim (underscores → hyphens).
        """
        model_path = self._latest_model_path()
        if not model_path:
            print(f"{self.__class__.__name__}: no model checkpoint found, skipping render")
            return

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/render_scenario.py",
            "--checkpoint", model_path,
            "--map-dir", map_dir,
            "--simulation-mode", simulation_mode,
            "--steps", str(scenario_length),
            "--output-dir", output_dir,
            "--view", view,
        ]
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        label = self.__class__.__name__
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, cwd=os.getcwd()
            )
            if result.returncode != 0:
                print(
                    f"{label}: render failed (exit {result.returncode}):\n"
                    f"{result.stderr[-500:]}"
                )
        except subprocess.TimeoutExpired:
            print(f"{label}: render timed out after 300s")
        except Exception as e:
            print(f"{label}: render error: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HumanReplayEvaluator
# ---------------------------------------------------------------------------

_HUMAN_REPLAY_METRICS_START = "HUMAN_REPLAY_METRICS_START"
_HUMAN_REPLAY_METRICS_END = "HUMAN_REPLAY_METRICS_END"

# Wandb keys for each metric emitted by the C binding in replay mode.
_HUMAN_REPLAY_WANDB_KEYS = {
    "collision_rate": "eval/human_replay_collision_rate",
    "offroad_rate": "eval/human_replay_offroad_rate",
    "completion_rate": "eval/human_replay_completion_rate",
}


class HumanReplayEvaluatorSubprocess(BaseEvaluatorSubprocess):
    """Evaluates the policy in log-replay mode against human trajectories.

    Configures the env subprocess for:
      - simulation_mode
      - control_mode
      - episode_length
      - map_dir (dataset of .bin scenarios to replay)
      - num_agents(1 per map, constrict by map_dir size)
      - disable all other evals

    The subprocess prints HUMAN_REPLAY_METRICS_START/END JSON containing
    collision_rate, offroad_rate, and completion_rate; these are logged to
    wandb under the eval/ namespace.
    """

    def __init__(self, config: dict, logger, global_step: int, run_id: str):
        super().__init__(config, logger, global_step, run_id)
        eval_cfg = config.get("eval", {})

        map_dir = eval_cfg.get("map_dir", "")
        if isinstance(map_dir, str):
            map_dir = map_dir.strip("\"'")
        self.map_dir = map_dir

        self.num_agents = min(eval_cfg.get("human_replay_num_agents"), len([f for f in os.listdir(self.map_dir) if f.endswith(".bin")]))
        self.simulation_mode = eval_cfg.get("human_replay_simulation_mode")
        self.control_mode = eval_cfg.get("human_replay_control_mode")
        self.scenario_length = eval_cfg.get("scenario_length")

        # Register subprocess overrides.
        self.add_override_configs("eval", {
            "wosac_realism_eval": False,
            "human_replay_eval": True,
            "human_replay_num_agents": self.num_agents,
            "human_replay_control_mode": self.control_mode,
        })
        if map_dir:
            self.add_override_configs("eval", {"map_dir": map_dir})
        self.add_override_configs("env", {
            "simulation_mode": self.simulation_mode,
            "control_mode": self.control_mode,
            "episode_length": self.scenario_length,
        })

    def run(self) -> None:
        """Run human replay eval in a subprocess and log results to wandb."""
        model_path = self._latest_model_path()
        if not model_path:
            print("HumanReplayEvaluator: no checkpoint found, skipping.")
            return

        if self.map_dir and not os.path.isdir(self.map_dir):
            print(f"HumanReplayEvaluator: map_dir not found ({self.map_dir}), skipping.")
            return

        print(f"HumanReplayEvaluator: running with map_dir={self.map_dir or '(config default)'}")
        try:
            result = self._run_subprocess(model_path)
        except subprocess.TimeoutExpired:
            print("HumanReplayEvaluator: subprocess timed out after 600s")
            return
        except Exception as e:
            print(f"HumanReplayEvaluator: subprocess error: {type(e).__name__}: {e}")
            return

        if result.returncode != 0:
            print(
                f"HumanReplayEvaluator: subprocess failed (exit {result.returncode}):\n"
                f"{result.stderr[-1000:]}"
            )
            return

        metrics = self._parse_metrics(
            result.stdout, _HUMAN_REPLAY_METRICS_START, _HUMAN_REPLAY_METRICS_END
        )
        if metrics is None:
            print("HumanReplayEvaluator: no metrics found in subprocess output")
            return

        print(f"HumanReplayEvaluator: {metrics}")
        self._log_to_wandb(metrics)

    def _log_to_wandb(self, metrics: dict) -> None:
        if self.logger is None:
            return
        payload = {}
        for metric_key, wandb_key in _HUMAN_REPLAY_WANDB_KEYS.items():
            if metric_key in metrics:
                try:
                    payload[wandb_key] = float(metrics[metric_key])
                except (TypeError, ValueError):
                    pass
        if payload:
            log_fn = getattr(self.logger, "log", None)
            if log_fn is not None:
                log_fn(payload, self.global_step)
