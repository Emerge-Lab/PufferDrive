# PufferDrive policy under carla_garage's CARLA Leaderboard 1.0 (longest6 v1)

## Goal

Evaluate a PufferDrive-trained policy with **carla_garage's unmodified Leaderboard 1.0** pipeline
(`leaderboard_evaluator_local.py`, CARLA 0.9.10.1, `longest6.xml` + `eval_scenarios.json`,
`BENCHMARK=longest6`) — the evaluator, scenarios, scoring and `tools/result_parser.py` that produced
the TransFuser++ longest6 numbers. The Leaderboard 2.0 counterpart (CaRL, CARLA 0.9.15) is
`../leaderboard_agent.py`; both share the same shadow env core.

## Design

Zero changes to carla_garage. The leaderboard imports any agent file via `--agent`; the difference to
the Leaderboard 2.0 agent is the interpreter: the `garage` conda env is **Python 3.7** (the only Python
the CARLA 0.9.10 egg ships for) and PufferDrive needs 3.9+, so one process cannot hold both. The
integration is therefore a client/server pair around the shared core:

- `../shadow_ego.py` — `ShadowEgo`, the carla-free core used by both leaderboards: loads the checkpoint
  and its `config.yaml`, builds the shadow `Drive` env (clean-eval profile, `cosim/arch.py`), calibrates
  the bin frame against CARLA's driving waypoints, maps CARLA lights/stop signs onto the bin, turns the
  leaderboard's target points into route goals, and per tick overwrites ALL shadow agents from the
  snapshot, runs the policy and integrates one dt. All inputs/outputs are CARLA-frame plain data.
- `leaderboard_agent.py` (this directory) — the `AutonomousAgent` for the garage evaluator (entry point
  `PufferAgentLB1`), Python 3.7, imports nothing from `pufferlib`. On the first tick it reads the town, the
  ego's box/wheels, the route (`set_global_plan` sparse target points + the dense 1 m route), the driving
  waypoints, every traffic light's stop waypoints (from the trigger volume: 0.9.10 has no
  `get_stop_waypoints`) and every `traffic.stop` trigger volume, and sends them to the server. Every tick
  it ships ego/partner states and light states, and teleports the ego to the returned pose
  (`COSIM_DYNAMICS_SOURCE=pufferdrive`) or applies the returned throttle/brake/steer (`carla`). Chase-cam
  mp4 (`COSIM_DEBUG_CARLA_VIEW`) and infraction clips (`COSIM_RECORD_INFRACTIONS`) are written here.
- `policy_server.py` — started by the agent in the PufferDrive venv (`COSIM_SERVER_PYTHON`, default
  `<repo>/.venv/bin/python`), one process per route, `ShadowEgo` behind `lb1_protocol.py`
  (length-prefixed JSON over an inherited socket). Telemetry, world log, obs replay and obs dump come out
  of this process (same `COSIM_*` variables and file names as the Leaderboard 2.0 agent).
- `ground_truth.py` — the Leaderboard 1.0 criteria (red light, stop sign, outside lanes, collision via a
  collision sensor on the client) recomputed from the client's observations, for the `carla_*` telemetry
  columns and infraction clips. The leaderboard's own `result.json` is the score.

Verified 2026-10-05 locally with `k_scaled_0040_1000` on a CARLA 0.9.10.1 offscreen-OpenGL server, policy
on the CPU at the 20 Hz tick: Town01 route 0 scenario-free completed with DS 100 / RC 100 and no
infraction, 655 s of game time in 22 min wall-clock (0.5x real time, 256 background cars), telemetry,
world log, obs replay (13096 frames) and chase-cam mp4 written, and both `analyze_carla_cosim.py` and
`render_carla_obs_html.py` accept the run.

## How to run

Prerequisites (this machine): CARLA 0.9.10.1 at `~/ordnung/internal/CARLA_0.9.10`, carla_garage at
`~/ordnung/internal/carla_garage_1`, the `garage` conda env (`environment.yml`, Python 3.7), the repo's
`.venv` for the server. The 0.9.10 egg links `libtiff.so.5`, which newer distros no longer ship (Ubuntu
24.04: `libtiff.so.6` only). The launcher scripts test `import carla` in the garage env and, when it fails,
first put the env's own lib dir on `LD_LIBRARY_PATH` (`conda install -n garage "libtiff<4.5"` installs a
`.so.5` there; the egg does not find it without the path) and otherwise link the system `libtiff.so.6`
under that name next to the results (`LIBTIFF5_COMPAT=0` disables the link). On a distro that still ships
`libtiff.so.5`, nothing happens.

One route, own CARLA server (the script starts and stops it):

```bash
CKPT=experiments/k_scaled_0040_1000/final_model.pt \
ROUTES=~/ordnung/internal/carla_garage_1/leaderboard/data/longest6_split/longest_weathers_0.xml \
SCENARIOS=~/ordnung/internal/carla_garage_1/leaderboard/data/scenarios/no_scenarios.json \
bash pufferlib/ocean/cosim/carla/lb1/run_leaderboard.sh
```

Against a server you started yourself (`DISPLAY= ./CarlaUE4.sh -opengl -carla-rpc-port=2000`), from VS Code:
launch config "Cosim: CARLA leaderboard 1.0 agent (garage)" runs `run_evaluator.py --model-dir <dir>
--routes <xml> ...` in the garage env with outputs under `<dir>/eval/carla_leaderboard_v1/`. It points
`LD_LIBRARY_PATH` at `experiments/compat_lib` (gitignored), which holds the `libtiff.so.5` link:
`mkdir -p experiments/compat_lib && ln -s /usr/lib/x86_64-linux-gnu/libtiff.so.6 experiments/compat_lib/libtiff.so.5`.

All 36 longest6 routes on the cluster: `scripts/kesai/14_carla_longest6_v1.sh` (one CARLA 0.9.10 server
+ evaluator per GPU over the `longest6_split` files, retries, garage's `result_parser.py` aggregation,
the nuPlan-style HTML report and the obs replay gallery, like `11_carla_longest6.sh`).

The evaluator reads `BENCHMARK=longest6` (background traffic on every spawn point, stop-sign penalty
1.0) and `ROUTES` (names the route records) from the environment; `run_leaderboard.sh`,
`run_evaluator.py` and script 14 export them.

## Differences to the Leaderboard 2.0 co-sim

- Target points every **50 m** (`downsample_route(route, 50)`) instead of 200 m, so the sliding goal
  window is denser; junction entries/exits and lane changes are target points in both.
- Scoring is Leaderboard 1.0's: penalties 0.50/0.60/0.65/0.70 (pedestrian/vehicle/layout/red light),
  stop sign 1.0 under `BENCHMARK=longest6`, no min-speed or scenario-timeout infractions, blocked after
  180 s (`BLOCKED_THRESHOLD`), route timeout 0.8 s/m. `result.json` records carry nine infraction keys.
- The ego is `vehicle.lincoln.mkz2017`; routes carry weather.
- CARLA 0.9.10: no `World.cast_ray`, so the teleported ego rests on the lane waypoint's height and road
  plane (the 0.9.15 agent additionally samples the road mesh); stop waypoints from the trigger volume.
- The `carla_*` telemetry columns come from `ground_truth.py` (straight-ahead probes in place of lane
  waypoints for the stop sign, no same-lane check for the red light), not from CaRL's criteria.

## Outputs

As for the Leaderboard 2.0 agent: `result.json` (the leaderboard's), `telemetry/<route>.csv`,
`world_log/<route>.npz`, `obs_html/<route>.replay.zlib` (rendered by
`scripts/eval/render_carla_obs_html.py`), `carla_view/<route>.mp4`. `scripts/eval/analyze_carla_cosim.py`
builds the HTML report from a run folder holding `result.json`, `world_log/`, `telemetry/`, `carla_view/`.
