# OpenGigaFlow

OpenGigaFlow is an open-source re-implementation of GIGAFLOW, the batched self-play driving simulator from
[*Robust Autonomy Emerges from Self-Play*](https://arxiv.org/abs/2502.03349) (Cusumano-Towner et al., 2025).
A C simulation engine runs dense traffic on CARLA town maps, and a PyTorch PPO trainer learns one shared driving
policy for every agent from self-play alone, without human driving data. The trained policy is evaluated in
self-play and, through co-simulation, in nuPlan and the CARLA leaderboard.

- **Simulator:** C engine with a Python binding. Up to 150 agents per environment, with collisions, off-road
  detection, traffic lights, stop signs, speed limits and scripted pedestrians. `gigaflow` mode populates a map
  with agents and random goals; `replay` mode plays back logged scenarios.
- **Training:** PPO with advantage filtering, on one GPU or several via `torchrun`, in bfloat16 with fused CUDA
  encoder kernels. The policy is conditioned on randomized reward weights and speed caps, so one network covers a
  range of driving styles.
- **Maps:** 128 augmented CARLA town maps for training and the 8 original towns for evaluation ship with the
  repository.
- **Evaluation:** a built-in self-play benchmark, plus co-simulation with nuPlan's closed-loop simulator (Val14)
  and CARLA Leaderboard 1.0 (longest6 v1).
- **Replays:** evaluated episodes render to self-contained interactive HTML pages, optionally with the
  observations the policy saw.

## Install

```bash
# Python env
uv venv && source .venv/bin/activate
uv pip install -e .

# Build C extensions (required after any .h/.c change)
python setup.py build_ext --inplace --force
```

## Data

Maps are `.bin` files under `opengigaflow/resources/`; the CARLA maps are converted from the towns' OpenDRIVE
files. The repository ships:

- `carla_128_affine_aug/`: the training maps (`env.map_dir` default), the 8 CARLA towns each mirrored, scaled
  and sheared into 16 variants.
- `carla/`: the 8 CARLA towns (Town01-07, Town10HD), used by the self-play benchmarks and the CARLA
  co-simulation.
- `nuplan/`, `sdc_replay_test/`: one nuPlan scenario each, used by the tests.

Not shipped: the converted scenarios behind the replay benchmarks in
`opengigaflow/config/evaluation/benchmark.yaml` (`nuplan_*` read the full set from `nuplan/`, `womd_*` read
Waymo Open Motion validation scenarios from `wod-motion_val/`), and the nuPlan city maps that the nuPlan
co-simulation needs (`CITY_BIN_DIR`).

## Train

All settings, with comments, are in `opengigaflow/config/opengigaflow.yaml`.

```bash
# Single node; train.data_dir is the run's folder
opengigaflow train train.data_dir=experiments/<run>

# Override config on the fly (Hydra syntax: dotted.key=value). An override for a setting the config
# does not have is ignored with a WARNING on stderr, so check the log for typos.
opengigaflow train train.data_dir=experiments/<run> train.learning_rate=0.001 env.num_agents=512
```

The run folder receives `config.yaml`, the latest checkpoint (`models/`), `trainer_state.pt` and, when training
ends, `final_model.pt`. `train.data_dir` defaults to `experiments/` itself, and a run started in a folder that
already holds a `trainer_state.pt` resumes it, so give every run its own folder. `wandb=True` (with
`wandb_project` and `run_name`) and `tb=True` log to Weights & Biases and TensorBoard.

Short local run on a machine without CUDA:

```bash
opengigaflow train train.data_dir=experiments/<run> \
    train.device=cpu vec.backend=Serial vec.num_envs=2 vec.batch_size=2 env.num_agents=64 \
    train.total_timesteps=8192 train.batch_size=2048 train.minibatch_size=1024 train.bptt_horizon=16
```

8-GPU server:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m opengigaflow.rl train train.data_dir=experiments/<run>
```

On SLURM we train with `scripts/kesai/0_train_8gpu.sh` (one 8-GPU node per seed, then the evaluations);
`scripts/kesai/1_train_multinode.sh` spans several nodes. Like all `scripts/kesai/` jobs they are examples from
our cluster to copy and adapt.

## Evaluation

Each evaluation loads a checkpoint from a training run, e.g. `experiments/<run>/final_model.pt`, with the run's
`config.yaml` next to it.

The `scripts/kesai/` jobs referenced below are the ones we submit on our own SLURM clusters. Treat them as
examples to copy and adapt, not as a general interface: next to the evaluation settings they hold our partition
and resource requests, our data layout (`scripts/kesai/cluster_env.sh`), our conda environment names and the run
that was current when they were committed.

### Self-play

Runs inside OpenGigaFlow on the 8 CARLA towns. Short local run (`selfplay_fast`: 250 scenarios of 33 s, failed
episodes rendered as HTML; add `train.device=cpu` without a GPU):

```bash
opengigaflow eval selfplay_fast \
    load_model_path=experiments/<run>/final_model.pt \
    eval.render_filter=all_infractions \
    eval.max_rendered_failures=10
```

8-GPU server, as in the KE:SAI scripts (80,000 ten-minute `selfplay` scenarios, one shard per GPU):

```bash
python scripts/parallel_eval.py selfplay --total-scenarios 80000 --num-gpus 8 \
    load_model_path=experiments/<run>/final_model.pt \
    vec.num_envs=60 \
    eval.render_filter=all_infractions
```

`vec.num_envs` is the number of environment workers per GPU; 60 fits a node with 480 cores. The multi-node SLURM
version is `scripts/kesai/2_eval_selfplay.sh`. Benchmark definitions, filtered replays and the output
layout are described in [docs/evaluation.md](docs/evaluation.md).

### nuPlan (Val14)

Runs the policy as a planner inside nuPlan's unmodified closed-loop simulator, with CaRL's Val14 split and
controller. Setup:

- The nuPlan dataset and maps ([dataset setup](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html))
  and a [nuplan-devkit](https://github.com/motional/nuplan-devkit) checkout.
- [CaRL's nuPlan code](https://github.com/autonomousvision/CaRL/tree/main/nuPlan) (`carl_nuplan`) and its conda
  environment.
- This repository's simulator binding, built for that environment:
  `NO_TRAIN=1 /path/to/nuplan_env/bin/python setup.py build_ext --inplace --force`.
- The nuPlan city maps as `.bin` files.
- Details: [nuPlan co-simulation README](opengigaflow/ocean/cosim/nuplan/README.md).

Short local run (10 scenarios, reactive agents):

```bash
CKPT=experiments/<run>/final_model.pt \
PY=/path/to/nuplan_env/bin/python PD=$PWD \
NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit \
NUPLAN_DATA_ROOT=/path/to/nuplan NUPLAN_MAPS_ROOT=/path/to/nuplan/maps NUPLAN_EXP_ROOT=/path/to/nuplan_exp \
CITY_BIN_DIR=/path/to/nuplan_city_bins \
PYTHONPATH=$PWD:/path/to/CaRL/nuPlan:/path/to/nuplan-devkit \
CHALLENGES=closed_loop_reactive_agents_opengigaflow WORKER=sequential LIMIT_TOTAL_SCENARIOS=10 \
bash opengigaflow/ocean/cosim/nuplan/run_nuplan_planner.sh
```

All 1,118 Val14 scenarios on an 8-GPU node: `RUN_DIR=experiments/<run> sbatch scripts/kesai/3_eval_nuplan_reactive.sh`
(`4_eval_nuplan_nonreactive.sh` for log-replay agents). The script also sets the planner options behind our scores
(goal window, start-up jerk caps, pedestrian box padding, 40 partner slots, lane-width clipping); use the same
values for comparable numbers.

### CARLA longest6 v1 (Leaderboard 1.0)

Drives the policy through carla_garage's unmodified Leaderboard 1.0 evaluator on the 36 longest6 routes. Setup:

- [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1) and
  [carla_garage](https://github.com/autonomousvision/carla_garage) with its `garage` conda environment
  (Python 3.7). The routes are in its
  [`leaderboard/data/longest6_split/`](https://github.com/autonomousvision/carla_garage/tree/main/leaderboard/data/longest6_split).
- The policy runs in a separate server process in this repository's `.venv`.
- Details: [CARLA Leaderboard 1.0 co-simulation README](opengigaflow/ocean/cosim/carla/lb1/README.md).

Short local run (one route; the script starts and stops its own CARLA server):

```bash
CKPT=experiments/<run>/final_model.pt \
CARLA_ROOT=/path/to/CARLA_0.9.10.1 \
GARAGE_WORK_DIR=/path/to/carla_garage \
GARAGE_PY=/path/to/envs/garage/bin/python \
ROUTES=/path/to/carla_garage/leaderboard/data/longest6_split/longest_weathers_0.xml \
bash opengigaflow/ocean/cosim/carla/lb1/run_leaderboard.sh
```

All 36 routes with 3 seeds on an 8-GPU node:
`RUN_DIR=experiments/<run> sbatch scripts/kesai/5_eval_longest6_v1.sh`. The script also sets the co-simulation
options behind our scores (`COSIM_*`: CARLA pedal control, 40 partner slots, pedestrian box padding, 30 m/s speed
cap); use the same values for comparable numbers.

## Debug

```bash
# Debug build (-O0, -g) with address, undefined-behavior and leak sanitizers
DEBUG=1 python setup.py build_ext --inplace --force

# Build only the simulator binding, without the torch training extension
NO_TRAIN=1 python setup.py build_ext --inplace --force

# Debug training under AddressSanitizer or gdb
CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python -m opengigaflow.rl train train.device=cpu vec.backend=Serial
gdb --args python -m opengigaflow.rl train train.device=cpu vec.backend=Serial
```

Step-through debugging of the C and Python code in VS Code:
[docs/debugger/setup_your_debugger.md](docs/debugger/setup_your_debugger.md).

## Tests

```bash
# Run the local suites: Python unit tests, eval integration tests, C sim tests.
# Rebuilds the C extension automatically when a .c/.h file changed.
make test

# Individual suites
make test-unit       # tests/unit_tests
make test-eval       # tests/eval (evaluation integration)
make test-c          # tests/drive (dynamics, geometry, IDM, collisions, ...)

# Force-rebuild the C extension regardless of timestamps
make rebuild

# Opt-in: full smoke suite in Docker (train/rollout goldens)
make test-docker-smoke
```

`make help` (or bare `make`) prints this list. Targets call `.venv/bin/python`
directly, so no venv activation is needed. If your interpreter lives elsewhere
(conda, a cluster venv), override it: `make PYTHON=/path/to/python test`.
