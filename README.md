# OpenGigaFlow

OpenGigaFlow is an open-source re-implementation of GIGAFLOW, the batched self-play driving simulator from
[*Robust Autonomy Emerges from Self-Play*](https://arxiv.org/abs/2502.03349) (Cusumano-Towner et al., 2025).
A C simulation engine runs dense traffic on CARLA town maps, and a PyTorch PPO trainer learns one shared driving
policy for every agent from self-play alone, without human driving data. The trained policy is evaluated in
self-play and, through co-simulation, in nuPlan and the CARLA leaderboard.

- **Simulator:** C engine with a Python binding. Up to 150 agents per environment, with collisions, off-road
  detection, traffic lights, stop signs, speed limits and scripted pedestrians. `gigaflow` mode populates a map
  with agents and random goals; `replay` mode plays back logged scenarios. Besides the policy, agents can be
  driven by two built-in rule-based controllers, IDM and a PDM-Closed style planner (`env.sdc_controller`,
  `env.non_sdc_controller`).
- **Training:** PPO with advantage filtering, on one GPU or several via `torchrun`, in bfloat16 with fused CUDA
  encoder kernels. The policy is conditioned on randomized reward weights and speed caps, so one network covers a
  range of driving styles.
- **Maps:** 128 augmented CARLA town maps for training and the 8 original towns for evaluation ship with the
  repository.
- **Evaluation:** a built-in self-play benchmark, plus co-simulation with nuPlan's closed-loop simulator (Val14),
  CARLA Leaderboard 1.0 (longest6 v1) and CARLA Leaderboard 2.0 (longest6 v2).
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

All settings, with comments, are in `opengigaflow/config/opengigaflow.yaml`. The defaults are sized for the server
under [Recommended hardware](#recommended-hardware): 128 environment workers and a 393,216-sample minibatch per GPU. On a smaller machine, lower `vec.num_envs`,
`train.minibatch_size` and `train.max_minibatch_size`.

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

On SLURM, `RUN_NAME=<run> sbatch scripts/slurm/train.sh` trains one run on the nodes of its allocation and then
runs the self-play evaluation; its arguments are config overrides.

## Evaluation

Each evaluation loads a checkpoint from a training run, e.g. `experiments/<run>/final_model.pt`, with the run's
`config.yaml` next to it.

The SLURM jobs referenced below are in `scripts/slurm/`. Each lists its settings at the top (the run, paths, GPUs
per node, what to render): set them in the environment or edit them there, adjust the `#SBATCH` lines to your
cluster and submit from the checkout root. `scripts/kesai/` holds the same jobs filled in for our cluster.

### Self-play

Runs inside OpenGigaFlow on the 8 CARLA towns. Short local run (`selfplay_fast`: 250 scenarios of 33 s, failed
episodes rendered as HTML; add `train.device=cpu` without a GPU):

```bash
opengigaflow eval selfplay_fast \
    load_model_path=experiments/<run>/final_model.pt \
    eval.render_filter=all_infractions \
    eval.max_rendered_failures=10
```

8-GPU server (the `selfplay` benchmark: 80,000 ten-minute scenarios, one shard per GPU):

```bash
python scripts/parallel_eval.py selfplay --num-gpus 8 \
    load_model_path=experiments/<run>/final_model.pt \
    vec.num_envs=60 \
    eval.render_filter=all_infractions
```

`vec.num_envs` is the number of environment workers per GPU (default 128, the training value). The multi-node SLURM version is
`scripts/slurm/eval_selfplay.sh`. Benchmark definitions, filtered replays and the output layout are described in
[docs/evaluation.md](docs/evaluation.md).

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

All 1,118 Val14 scenarios on an 8-GPU node: `RUN_DIR=experiments/<run> sbatch scripts/slurm/eval_nuplan.sh`
(with `CHALLENGES=closed_loop_nonreactive_agents_opengigaflow` for log-replay agents). The planner options behind our scores (goal window, start-up
jerk caps, pedestrian box padding, 40 partner slots, lane-width clipping) are the defaults in
`opengigaflow/ocean/cosim/nuplan/config/planner/opengigaflow_planner.yaml`.

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
`RUN_DIR=experiments/<run> sbatch scripts/slurm/eval_longest6_v1.sh`. The co-simulation options behind our scores
(40 partner slots, pedestrian box padding, 30 m/s speed cap) are the defaults, `COSIM_EVAL_DEFAULTS` in
`opengigaflow/ocean/cosim/carla/shadow_ego.py`.

### CARLA longest6 v2 (Leaderboard 2.0)

Drives the policy through CaRL's unmodified Leaderboard 2.0 evaluator on the 36 longest6 v2 routes. Setup:

- [CARLA 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15) and
  [CaRL](https://github.com/autonomousvision/CaRL) with its `carl` conda environment (Python 3.10), set up as in
  [its CARLA README](https://github.com/autonomousvision/CaRL/tree/main/CARLA). The routes are its
  [`CARLA/custom_leaderboard/leaderboard/data/longest6.xml`](https://github.com/autonomousvision/CaRL/blob/main/CARLA/custom_leaderboard/leaderboard/data/longest6.xml).
- The policy runs inside the evaluator, in the `carl` environment; the job builds this repository's simulator
  binding for it.
- Details: [CARLA Leaderboard 2.0 co-simulation README](opengigaflow/ocean/cosim/carla/README.md).

Short local run (one route, one seed; the job starts and stops its own CARLA server):

```bash
RUN_DIR=experiments/<run> \
CARLA_ROOT=/path/to/CARLA_0.9.15 \
CARL_WORK_DIR=/path/to/CaRL/CARLA \
PY=/path/to/envs/carl/bin/python \
ROUTE_SUBSET=0 SEEDS=0 WORKERS=1 \
bash scripts/slurm/eval_longest6_v2.sh
```

All 36 routes with 3 seeds on an 8-GPU node:
`RUN_DIR=experiments/<run> sbatch scripts/slurm/eval_longest6_v2.sh`. The co-simulation options behind our scores
are the same `COSIM_EVAL_DEFAULTS`, except that the policy keeps the partner slots it was trained with instead of 40.

## Performance

### GIGAFLOW benchmarks

In the following we compare the performance of OpenGigaFlow to GIGAFLOW on benchmarks reported in the GIGAFLOW paper. We report the mean performance over 3 training seeds.

#### nuPlan Val14 (reactive)
nuPlan Val14 (reactive) is the most common planning benchmark in the literature and measure peformance of the policy on nuPlan maps in 15 second scenarios, with log-replay initialization and other agents being controlled by IDM.
OpenGigaFlow is the first open-model that can match PDM-Closed zero-shot without training on nuPlan data or maps.

| Method             | Type | Zero-shot | Open |  CLS |
|--------------------|:----:|:---------:|:----:|-----:|
| PDM-Closed         | Rule |     ❌     |  ✅   | 92.1 |
| PlanTF             |  IL  |     ❌     |  ✅   | 76.1 |
| Diffusion Planner  |  IL  |     ❌     |  ✅   | 82.7 |
| CaRL, tuned        |  RL  |     ❌     |  ✅   | 93.1 |
| GIGAFLOW           |  RL  |     ✅     |  ❌   | 93.8 |
| TerraZero          |  RL  |     ❌     |  ❌   | 94.2 |
| OpenGigaFlow 1.0.0 |  RL  |     ✅     |  ✅   | 91.8 |

#### CARLA longest6 v1 (Leaderboard 1.0)
longest6 v1 is a classic CARLA benchmark, that test the models ability to drive longer distances (much longer than nuPlan around ~1.6 km per route) consistently. Additionally it tests the agent in safety critical scenarios such as other cars running red lights, that are not observed in everyday driving logs. OpenGigaFlow is the state-of-the-art open model on this benchmark.

| Method             | Type | Zero-shot | Open | Driving score |
|--------------------|:----:|:---------:|:----:|--------------:|
| TransFuser Expert  | Rule |     ❌     |  ✅   |        77 ± 2 |
| TF++ Expert        | Rule |     ❌     |  ✅   |        83 ± 1 |
| PlanT              |  IL  |     ❌     |  ✅   |        81 ± 7 |
| GIGAFLOW           |  RL  |     ✅     |  ❌   |        92 ± 2 |
| OpenGigaFlow 1.0.0 |  RL  |     ✅     |  ✅   |               |

#### Self-play

This is an benchmark where all 50 cars on 8 CARLA towns are driven by the same method, therefore measuring performance in full autonomy traffic. An infraction is a collision or leaving the road. This is the benchmark the GigaFlow policies are optimized for and they outperform other methods by more than 3 orders of magnitude here.
GIGAFLOW numbers are italic because the original benchmark is not open so there might be difference in the evaluation setup compared to our results.

| Method                | Type | Open | km / infraction |
|-----------------------|:----:|:----:|----------------:|
| IDM                   | Rule |  ✅   |              11 |
| PDM                   | Rule |  ✅   |             316 |
| OpenGigaFlow 1.0.0    |  RL  |  ✅   |         678,441 |
| *GIGAFLOW (reported)* |  RL  |  ❌   |     *3,000,000* |


### Additional benchmarks

Here we report some additional evals that are not part of the original GIGAFLOW paper, so the training recipe and method is not tuned for these benchmarks. OpenGigaFlow achieves reasonable performance zero-shot on them but is not state-of-the-art.

#### nuPlan Val14 (non-reactive)
The same as Val14 Reactive except other cars are driven by log replay and do not react to the ego agent. We observe that our policy trained in self-play where most cars react to you sometime drive assuming the other car will react to it, leading to a lower score on this benchmark.

| Method             | Type | Zero-shot | Open |  CLS |
|--------------------|:----:|:---------:|:----:|-----:|
| PDM-Closed         | Rule |     ❌     |  ✅   | 92.8 |
| PlanTF             |  IL  |     ❌     |  ✅   | 84.6 |
| Diffusion Planner  |  IL  |     ❌     |  ✅   | 89.6 |
| GIGAFLOW           |  RL  |     ✅     |  ❌   |    – |
| CaRL, tuned        |  RL  |     ❌     |  ✅   | 93.9 |
| OpenGigaFlow 1.0.0 |  RL  |     ✅     |  ✅   | 87.5 |

#### CARLA longest6 v2 (Leaderboard 2.0)
A newer version of the classic longest6 updated with the CARLA leaderboard 2.0 scenarios and newer CARLA version. Unlike the other benchmarks it features driving at higher speeds (up to 80 km/h) as well as some extremely challenging scenarios where the model needs to merge into an infinite stream of non-reactive cars driving at high speeds through an intersection. Solving this requires exact timing and driving the car at its limits. No method today can solve all of these scenarios reliably.

| Method             | Type | Zero-shot | Open | Driving score |
|--------------------|:----:|:---------:|:----:|--------------:|
| PDM-Lite           | Rule |     ❌     |  ✅   |            73 |
| PlanT              |  IL  |     ❌     |  ✅   |        62 ± 2 |
| Roach              |  RL  |     ❌     |  ✅   |       22 ± 14 |
| CaRL               |  RL  |     ❌     |  ✅   |        64 ± 2 |
| OpenGigaFlow 1.0.0 |  RL  |     ✅     |  ✅   |               |

## Recommended hardware

We optimized the code and the default configuration for the server we train on:

- **Server:** [Supermicro A+ Server AS-5126GS-TNRT2](https://www.supermicro.com/en/products/system/datasheet/as-5126gs-tnrt2)
- **CPU:** 2 x AMD EPYC 9755 (256 cores, 512 threads)
- **RAM:** 1,024 GB (16 x 64 GB DDR5-6400)
- **GPU:** 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each)

One such server trains the default configuration at about 10.5 million agent samples per second (SPS): a full run of 1 trillion steps takes about 26 hours. For comparison this is ~8x faster than the original GIGAFLOW training which achieved 1.2 million SPS on an 8x A100 server. As of November 2026, 8x RTX PRO 6000 instances and 8x A100 (80GB) instances cost roughly the same on cloud providers.

The closest cloud instances with the same GPU:

- **Google Cloud `g4-standard-384`:** 8 GPUs, 384 vCPUs (AMD EPYC Turin), 1,440 GB RAM.
- **AWS `g7e.48xlarge`:** 8 GPUs, 192 vCPUs (Intel Xeon Emerald Rapids), 2,048 GiB RAM.

On the Google Cloud instance we reached a 10 million SPS after some tuning. The AWS instance is untested: it has
half as many vCPUs (a vCPU is one hardware thread) and the simulator runs on the CPU, so its SPS may be lower.

The fused encoder kernels (`opengigaflow/ocean/cuda_slot_encoder.py`) are compiled at run time for the installed
GPU, but their tile sizes and shared-memory layout are hand-tuned for the GPU above. On a different GPU model it
is worth letting a coding agent retune them: `tests/unit_tests/test_fused_slot_encoder.py` checks the kernels
against the PyTorch reference, and the `SPS` and `MFU_train` values logged during training show the effect.
`policy.fused_slot_encoder=false` turns the kernels off and runs the model in plain PyTorch. The Pytorch implementation is significantly (about 2x) slower than the custom CUDA implementation.

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
make test-c          # tests/drive (dynamics, geometry, IDM, PDM, collisions, ...)

# Force-rebuild the C extension regardless of timestamps
make rebuild

# Opt-in: full smoke suite in Docker (train/rollout goldens)
make test-docker-smoke
```

`make help` (or bare `make`) prints this list. Targets call `.venv/bin/python`
directly, so no venv activation is needed. If your interpreter lives elsewhere
(conda, a cluster venv), override it: `make PYTHON=/path/to/python test`.
