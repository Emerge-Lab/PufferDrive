# Waymo Open Sim Agent Challenge (WOSAC) Benchmark

We provide a re-implementation of the [Waymo Open Sim Agent Challenge (WOSAC)](https://waymo.com/research/the-waymo-open-sim-agents-challenge/), which measures _distributional realism_ of simulated trajectories compared to logged human trajectories. Our version preserves the original logic and metric weighting but uses PyTorch on GPU for the metrics computation, unlike the original TensorFlow CPU implementation. The code is also simplified for clarity, making it easier to understand, adapt, and extend.

**Note:** In PufferDrive, agents are conditioned on a "goal" represented as a single (x, y) position, reflecting that drivers typically have a high-level destination in mind. Evaluating whether an agent matches human distributional properties can be decomposed into: (1) inferring a person's intended direction from context (1 second in WOSAC) and (2) navigating toward that goal in a human-like manner. We focus on the second component, though the evaluation could be adapted to include behavior prediction as in the original WOSAC.

[TODO: ADD bar graphs]

## Usage

### Running a single evaluation from a checkpoint

The `[eval]` section in `drive.ini` contains all relevant configurations. To run the WOSAC eval once:

```bash
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <your-trained-policy>.pt
```

The default configs aim to emulate the WOSAC settings as closely as possible, but you can adjust them:

```ini
[eval]
map_dir = "resources/drive/binaries/validation" # Dataset to use
num_maps = 100  # Number of maps to run evaluation on. (It will alwasys be the first num_maps maps of the map_dir)
wosac_num_rollouts = 32      # Number of policy rollouts per scene
wosac_init_steps = 10        # When to start the simulation
wosac_control_mode = "control_wosac"  # Control the tracks to predict
wosac_init_mode = "create_all_valid"  # Initialize from the tracks to predict
wosac_goal_behavior = 2      # Stop when reaching the goal
wosac_goal_radius = 2.0      # Can shrink goal radius for WOSAC evaluation
```

### Log evals to W&B during training

During experimentation, logging key metrics directly to W&B avoids a post-training step. Evaluations can be enabled during training, with results logged under a separate `eval/` section. The main configuration options:

```ini
[train]
checkpoint_interval = 500    # Set equal to eval_interval to use the latest checkpoint

[eval]
eval_interval = 500          # Run eval every N epochs
map_dir = "resources/drive/binaries/training"  # Dataset to use
num_maps = 20 # Number of maps to run evaluation on. (It will alwasys be the first num_maps maps of the map_dir)
```

## Baselines

We provide baselines on a small curated dataset from the WOMD validation set with perfect ground-truth (no collisions or off-road events from labeling mistakes).

| Method | Realism meta-score | Kinematic metrics | Interactive metrics | Map-based metrics | minADE | ADE |
|--------|-------------------|-------------------|---------------------|-------------------|--------|------|
| Ground-truth (UB) | 0.832 | 0.606 | 0.846 | 0.961 | 0 | 0 |
| π_Base self-play RL | 0.737 | 0.319 | 0.789 | 0.938 | 10.834 | 11.317 |
| [SMART-tiny-CLSFT](https://arxiv.org/abs/2412.05334) | 0.805 | 0.534 | 0.830 | 0.949 | 1.124 | 3.123 |
| π_Random | 0.485 | 0.214 | 0.657 | 0.408 | 6.477 | 18.286 |

*Table: WOSAC baselines in PufferDrive on 229 selected clean held-out validation scenarios.*

---

> ✏️ Download the dataset from [Hugging Face](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_wosac_val_clean) to reproduce these results or benchmark your policy.

---

## Useful links

- [WOSAC challenge and leaderboard](https://waymo.com/open/challenges/2025/sim-agents/)
- [Sim agent challenge tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_sim_agents.ipynb)
- [Reference paper introducing WOSAC](https://arxiv.org/pdf/2305.12032)
- [Metrics entry point](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py)
- [Log-likelihood estimators](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py)
- [Configurations proto file](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/sim_agents_metrics.proto#L51)
- [Default sim agent challenge configs](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto)
