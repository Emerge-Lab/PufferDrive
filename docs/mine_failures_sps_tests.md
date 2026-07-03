# mine_failures throughput (SPS) tests

Quick experiments to understand what drives wall-clock time in failure mining /
evaluation. All runs: **160 episodes/scenarios, 6000 steps, 50 agents, seed 2904,
16 envs**, gigaflow, homogeneous (where applicable).

## Results

| # | Pipeline | Policy | Backbone | Replay zlib capture | Wall time |
|---|----------|--------|----------|:-------------------:|-----------|
| 1 | mine_failures | self_play_target | 512×4 TargetDrive | **on**  | **1364.51 s** |
| 2 | mine_failures | self_play_target | 512×4 TargetDrive | off     | **406.68 s** |
| 3 | mine_failures | test_sps (random) | 1024×3 TargetDrive | off    | **545.56 s** |
| 4 | benchmark     | big-model-no-stride2 | 1024×3 Drive | off (n/a) | **589.55 s** |

`test_sps` is a randomly-initialized 1024-wide TargetDrive built only to measure a
bigger policy's throughput in the same pipeline (its metrics are meaningless).

## What each delta isolates

| Comparison | Variable changed | Cost |
|------------|------------------|------|
| #1 − #2 | replay zlib capture on→off | **+957.8 s** |
| #3 − #2 | policy size 512×4 → 1024×3 | +138.9 s |
| #4 − #3 | mine_failures → benchmark pipeline (same 1024×3) | +44.0 s |

## Findings

- **Replay capture dominates (~70% of run #1).** The cost is the *per-step*
  recording of agent/traffic frames into Python buffers — ~960,000 records for the
  run — not the `zlib.compress` itself. With `--capture-mining-replay-failures-only 1`
  only the **27** failure episodes were compressed to disk (111.5 MB). Disabling
  `--capture-mining-replay` took 1364 → 407 s.
- **Policy size** (512×4 → 1024×3 TargetDrive, same pipeline, capture off):
  **+139 s (+34%)**.
- **Pipeline** (same 1024×3 net, mine_failures vs benchmark): **+44 s** — the
  benchmark's leaner-vs-heavier rollout structure / branch.
- The original benchmark-vs-mine gap (590 vs 407 s) decomposes as
  **~139 s policy size + ~44 s pipeline ≈ 183 s** (≈76% policy size, ≈24% pipeline).
- **Benchmark wall time scales with batches, not scenario count.** It runs
  `n_work = min(vec.num_envs=16, num_scenarios)` envs in batches of one 6000-step
  rollout, so wall ≈ ⌈scenarios/16⌉ × (6000-step rollout). 64 scenarios (4 batches)
  ≈ 160 scenarios (10 batches) only when the machine is contended; on an idle box
  64 scenarios should be ~2.5× faster.

## Commands (for reproduction)

mine_failures (flip `--capture-mining-replay 0/1`, swap policy paths for #1–#3):

```bash
python -m pufferlib.pufferl mine_failures puffer_drive \
  --num-episodes 160 --eval-simulation gigaflow --eval-scenario-length 6000 \
  --seed 2904 --mine-env-config large --mine-eval-mode homogeneous \
  --capture-mining-replay 0 --capture-mining-replay-failures-only 0 \
  --env.sdc-controller policy --env.non-sdc-controller policy \
  --target-policy-path experiments/self_play_target.pt \
  --target-policy-config experiments/self_play_target_config.yaml \
  --load-model-path experiments/self_play_target.pt \
  --mine-policy-homogeneous-target-actor 1 --train.device cuda \
  --vec.num-envs 16 --vec.num-workers 16 --vec.batch-size 16 --vec.zero-copy True
```

benchmark (#4):

```bash
./scripts/eval/run_all_eval.sh --benchmark-datasets carla --render 0 --render-obs 0 --render-only 0
```
