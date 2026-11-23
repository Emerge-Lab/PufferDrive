# Reproduce the spiced self-play result

Small starter for the key result in [Human-like autonomy emerges from self-play and a pinch of human data](https://arxiv.org/abs/2606.19370): a regularized self-play policy keeps a high self-play score and a low self-play collision rate, and it collides less often, at-fault, with held-out human replays.

```bash
python reproduce/eval.py
```

The script evaluates both checkpoints below on the first 128 maps of the held-out validation split (`--num-maps`, `--num-agents`). It prints score, collision rate, and at-fault collision rate for self-play (`control_vehicles`) and human replay (`control_sdc_only`). Settings match `examples/analyze.py`: stochastic actions, episode length 200, no observation noise, lambda fixed to the value stored in the checkpoint.

## Policies

Both files are the 20B-step runs behind the result tables. Each is about 2.5 MB.

| Role | File | Training |
| --- | --- | --- |
| Regularized (ours) | `reproduce/policies/reg_delta_50k_maps_anchor_200_maps.pt` | 50k self-play maps, lambda **0.075**, behavioral-cloning anchor on 200 scenarios (about 30 minutes, 18k human transitions) |
| Unregularized | `reproduce/policies/unreg_delta_50k_maps.pt` | 50k self-play maps, lambda 0 |

The regularized file is the same weights as `models/scaling_cpts/reg_delta_50k_maps_anchor_200_maps.pt` and `models/lambda_ablation/reg_lam_0.075_20B.pt`. The unregularized file is the same weights as `models/scaling_cpts/unreg_delta_50k_maps.pt` and `models/lambda_ablation/reg_lam_0.0_20B.pt`.

`models/best_cpts/` has older copies with the same names. Those are not the table checkpoints: the 1,200-map and 12,000-map files there were trained for 10B steps, and the 200-map and unregularized files there are different 15B–20B runs.

Why this regularized checkpoint: the paper's method is the 30-minute anchor with lambda 0.075. The repo's main comparison labels `reg_delta_50k_maps_anchor_200_maps.pt` as "Regularized (ours)". On the full 10k validation split (`results/checkpoint_eval_result_20B_all_det_false.csv`, same protocol as `examples/analyze.py`) that policy scores 0.975 in self-play with a 0.47% collision rate, and its human-replay at-fault rate is 0.91%. The unregularized policy's human-replay at-fault rate on that split is 1.60%. The paper's Table 1 reports the 30-minute policy near 0.2% self-play collision and 0.7% human-replay at-fault, and Figure 1 quotes 0.6%.

## Scenarios

The starter samples `resources/drive/binaries/validation/map_000.bin` through `map_127.bin`.

That directory is the 10k held-out Waymo Open Motion validation split used for the paper's human-replay and self-play test numbers. The scenarios are the [pufferdrive_womd_val](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_val) release (10k WOMD validation scenarios). Convert JSON to binaries with `python pufferlib/ocean/drive/drive.py` if `map_000.bin` is not already there. See `docs/src/data.md`.

The full paper table uses all 10k maps and about 10k human-replay episodes (`NUM_TOTAL_EVAL_AGENTS` in `examples/analyze.py`). This script uses the first 128 maps so it can run as a starter. Rates on that slice move around the full-split numbers above.

One stochastic pass of `python reproduce/eval.py` (128 maps, 1024 agents, seed 1) produced:

| Policy | Self-play score | Self-play collision | Human-replay at-fault |
| --- | --- | --- | --- |
| Regularized | 0.971 | 0.38% | 0.29% |
| Unregularized | 0.981 | 1.40% | 1.37% |

Self-play used 153 scenes. Human replay used 1024 scenes. The regularized policy is the one with the low self-play collision rate and the low human-replay at-fault rate.
