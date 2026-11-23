# Reproduce the spiced self-play result

Regularized self-play (lambda 0.075, about 30 minutes of human anchor data) keeps a high self-play score and a low self-play collision rate, and it collides less often, at-fault, with human replays than unregularized self-play. Paper: [arXiv:2606.19370](https://arxiv.org/abs/2606.19370).

```bash
python reproduce/eval.py
python reproduce/eval.py --num-maps 10000 --num-agents 1024 --target-scenes 10000
python reproduce/render.py
```

Eval uses stochastic actions, episode length 200, and no observation noise. Lambda stays at the value stored in the checkpoint. Self-play controls every vehicle. Human replay controls only the ego; other agents follow the log.

## Results

First 128 validation maps, 1024 agents, seed 1. Self-play is 153 scenes. Human replay is 1024 scenes.

| Policy | Self-play score | Self-play collision | Human-replay at-fault |
| --- | --- | --- | --- |
| Regularized | 0.971 | 0.38% | 0.29% |
| Unregularized | 0.981 | 1.40% | 1.37% |

Full 10k validation split, 1024 agents per rollout, new seeds until 10,000 scenes in each mode:

| Policy | Self-play score | Self-play collision | Human-replay at-fault |
| --- | --- | --- | --- |
| Regularized | 0.977 | 0.45% | 0.93% |
| Unregularized | 0.984 | 0.86% | 1.86% |

The paper's Table 1 reports the 30-minute policy near 0.2% self-play collision and 0.7% human-replay at-fault. Figure 1 quotes 0.6% at-fault.

## Policies

Both files are the 20B-step runs. Each is about 2.5 MB.

| Role | File | Training |
| --- | --- | --- |
| Regularized (ours) | `reproduce/policies/reg_delta_50k_maps_anchor_200_maps.pt` | 50k self-play maps, lambda **0.075**, anchor on 200 scenarios (about 30 minutes) |
| Unregularized | `reproduce/policies/unreg_delta_50k_maps.pt` | 50k self-play maps, lambda 0 |

The regularized weights match `models/scaling_cpts/reg_delta_50k_maps_anchor_200_maps.pt` and `models/lambda_ablation/reg_lam_0.075_20B.pt`. The unregularized weights match `models/scaling_cpts/unreg_delta_50k_maps.pt` and `models/lambda_ablation/reg_lam_0.0_20B.pt`. Copies under `models/best_cpts/` with the same names are earlier runs.

## Validation split

The 10k held-out Waymo Open Motion validation scenarios are [pufferdrive_womd_val](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_val).

```bash
uv pip install -U "huggingface_hub[cli]"
huggingface-cli download daphne-cornelisse/pufferdrive_womd_val \
  --repo-type dataset \
  --local-dir data/processed/validation
python -c "from pufferlib.ocean.drive.drive import process_all_maps; process_all_maps(data_folder='data/processed/validation', max_maps=10000)"
```

Binaries land in `resources/drive/binaries/validation/` as `map_000.bin` onward. `eval.py` samples that directory. `--num-maps` and `--num-agents` change the slice. `--target-scenes` draws extra seeds until each mode has that many scenes.

Rendering does not need this download.

## Rendering

`reproduce/data/` has three interactive validation intersections. Ids are in `scenes.json`.

| Map | Waymo scenario id |
| --- | --- |
| `map_000.bin` | `f2e32390fb9fc73a` |
| `map_001.bin` | `3810e617bb1f0c2d` |
| `map_002.bin` | `be141957a08e978e` |

```bash
python reproduce/render.py
python reproduce/render.py --scene 1
python reproduce/render.py --mode self_play --policy unregularized
```

The default is human replay with the regularized policy, greedy actions, 91 steps. ffmpeg writes `reproduce/videos/`. The blue vehicle is the ego. Pale green vehicles are human replays. The green circle is the ego goal.

Scene 1, mid-clip:

![Human replay of the regularized policy on scene 1](snapshot.png)
