# Evaluation

Benchmarks are provided to measure how closely agents match real-world driving behavior and how well they pair with human trajectories.

## WOSAC distributional realism
Evaluate how realistic your policy behaves compared to the Waymo Open Sim Agents Challenge (WOSAC):

```bash
puffer eval puffer_drive --eval.wosac-realism-eval True
```

Add `--load-model-path <path_to_checkpoint>.pt` to score a trained policy instead of a random baseline.

## Human-compatibility
Test how a policy coexists with human-controlled agents:

```bash
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <path_to_checkpoint>.pt
```

During this evaluation the self-driving car (SDC) is controlled by your policy while other agents replay log data.
