# Waymo Open Sim Agent Challenge (WOSAC) benchmark

## Usage

Standard WOSAC evaluation
```bash
puffer eval puffer_drive --wosac.enabled True
```

Run WOSAC with additional sanity checks
```bash
puffer eval puffer_drive --wosac.enabled True --wosac.dashboard True
```

## Links

- [Challenge and leaderboard](https://waymo.com/open/challenges/2025/sim-agents/)
- [Metrics entry point](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py)
- [Log-likelihood estimators](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py)
- Configurations [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/sim_agents_metrics.proto#L51) [default sim agent challenge configs](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto)
