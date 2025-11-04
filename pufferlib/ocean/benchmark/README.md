# Waymo Open Sim Agent Challenge (WOSAC) benchmark

## Usage

WOSAC evaluation with random policy
```bash
puffer eval puffer_drive --wosac.enabled True
```

WOSAC evaluation with your checkpoint
```bash
puffer eval puffer_drive --wosac.enabled True --load-model-path <your-trained-policy>.pt
```

## Links

- [Challenge and leaderboard](https://waymo.com/open/challenges/2025/sim-agents/)
- [Sim agent challenge tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_sim_agents.ipynb)
- [Metrics entry point](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py)
- [Log-likelihood estimators](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py)
- Configurations [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/sim_agents_metrics.proto#L51) [default sim agent challenge configs](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto)


## Notes

- For the sim agent challenge we compute the log likelihood with `aggregate_objects=False`, which means that we use [`_log_likelihood_estimate_timeseries_agent_level()`](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py#L17)
- As such, the interpretation is as follows:

Steps [for every scene]:
1. Rollout policy in environment K times → (n_agents, n_rollouts, n_steps)
2. Obtain log data → (n_agents, 1, n_steps)
3. Obtain features from (x, y, z, heading tuples)
4. Compute log-likelihood metrics from features
    - a. Flatten across time (assume independence) → (n_agents, n_rollouts * n_steps)
    - b. Use the per-agent simulated features to construct a probability distribution
    - c. Take the per-agent ground-truth values and find the bin that is closed for each
    - d. Take log of the probability for each bin → (n_agents, n_steps)
5. Likelihood score is exp(sum(log_probs)) → (n_agents, 1) \in [0, 1]
