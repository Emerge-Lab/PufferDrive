# Waymo Open Sim Agent Challenge (WOSAC) benchmark

## Usage

WOSAC evaluation with random policy
```bash
puffer eval puffer_drive --eval.wosac-realism-eval True
```

WOSAC evaluation with your checkpoint
```bash
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <your-trained-policy>.pt
```

## Links

- [Challenge and leaderboard](https://waymo.com/open/challenges/2025/sim-agents/)
- [Sim agent challenge tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_sim_agents.ipynb)
- [Reference paper introducing WOSAC](https://arxiv.org/pdf/2305.12032)
- [Metrics entry point](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py)
- [Log-likelihood estimators](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py)
- Configurations [proto file](https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/protos/sim_agents_metrics.proto#L51) [default sim agent challenge configs](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto)


## Implementation

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
5. Likelihood score is exp(sum(log_probs)/n_steps) → (n_agents, 1) \in [0, 1]

## Notes

- Currently, only kinematics realism score is implemented. Next steps would be to add the interactive realism score, and the map realism score:

    - Interactive realism score: requires grouping agents per scenario, and computing pairwise distances between agents over time.
    - Map realism score: requires access to the map and computing offroad status.

    Those two scores might require heavy computations, so we will consider reimplementic all the metrics in torch.

- About the time-independence assumption:

    1. This is the assumption used in the official WOSAC evaluation, their argument is that it would give more flexibility to the sim agents models:

        > Given the time series nature of simulation data, two choices emerge for how to treat samples over multiple timesteps for a given object for a given run segment: to treat them as time-independent or time-dependent samples. In the latter case, users would be expected to not only reconstruct the general behaviors present in the logged data in one rollout, but also recreate those behaviors over the exact same time intervals. To allow more flexibility in agent behavior, we use the former formulation when computing NLLs, defining each component metric m as an average (in log-space) over the time-axis, masked by validity.

    2. However this will lead to the score of a perfect logged oracle being inferior to 1.0, and makes it less interpretable. Here are the scores of a logged oracle using the time-independence assumption (setup: 1024 agents, 48 rollouts):

        ```
        Linear speed: 0.5640
        Linear acceleration: 0.4658
        Angular speed: 0.5543
        Angular acceleration: 0.6589

        Kinematics realism score: 0.5607
        ```
    These scores go to 1.0 if we use the time-dependent estimator, execpt for the smoothing factor that is used to avoid bins with 0 probability.

    Using the time-dependent estimator means generating n_steps histograms per agent, using num_rollouts samples per histogram, while time-independence means generating one histogram per agent using n_rollouts * n_steps samples. With the speed of PufferDrive,  we might be able to increase n_rollouts to have more samples per histogram.
