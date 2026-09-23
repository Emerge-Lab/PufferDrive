"""Env log dicts are weighted by their episode count when reduced to one metric per key."""

import numpy as np

from pufferlib.utils import accumulate_environment_metric, finalize_environment_metrics


def _accumulate(logs):
    sums = {}
    for log in logs:
        weight = float(log["n"])
        for key, value in log.items():
            accumulate_environment_metric(sums, key, value, weight)
    return sums


def test_weighted_mean_over_completed_episodes():
    logs = [
        {"n": 10.0, "episode_return": 1.0, "total_distance_travelled_sum": 100.0, "total_infraction_count": 1.0},
        {"n": 990.0, "episode_return": 3.0, "total_distance_travelled_sum": 9900.0, "total_infraction_count": 9.0},
    ]
    metrics = finalize_environment_metrics(_accumulate(logs))
    assert metrics["n"] == 1000.0
    assert np.isclose(metrics["episode_return"], (10 * 1.0 + 990 * 3.0) / 1000)
    assert metrics["total_distance_travelled"] == 10000.0
    assert metrics["total_infractions"] == 10.0
    assert np.isclose(metrics["distance_per_agent"], 10.0)


def test_unweighted_batch_stats_keep_plain_mean():
    sums = {}
    for value in (0.5, 1.5):
        accumulate_environment_metric(sums, "obs/mean", value, 1.0)
    assert finalize_environment_metrics(sums)["obs/mean"] == 1.0


def test_array_values_count_each_element():
    sums = {}
    accumulate_environment_metric(sums, "score", np.array([1.0, 3.0]), 2.0)
    assert finalize_environment_metrics(sums)["score"] == 2.0
