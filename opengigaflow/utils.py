import numbers

import numpy as np


EPISODE_COUNT_KEY = "n"
DISTANCE_SUM_KEY = "total_distance_travelled_sum"
INFRACTION_COUNT_KEY = "total_infraction_count"
WINDOW_SUM_KEYS = (DISTANCE_SUM_KEY, INFRACTION_COUNT_KEY)
EPISODE_SUM_KEYS = (EPISODE_COUNT_KEY, *WINDOW_SUM_KEYS)


def accumulate_environment_metric(metric_sums, key, value, weight):
    """Per-key [weighted_sum, weight_sum]; episode-count and window-sum keys accumulate plain sums."""
    values = value.tolist() if isinstance(value, np.ndarray) else value
    if not isinstance(values, (list, tuple)):
        values = [values]
    for item in values:
        if not isinstance(item, numbers.Number):
            continue
        entry = metric_sums.setdefault(key, [0.0, 0.0])
        if key in EPISODE_SUM_KEYS:
            entry[0] += float(item)
            entry[1] += 1
        else:
            entry[0] += float(item) * weight
            entry[1] += weight


def finalize_environment_metrics(metric_sums):
    reduced_metrics = {}
    total_distance = metric_sums.get(DISTANCE_SUM_KEY)
    total_infractions = metric_sums.get(INFRACTION_COUNT_KEY)
    for key, (value_sum, value_count) in metric_sums.items():
        if key in WINDOW_SUM_KEYS:
            continue
        if key == EPISODE_COUNT_KEY:
            reduced_metrics[key] = value_sum
            continue
        reduced_metrics[key] = value_sum / max(value_count, 1)

    if total_distance is not None and total_infractions is not None:
        # Sum of the "n" per-dict episode counts; normalizes the window sums below
        agent_episode_count = metric_sums.get(EPISODE_COUNT_KEY, [0.0, 0])[0]
        reduced_metrics["total_distance_travelled"] = total_distance[0]
        reduced_metrics["total_infractions"] = total_infractions[0]
        reduced_metrics["distance_per_agent"] = total_distance[0] / max(agent_episode_count, 1.0)
        reduced_metrics["infractions_per_agent"] = total_infractions[0] / max(agent_episode_count, 1.0)
        reduced_metrics["avg_distance_per_infraction"] = total_distance[0] / max(total_infractions[0], 1.0)

    return reduced_metrics


def reduce_environment_metrics(metric_lists):
    metric_sums = {
        key: [float(np.sum(values)), len(values)]
        for key, values in metric_lists.items()
        if values and isinstance(values[0], numbers.Number)
    }
    return finalize_environment_metrics(metric_sums)
