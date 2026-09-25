import numbers

import numpy as np


def environment_metric_sums(metric_lists):
    """Per-key [value_sum, sample_count] over the collected log dicts."""
    metric_sums = {}
    for key, values in metric_lists.items():
        if not values or not isinstance(values[0], numbers.Number):
            continue
        metric_sums[key] = [float(np.sum(values)), len(values)]
    return metric_sums


def finalize_environment_metrics(metric_sums):
    reduced_metrics = {}
    total_distance = metric_sums.get("total_distance_travelled_sum")
    total_infractions = metric_sums.get("total_infraction_count")
    for key, (value_sum, value_count) in metric_sums.items():
        if key in ("total_distance_travelled_sum", "total_infraction_count"):
            continue
        reduced_metrics[key] = value_sum / max(value_count, 1)

    if total_distance is not None and total_infractions is not None:
        # Sum of the "n" per-dict episode counts; normalizes the window sums below
        agent_episode_count = metric_sums.get("n", [0.0, 0])[0]
        reduced_metrics["total_distance_travelled"] = total_distance[0]
        reduced_metrics["total_infractions"] = total_infractions[0]
        reduced_metrics["distance_per_agent"] = total_distance[0] / max(agent_episode_count, 1.0)
        reduced_metrics["infractions_per_agent"] = total_infractions[0] / max(agent_episode_count, 1.0)
        reduced_metrics["avg_distance_per_infraction"] = total_distance[0] / max(total_infractions[0], 1.0)

    return reduced_metrics


def reduce_environment_metrics(metric_lists):
    # Preserve raw sums and sample counts so log batches with different numbers
    # of completed episodes contribute with the correct weight.
    return finalize_environment_metrics(environment_metric_sums(metric_lists))
