import numbers

import numpy as np


def reduce_environment_metrics(metric_lists):
    # Preserve raw sums and sample counts so log batches with different numbers
    # of completed episodes contribute with the correct weight.
    local_metrics = {}
    for key, values in metric_lists.items():
        if not values or not isinstance(values[0], numbers.Number):
            continue
        local_metrics[key] = (float(np.sum(values)), len(values))

    reduced_metrics = {}
    total_distance = local_metrics.get("total_distance_travelled_sum")
    total_infractions = local_metrics.get("total_infraction_count")
    for key, (value_sum, value_count) in local_metrics.items():
        if key in ("total_distance_travelled_sum", "total_infraction_count"):
            continue
        reduced_metrics[key] = value_sum / max(value_count, 1)

    if total_distance is not None and total_infractions is not None:
        reduced_metrics["total_distance_travelled"] = total_distance[0]
        reduced_metrics["total_infractions"] = total_infractions[0]
        reduced_metrics["avg_distance_per_infraction"] = total_distance[0] / max(total_infractions[0], 1.0)

    return reduced_metrics
