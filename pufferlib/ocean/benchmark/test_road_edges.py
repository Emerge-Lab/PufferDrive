"""Test script for road edge extraction.

Run with: python -m pufferlib.ocean.benchmark.test_road_edges
"""

import numpy as np
import matplotlib.pyplot as plt
import pufferlib
import pufferlib.vector
from pufferlib.pufferl import load_config


def main():
    env_name = "puffer_drive"
    args = load_config(env_name)

    args["vec"] = dict(backend="PufferEnv", num_envs=1)
    args["env"]["num_agents"] = 32

    from pufferlib.ocean import env_creator

    make_env = env_creator(env_name)
    vecenv = pufferlib.vector.make(make_env, env_kwargs=args["env"], **args["vec"])
    vecenv.reset()

    polylines = vecenv.driver_env.get_road_edge_polylines()

    print("\n=== Road Edge Statistics ===")
    print(f"num_polylines: {len(polylines['lengths'])}")
    print(f"total_points: {len(polylines['x'])}")
    print(
        f"points per polyline: min={polylines['lengths'].min()}, max={polylines['lengths'].max()}, mean={polylines['lengths'].mean():.1f}"
    )
    print(f"x range: [{polylines['x'].min():.1f}, {polylines['x'].max():.1f}]")
    print(f"y range: [{polylines['y'].min():.1f}, {polylines['y'].max():.1f}]")

    unique_scenarios = np.unique(polylines["scenario_id"])
    print(f"unique scenarios: {len(unique_scenarios)} -> {unique_scenarios}")

    for sid in unique_scenarios[:3]:
        mask = polylines["scenario_id"] == sid
        n_polys = mask.sum()
        pts = polylines["lengths"][mask].sum()
        print(f"  scenario {sid}: {n_polys} polylines, {pts} points")

    # Plot first scenario
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    sid = unique_scenarios[0]
    mask = polylines["scenario_id"] == sid
    poly_indices = np.where(mask)[0]

    boundaries = np.cumsum(np.concatenate([[0], polylines["lengths"]]))

    for i, idx in enumerate(poly_indices):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        x = polylines["x"][start:end]
        y = polylines["y"][start:end]

        ax.plot(x, y, "b-", linewidth=0.5, alpha=0.7)

        # Mark direction with arrow on first segment
        if len(x) >= 2:
            mid = len(x) // 2
            dx = x[mid] - x[mid - 1]
            dy = y[mid] - y[mid - 1]
            ax.annotate(
                "",
                xy=(x[mid], y[mid]),
                xytext=(x[mid - 1], y[mid - 1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=0.5),
            )

    ax.set_aspect("equal")
    ax.set_title(f"Road edges for scenario {sid} ({len(poly_indices)} polylines)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.tight_layout()
    plt.savefig("road_edges_test.png", dpi=150)
    print(f"\nPlot saved to road_edges_test.png")

    vecenv.close()


if __name__ == "__main__":
    main()
