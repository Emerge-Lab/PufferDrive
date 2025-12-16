import pandas as pd
import glob
import os
import numpy as np


def process_results(gt_val, base_dir="resources/drive/binaries/validation_10k"):
    print(f"Processing model results...")
    pattern = os.path.join(base_dir, "*", f"results_*_model.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No model result files found.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No dataframes loaded.")
        return

    full_df = pd.concat(dfs, ignore_index=True)

    output_filename = "model_results.csv"
    full_df.to_csv(output_filename, index=False)
    print(f"Saved concatenated results to {output_filename}")

    num_rows = len(full_df)
    print(f"Total rows: {num_rows}")
    if num_rows != 10000:
        print(f"WARNING: Expected 10000 rows, got {num_rows}")
    else:
        print("Row count check passed (10000).")

    # Compute metrics
    metrics = [
        "likelihood_linear_speed",
        "likelihood_linear_acceleration",
        "likelihood_angular_speed",
        "likelihood_angular_acceleration",
        "likelihood_distance_to_nearest_object",
        "likelihood_time_to_collision",
        "likelihood_collision_indication",
        "likelihood_distance_to_road_edge",
        "likelihood_offroad_indication",
        "ade",
        "min_ade",
        "realism_meta_score",
    ]

    print("\n--- Mean Metrics ---")
    means = full_df[metrics].mean()
    print(means)

    # Custom Scores
    kinematics_score = (
        full_df[
            [
                "likelihood_linear_speed",
                "likelihood_linear_acceleration",
                "likelihood_angular_speed",
                "likelihood_angular_acceleration",
            ]
        ]
        .mean(axis=1)
        .mean()
    )

    interaction_score = (
        0.55555556 * full_df["likelihood_collision_indication"]
        + 0.22222222 * full_df["likelihood_time_to_collision"]
        + 0.22222222 * full_df["likelihood_distance_to_nearest_object"]
    ).mean()

    map_score = (
        0.16666666 * full_df["likelihood_distance_to_road_edge"] + 0.83333333 * full_df["likelihood_offroad_indication"]
    ).mean()

    print("\n--- Aggregate Scores ---")
    print(f"Kinematics Score: {kinematics_score:.6f}")
    print(f"Interaction Score: {interaction_score:.6f}")
    print(f"Map Score:         {map_score:.6f}")

    # Also print the meta score from the file for comparison
    print(f"Realism Meta Score (from file): {full_df['realism_meta_score'].mean():.6f}")


def main():
    process_results(None)


if __name__ == "__main__":
    main()
