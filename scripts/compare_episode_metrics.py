#!/usr/bin/env python3
"""Compare replay episode metrics against their source evaluation rows."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RELATIVE_TOLERANCE = 1e-5
DEFAULT_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_MAX_MISMATCHES = 20
MATCH_COLUMNS = ("map_name", "seed")


def _validate_keys(frame, path):
    missing_columns = [column for column in MATCH_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing match columns: {', '.join(missing_columns)}")
    if frame[list(MATCH_COLUMNS)].isna().any(axis=None):
        raise ValueError(f"{path} contains null values in match columns: {', '.join(MATCH_COLUMNS)}")
    duplicated = frame.duplicated(list(MATCH_COLUMNS), keep=False)
    if duplicated.any():
        duplicate_keys = frame.loc[duplicated, list(MATCH_COLUMNS)].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{path} contains duplicate match keys: {duplicate_keys}")


def compare_episode_metrics(eval_path, replay_path, relative_tolerance, absolute_tolerance, max_mismatches):
    eval_metrics = pd.read_csv(eval_path)
    replay_metrics = pd.read_csv(replay_path)
    _validate_keys(eval_metrics, eval_path)
    _validate_keys(replay_metrics, replay_path)

    comparison_columns = [column for column in eval_metrics.columns if column not in MATCH_COLUMNS]
    missing_replay_columns = [column for column in comparison_columns if column not in replay_metrics.columns]
    if missing_replay_columns:
        raise ValueError(f"{replay_path} is missing evaluation columns: {', '.join(missing_replay_columns)}")

    merged = replay_metrics.merge(
        eval_metrics,
        on=list(MATCH_COLUMNS),
        how="left",
        suffixes=("_replay", "_eval"),
        validate="one_to_one",
        indicator=True,
    )
    unmatched_rows = merged[merged["_merge"] != "both"]
    matched_row_mask = merged["_merge"].eq("both").to_numpy()
    mismatch_records = []

    for column in comparison_columns:
        replay_column = merged[f"{column}_replay"]
        eval_column = merged[f"{column}_eval"]
        if pd.api.types.is_numeric_dtype(eval_metrics[column]):
            matches = np.isclose(
                replay_column.to_numpy(dtype=np.float64),
                eval_column.to_numpy(dtype=np.float64),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                equal_nan=True,
            )
        else:
            matches = (replay_column.eq(eval_column) | (replay_column.isna() & eval_column.isna())).to_numpy()

        matches = np.asarray(matches) | ~matched_row_mask

        for row_idx in np.flatnonzero(~matches):
            mismatch_records.append(
                {
                    "map_name": merged.iloc[row_idx]["map_name"],
                    "seed": merged.iloc[row_idx]["seed"],
                    "metric": column,
                    "eval": eval_column.iloc[row_idx],
                    "replay": replay_column.iloc[row_idx],
                }
            )

    print(f"Evaluation rows: {len(eval_metrics)}")
    print(f"Replay rows: {len(replay_metrics)}")
    print(f"Matched replay rows: {len(merged) - len(unmatched_rows)}")
    print(f"Numeric tolerance: rtol={relative_tolerance:g}, atol={absolute_tolerance:g}")

    if not unmatched_rows.empty:
        print(f"Unmatched replay rows: {len(unmatched_rows)}")
        print(unmatched_rows[list(MATCH_COLUMNS)].head(max_mismatches).to_string(index=False))
    if mismatch_records:
        print(f"Metric mismatches: {len(mismatch_records)}")
        print(pd.DataFrame(mismatch_records).head(max_mismatches).to_string(index=False))

    if not unmatched_rows.empty or mismatch_records:
        print("FAIL: replay metrics differ from the matching evaluation rows")
        return 1

    print("PASS: all replay metrics match their evaluation rows within tolerance")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match replay rows to evaluation rows by map_name and seed, then compare episode metrics."
    )
    parser.add_argument("eval_csv", type=Path, help="Source evaluation episode_metrics.csv")
    parser.add_argument("replay_csv", type=Path, help="Replay episode_metrics.csv")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--atol", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)
    parser.add_argument("--max-mismatches", type=int, default=DEFAULT_MAX_MISMATCHES)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")
    if args.max_mismatches < 1:
        raise ValueError("--max-mismatches must be >= 1")
    return compare_episode_metrics(
        args.eval_csv,
        args.replay_csv,
        args.rtol,
        args.atol,
        args.max_mismatches,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
