from __future__ import annotations

import argparse

from .plots import generate_analysis_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="MF-PBT run directory")
    parser.add_argument(
        "--hyperparameter",
        type=str,
        default="learning_rate",
        help="Hyperparameter column to plot on the distribution chart",
    )
    args = parser.parse_args()

    output_paths = generate_analysis_plots(
        run_dir=args.run_dir,
        hyperparameter_name=args.hyperparameter,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
