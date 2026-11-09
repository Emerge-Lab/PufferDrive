"""Pack trained PufferDrive checkpoints into one ensemble checkpoint folder for the evals.

Usage (from the repo root, venv active):
    python scripts/eval/build_ensemble_checkpoint.py --output-dir experiments/k_scaled_0073_ensemble \
        experiments/k_scaled_0073_1000/final_model.pt experiments/k_scaled_0073_2000/final_model.pt

The folder gets final_model.pt with every member's weights and config.yaml, the first member's training config, so the
evals build the env and the policy architecture from it; point RUN_DIR / load_model_path at the folder like at a
training run. At every policy step the members' action distributions and values are averaged and the action is
selected from the averaged distribution.
"""

import argparse
import os

from pufferlib.ensemble import write_ensemble_checkpoint


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("member_checkpoints", nargs="+", help="Trained .pt checkpoints, config.yaml beside each")
    parser.add_argument("--output-dir", required=True, help="Folder to write final_model.pt and config.yaml into")
    parser.add_argument("--run-name", help="run_name of the written config.yaml (default: the output folder name)")
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="Accept members trained with different env configs; the evals use the first member's env config",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    run_name = arguments.run_name or os.path.basename(os.path.normpath(arguments.output_dir))
    try:
        checkpoint_path, config_path = write_ensemble_checkpoint(
            arguments.member_checkpoints, arguments.output_dir, run_name, arguments.allow_env_mismatch
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f"build_ensemble_checkpoint: {error}")
    print(f"ensemble of {len(arguments.member_checkpoints)} checkpoints -> {checkpoint_path} (config {config_path})")


if __name__ == "__main__":
    main()
