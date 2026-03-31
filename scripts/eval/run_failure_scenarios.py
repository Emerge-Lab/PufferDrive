import pandas as pd
import subprocess

# --- Configuration ---
# Set the path to your CSV file
CSV_FILE_PATH = "benchmark/puffer_drive_0jb42gn1/model_puffer_drive_000700/episode_metrics.csv"

# Base command components
BASE_COMMAND_PREFIX = "puffer eval_multi_scenarios puffer_drive --load-model-path experiments/puffer_drive_0jb42gn1/best_models/model_puffer_drive_000700.pt --num_scenarios 500 --render 1 --render_obs 1"
BASE_COMMAND_SUFFIX = ""  # Can be used for extra args if needed


def get_failed_scenario_indices(csv_path):
    """
    Reads the episode metrics CSV and returns a list of scenario indices
    (episode_id) where a failure (collision, offroad, or traffic light violation) occurred.
    """
    try:
        # 1. Read the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Ensure required columns exist
    required_columns = ["episode_id", "offroad_rate", "collision_rate", "red_light_violation_rate"]

    # Check if all necessary columns are present
    if not all(col in df.columns for col in required_columns if col != "traffic_light_rate"):
        print(
            f"Error: CSV is missing one or more required columns: {', '.join(col for col in ['episode_id', 'offroad_rate', 'collision_rate', 'traffic_light_rate'] if col not in df.columns)}"
        )
        return []

    # 2. Define the failure condition
    # Check if any of the specified rates is greater than 0
    failure_condition = (df["collision_rate"] > 0) | (df["offroad_rate"] > 0)

    if "traffic_light_rate" in df.columns:
        failure_condition = failure_condition | (df["traffic_light_rate"] > 0)

    # 3. Filter and extract the episode_id (which acts as the map index)
    failed_indices = df[failure_condition]["episode_id"].tolist()

    # The map indices for the command should be an integer list (or a list of strings of integers)
    # The 'episode_id' column in the screenshot is an integer index starting from 0.
    return [str(int(i)) for i in failed_indices]


def execute_puffer_command(indices):
    """
    Constructs and executes the puffer eval command.
    """
    if not indices:
        print("\n✅ No failed scenarios found. Nothing to render.")
        return

    # Join the indices with a space
    indices_str = " ".join(indices)

    # Construct the final command string
    final_command = f"{BASE_COMMAND_PREFIX} --map_indices {indices_str}"

    print("\n--- 🔨 Command to be executed ---")
    print(final_command)
    print("---------------------------------")

    # Execute the command
    try:
        print("\n🚀 Executing command via subprocess...")
        # Use shell=True for convenience, but it's generally safer to pass a list
        # For simplicity with a complex string command, we use shell=True here.
        process = subprocess.run(final_command, shell=True, check=True, text=True, capture_output=True)

        print("\n✅ Command execution finished.")
        print("--- Subprocess Output (Stdout) ---")
        print(process.stdout)

        if process.stderr:
            print("--- Subprocess Errors (Stderr) ---")
            print(process.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Command failed with exit code {e.returncode}")
        print("--- Subprocess Errors (Stderr) ---")
        print(e.stderr)
    except FileNotFoundError:
        print("\n❌ ERROR: 'puffer' or related command not found. Ensure your environment is correctly set up.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during command execution: {e}")


if __name__ == "__main__":
    failed_scenario_indices = get_failed_scenario_indices(CSV_FILE_PATH)

    if failed_scenario_indices is not None:
        print(f"\nFound {len(failed_scenario_indices)} failed scenarios: {', '.join(failed_scenario_indices)}")
        execute_puffer_command(failed_scenario_indices)
