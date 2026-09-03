import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "experiments/regents/regents_nuplan/npz"


def verify_scenario(scenario_idx):
    path = ARTIFACTS_DIR / f"scenario_{scenario_idx:05d}.npz"
    if not path.exists():
        print(f"Artifact not found: {path}")
        return

    print("\n" + "=" * 50)
    print(f" VERIFYING SCENARIO {scenario_idx}")
    print("=" * 50)

    with np.load(path, allow_pickle=False) as archive:
        metadata_bytes = archive["metadata_json_utf8"]
        metadata_json = metadata_bytes.tobytes().decode("utf-8")
        metadata = json.loads(metadata_json)

        opt_meta = metadata["optimization"]
        reasons_by_agent = opt_meta["failure_reasons_by_agent"]

        # Load agent_id array to see how many agents exist
        agent_id = archive["agent_id"][0]  # [num_agents]

        print(f"Total Agents in Dataset: {len(agent_id)}")
        print(f"Optimization Success:    {opt_meta['success']}")
        print(f"Failure Reason:          {opt_meta['failure_reason'] or 'None'}\n")

        # List of filter reasons for each agent
        for idx, reasons in enumerate(reasons_by_agent):
            if idx == 0:
                print(f"Agent {idx} (ID: {agent_id[idx]}): {reasons} (This is the Ego/SDC)")
            else:
                print(f"Agent {idx} (ID: {agent_id[idx]}): {reasons}")


def main():
    for idx in [2, 10, 15]:
        verify_scenario(idx)
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
