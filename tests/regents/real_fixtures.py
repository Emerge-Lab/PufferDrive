"""Stable real-data fixture identities shared by the ReGentS regressions."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/nuplan"
NUPLAN_FILE_PREFIX = "nuplan__"

# This is the exact cohort used to establish the Stage 3 reconstruction gates.
# Never substitute positional directory indices: the local NuPlan directory is
# ignored by Git and may grow as datasets are downloaded.
REGENTS_AUDIT_SCENARIO_IDS = (
    "00018a38-0063-54d1-a3c1-1ab931a4a1e5",
    "000a50a5-dd43-5c16-9ba6-85dca362ee38",
    "00a2de82-c3a3-5c9d-a5f6-c85b22be2bd7",
    "00ab6c77-cf03-5afe-a08f-70fddd56ff05",
    "00ae06cf-c9ac-527b-ba2d-0dde745ecd8b",
    "00af9733-d21f-5e64-9020-67394bacf03b",
    "0a0733bd-90c2-5eed-9888-40ad605d017c",
    "0a09c765-b5a3-581d-8c52-82484a437054",
    "0a09eab1-5fc7-5f39-afeb-502cb70e1844",
    "0a0c5301-7737-517e-9735-e7dee76d646a",
    "0a0c7c33-8df9-589a-8fdc-b4b5b24ab7de",
    "0a0e40df-f953-5e78-8946-cc43b12ff8ae",
    "0a11a5d3-5874-57d7-bca3-e0c6cade1c1d",
    "0a13529a-a21c-57b7-b5b2-e7d2f32da2fa",
    "0a16167d-e74a-5813-ab9d-ebdfb4bf616f",
    "0a172967-7aee-5333-abcc-4afa825f8fc4",
)


def resolve_nuplan_scenarios(scenario_ids=REGENTS_AUDIT_SCENARIO_IDS):
    """Return the full map catalog, requested indices, and requested paths."""
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Requested NuPlan fixture scenario IDs must be unique")
    map_paths = sorted(NUPLAN_MAP_DIR.glob("*.bin"))
    scenario_id_to_entry = {}
    for map_idx, map_path in enumerate(map_paths):
        if not map_path.stem.startswith(NUPLAN_FILE_PREFIX):
            raise ValueError(f"Unexpected NuPlan fixture filename: {map_path.name}")
        scenario_id = map_path.stem.removeprefix(NUPLAN_FILE_PREFIX)
        if scenario_id in scenario_id_to_entry:
            raise ValueError(f"Duplicate NuPlan fixture scenario ID: {scenario_id}")
        scenario_id_to_entry[scenario_id] = (map_idx, map_path)

    missing_ids = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenario_id_to_entry]
    if missing_ids:
        raise FileNotFoundError(f"Missing pinned NuPlan fixture scenarios: {', '.join(missing_ids)}")
    map_indices = [scenario_id_to_entry[scenario_id][0] for scenario_id in scenario_ids]
    requested_paths = [scenario_id_to_entry[scenario_id][1] for scenario_id in scenario_ids]
    return map_paths, map_indices, requested_paths
