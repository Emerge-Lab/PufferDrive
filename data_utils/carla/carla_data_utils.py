import os
import json
from typing import List


def replace_roads_in_json_files(source_dir: str, destination_dir: str, town_names: List[str]):
    for town_name in town_names:
        print(f"Replacing roads for town: {town_name}")
        source_file = os.path.join(source_dir, f"{town_name}.json")
        if not os.path.isfile(source_file):
            print(f"Source file for {town_name} not found, skipping.")
            continue
        with open(source_file, "r") as sf:
            source_data = json.load(sf)
        roads_value = source_data["roads"]
        if roads_value is None:
            print(f"No roads data found for {town_name}, skipping.")
            continue

        for filename in os.listdir(destination_dir):
            if filename.startswith(town_name) and filename.endswith(".json"):
                dest_file = os.path.join(destination_dir, filename)
                with open(dest_file, "r") as df:
                    dest_data = json.load(df)
                dest_data["roads"] = roads_value
                with open(dest_file, "w") as df:
                    json.dump(dest_data, df, indent=2)


def make_all_non_expert_agent_initial_velocity_zero(towns, destination_directory):
    for filename in os.listdir(destination_directory):
        print(f"Processing file: {filename}")
        for town in towns:
            if filename.startswith(town):
                json_file = os.path.join(destination_directory, filename)
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_file}, skipping.")
                    continue

                for agent in data.get("agents", []):
                    if agent.get("is_expert", False) == False:
                        agent["initial_velocity"] = 0
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=2)

                objects = data.get("objects", [])

                # Make initial velocity zero for all non_expert_agents
                for obj in objects:
                    if obj.get("mark_as_expert", False) == False:
                        obj["velocity"][0] = {"x": 0.0, "y": 0.0}

                data["objects"] = objects
                # Save changes to the JSON file
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=2)


if __name__ == "__main__":
    source_directory = "data/processed/carla"
    destination_directory = "data/processed/carla_data"
    towns = ["town01", "town02", "town04", "town06", "town10HD"]
    # replace_roads_in_json_files(source_directory, destination_directory, towns)
    make_all_non_expert_agent_initial_velocity_zero(towns, destination_directory)
