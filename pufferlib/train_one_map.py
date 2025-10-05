# train_one_map.py
from pufferlib.pufferl import load_config, train

# Load default config for 'drive'
args = load_config("puffer_drive")

# Force single map and stable sampling
args["env"]["num_maps"] = 1          # only one map available
args["env"]["resample_frequency"] = 0  # do not resample maps during training
args["env"]["num_agents"] = 64      # fewer agents -> faster iteration
args["vec"]["num_envs"] = 6
args["train"]["total_timesteps"] = 200  # adjust as needed
args["train"]["device"] = "cpu"     # or "cuda"
args["package"] = "ocean"
# If you want to fine-tune an existing checkpoint:
# args['load_model_path'] = "path/to/checkpoint.pt"

def main():
	# Start training
	train("puffer_drive", args=args)


if __name__ == '__main__':
	# On platforms that use 'spawn' (macOS, Windows), protect the
	# program entry point so child processes aren't started during import.
	try:
		from multiprocessing import freeze_support
		freeze_support()
	except Exception:
		pass
	main()