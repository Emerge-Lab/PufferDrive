"""Render a video from a saved checkpoint using headless Raylib + ffmpeg.

Designed to run as a subprocess from the training loop:
    python -m pufferlib.render_video --model-path ... --map-path ... --output-path ...
"""

import argparse
import os
import sys
import glob
import shutil

import numpy as np
import torch

import pufferlib.pytorch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamics-model", default="jerk")
    args = parser.parse_args()

    from pufferlib.ocean.drive.drive import Drive, RenderView
    from pufferlib.ocean.torch import Drive as DrivePolicy
    from pufferlib.models import LSTMWrapper

    # Create a temp directory with a single map
    map_dir_tmp = args.output_path + "_render_tmp_map"
    os.makedirs(map_dir_tmp, exist_ok=True)
    shutil.copy2(args.map_path, os.path.join(map_dir_tmp, os.path.basename(args.map_path)))

    try:
        env = Drive(
            render_mode=1,  # RENDER_HEADLESS
            num_agents=64,
            num_maps=1,
            map_dir=map_dir_tmp,
            dynamics_model=args.dynamics_model,
            resample_frequency=0,
        )

        # Build policy and load weights
        policy = DrivePolicy(env)
        policy = LSTMWrapper(env, policy)
        state_dict = torch.load(args.model_path, map_location=args.device, weights_only=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        policy = policy.to(args.device)
        policy.eval()

        obs, _ = env.reset()
        ep_length = env.episode_length if env.episode_length else 1000
        num_agents = env.num_agents

        # LSTM state
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=args.device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=args.device),
        )

        for step in range(int(ep_length)):
            env.render(RenderView.FULL_SIM_STATE, draw_traces=True)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs.copy()).to(args.device)
                logits, _ = policy.forward_eval(obs_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy()

            obs, rewards, terminals, truncations, info = env.step(action)

        env.close()  # finalizes ffmpeg -> mp4

        # Find and move the mp4
        mp4_files = glob.glob("*.mp4")
        if mp4_files:
            latest_mp4 = max(mp4_files, key=os.path.getctime)
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            shutil.move(latest_mp4, args.output_path)
        else:
            print("Warning: No mp4 file produced", file=sys.stderr)
            sys.exit(1)

    finally:
        shutil.rmtree(map_dir_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
