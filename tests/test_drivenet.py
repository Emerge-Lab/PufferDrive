import os
import sys
import numpy as np
import torch

from pufferlib.ocean.torch import Drive, Recurrent
from pufferlib.ocean import env_creator
from pufferlib.ocean.drive import binding


def test_drivenet(
    pt_file="resources/drive/puffer_drive_weights.pt",
    bin_file="resources/drive/puffer_drive_weights.bin",
    batch_size=4,
    seed=42,
):
    """Compare logits from PyTorch and C implementations."""

    assert os.path.exists(bin_file), f"{bin_file} not found"
    assert os.path.exists(pt_file), f"{pt_file} not found"

    env = env_creator("puffer_drive")(num_maps=1, num_agents=batch_size, scenario_length=91)
    policy = Drive(env, input_size=64, hidden_size=256)
    model = Recurrent(env, policy=policy, input_size=256, hidden_size=256)

    state_dict = torch.load(pt_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = np.random.randn(batch_size, env.num_obs).astype(np.float32)

    # Categorical road type features must be integers 0-6
    road_start = 7 + 63 * 7
    for i in range(200):
        obs[:, road_start + i * 7 + 6] = np.random.randint(0, 7, size=batch_size)

    with torch.no_grad():
        lstm_state = {
            "lstm_h": torch.zeros(1, batch_size, 256),
            "lstm_c": torch.zeros(1, batch_size, 256),
        }
        actions_torch, _ = model.forward(torch.from_numpy(obs), lstm_state)

    logits_torch = torch.cat(actions_torch, dim=1).cpu().numpy()

    # C forward pass
    _, logits_c = binding.test_forward(observations=obs, weights_file=bin_file)

    diff = np.abs(logits_torch - logits_c)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"First batch:")
    print(f"Logits PyTorch: {logits_torch[0, :10]}")
    print(f"Logits C:       {logits_c[0, :10]}")
    print(f"  Diff:    {diff[0, :10]}")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-2:
        return True
    else:
        return False


if __name__ == "__main__":
    success = test_drivenet()
    sys.exit(0 if success else 1)
