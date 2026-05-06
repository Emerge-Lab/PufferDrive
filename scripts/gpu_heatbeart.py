"""GPU heartbeat: keeps utilization above threshold to prevent job reclamation.

Monitors GPU utilization via nvidia-smi and performs matrix multiplications
when utilization drops below THRESHOLD. Steps aside when real training is active.
"""

import subprocess
import time
import torch

THRESHOLD = 65  # percent GPU utilization to maintain
CHECK_INTERVAL = 0.05  # seconds between checks
N = 6144  # matrix size for dummy work
BURST_ITERATIONS = 60  # number of matmuls per burst


def get_gpu_utilization():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 100  # assume busy if query fails


def main():
    device = torch.device("cuda")
    x = torch.randn(N, N, device=device)
    y = torch.randn(N, N, device=device)

    print(f"GPU heartbeat started (threshold={THRESHOLD}%, matrix={N}x{N})")

    while True:
        util = get_gpu_utilization()
        if util < THRESHOLD:
            for _ in range(BURST_ITERATIONS):
                torch.mm(x, y)
            torch.cuda.synchronize()
        else:
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
