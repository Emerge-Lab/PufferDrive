"""NCCL smoke test for a torchrun launch: one tiny all_reduce (hangs = transport broken), then a 256 MiB all_reduce
bandwidth figure. 2 min timeout instead of NCCL's 10 min default.
  one node:  srun -p dev -N1 --gres=gpu:8 --exclusive -t 10 torchrun --standalone --nproc-per-node=8 scripts/kesai/nccl_check.py
  two nodes: see 3_train_64GPU_multinode_slurm.sh for the rdzv flags; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET shows the transport.
"""

import datetime
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
node = os.uname().nodename

x = torch.ones(1, device="cuda")
t = time.time()
dist.all_reduce(x)
torch.cuda.synchronize()
print(
    f"rank {rank} on {node}: all_reduce ok, sum={x.item():.0f} of {dist.get_world_size()}, {time.time() - t:.2f}s",
    flush=True,
)

y = torch.empty(64 * 1024 * 1024, device="cuda")  # 256 MiB
dist.barrier()
torch.cuda.synchronize()
t = time.time()
for _ in range(5):
    dist.all_reduce(y)
torch.cuda.synchronize()
dt = (time.time() - t) / 5
if rank == 0:
    print(f"all_reduce 256 MiB: {dt * 1000:.0f} ms per call, {y.numel() * 4 / dt / 1e9:.2f} GB/s", flush=True)
dist.destroy_process_group()
