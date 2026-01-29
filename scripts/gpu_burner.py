import time

import torch


size = 512
a = torch.randn(size, size, device="cuda", dtype=torch.float16)
b = torch.randn(size, size, device="cuda", dtype=torch.float16)

while True:
    start = time.perf_counter()
    while time.perf_counter() - start < 0.9:
        for _ in range(50):
            c = a @ b
        torch.cuda.synchronize()
    time.sleep(0.1)