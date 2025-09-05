import time
import psutil
import os
from multiprocessing import Process, Queue
from tqdm import tqdm
import csv
from datetime import datetime

import torch
import numpy as np
import gymnasium
import json
import struct
import random
import pufferlib
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
import pandas as pd


def print_memory(process, step=None, print_flag=True):
    cpu_mem = process.memory_info().rss / (1024 ** 2)  # MB
    msg = f"CPU Memory: {cpu_mem:.2f} MB"
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        msg += f", GPU Memory: {gpu_mem:.2f} MB"
    if step is not None:
        msg = f"[Step {step}] " + msg
        if print_flag:
            print(msg)
    return cpu_mem

def run_simulation(num_maps=1000, atn_cache=1024, num_agents=1024, num_steps=1000, q = Queue(), print_flag=True):
    env = Drive(num_agents=num_agents, num_maps=num_maps)
    print(f'num_agents: {env.num_agents}, num_maps: {env.num_maps}, num_envs: {env.num_envs}')
    env.reset()
    actions = np.stack([
        np.random.randint(0, space.n + 1, (atn_cache, num_agents))
        for space in env.single_action_space
    ], axis=-1)

    process = psutil.Process(os.getpid())

    start = time.time()
    tick = 0
    cpu_mem = 0
    cpu_mem = max(print_memory(process=process,step="start", print_flag=print_flag), cpu_mem)

    for _ in range(num_steps):
        atn = actions[tick % atn_cache]
        env.step(atn)
        # What Could cause a major decrease in CPU memory usage?: Probably a reset on SDC frees the pos arrays and reallocates it in next step
        # Why does it happen at the end of the num_steps?
        tick += 1
        if tick % 100 == 0:
            if print_flag:
                print(f"Step {tick} done")
            cpu_mem = max(print_memory(process=process,step=tick,print_flag=print_flag), cpu_mem)
    end_time = time.time()
    print(f'Finished {num_steps} steps in {end_time - start:.2f} seconds')
    print(f'Average steps per second: {num_steps / (end_time - start):.2f}')
    cpu_mem = max(print_memory(process=process,step="end",print_flag=print_flag), cpu_mem)

    env.close()     # Does it deallocate memory?
    q.put(
        (
            num_steps,
            num_agents,
            num_maps,
            env.num_envs,
            cpu_mem,
            end_time - start
        )
    )

def test_performance(num_maps=1000, num_agents=1024, num_steps=1000, print_flag=True):

    q = Queue()
    p = Process(
        target=run_simulation,
        args=(
            num_maps,  # num_maps
            1024,  # atn_cache
            num_agents,  # num_agents
            num_steps,  # num_steps
            q,  # Queue to return results
            print_flag
        ),
    )
    p.start()
    p.join()  # Wait for the process to finish
    return q.get()

if __name__ == '__main__':
    # BATCH_SIZE_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    BATCH_SIZE_LIST = [2]
    num_agents_list = [512]    # [512, 4096, 16384, 32768]

    # create a DataFrame to store results
    results_df = pd.DataFrame(columns=['num_steps', 'num_agents', 'num_maps', 'num_envs', 'cpu_mem_MB', 'elapsed_time_s'])

    pbar = tqdm(num_agents_list, colour="green")
    for idx, num_agents in enumerate(pbar):
        pbar.set_description(
            f"Profiling puffer pufferdrive with num_agents {num_agents}, num_steps 1000, num_maps 1000"
        )

        res = test_performance(num_agents=num_agents, print_flag=False)
        (
            num_steps,
            num_agents,
            num_maps,
            num_envs,
            cpu_mem,
            elapsed_time
        ) = res

        # Create or append to a DataFrame
        row = [num_steps, num_agents, num_maps, num_envs, cpu_mem, elapsed_time]
        
        results_df = pd.concat(
            [results_df, pd.DataFrame([row], columns=results_df.columns)],
            ignore_index=True
        )

    csv_filename = f"profilePufferPort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_filename, index=False)
