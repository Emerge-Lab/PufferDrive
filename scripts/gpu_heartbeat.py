#!/usr/bin/env python3
"""Hold GPU utilization above a floor so the cluster's idle-GPU reclaimer does not scancel long jobs."""

import argparse
import signal
import sys
import time

import torch

DEFAULT_TARGET_UTILIZATION_PERCENT = 85.0
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_MATRIX_SIZE = 4096
MIN_MATRIX_SIZE = 256
MAX_MATRIX_SIZE = 32768
MIN_POLL_INTERVAL_SECONDS = 0.5
MAX_POLL_INTERVAL_SECONDS = 60.0
MATMULS_PER_CHUNK = 8
MAX_MATMULS_PER_WINDOW = 1 << 20
EVENT_POLL_SECONDS = 0.001
MAX_EVENT_POLLS = 10000
STATUS_INTERVAL_SECONDS = 300.0

terminate_requested = False


def log(message):
    print(f"[gpu_heartbeat] {message}", file=sys.stderr, flush=True)


def request_termination(signal_number, frame):
    global terminate_requested
    terminate_requested = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-utilization-percent", type=float,
                        default=DEFAULT_TARGET_UTILIZATION_PERCENT)
    parser.add_argument("--poll-interval-seconds", type=float,
                        default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--matrix-size", type=int, default=DEFAULT_MATRIX_SIZE)
    args = parser.parse_args()

    if not 0.0 < args.target_utilization_percent <= 100.0:
        parser.error("--target-utilization-percent must be in (0, 100]")
    if not MIN_POLL_INTERVAL_SECONDS <= args.poll_interval_seconds <= MAX_POLL_INTERVAL_SECONDS:
        parser.error(f"--poll-interval-seconds must be in "
                     f"[{MIN_POLL_INTERVAL_SECONDS}, {MAX_POLL_INTERVAL_SECONDS}]")
    if not MIN_MATRIX_SIZE <= args.matrix_size <= MAX_MATRIX_SIZE:
        parser.error(f"--matrix-size must be in [{MIN_MATRIX_SIZE}, {MAX_MATRIX_SIZE}]")
    return args


def read_utilization_percent():
    try:
        return float(torch.cuda.utilization())
    except Exception:
        return None  # unreadable utilization must fall through to generating load, never to idling


def wait_for_chunk(completion_event):
    for _ in range(MAX_EVENT_POLLS):
        if completion_event.query():
            return True
        time.sleep(EVENT_POLL_SECONDS)
    return False


def generate_load(left, right, completion_event, window_end_seconds):
    matmul_count = 0
    while time.monotonic() < window_end_seconds and matmul_count < MAX_MATMULS_PER_WINDOW:
        for _ in range(MATMULS_PER_CHUNK):
            torch.mm(left, right)
        completion_event.record()
        if not wait_for_chunk(completion_event):
            return -1
        matmul_count += MATMULS_PER_CHUNK
    return matmul_count


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        log("ERROR: no CUDA device visible; refusing to start")
        return 1

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)

    device = torch.device("cuda")
    left = torch.randn(args.matrix_size, args.matrix_size, device=device)
    right = torch.randn(args.matrix_size, args.matrix_size, device=device)
    completion_event = torch.cuda.Event()

    log(f"started on {torch.cuda.get_device_name(device)} "
        f"target={args.target_utilization_percent:.0f}% "
        f"poll={args.poll_interval_seconds:.1f}s matrix={args.matrix_size}")
    if read_utilization_percent() is None:
        log("WARNING: utilization is unreadable (pynvml missing?); will generate load continuously")

    windows_total = 0
    windows_fired = 0
    windows_unreadable = 0
    utilization_sum = 0.0
    next_status_seconds = time.monotonic() + STATUS_INTERVAL_SECONDS

    while not terminate_requested:
        window_end_seconds = time.monotonic() + args.poll_interval_seconds
        utilization_percent = read_utilization_percent()
        windows_total += 1
        if utilization_percent is None:
            windows_unreadable += 1
        else:
            utilization_sum += utilization_percent

        if utilization_percent is None or utilization_percent < args.target_utilization_percent:
            if generate_load(left, right, completion_event, window_end_seconds) < 0:
                log("ERROR: matmul chunk did not complete within the poll budget")
                return 1
            windows_fired += 1

        idle_seconds = window_end_seconds - time.monotonic()
        if idle_seconds > 0.0:
            time.sleep(idle_seconds)

        now_seconds = time.monotonic()
        if now_seconds < next_status_seconds:
            continue

        windows_readable = windows_total - windows_unreadable
        mean_utilization = utilization_sum / windows_readable if windows_readable else float("nan")
        log(f"mean_utilization={mean_utilization:.1f}% fired={windows_fired}/{windows_total} "
            f"unreadable={windows_unreadable}")
        windows_total = windows_fired = windows_unreadable = 0
        utilization_sum = 0.0
        next_status_seconds = now_seconds + STATUS_INTERVAL_SECONDS

    log("terminating on signal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
