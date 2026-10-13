#!/usr/bin/env python3
"""Hold smoothed GPU utilization at a target so the cluster's idle-GPU reclaimer does not scancel long jobs."""

import argparse
import math
import signal
import sys
import time

import torch

DEFAULT_TARGET_UTILIZATION_PERCENT = 80.0
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_MATRIX_SIZE = 4096
MIN_MATRIX_SIZE = 256
MAX_MATRIX_SIZE = 32768
MIN_POLL_INTERVAL_SECONDS = 0.5
MAX_POLL_INTERVAL_SECONDS = 10.0
CHUNK_TARGET_SECONDS = 0.25
CHUNKS_IN_FLIGHT = 2
CALIBRATION_MATMULS = 16
MAX_MATMULS_PER_CHUNK = 1024
MAX_MATMULS_PER_WINDOW = 1 << 20
EVENT_POLL_SECONDS = 0.001
MAX_EVENT_POLLS = 10000
UTILIZATION_SMOOTHING_SECONDS = 300.0
UNSEEN_STARTUP_SECONDS = 300.0  # job time before this process runs (container start, extension build)
SAMPLE_INTERVAL_SECONDS = 0.25
MAX_SAMPLES_PER_WINDOW = math.ceil(MAX_POLL_INTERVAL_SECONDS / SAMPLE_INTERVAL_SECONDS) + 1
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


def window_reading_percent(readings):
    """Mean of a window's samples; None when any is unreadable, so the window counts as idle."""
    if any(reading is None for reading in readings):
        return None
    return sum(readings) / len(readings)


def sample_utilization_until(window_end_seconds):
    readings = []
    for _ in range(MAX_SAMPLES_PER_WINDOW):
        idle_seconds = window_end_seconds - time.monotonic()
        if idle_seconds <= 0.0:
            break
        time.sleep(min(SAMPLE_INTERVAL_SECONDS, idle_seconds))
        readings.append(read_utilization_percent())
    return readings


class FireController:
    """Fires through the startup span, then whenever the exponentially smoothed utilization is below the target."""

    def __init__(self, poll_interval_seconds, target_utilization_percent):
        self.target_utilization_percent = target_utilization_percent
        self.smoothing_weight = 1.0 - math.exp(-poll_interval_seconds / UTILIZATION_SMOOTHING_SECONDS)
        # smoothing starts at the target after the startup span, so it does not coast down from 100%
        self.smoothed_utilization_percent = target_utilization_percent
        # firing at 100% for this span brings the unseen idle start back up to the target on average
        startup_seconds = (UNSEEN_STARTUP_SECONDS * target_utilization_percent / (100.0 - target_utilization_percent)
                           if target_utilization_percent < 100.0 else 0.0)
        self.startup_windows_remaining = math.ceil(startup_seconds / poll_interval_seconds)

    # a reading after a fired window measures our own load, which the reclaimer sees too
    def record_and_decide_fire(self, utilization_percent):
        if self.startup_windows_remaining > 0:
            self.startup_windows_remaining -= 1
            return True
        reading_percent = 0.0 if utilization_percent is None else utilization_percent
        deviation_percent = reading_percent - self.smoothed_utilization_percent
        self.smoothed_utilization_percent += self.smoothing_weight * deviation_percent
        return self.smoothed_utilization_percent < self.target_utilization_percent


def wait_for_chunk(completion_event):
    for _ in range(MAX_EVENT_POLLS):
        if completion_event.query():
            return True
        time.sleep(EVENT_POLL_SECONDS)
    return False


def calibrate_chunk_matmuls(left, right):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(CALIBRATION_MATMULS):
        torch.mm(left, right)  # queued ahead so a CPU stall while enqueueing the timed span leaves no GPU gap in it
    start_event.record()
    for _ in range(CALIBRATION_MATMULS):
        torch.mm(left, right)
    end_event.record()
    end_event.synchronize()
    matmul_seconds = start_event.elapsed_time(end_event) / 1000.0 / CALIBRATION_MATMULS
    return max(1, min(MAX_MATMULS_PER_CHUNK, math.ceil(CHUNK_TARGET_SECONDS / matmul_seconds)))


# chunk_events is oldest first; two queued chunks keep the GPU busy while this nice-19 process waits for CPU
def generate_load(left, right, chunk_events, chunk_matmuls, window_end_seconds):
    matmul_count = 0
    while time.monotonic() < window_end_seconds and matmul_count < MAX_MATMULS_PER_WINDOW:
        oldest_event = chunk_events.pop(0)
        if not wait_for_chunk(oldest_event):
            return -1
        for _ in range(chunk_matmuls):
            torch.mm(left, right)
        oldest_event.record()
        chunk_events.append(oldest_event)
        matmul_count += chunk_matmuls
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
    chunk_events = [torch.cuda.Event() for _ in range(CHUNKS_IN_FLIGHT)]
    chunk_matmuls = calibrate_chunk_matmuls(left, right)

    log(f"started on {torch.cuda.get_device_name(device)} "
        f"target={args.target_utilization_percent:.0f}% "
        f"poll={args.poll_interval_seconds:.1f}s matrix={args.matrix_size} chunk_matmuls={chunk_matmuls}")
    if read_utilization_percent() is None:
        log("WARNING: utilization is unreadable (pynvml missing?); will generate load continuously")

    windows_total = 0
    windows_fired = 0
    windows_unreadable = 0
    windows_training_only = 0
    utilization_sum = 0.0
    training_utilization_sum = 0.0
    controller = FireController(args.poll_interval_seconds, args.target_utilization_percent)
    previous_window_fired = False
    pending_readings = []
    next_status_seconds = time.monotonic() + STATUS_INTERVAL_SECONDS

    while not terminate_requested:
        window_end_seconds = time.monotonic() + args.poll_interval_seconds
        if not pending_readings:
            pending_readings.append(read_utilization_percent())
        utilization_percent = window_reading_percent(pending_readings)
        pending_readings = []
        windows_total += 1
        if utilization_percent is None:
            windows_unreadable += 1
        else:
            utilization_sum += utilization_percent
        if utilization_percent is not None and not previous_window_fired:
            windows_training_only += 1
            training_utilization_sum += utilization_percent

        previous_window_fired = controller.record_and_decide_fire(utilization_percent)
        if previous_window_fired:
            if generate_load(left, right, chunk_events, chunk_matmuls, window_end_seconds) < 0:
                log("ERROR: matmul chunk did not complete within the poll budget")
                return 1
            windows_fired += 1
        else:
            pending_readings = sample_utilization_until(window_end_seconds)

        idle_seconds = window_end_seconds - time.monotonic()
        if idle_seconds > 0.0:
            time.sleep(idle_seconds)

        now_seconds = time.monotonic()
        if now_seconds < next_status_seconds:
            continue

        windows_readable = windows_total - windows_unreadable
        mean_utilization = utilization_sum / windows_readable if windows_readable else float("nan")
        training_utilization = (training_utilization_sum / windows_training_only
                                if windows_training_only else float("nan"))
        log(f"mean_utilization={mean_utilization:.1f}% training_utilization={training_utilization:.1f}% "
            f"smoothed_utilization={controller.smoothed_utilization_percent:.1f}% "
            f"startup_windows_left={controller.startup_windows_remaining} "
            f"fired={windows_fired}/{windows_total} unreadable={windows_unreadable}")
        windows_total = windows_fired = windows_unreadable = windows_training_only = 0
        utilization_sum = training_utilization_sum = 0.0
        next_status_seconds = now_seconds + STATUS_INTERVAL_SECONDS

    log("terminating on signal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
