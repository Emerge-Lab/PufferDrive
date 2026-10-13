"""The heartbeat's fire controller: holds smoothed utilization at the target, firing only the shortfall."""

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gpu_heartbeat", REPO_ROOT / "scripts" / "gpu_heartbeat.py")
gpu_heartbeat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_heartbeat)

POLL_SECONDS = 2.0
TARGET_PERCENT = gpu_heartbeat.DEFAULT_TARGET_UTILIZATION_PERCENT
MINUTE_WINDOWS = math.ceil(60.0 / POLL_SECONDS)
FIVE_MINUTE_WINDOWS = 5 * MINUTE_WINDOWS
TEN_MINUTE_WINDOWS = 10 * MINUTE_WINDOWS
HOUR_WINDOWS = 60 * MINUTE_WINDOWS
UNSEEN_IDLE_WINDOWS = math.ceil(gpu_heartbeat.UNSEEN_STARTUP_SECONDS / POLL_SECONDS)
STARTUP_SECONDS = gpu_heartbeat.UNSEEN_STARTUP_SECONDS * TARGET_PERCENT / (100.0 - TARGET_PERCENT)
STARTUP_WINDOWS = math.ceil(STARTUP_SECONDS / POLL_SECONDS)
# strictest published cancel line is 60 % (gh* nodes); stall onsets must stay 10 points above it
STALL_FLOOR_PERCENT = 70.0
FIRED_READING_PERCENT = 100.0
READING_NOISE_PERCENT = 8.0
SEED = 3


def _simulate(training_percent_by_window, unreadable_windows=()):
    """Readings follow the previous window: our own load reads 100 %, otherwise training alone (with noise)."""
    rng = random.Random(SEED)
    controller = gpu_heartbeat.FireController(POLL_SECONDS, TARGET_PERCENT)
    seen, fired = [], []
    previous_fired = False
    for window_idx, training_percent in enumerate(training_percent_by_window):
        if window_idx in unreadable_windows:
            reading = None
        elif previous_fired:
            reading = FIRED_READING_PERCENT
        else:
            noisy = training_percent + rng.uniform(-READING_NOISE_PERCENT, READING_NOISE_PERCENT)
            reading = min(100.0, max(0.0, noisy))
        previous_fired = controller.record_and_decide_fire(reading)
        seen.append(0.0 if reading is None else reading)
        fired.append(previous_fired)
    return seen, fired


def _sliding_means(values, width):
    """Means of every width-long stretch, one starting each minute."""
    return [sum(values[start : start + width]) / width for start in range(0, len(values) - width + 1, MINUTE_WINDOWS)]


def _phases(high_percent, high_windows, low_percent, low_windows, total_windows):
    period = [high_percent] * high_windows + [low_percent] * low_windows
    return (period * (total_windows // len(period) + 1))[:total_windows]


@pytest.mark.parametrize("training_percent", [55.0, 75.0])
def test_a_busy_but_short_run_fires_only_the_shortfall(training_percent):
    seen, fired = _simulate([training_percent] * (4 * HOUR_WINDOWS))
    settled_seen, settled_fired = seen[HOUR_WINDOWS:], fired[HOUR_WINDOWS:]
    expected_fraction = (TARGET_PERCENT - training_percent) / (FIRED_READING_PERCENT - training_percent)
    assert sum(settled_fired) / len(settled_fired) == pytest.approx(expected_fraction, abs=0.03)
    assert min(_sliding_means(settled_seen, HOUR_WINDOWS)) >= TARGET_PERCENT - 1.0
    assert min(_sliding_means(settled_seen, FIVE_MINUTE_WINDOWS)) >= TARGET_PERCENT - 3.0


def test_phased_training_holds_every_five_minutes_at_the_target():
    seen, _ = _simulate(_phases(95.0, 5, 30.0, 5, 4 * HOUR_WINDOWS))
    assert min(_sliding_means(seen[HOUR_WINDOWS:], FIVE_MINUTE_WINDOWS)) >= TARGET_PERCENT - 3.0


def test_a_run_above_the_target_fires_only_at_startup():
    _, fired = _simulate([84.0] * (2 * HOUR_WINDOWS))
    assert all(fired[:STARTUP_WINDOWS])
    assert not any(fired[STARTUP_WINDOWS:])


def test_a_run_above_the_target_fires_through_its_dips():
    seen, fired = _simulate(_phases(90.0, 25 * MINUTE_WINDOWS, 30.0, FIVE_MINUTE_WINDOWS, 4 * HOUR_WINDOWS))
    settled_seen, settled_fired = seen[HOUR_WINDOWS:], fired[HOUR_WINDOWS:]
    assert any(settled_fired)
    assert min(_sliding_means(settled_seen, TEN_MINUTE_WINDOWS)) >= STALL_FLOOR_PERCENT


def test_after_the_unseen_start_hours_and_ten_minute_windows_hold_the_target():
    seen, _ = _simulate([0.0] * (5 * MINUTE_WINDOWS) + [55.0] * (2 * HOUR_WINDOWS))
    reclaimer_view = [0.0] * UNSEEN_IDLE_WINDOWS + seen
    assert sum(reclaimer_view[:HOUR_WINDOWS]) / HOUR_WINDOWS >= TARGET_PERCENT - 1.0
    assert min(_sliding_means(reclaimer_view[TEN_MINUTE_WINDOWS:], TEN_MINUTE_WINDOWS)) >= TARGET_PERCENT - 3.0


def test_a_stall_in_a_busy_run_fires_within_a_minute_or_two():
    stall_start = 2 * HOUR_WINDOWS
    seen, fired = _simulate([95.0] * stall_start + [0.0] * (15 * MINUTE_WINDOWS) + [95.0] * HOUR_WINDOWS)
    assert fired[stall_start:].index(True) <= 2 * MINUTE_WINDOWS
    assert min(_sliding_means(seen[HOUR_WINDOWS:], TEN_MINUTE_WINDOWS)) >= STALL_FLOOR_PERCENT


def test_unreadable_utilization_always_fires():
    windows = 2 * HOUR_WINDOWS
    _, fired = _simulate([90.0] * windows, unreadable_windows=set(range(windows)))
    assert all(fired)


def test_an_idle_job_reads_busy():
    seen, _ = _simulate([0.0] * HOUR_WINDOWS)
    assert min(_sliding_means(seen, TEN_MINUTE_WINDOWS)) >= TARGET_PERCENT - 3.0


@pytest.mark.parametrize("poll_seconds", [0.5, 2.0, 10.0])
def test_startup_fires_then_smoothing_starts_at_the_target_with_its_time_constant(poll_seconds):
    controller = gpu_heartbeat.FireController(poll_seconds, TARGET_PERCENT)
    startup_windows = math.ceil(
        gpu_heartbeat.UNSEEN_STARTUP_SECONDS * TARGET_PERCENT / (100.0 - TARGET_PERCENT) / poll_seconds
    )
    assert all(controller.record_and_decide_fire(FIRED_READING_PERCENT) for _ in range(startup_windows))
    assert controller.smoothed_utilization_percent == TARGET_PERCENT
    idle_windows = math.ceil(gpu_heartbeat.UTILIZATION_SMOOTHING_SECONDS / poll_seconds)
    assert all(controller.record_and_decide_fire(0.0) for _ in range(idle_windows))
    decay = math.exp(-idle_windows * poll_seconds / gpu_heartbeat.UTILIZATION_SMOOTHING_SECONDS)
    assert controller.smoothed_utilization_percent == pytest.approx(TARGET_PERCENT * decay, rel=1e-9)


@pytest.mark.parametrize("target_percent, startup_minutes", [(80.0, 20.0), (90.0, 45.0), (100.0, 0.0)])
def test_the_startup_span_repays_the_unseen_start_at_any_target(target_percent, startup_minutes):
    controller = gpu_heartbeat.FireController(POLL_SECONDS, target_percent)
    assert controller.startup_windows_remaining == math.ceil(startup_minutes * 60.0 / POLL_SECONDS)


def test_a_window_reading_is_the_mean_of_its_samples_and_unreadable_if_any_sample_is():
    assert gpu_heartbeat.window_reading_percent([90.0, 30.0, 60.0]) == pytest.approx(60.0)
    assert gpu_heartbeat.window_reading_percent([90.0]) == 90.0
    assert gpu_heartbeat.window_reading_percent([90.0, None, 60.0]) is None


class _FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, end_event):
        return FAKE_SPAN_MILLISECONDS


class _FakeCuda:
    Event = _FakeEvent

    def __init__(self, readings):
        self.readings = list(readings)
        self.reads = 0

    def is_available(self):
        return True

    def get_device_name(self, device):
        return "fake"

    def utilization(self):
        self.reads += 1
        if self.reads > len(self.readings):
            gpu_heartbeat.terminate_requested = True
            return 0
        return self.readings[self.reads - 1]


class _FakeTorch:
    def __init__(self, readings):
        self.cuda = _FakeCuda(readings)
        self.matmuls = 0

    def device(self, name):
        return name

    def randn(self, *shape, device=None):
        return None

    def mm(self, left, right):
        self.matmuls += 1


FAKE_SPAN_MILLISECONDS = 4000.0
MAIN_POLL_SECONDS = 0.5
MAIN_UNSEEN_SECONDS = 1.0
MAIN_READINGS = 40


def test_main_runs_the_loop_fires_through_startup_then_samples_idle_windows(monkeypatch, capsys):
    fake_torch = _FakeTorch([95.0] * MAIN_READINGS)
    monkeypatch.setattr(gpu_heartbeat, "torch", fake_torch)
    monkeypatch.setattr(gpu_heartbeat, "UNSEEN_STARTUP_SECONDS", MAIN_UNSEEN_SECONDS)
    monkeypatch.setattr(gpu_heartbeat, "STATUS_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(gpu_heartbeat, "terminate_requested", False)
    monkeypatch.setattr(gpu_heartbeat.signal, "signal", lambda signal_number, handler: None)
    monkeypatch.setattr(sys, "argv", ["gpu_heartbeat.py", "--poll-interval-seconds", str(MAIN_POLL_SECONDS)])
    assert gpu_heartbeat.main() == 0
    status_lines = [line for line in capsys.readouterr().err.splitlines() if "smoothed_utilization=" in line]
    startup_windows = math.ceil(MAIN_UNSEEN_SECONDS * TARGET_PERCENT / (100.0 - TARGET_PERCENT) / MAIN_POLL_SECONDS)
    fired_by_window = [int(line.split("fired=")[1].split("/")[0]) for line in status_lines]
    assert fired_by_window[:startup_windows] == [1] * startup_windows
    assert sum(fired_by_window[startup_windows:]) == 0 and len(fired_by_window) > startup_windows + 2
    # an idle window takes several samples, so reads outrun windows once startup ends
    assert fake_torch.cuda.reads > len(status_lines) + 2 and fake_torch.matmuls > 0
