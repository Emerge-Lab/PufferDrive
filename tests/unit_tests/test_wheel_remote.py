"""The remote (phone / browser) input: its HTTP contract on 127.0.0.1, the key file, control validation, the
pedal curves and timeout, the MJPEG frame sink, the page, and run_demo / window wiring. Nothing here starts
CARLA, opens the G920 or runs the demo loop."""

import http.client
import importlib.util
import io
import json
import math
import os
import re
import socket
import stat
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo.input import remote as remote_module
from pufferlib.wheel_demo.input.base import HAT_NEUTRAL, WheelCalibration, WheelInput, WheelState
from pufferlib.wheel_demo.input.remote import (
    REMOTE_CONTROL_TIMEOUT_SECONDS,
    REMOTE_DEFAULT_PORT,
    REMOTE_MAX_BODY_BYTES,
    REMOTE_PAGE_PATH,
    REMOTE_STREAM_BOUNDARY,
    REMOTE_STREAM_FPS,
    REMOTE_STREAM_WIDTH_PX,
    RemoteWheelInput,
    load_or_create_key,
    parse_control,
)
from pufferlib.wheel_demo.input.trace import SyntheticWheelInput

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_HOST = "127.0.0.1"
TEST_KEY = "test-remote-key"
WRONG_KEY = "not-the-key"
TEST_COMMANDS = frozenset({"cycle_camera", "toggle_night", "restart"})
CONTRACT_DEFAULT_PORT = 8765
CONTRACT_GEAR_COMMANDS = ("shift_up", "shift_down")
WAIT_TIMEOUT_SECONDS = 5.0
WAIT_POLL_SECONDS = 0.01
HTTP_TIMEOUT_SECONDS = 5.0
KEY_FILE_MODE = 0o600
SOURCE_FRAME_SIZE_PX = (1280, 720)
FAKE_JPEG = b"\xff\xd8fake-jpeg\xff\xd9"
JPEG_MAGIC = b"\xff\xd8"
CUSTOM_THROTTLE_GAMMA = 2.0
CUSTOM_BRAKE_GAMMA = 0.5
HALF_TRAVEL = 0.5
FLOAT_TOLERANCE = 1e-9
VALID_CONTROL = {"steering": 0.25, "throttle": 0.5, "brake": 0.0, "clutch": 0.0}
MPS_PER_KMH = 1.0 / 3.6
WINDOW_SUBSTEP_SECONDS = 0.02


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((TEST_HOST, 0))
        return probe.getsockname()[1]


def wait_for(predicate, timeout_seconds=WAIT_TIMEOUT_SECONDS):
    max_iteration_count = int(timeout_seconds / WAIT_POLL_SECONDS)
    for _ in range(max_iteration_count):
        if predicate():
            return True
        time.sleep(WAIT_POLL_SECONDS)
    return predicate()


def load_script(name):
    path = os.path.join(REPO_ROOT, "scripts", "wheel_demo", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"wheel_demo_remote_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_demo():
    return load_script("run_demo")


@pytest.fixture
def remote():
    wheel_input = RemoteWheelInput(TEST_KEY, TEST_COMMANDS, port=free_port(), host=TEST_HOST).start()
    yield wheel_input
    wheel_input.stop()


def request(remote_input, method, path, body=None, key=TEST_KEY, raw_body=None):
    connection = http.client.HTTPConnection(TEST_HOST, remote_input.port, timeout=HTTP_TIMEOUT_SECONDS)
    try:
        query = "" if key is None else f"?key={key}"
        payload = raw_body if raw_body is not None else (None if body is None else json.dumps(body).encode("utf-8"))
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path + query, body=payload, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def post_control(remote_input, **overrides):
    control = dict(VALID_CONTROL)
    control.update(overrides)
    return request(remote_input, "POST", "/control", control)


class FakeSurface:
    def __init__(self, size_px=SOURCE_FRAME_SIZE_PX):
        self.size_px = size_px

    def get_size(self):
        return self.size_px


class FakePygame:
    """Records every scale and save so a test can see whether a frame was encoded."""

    def __init__(self):
        self.scaled_sizes = []
        self.saved_count = 0
        self.transform = SimpleNamespace(scale=self._scale)
        self.image = SimpleNamespace(save=self._save)

    def _scale(self, surface, size_px):
        self.scaled_sizes.append(tuple(size_px))
        return FakeSurface(tuple(size_px))

    def _save(self, surface, handle, namehint=""):
        self.saved_count += 1
        handle.write(FAKE_JPEG)


class StreamClient:
    """A raw socket reading /stream, so the test controls exactly when it connects and disconnects."""

    def __init__(self, remote_input, key=TEST_KEY):
        self.sock = socket.create_connection((TEST_HOST, remote_input.port), timeout=HTTP_TIMEOUT_SECONDS)
        self.sock.sendall(f"GET /stream?key={key} HTTP/1.1\r\nHost: {TEST_HOST}\r\n\r\n".encode("ascii"))
        self.buffer = b""

    def read_until(self, marker):
        max_read_count = 10000
        for _ in range(max_read_count):
            if marker in self.buffer:
                return True
            chunk = self.sock.recv(65536)
            if not chunk:
                return marker in self.buffer
            self.buffer += chunk
        return False

    def read_part(self):
        assert self.read_until(b"\r\n\r\n")
        header_end = self.buffer.index(b"\r\n\r\n")
        response_header = self.buffer[:header_end].decode("ascii")
        self.buffer = self.buffer[header_end + 4:]
        part_marker = f"--{REMOTE_STREAM_BOUNDARY}\r\n".encode("ascii")
        assert self.read_until(part_marker)
        self.buffer = self.buffer[self.buffer.index(part_marker):]
        assert self.read_until(b"\r\n\r\n")
        part_header_end = self.buffer.index(b"\r\n\r\n")
        part_header = self.buffer[:part_header_end].decode("ascii")
        length = int(re.search(r"Content-Length: (\d+)", part_header).group(1))
        self.buffer = self.buffer[part_header_end + 4:]
        max_read_count = 10000
        for _ in range(max_read_count):
            if len(self.buffer) >= length:
                break
            self.buffer += self.sock.recv(65536)
        return response_header, part_header, self.buffer[:length]

    def close(self):
        self.sock.close()


class BackgroundFrames:
    """Calls `offer` repeatedly on a thread until closed, so the test thread can block reading the stream."""

    def __init__(self, offer):
        self.offer = offer
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        max_offer_count = 100000
        for _ in range(max_offer_count):
            if self.stopped.is_set():
                return
            self.offer()
            time.sleep(WAIT_POLL_SECONDS)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc_info):
        self.stopped.set()
        self.thread.join(WAIT_TIMEOUT_SECONDS)


def offer_until(remote_input, pygame_module, predicate, surface=None):
    surface = surface or FakeSurface()

    def offer_and_check():
        remote_input.offer_frame(surface, pygame_module)
        return predicate()

    return wait_for(offer_and_check)


# ---------------------------------------------------------------------------------------------
# Construction and interface
# ---------------------------------------------------------------------------------------------


def test_default_port_is_the_contract():
    assert REMOTE_DEFAULT_PORT == CONTRACT_DEFAULT_PORT


def test_remote_is_a_wheel_input():
    assert issubclass(RemoteWheelInput, WheelInput)


@pytest.mark.parametrize("key", ["", None])
def test_an_empty_key_is_refused(key):
    with pytest.raises(ValueError):
        RemoteWheelInput(key, TEST_COMMANDS, port=free_port(), host=TEST_HOST)


def test_url_carries_the_port_and_key():
    port = free_port()
    wheel_input = RemoteWheelInput(TEST_KEY, TEST_COMMANDS, port=port, host=TEST_HOST)
    assert wheel_input.url("192.0.2.7") == f"http://192.0.2.7:{port}/?key={TEST_KEY}"


def test_stop_frees_the_port():
    port = free_port()
    RemoteWheelInput(TEST_KEY, TEST_COMMANDS, port=port, host=TEST_HOST).start().stop()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        with pytest.raises(OSError):
            probe.connect((TEST_HOST, port))


def test_before_any_control_the_state_is_neutral(remote):
    assert wait_for(lambda: remote.latest().report_count > 0)
    state = remote.latest()
    assert (state.steering_norm, state.throttle_norm, state.brake_norm, state.clutch_norm) == (0.0, 0.0, 0.0, 0.0)
    assert (state.buttons, state.hat) == (0, HAT_NEUTRAL)
    assert not remote.is_stale()


# ---------------------------------------------------------------------------------------------
# Key: every route refuses a wrong or missing key
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", [WRONG_KEY, None, ""])
@pytest.mark.parametrize(
    "method,path,body",
    [("GET", "/", None), ("GET", "/stream", None), ("GET", "/nowhere", None),
     ("POST", "/control", VALID_CONTROL), ("POST", "/command", {"name": "cycle_camera"})],
)
def test_wrong_or_missing_key_is_forbidden(remote, key, method, path, body):
    status, _content_type, _body = request(remote, method, path, body, key=key)
    assert status == 403


def test_a_forbidden_control_changes_nothing(remote):
    status, _content_type, _body = request(remote, "POST", "/control", dict(VALID_CONTROL, throttle=1.0), key=WRONG_KEY)
    assert status == 403
    time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS / 5)
    state = remote.latest()
    assert (state.steering_norm, state.throttle_norm) == (0.0, 0.0)


def test_a_forbidden_command_queues_nothing(remote):
    request(remote, "POST", "/command", {"name": "cycle_camera"}, key=WRONG_KEY)
    request(remote, "POST", "/command", {"name": "shift_up"}, key=WRONG_KEY)
    assert remote.poll() == {}
    time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS / 5)
    assert remote.latest().shift_up_count == 0


# ---------------------------------------------------------------------------------------------
# GET routes
# ---------------------------------------------------------------------------------------------


def test_root_serves_the_page(remote):
    status, content_type, body = request(remote, "GET", "/")
    assert status == 200
    assert content_type.startswith("text/html")
    with open(REMOTE_PAGE_PATH, encoding="utf-8") as handle:
        assert body.decode("utf-8") == handle.read()


@pytest.mark.parametrize("method,path,body", [("GET", "/nowhere", None), ("POST", "/nowhere", VALID_CONTROL)])
def test_unknown_routes_are_not_found(remote, method, path, body):
    status, _content_type, _body = request(remote, method, path, body)
    assert status == 404


# ---------------------------------------------------------------------------------------------
# POST /control
# ---------------------------------------------------------------------------------------------


def test_control_replies_with_the_published_status(remote):
    status_line = {"speed_kmh": 42, "gear": "D", "assist": "blend 30%"}
    remote.publish_status(status_line)
    status, content_type, body = post_control(remote)
    assert status == 200
    assert content_type == "application/json"
    assert json.loads(body) == status_line
    assert json.loads(remote.status_json()) == status_line


def test_control_replies_an_empty_object_before_any_status(remote):
    status, _content_type, body = post_control(remote)
    assert status == 200
    assert json.loads(body) == {}


def test_steering_passes_through_and_pedals_follow_the_default_curves(remote):
    status, _content_type, _body = post_control(remote, steering=-0.75, throttle=HALF_TRAVEL, brake=HALF_TRAVEL,
                                                clutch=HALF_TRAVEL)
    assert status == 200
    calibration = WheelCalibration()
    expected_throttle = HALF_TRAVEL**calibration.throttle_response_gamma
    expected_brake = HALF_TRAVEL**calibration.brake_response_gamma
    assert wait_for(lambda: abs(remote.latest().throttle_norm - expected_throttle) < FLOAT_TOLERANCE)
    state = remote.latest()
    assert state.steering_norm == pytest.approx(-0.75)
    assert state.brake_norm == pytest.approx(expected_brake)
    assert state.clutch_norm == pytest.approx(HALF_TRAVEL)


def test_pedals_follow_a_custom_calibration_curve():
    calibration = WheelCalibration(throttle_response_gamma=CUSTOM_THROTTLE_GAMMA, brake_response_gamma=CUSTOM_BRAKE_GAMMA)
    wheel_input = RemoteWheelInput(TEST_KEY, TEST_COMMANDS, port=free_port(), host=TEST_HOST,
                                   calibration=calibration).start()
    try:
        post_control(wheel_input, throttle=HALF_TRAVEL, brake=HALF_TRAVEL)
        expected_throttle = HALF_TRAVEL**CUSTOM_THROTTLE_GAMMA
        assert wait_for(lambda: abs(wheel_input.latest().throttle_norm - expected_throttle) < FLOAT_TOLERANCE)
        assert wheel_input.latest().brake_norm == pytest.approx(HALF_TRAVEL**CUSTOM_BRAKE_GAMMA)
    finally:
        wheel_input.stop()


@pytest.mark.parametrize("travel", [0.0, 1.0])
def test_pedal_travel_ends_map_to_the_ends(remote, travel):
    post_control(remote, throttle=travel, brake=travel, clutch=travel, steering=travel)
    assert wait_for(lambda: remote.latest().steering_norm == travel)
    state = remote.latest()
    assert (state.throttle_norm, state.brake_norm, state.clutch_norm) == (travel, travel, travel)


def test_silence_lifts_the_pedals_but_holds_the_steering(remote):
    post_control(remote, steering=0.4, throttle=1.0, brake=1.0, clutch=1.0)
    assert wait_for(lambda: remote.latest().throttle_norm == 1.0)
    time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS * 1.5)
    assert wait_for(lambda: remote.latest().throttle_norm == 0.0)
    state = remote.latest()
    assert (state.brake_norm, state.clutch_norm) == (0.0, 0.0)
    assert state.steering_norm == pytest.approx(0.4)


def test_controls_well_inside_the_timeout_keep_the_pedals(remote):
    post_control(remote, throttle=1.0)
    assert wait_for(lambda: remote.latest().throttle_norm == 1.0)
    post_count = 10
    for _ in range(post_count):
        post_control(remote, throttle=1.0)
        time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS / 5)
        assert remote.latest().throttle_norm == 1.0


def test_a_control_after_the_timeout_presses_again(remote):
    post_control(remote, throttle=1.0)
    time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS * 1.5)
    assert wait_for(lambda: remote.latest().throttle_norm == 0.0)
    post_control(remote, throttle=1.0)
    assert wait_for(lambda: remote.latest().throttle_norm == 1.0)


@pytest.mark.parametrize(
    "raw_body",
    [b"not json", b"[0.1, 0.2, 0.3, 0.4]", b'"steering"',
     b'{"steering": NaN, "throttle": 0, "brake": 0, "clutch": 0}',
     b'{"steering": 0, "throttle": Infinity, "brake": 0, "clutch": 0}',
     b'{"steering": 1.5, "throttle": 0, "brake": 0, "clutch": 0}',
     b'{"steering": 0, "throttle": -0.1, "brake": 0, "clutch": 0}',
     b'{"steering": 0, "throttle": 0, "brake": 0}',
     b'{"steering": true, "throttle": 0, "brake": 0, "clutch": 0}'],
)
def test_bad_control_bodies_are_rejected_and_change_nothing(remote, raw_body):
    status, _content_type, _body = request(remote, "POST", "/control", raw_body=raw_body)
    assert status == 400
    time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS / 5)
    state = remote.latest()
    assert (state.steering_norm, state.throttle_norm) == (0.0, 0.0)


@pytest.mark.parametrize("raw_body", [b"", b"x" * (REMOTE_MAX_BODY_BYTES + 1)])
def test_empty_or_oversized_bodies_are_refused(remote, raw_body):
    status, _content_type, _body = request(remote, "POST", "/control", raw_body=raw_body)
    assert status == 413


def test_the_server_keeps_serving_after_bad_requests(remote):
    request(remote, "POST", "/control", raw_body=b"not json")
    request(remote, "POST", "/command", {"name": "format_disk"})
    status, _content_type, _body = post_control(remote)
    assert status == 200


# ---------------------------------------------------------------------------------------------
# parse_control
# ---------------------------------------------------------------------------------------------


def test_parse_control_returns_floats_in_contract_order():
    body = json.dumps({"clutch": 0.4, "brake": 0.3, "throttle": 0.2, "steering": -0.1})
    assert parse_control(body) == (-0.1, 0.2, 0.3, 0.4)


def test_parse_control_accepts_integers_and_bytes():
    parsed = parse_control(b'{"steering": -1, "throttle": 1, "brake": 0, "clutch": 0}')
    assert parsed == (-1.0, 1.0, 0.0, 0.0)
    assert all(type(value) is float for value in parsed)


@pytest.mark.parametrize("name", ["steering", "throttle", "brake", "clutch"])
@pytest.mark.parametrize("bad_value", ["0.5", None, True, False, [0.5], {"value": 0.5}])
def test_parse_control_rejects_non_numbers(name, bad_value):
    with pytest.raises(ValueError):
        parse_control(json.dumps(dict(VALID_CONTROL, **{name: bad_value})))


@pytest.mark.parametrize("name", ["steering", "throttle", "brake", "clutch"])
def test_parse_control_rejects_a_missing_field(name):
    control = dict(VALID_CONTROL)
    del control[name]
    with pytest.raises(ValueError):
        parse_control(json.dumps(control))


@pytest.mark.parametrize("name", ["steering", "throttle", "brake", "clutch"])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_parse_control_rejects_non_finite(name, bad_value):
    with pytest.raises(ValueError):
        parse_control(json.dumps(dict(VALID_CONTROL, **{name: bad_value})))


@pytest.mark.parametrize(
    "name,bad_value",
    [("steering", -1.0001), ("steering", 1.0001), ("throttle", -0.01), ("throttle", 1.01), ("brake", -0.01),
     ("brake", 1.01), ("clutch", -0.01), ("clutch", 1.01)],
)
def test_parse_control_rejects_out_of_range(name, bad_value):
    with pytest.raises(ValueError):
        parse_control(json.dumps(dict(VALID_CONTROL, **{name: bad_value})))


@pytest.mark.parametrize("body", ["[]", "1", '"x"', "null", "not json", ""])
def test_parse_control_rejects_non_objects(body):
    with pytest.raises(ValueError):
        parse_control(body)


# ---------------------------------------------------------------------------------------------
# POST /command
# ---------------------------------------------------------------------------------------------


def test_an_allowed_command_is_queued_once(remote):
    status, _content_type, _body = request(remote, "POST", "/command", {"name": "cycle_camera"})
    assert status == 204
    assert remote.poll() == {"cycle_camera": True}
    assert remote.poll() == {}


def test_several_commands_arrive_together(remote):
    for name in ("cycle_camera", "toggle_night", "restart"):
        assert request(remote, "POST", "/command", {"name": name})[0] == 204
    assert remote.poll() == {"cycle_camera": True, "toggle_night": True, "restart": True}


def test_poll_takes_a_wheel_state_like_the_wheel_buttons(remote):
    request(remote, "POST", "/command", {"name": "toggle_night"})
    assert remote.poll(WheelState()) == {"toggle_night": True}


@pytest.mark.parametrize(
    "raw_body",
    [json.dumps({"name": "format_disk"}).encode(), json.dumps({"name": "ffb_gain_up"}).encode(),
     json.dumps({"name": 3}).encode(), json.dumps({"name": None}).encode(), json.dumps({}).encode(),
     json.dumps(["cycle_camera"]).encode(), b"not json"],
)
def test_unknown_or_malformed_commands_are_rejected(remote, raw_body):
    status, _content_type, _body = request(remote, "POST", "/command", raw_body=raw_body)
    assert status == 400
    assert remote.poll() == {}


@pytest.mark.parametrize("name,field", [("shift_up", "shift_up_count"), ("shift_down", "shift_down_count")])
def test_gear_commands_become_paddle_counts_not_demo_commands(remote, name, field):
    press_count = 3
    for _ in range(press_count):
        assert request(remote, "POST", "/command", {"name": name})[0] == 204
    assert wait_for(lambda: getattr(remote.latest(), field) == press_count)
    assert remote.poll() == {}
    other_field = "shift_down_count" if field == "shift_up_count" else "shift_up_count"
    assert getattr(remote.latest(), other_field) == 0


def test_paddle_counts_are_monotonic(remote):
    request(remote, "POST", "/command", {"name": "shift_up"})
    assert wait_for(lambda: remote.latest().shift_up_count == 1)
    time.sleep(REMOTE_CONTROL_TIMEOUT_SECONDS / 5)
    assert remote.latest().shift_up_count == 1
    request(remote, "POST", "/command", {"name": "shift_up"})
    assert wait_for(lambda: remote.latest().shift_up_count == 2)


def test_gear_commands_are_accepted_even_when_not_listed():
    assert not set(CONTRACT_GEAR_COMMANDS) & TEST_COMMANDS
    wheel_input = RemoteWheelInput(TEST_KEY, TEST_COMMANDS, port=free_port(), host=TEST_HOST)
    assert all(wheel_input.receive_command(name) for name in CONTRACT_GEAR_COMMANDS)


# ---------------------------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------------------------


def test_status_json_is_the_latest_published(remote):
    assert json.loads(remote.status_json()) == {}
    remote.publish_status({"assist": "off"})
    remote.publish_status({"assist": "full", "speed_kmh": 3})
    assert json.loads(remote.status_json()) == {"assist": "full", "speed_kmh": 3}


def test_status_published_from_another_thread_is_seen(remote):
    publisher = threading.Thread(target=remote.publish_status, args=({"gear": "R"},))
    publisher.start()
    publisher.join()
    assert json.loads(post_control(remote)[2]) == {"gear": "R"}


# ---------------------------------------------------------------------------------------------
# Key file
# ---------------------------------------------------------------------------------------------


def test_key_file_is_created_owner_only_and_stable(tmp_path):
    path = tmp_path / "nested" / "dir" / "remote_key"
    first_key = load_or_create_key(str(path))
    assert first_key
    assert stat.S_IMODE(os.stat(path).st_mode) == KEY_FILE_MODE
    assert load_or_create_key(str(path)) == first_key
    assert path.read_text().strip() == first_key


def test_key_is_url_safe(tmp_path):
    key = load_or_create_key(str(tmp_path / "remote_key"))
    assert re.fullmatch(r"[A-Za-z0-9_-]+", key)


def test_two_new_key_files_get_different_keys(tmp_path):
    assert load_or_create_key(str(tmp_path / "first")) != load_or_create_key(str(tmp_path / "second"))


def test_an_existing_key_file_is_read_and_stripped(tmp_path):
    path = tmp_path / "remote_key"
    path.write_text("  handmade-key \n")
    assert load_or_create_key(str(path)) == "handmade-key"


@pytest.mark.parametrize("content", ["", "\n", "   \n"])
def test_an_empty_key_file_is_an_error(tmp_path, content):
    path = tmp_path / "remote_key"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_or_create_key(str(path))
    assert path.read_text() == content


# ---------------------------------------------------------------------------------------------
# Frame sink and the MJPEG stream
# ---------------------------------------------------------------------------------------------


def test_no_frame_is_encoded_without_a_stream_client(remote):
    fake_pygame = FakePygame()
    offer_count = 5
    for _ in range(offer_count):
        remote.offer_frame(FakeSurface(), fake_pygame)
        time.sleep(1.0 / REMOTE_STREAM_FPS)
    assert fake_pygame.scaled_sizes == []
    assert fake_pygame.saved_count == 0


def test_a_stream_client_receives_jpeg_parts(remote):
    fake_pygame = FakePygame()
    client = StreamClient(remote)
    try:
        assert offer_until(remote, fake_pygame, lambda: fake_pygame.saved_count > 0)
        response_header, part_header, jpeg = client.read_part()
    finally:
        client.close()
    assert response_header.startswith("HTTP/1.1 200")
    assert f"multipart/x-mixed-replace; boundary={REMOTE_STREAM_BOUNDARY}" in response_header
    assert "Content-Type: image/jpeg" in part_header
    assert jpeg == FAKE_JPEG


def test_frames_are_scaled_to_the_stream_width_keeping_the_aspect(remote):
    fake_pygame = FakePygame()
    client = StreamClient(remote)
    try:
        assert offer_until(remote, fake_pygame, lambda: fake_pygame.saved_count > 0)
    finally:
        client.close()
    source_width_px, source_height_px = SOURCE_FRAME_SIZE_PX
    expected_height_px = round(source_height_px * REMOTE_STREAM_WIDTH_PX / source_width_px)
    assert fake_pygame.scaled_sizes[0] == (REMOTE_STREAM_WIDTH_PX, expected_height_px)


def test_frames_are_encoded_at_most_at_the_stream_rate(remote):
    fake_pygame = FakePygame()
    client = StreamClient(remote)
    try:
        assert offer_until(remote, fake_pygame, lambda: fake_pygame.saved_count > 0)
        encoded_before = fake_pygame.saved_count
        burst_seconds = 0.5
        burst_started_s = time.monotonic()
        burst_offer_interval_seconds = 0.001
        max_offer_count = int(burst_seconds / burst_offer_interval_seconds)
        for _ in range(max_offer_count):
            if time.monotonic() - burst_started_s >= burst_seconds:
                break
            remote.offer_frame(FakeSurface(), fake_pygame)
            time.sleep(burst_offer_interval_seconds)
        encoded_in_burst = fake_pygame.saved_count - encoded_before
    finally:
        client.close()
    assert 1 <= encoded_in_burst <= math.ceil(burst_seconds * REMOTE_STREAM_FPS) + 1


def test_encoding_stops_after_the_stream_client_leaves(remote):
    fake_pygame = FakePygame()
    client = StreamClient(remote)
    assert offer_until(remote, fake_pygame, lambda: fake_pygame.saved_count > 0)
    client.close()
    previous = {"count": -1}

    def encoding_stopped():
        settled = fake_pygame.saved_count == previous["count"]
        previous["count"] = fake_pygame.saved_count
        time.sleep(2.0 / REMOTE_STREAM_FPS)
        return settled

    assert offer_until(remote, fake_pygame, encoding_stopped)


def test_stop_ends_an_open_stream(remote):
    fake_pygame = FakePygame()
    client = StreamClient(remote)
    try:
        assert offer_until(remote, fake_pygame, lambda: fake_pygame.saved_count > 0)
        stopper = threading.Thread(target=remote.stop)
        stopper.start()
        stopper.join(WAIT_TIMEOUT_SECONDS)
        assert not stopper.is_alive()
        client.sock.settimeout(WAIT_TIMEOUT_SECONDS)
        max_read_count = 10000
        for _ in range(max_read_count):
            if not client.sock.recv(65536):
                break
        else:
            pytest.fail("the stream never closed after stop()")
    finally:
        client.close()


def test_a_real_pygame_frame_is_a_scaled_jpeg(remote):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    surface = pygame.Surface(SOURCE_FRAME_SIZE_PX)
    surface.fill((200, 30, 60))
    client = StreamClient(remote)
    try:
        with BackgroundFrames(lambda: remote.offer_frame(surface, pygame)):
            _response_header, _part_header, jpeg = client.read_part()
    finally:
        client.close()
    assert jpeg.startswith(JPEG_MAGIC)
    decoded = pygame.image.load(io.BytesIO(jpeg), "frame.jpg")
    source_width_px, source_height_px = SOURCE_FRAME_SIZE_PX
    assert decoded.get_size() == (REMOTE_STREAM_WIDTH_PX, round(source_height_px * REMOTE_STREAM_WIDTH_PX / source_width_px))


# ---------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def page_source():
    with open(REMOTE_PAGE_PATH, encoding="utf-8") as handle:
        return handle.read()


def test_every_page_command_is_one_run_demo_accepts(page_source, run_demo):
    page_commands = set(re.findall(r'data-command="([^"]+)"', page_source))
    accepted = set(run_demo.WHEEL_BUTTON_COMMANDS.values()) | set(run_demo.WHEEL_BUTTON_COMBOS.values())
    assert page_commands
    assert page_commands <= accepted


def test_every_command_the_page_script_sends_is_accepted(page_source, run_demo):
    sent_names = set(re.findall(r'sendCommand\("([a-z_]+)"\)', page_source))
    sent_names |= set(re.findall(r'"(shift_(?:up|down))"', page_source))
    accepted = set(run_demo.WHEEL_BUTTON_COMMANDS.values()) | set(run_demo.WHEEL_BUTTON_COMBOS.values())
    assert set(CONTRACT_GEAR_COMMANDS) <= sent_names
    assert sent_names - set(CONTRACT_GEAR_COMMANDS) <= accepted


def test_page_reads_the_key_from_the_query_string(page_source):
    assert re.search(r'new URLSearchParams\(location\.search\)\.get\("key"\)', page_source)
    assert "encodeURIComponent(KEY)" in page_source


def test_page_posts_to_the_control_and_command_routes_and_streams(page_source):
    assert 'post("/control"' in page_source
    assert 'post("/command"' in page_source
    assert '"/stream" + query' in page_source
    for field in ("steering", "throttle", "brake", "clutch"):
        assert re.search(rf"\b{field}:", page_source)


def test_page_has_no_external_resources(page_source):
    assert not re.search(r'(src|href)="https?://', page_source)


# ---------------------------------------------------------------------------------------------
# run_demo wiring
# ---------------------------------------------------------------------------------------------


def test_synthetic_input_is_unchanged(run_demo):
    wheel_input, button_map = run_demo.build_wheel_input("synthetic", None, None)
    assert isinstance(wheel_input, SyntheticWheelInput)
    assert button_map is None


def test_wheel_input_still_opens_the_g920_with_its_button_map(run_demo, monkeypatch):
    from pufferlib.wheel_demo.input import g920

    opened = []

    class FakeG920:
        def __init__(self, calibration, button_map):
            opened.append((calibration, button_map))

        def start(self):
            return self

    monkeypatch.setattr(g920, "G920HidInput", FakeG920)
    wheel_input, button_map = run_demo.build_wheel_input("wheel", None, None)
    assert isinstance(wheel_input, FakeG920)
    assert isinstance(button_map, g920.ButtonMap)
    assert opened == [(WheelCalibration(), button_map)]


def test_remote_input_is_built_started_with_the_demo_commands(run_demo):
    port = free_port()
    wheel_input, button_map = run_demo.build_wheel_input("remote", None, None, port, TEST_KEY)
    try:
        assert isinstance(wheel_input, RemoteWheelInput)
        assert button_map is None
        assert wheel_input.key == TEST_KEY
        assert wheel_input.port == port
        expected = set(run_demo.WHEEL_BUTTON_COMMANDS.values()) | set(run_demo.WHEEL_BUTTON_COMBOS.values())
        assert wheel_input.allowed_commands == expected
        assert request(wheel_input, "POST", "/command", {"name": "restart"})[0] == 204
        assert wheel_input.poll() == {"restart": True}
    finally:
        wheel_input.stop()


def test_remote_input_uses_the_calibration_file(run_demo, tmp_path):
    path = tmp_path / "calibration.json"
    calibration = WheelCalibration(throttle_response_gamma=CUSTOM_THROTTLE_GAMMA)
    calibration.to_json(path)
    wheel_input, _button_map = run_demo.build_wheel_input("remote", str(path), None, free_port(), TEST_KEY)
    try:
        assert wheel_input.calibration == calibration
    finally:
        wheel_input.stop()


def test_remote_input_without_a_key_uses_the_kept_key_file(run_demo, monkeypatch, tmp_path):
    key_path = tmp_path / "cache" / "remote_key"
    monkeypatch.setattr(run_demo, "REMOTE_KEY_PATH", str(key_path))
    wheel_input, _button_map = run_demo.build_wheel_input("remote", None, None, free_port(), None)
    try:
        assert wheel_input.key == key_path.read_text().strip()
    finally:
        wheel_input.stop()


def test_remote_cli_flags_default_to_the_contract(run_demo):
    parser = capture_parser(run_demo)
    args = parser.parse_args([])
    assert args.input == "wheel"
    assert args.remote_port == CONTRACT_DEFAULT_PORT
    assert args.remote_key is None
    args = parser.parse_args(["--input", "remote", "--remote-port", "9100", "--remote-key", "abc"])
    assert (args.input, args.remote_port, args.remote_key) == ("remote", 9100, "abc")


class ParserCaptured(Exception):
    pass


def capture_parser(run_demo):
    import argparse

    captured = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise ParserCaptured

    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(ParserCaptured):
            run_demo.main()
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]


def fake_dynamic(active=False, share=0.0):
    return SimpleNamespace(active=active, share=share)


def fake_coadapt(active=False, share=0.0):
    return SimpleNamespace(active=active, share=share)


def test_remote_status_shows_speed_gear_and_assist(run_demo):
    human = {"speed_mps": -36.0 * MPS_PER_KMH, "gear": 1, "gearbox_mode": run_demo.GEARBOX_MODE_AUTOMATIC}
    assist = {"mode": binding.ASSIST_MODE_OFF, "blend_target": 0.0}
    status = run_demo.remote_status(human, assist, fake_coadapt(), fake_dynamic())
    assert status == {
        "speed_kmh": 36,
        "gear": run_demo.gear_label(1, run_demo.GEARBOX_MODE_AUTOMATIC),
        "assist": run_demo.ASSIST_MODE_NAMES[binding.ASSIST_MODE_OFF],
    }
    json.dumps(status)


def test_remote_status_without_a_human_is_only_the_assist(run_demo):
    assist = {"mode": binding.ASSIST_MODE_FULL, "blend_target": 0.0}
    status = run_demo.remote_status(None, assist, fake_coadapt(), fake_dynamic())
    assert status == {"assist": run_demo.ASSIST_MODE_NAMES[binding.ASSIST_MODE_FULL]}


@pytest.mark.parametrize(
    "assist_mode,blend_target,coadapt,dynamic,expected",
    [
        (binding.ASSIST_MODE_BLEND, 0.3, fake_coadapt(), fake_dynamic(), "blend 30%"),
        (binding.ASSIST_MODE_BLEND, 0.2, fake_coadapt(True, 0.6), fake_dynamic(), "co-adapt 60%"),
        (binding.ASSIST_MODE_BLEND, 0.2, fake_coadapt(True, 0.6), fake_dynamic(True, 0.45), "dynamic blend 45%"),
        (binding.ASSIST_MODE_NUDGE, 0.5, fake_coadapt(), fake_dynamic(), None),
    ],
)
def test_remote_status_names_the_assist_like_the_hud(run_demo, assist_mode, blend_target, coadapt, dynamic, expected):
    assist = {"mode": assist_mode, "blend_target": blend_target}
    status = run_demo.remote_status(None, assist, coadapt, dynamic)
    assert status["assist"] == (expected or run_demo.ASSIST_MODE_NAMES[assist_mode])


class MainReached(Exception):
    pass


class RemoteFakeLoop:
    def __init__(self, config, wheel):
        self.env = SimpleNamespace(set_assist_mode=lambda mode, blend: None)
        self.human_agent_idx = 0
        self.substep_dt = 0.05
        self.dt = 0.3

    @property
    def observations(self):
        raise MainReached


class FakeRemote:
    def url(self, address=None):
        return f"http://test/?key={TEST_KEY}"


def run_main_until_reached(run_demo, monkeypatch, tmp_path, extra_args, wheel_input, built_window=None):
    def refuse_force_feedback(*args, **kwargs):
        raise AssertionError("force feedback must not start")

    def fake_window(*args, **kwargs):
        built_window.update(kwargs)
        raise MainReached

    monkeypatch.setattr(run_demo, "build_wheel_input", lambda *args: (wheel_input, None))
    monkeypatch.setattr(run_demo, "WheelDemoLoop", RemoteFakeLoop)
    monkeypatch.setattr(run_demo, "choose_town", lambda town: town or "Town01")
    monkeypatch.setattr(run_demo, "CarlaDemoWindow", fake_window)
    monkeypatch.setattr(run_demo, "ForceFeedbackLoop", refuse_force_feedback)
    monkeypatch.setattr(run_demo, "platform_force_output", refuse_force_feedback)
    argv = ["run_demo.py", "--town", "Town01", "--calibration", str(tmp_path / "none.json"),
            "--buttons", str(tmp_path / "none_buttons.json")] + extra_args
    monkeypatch.setattr(run_demo.sys, "argv", argv)
    with pytest.raises(MainReached):
        run_demo.main()


def test_main_refuses_force_feedback_with_remote_input(run_demo, monkeypatch, tmp_path, capsys):
    run_main_until_reached(run_demo, monkeypatch, tmp_path, ["--input", "remote", "--ffb"], FakeRemote())
    captured = capsys.readouterr()
    assert "force feedback is off" in captured.err
    assert f"key={TEST_KEY}" in captured.out


@pytest.mark.parametrize("input_name,expects_sink", [("remote", True), ("synthetic", False)])
def test_main_gives_the_window_the_remote_as_frame_sink(run_demo, monkeypatch, tmp_path, input_name, expects_sink):
    wheel_input = FakeRemote()
    built_window = {}
    run_main_until_reached(run_demo, monkeypatch, tmp_path, ["--input", input_name, "--render"], wheel_input,
                           built_window)
    assert built_window["frame_sink"] is (wheel_input if expects_sink else None)


# ---------------------------------------------------------------------------------------------
# The window hands each drawn frame to its sink
# ---------------------------------------------------------------------------------------------


class DrawScene:
    def __init__(self, carla, town, env, human_agent_idx, substep_dt_s, host, port, image_size_px):
        self.closed = False

    def set_night(self, night):
        pass

    def set_camera_mode(self, camera_mode):
        pass

    def update(self, human, wheel_state, elapsed_seconds):
        return {"main": object()}

    def goal_marker(self, human, image):
        return None

    def skid_marks(self, image, depth):
        return None

    def close(self):
        self.closed = True


class RecordingSink:
    def __init__(self):
        self.offers = []

    def offer_frame(self, surface, pygame):
        self.offers.append((surface, pygame))


@pytest.fixture
def drawable_window_module(monkeypatch):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from pufferlib.wheel_demo import carla_window

    monkeypatch.setitem(sys.modules, "carla", types.ModuleType("carla"))
    monkeypatch.setattr(carla_window, "CarlaScene", DrawScene)
    monkeypatch.setattr(pygame.display, "set_mode", lambda size_px, flags=0, *args, **kwargs: pygame.Surface(size_px))
    monkeypatch.setattr(pygame.display, "set_caption", lambda *args: None)
    monkeypatch.setattr(pygame.display, "flip", lambda: None)
    return carla_window, pygame


def make_drawable_window(carla_window, pygame, frame_sink):
    window = carla_window.CarlaDemoWindow(object(), "Town01", 0, WINDOW_SUBSTEP_SECONDS, frame_sink=frame_sink)
    window._surface = lambda image, mirrored=False: pygame.Surface(window.size_px)
    window.hud = SimpleNamespace(draw=lambda *args, **kwargs: None)
    return window


def test_window_defaults_to_no_frame_sink(drawable_window_module):
    carla_window, pygame = drawable_window_module
    window = carla_window.CarlaDemoWindow(object(), "Town01", 0, WINDOW_SUBSTEP_SECONDS)
    assert window.frame_sink is None


def test_window_offers_each_drawn_frame_to_its_sink(drawable_window_module):
    carla_window, pygame = drawable_window_module
    sink = RecordingSink()
    window = make_drawable_window(carla_window, pygame, sink)
    draw_count = 3
    drawn = [window.draw(None, WheelState(), 0.0) for _ in range(draw_count)]
    assert sink.offers == [(window.screen, pygame)] * drawn.count(True)
    assert len(sink.offers) == window.frames_drawn


def test_window_without_a_sink_draws_as_before(drawable_window_module):
    carla_window, pygame = drawable_window_module
    window = make_drawable_window(carla_window, pygame, None)
    assert window.draw(None, WheelState(), 0.0) is True
    assert window.frames_drawn == 1


def test_window_streams_into_a_real_remote(drawable_window_module, remote):
    carla_window, pygame = drawable_window_module
    window = make_drawable_window(carla_window, pygame, remote)
    client = StreamClient(remote)
    try:
        with BackgroundFrames(lambda: window.draw(None, WheelState(), 0.0)):
            _response_header, _part_header, jpeg = client.read_part()
    finally:
        client.close()
    assert jpeg.startswith(JPEG_MAGIC)
