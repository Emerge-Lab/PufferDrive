"""Remote input: a touch steering wheel and pedals in a phone or desktop browser, with the demo window streamed back.

The demo serves one page (remote_page.html) and a motion-JPEG stream of what its window shows. The page posts the
wheel angle and pedal travel many times a second; every request must carry the session key, so only someone who was
given the URL can drive. Each reply carries the demo's status line (speed, gear, assist). Losing the page for REMOTE_CONTROL_TIMEOUT_SECONDS lifts both feet off the pedals.
"""

import hmac
import io
import json
import math
import os
import secrets
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pufferlib.wheel_demo.input.base import HAT_NEUTRAL, WheelInput

REMOTE_DEFAULT_PORT = 8765
REMOTE_BIND_HOST = "0.0.0.0"
REMOTE_KEY_BYTES = 9
REMOTE_PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_page.html")
# The page posts at least this often while open, so a longer silence means it was closed or lost.
REMOTE_CONTROL_TIMEOUT_SECONDS = 0.5
REMOTE_PUBLISH_INTERVAL_S = 0.005
REMOTE_MAX_BODY_BYTES = 1024
REMOTE_MAX_QUEUED_COMMANDS = 32
REMOTE_STREAM_FPS = 20.0
REMOTE_STREAM_WIDTH_PX = 960
REMOTE_STREAM_WAIT_SECONDS = 1.0
REMOTE_STREAM_BOUNDARY = "frame"
REMOTE_GEAR_COMMANDS = ("shift_up", "shift_down")


def load_or_create_key(path):
    """The session key kept in `path` (made on first use, readable by its owner only), so the URL stays the same."""
    if os.path.isfile(path):
        with open(path) as handle:
            key = handle.read().strip()
        if not key:
            raise ValueError(f"{path} is empty; delete it to make a new remote key")
        return key
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    key = secrets.token_urlsafe(REMOTE_KEY_BYTES)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(key + "\n")
    return key


def lan_address():
    """This machine's address on the network it routes through, for the printed URL; no packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "localhost"


def parse_control(body):
    """(steering_norm, throttle_travel, brake_travel, clutch_travel) from a posted JSON body; ValueError when invalid."""
    message = json.loads(body)
    if not isinstance(message, dict):
        raise ValueError("control must be a JSON object")
    ranges = {"steering": (-1.0, 1.0), "throttle": (0.0, 1.0), "brake": (0.0, 1.0), "clutch": (0.0, 1.0)}
    values = []
    for name, (low, high) in ranges.items():
        value = message.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name}={value} is outside [{low}, {high}]")
        values.append(float(value))
    return tuple(values)


class RemoteWheelInput(WheelInput):
    """Controls from the browser page. Also the demo's command source (poll) and the window's frame sink."""

    def __init__(self, key, allowed_commands, port=REMOTE_DEFAULT_PORT, host=REMOTE_BIND_HOST, calibration=None):
        super().__init__(calibration)
        if not key:
            raise ValueError("the remote input needs a non-empty key")
        self.key = key
        self.port = int(port)
        self.host = host
        self.allowed_commands = frozenset(allowed_commands)
        self._lock = threading.Lock()
        self._controls = (0.0, 0.0, 0.0, 0.0)
        self._last_control_s = None
        self._shift_up = 0
        self._shift_down = 0
        self._commands = deque(maxlen=REMOTE_MAX_QUEUED_COMMANDS)
        self._status_json = b"{}"
        self._frame_condition = threading.Condition()
        self._frame_jpeg = None
        self._frame_idx = 0
        self._stream_clients = 0
        self._next_frame_s = 0.0
        self._server = None
        self._server_thread = None

    def url(self, address=None):
        return f"http://{address or lan_address()}:{self.port}/?key={self.key}"

    def _open(self):
        with open(REMOTE_PAGE_PATH, encoding="utf-8") as handle:
            self._page = handle.read().encode("utf-8")
        self._server = ThreadingHTTPServer((self.host, self.port), _handler_for(self))
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(target=self._server.serve_forever, name="remote-http", daemon=True)
        self._server_thread.start()

    def _close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._frame_condition:
            self._frame_condition.notify_all()

    def _read_state(self):
        now_s = time.monotonic()
        with self._lock:
            steering_norm, throttle_travel, brake_travel, clutch_travel = self._controls
            if self._last_control_s is None or now_s - self._last_control_s > REMOTE_CONTROL_TIMEOUT_SECONDS:
                throttle_travel = brake_travel = clutch_travel = 0.0
            shift_up, shift_down = self._shift_up, self._shift_down
            self._shift_up = self._shift_down = 0
        # the same pedal curves as the wheel, so a press of the same travel drives the same
        self._publish(
            self._state,
            steering_norm,
            throttle_travel**self.calibration.throttle_response_gamma,
            brake_travel**self.calibration.brake_response_gamma,
            clutch_travel,
            0,
            HAT_NEUTRAL,
            shift_up,
            shift_down,
            now_s,
        )
        time.sleep(REMOTE_PUBLISH_INTERVAL_S)

    def poll(self, _wheel_state=None):
        """Demo commands pressed on the page since the last poll, as {name: True} like the wheel's buttons."""
        with self._lock:
            names, self._commands = list(self._commands), deque(maxlen=REMOTE_MAX_QUEUED_COMMANDS)
        return {name: True for name in names}

    def publish_status(self, status):
        """What the page shows above the video, since its own wheel and buttons cover the HUD's readouts."""
        encoded = json.dumps(status).encode("utf-8")
        with self._lock:
            self._status_json = encoded

    def status_json(self):
        with self._lock:
            return self._status_json

    def receive_control(self, steering_norm, throttle_travel, brake_travel, clutch_travel):
        with self._lock:
            self._controls = (steering_norm, throttle_travel, brake_travel, clutch_travel)
            self._last_control_s = time.monotonic()

    def receive_command(self, name):
        """False for a name the demo does not know, so a stray request cannot queue anything."""
        if name not in self.allowed_commands and name not in REMOTE_GEAR_COMMANDS:
            return False
        with self._lock:
            if name == "shift_up":
                self._shift_up += 1
            elif name == "shift_down":
                self._shift_down += 1
            else:
                self._commands.append(name)
        return True

    def offer_frame(self, surface, pygame):
        """Called by the window after each drawn frame; keeps a scaled JPEG only while someone is watching."""
        now_s = time.monotonic()
        if self._stream_clients == 0 or now_s < self._next_frame_s:
            return
        self._next_frame_s = now_s + 1.0 / REMOTE_STREAM_FPS
        width_px, height_px = surface.get_size()
        scaled_height_px = round(height_px * REMOTE_STREAM_WIDTH_PX / width_px)
        scaled = pygame.transform.scale(surface, (REMOTE_STREAM_WIDTH_PX, scaled_height_px))
        encoded = io.BytesIO()
        pygame.image.save(scaled, encoded, "frame.jpg")
        with self._frame_condition:
            self._frame_jpeg = encoded.getvalue()
            self._frame_idx += 1
            self._frame_condition.notify_all()

    def _next_frame(self, last_frame_idx):
        with self._frame_condition:
            self._frame_condition.wait_for(
                lambda: self._frame_idx != last_frame_idx or self._server is None, timeout=REMOTE_STREAM_WAIT_SECONDS
            )
            return self._frame_idx, self._frame_jpeg


def _handler_for(remote):
    class RemoteHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def _authorised(self):
            given = parse_qs(urlparse(self.path).query).get("key", [""])[0]
            return hmac.compare_digest(given.encode("utf-8"), remote.key.encode("utf-8"))

        def _reply(self, status, body=b"", content_type="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authorised():
                self._reply(403, b"wrong or missing key\n")
                return
            route = urlparse(self.path).path
            if route == "/":
                self._reply(200, remote._page, "text/html; charset=utf-8")
            elif route == "/stream":
                self._stream()
            else:
                self._reply(404, b"not found\n")

        def do_POST(self):
            if not self._authorised():
                self._reply(403, b"wrong or missing key\n")
                return
            length_header = self.headers.get("Content-Length") or "0"
            length = int(length_header) if length_header.isdigit() else 0
            if not 0 < length <= REMOTE_MAX_BODY_BYTES:
                self._reply(413, b"body too large or empty\n")
                return
            body = self.rfile.read(length)
            route = urlparse(self.path).path
            try:
                if route == "/control":
                    remote.receive_control(*parse_control(body))
                    self._reply(200, remote.status_json(), "application/json")
                    return
                elif route == "/command":
                    message = json.loads(body)
                    name = message.get("name") if isinstance(message, dict) else None
                    if not isinstance(name, str) or not remote.receive_command(name):
                        raise ValueError(f"unknown command {name!r}")
                else:
                    self._reply(404, b"not found\n")
                    return
            except ValueError as error:
                self._reply(400, f"{error}\n".encode("utf-8"))
                return
            self._reply(204)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={REMOTE_STREAM_BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            with remote._frame_condition:
                remote._stream_clients += 1
            frame_idx = None
            try:
                while remote._server is not None:
                    frame_idx, jpeg = remote._next_frame(frame_idx)
                    if jpeg is None:
                        continue
                    header = (
                        f"--{REMOTE_STREAM_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                    )
                    self.wfile.write(header.encode("ascii") + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with remote._frame_condition:
                    remote._stream_clients -= 1

    return RemoteHandler
