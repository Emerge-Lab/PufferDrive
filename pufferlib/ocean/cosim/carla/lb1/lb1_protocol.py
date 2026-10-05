"""Length-prefixed JSON messages between lb1/leaderboard_agent.py (CARLA 0.9.10, Python 3.7) and
lb1/policy_server.py (PufferDrive venv). Standard library only: the agent imports this file as a
sibling module (the leaderboard puts the agent's directory on sys.path), the server as a package module.

Handshake, then one request/reply pair per CARLA tick:
  server -> client  {"type": "config", "partner_slots", "partner_max_abs_dz_m", "light_probe_steps_m",
                     "dynamics_source", "infraction_flags"}
  client -> server  {"type": "init", "route": shadow_ego.init_route dict}        -> {"type": "ready"}
  client -> server  {"type": "tick", "snapshot": shadow_ego.step dict,
                     "ground_truth": lb1/ground_truth.py observation or null}     -> {"type": "control",
                     "motion": [...] | null, "control": [steer, throttle, brake] | null,
                     "pd_flags": {...} | null, "carla_flags": {...} | null}
  client -> server  {"type": "finish"}                                            -> {"type": "done"}
"""

import json
import struct


HEADER = struct.Struct("!I")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
RECV_CHUNK_BYTES = 1 << 20


def send_message(sock, message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message of {len(payload)} bytes exceeds MAX_MESSAGE_BYTES")
    sock.sendall(HEADER.pack(len(payload)) + payload)


def recv_message(sock):
    """The next message, or None once the peer closed the connection between messages."""
    header = _recv_exact(sock, HEADER.size)
    if header is None:
        return None
    (length,) = HEADER.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"incoming message of {length} bytes exceeds MAX_MESSAGE_BYTES")
    payload = _recv_exact(sock, length)
    if payload is None:
        raise ConnectionError("peer closed the connection inside a message")
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock, byte_count):
    chunks = []
    remaining = byte_count
    while remaining > 0:  # every pass consumes >= 1 byte or returns
        chunk = sock.recv(min(remaining, RECV_CHUNK_BYTES))
        if not chunk:
            if chunks:
                raise ConnectionError("peer closed the connection inside a message")
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
