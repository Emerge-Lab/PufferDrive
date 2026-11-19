"""LB1 co-sim client: protocol round trips; it stays Python 3.7 and opengigaflow-free for CARLA 0.9.10's garage env."""

import ast
import socket
import subprocess
from pathlib import Path

import pytest

from opengigaflow.ocean.cosim.carla.lb1 import lb1_protocol


LB1_DIR = Path(__file__).resolve().parents[2] / "opengigaflow" / "ocean" / "cosim" / "carla" / "lb1"
CLIENT_SIDE_FILES = ("leaderboard_agent.py", "lb1_protocol.py", "run_evaluator.py")
CLIENT_PYTHON_FEATURE_VERSION = (3, 7)
GARAGE_PYTHON = Path.home() / "miniconda3" / "envs" / "garage" / "bin" / "python"
LARGE_PAYLOAD_FLOATS = 300_000  # a 5-route-km dense plan plus every Town04 driving waypoint


def test_round_trip_preserves_nested_message():
    client, server = socket.socketpair()
    message = {
        "type": "tick",
        "snapshot": {"ego": [1.5, -2.0, 0.0], "partners": [], "light_states": [[7, "Red"]]},
        "x": None,
    }
    lb1_protocol.send_message(client, message)
    assert lb1_protocol.recv_message(server) == message
    client.close()
    server.close()


def test_large_message_arrives_whole_over_chunked_reads():
    client, server = socket.socketpair()
    payload = {
        "type": "init",
        "route": {"dense_plan": [[float(k), 0.5 * k, 0.0] for k in range(LARGE_PAYLOAD_FLOATS // 3)]},
    }
    import threading

    sender = threading.Thread(target=lb1_protocol.send_message, args=(client, payload))
    sender.start()
    received = lb1_protocol.recv_message(server)
    sender.join()
    assert received == payload
    client.close()
    server.close()


def test_closed_peer_reads_as_none():
    client, server = socket.socketpair()
    client.close()
    assert lb1_protocol.recv_message(server) is None
    server.close()


@pytest.mark.parametrize("filename", CLIENT_SIDE_FILES)
def test_client_side_files_parse_as_python_37_without_opengigaflow(filename):
    source = (LB1_DIR / filename).read_text()
    tree = ast.parse(source, filename=filename, feature_version=CLIENT_PYTHON_FEATURE_VERSION)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name.split(".")[0] in ("opengigaflow", "data_utils", "torch")], imported


@pytest.mark.skipif(not GARAGE_PYTHON.is_file(), reason="garage conda env not installed on this machine")
@pytest.mark.parametrize("filename", CLIENT_SIDE_FILES)
def test_client_side_files_compile_with_garage_python(filename):
    subprocess.run([str(GARAGE_PYTHON), "-m", "py_compile", str(LB1_DIR / filename)], check=True)
