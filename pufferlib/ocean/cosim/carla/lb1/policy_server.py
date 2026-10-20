"""Shadow env + policy process behind lb1/leaderboard_agent.py (CARLA Leaderboard 1.0, CARLA 0.9.10).

Started by the agent in the PufferDrive venv, one process per route, talking lb1_protocol over an
inherited socket: `init` builds the ShadowEgo for the route, every `tick` runs one policy step and
answers with the ego command (teleport pose or control values) plus both infraction sources,
`finish` flushes the per-route outputs. Options come from the COSIM_* environment variables
(shadow_ego.py), inherited from the evaluator.

usage: policy_server.py --checkpoint <.pt or run dir> --socket-fd <fd> --route-tag <name>
"""

import argparse
import socket
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.carla.lb1 import lb1_protocol
from pufferlib.ocean.cosim.carla.lb1.ground_truth import LeaderboardOneGroundTruth
from pufferlib.ocean.cosim.carla.shadow_ego import MAX_POLICY_STEPS, PARTNER_MAX_ABS_DZ_M, ShadowEgo


MAX_MESSAGES = MAX_POLICY_STEPS + 16  # ticks plus the handshake; the shadow env caps the route before this


def reply_for_step(result, carla_flags):
    return {
        "type": "control",
        "motion": None if result.motion is None else [float(v) for v in result.motion],
        "control": None if result.control is None else [float(v) for v in result.control],
        "pd_flags": result.pd_flags,
        "carla_flags": carla_flags,
    }


def serve(sock, shadow):
    lb1_protocol.send_message(
        sock,
        {
            "type": "config",
            "partner_slots": shadow.partner_slots,
            "partner_max_abs_dz_m": PARTNER_MAX_ABS_DZ_M,
            "light_probe_steps_m": list(cb.LIGHT_PROBE_STEPS_M),
            "dynamics_source": shadow.dynamics_source,
        },
    )
    ground_truth = None
    for _ in range(MAX_MESSAGES):
        message = lb1_protocol.recv_message(sock)
        if message is None:
            print("[policy_server] client closed the connection", flush=True)
            return
        kind = message["type"]
        if kind == "init":
            route = message["route"]
            shadow.init_route(route)
            if route.get("ground_truth", False):
                ground_truth = LeaderboardOneGroundTruth(
                    route["lights"], route["stop_signs"], route["ego"]["extent"][0]
                )
            lb1_protocol.send_message(sock, {"type": "ready", "num_traffic": shadow.num_traffic})
        elif kind == "tick":
            snapshot = message["snapshot"]
            result = shadow.step(snapshot)
            carla_flags = None
            if ground_truth is not None and message.get("ground_truth") is not None:
                carla_flags = ground_truth.update(snapshot, message["ground_truth"])
            shadow.telemetry(result, carla_flags)
            lb1_protocol.send_message(sock, reply_for_step(result, carla_flags))
        elif kind == "finish":
            shadow.finish()
            lb1_protocol.send_message(sock, {"type": "done"})
            return
        else:
            raise ValueError(f"unknown message type {kind!r}")
    raise RuntimeError(f"more than MAX_MESSAGES={MAX_MESSAGES} messages on one route")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", required=True, help="PufferDrive checkpoint .pt or run dir (config.yaml beside)"
    )
    parser.add_argument("--socket-fd", type=int, required=True, help="inherited socket to the leaderboard agent")
    parser.add_argument("--route-tag", required=True, help="per-route file stem for every output")
    args = parser.parse_args()
    sock = socket.socket(fileno=args.socket_fd)
    try:
        serve(sock, ShadowEgo(args.checkpoint, args.route_tag))
    finally:
        sock.close()


if __name__ == "__main__":
    main()
