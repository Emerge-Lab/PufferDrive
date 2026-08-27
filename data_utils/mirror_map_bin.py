#!/usr/bin/env python3
"""mirror_map_bin.py — Reflect a PufferDrive map .bin across the x-axis to
flip driving convention (right-hand traffic → left-hand traffic).

CARLA's OpenDRIVE maps are right-hand-traffic by default. Negating y for every
position, vy for every velocity, and h for every heading produces an equivalent
map in which the same lane network is traversed on the left side of the road
instead of the right. Lane / road / agent / traffic-control IDs and the
lane-graph distance matrix are preserved unchanged — they're geometry-invariant.

Round-trips byte-exactly when mirror is applied twice.

Usage:
    # Mirror a whole directory of bins:
    python data_utils/mirror_map_bin.py \\
        --input  pufferlib/resources/drive/binaries/carla \\
        --output pufferlib/resources/drive/binaries/carla_lhs

    # Mirror a single file:
    python data_utils/mirror_map_bin.py \\
        --input  pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin \\
        --output pufferlib/resources/drive/binaries/carla_lhs/opendrive__Town01.bin
"""

import argparse
import struct
from pathlib import Path


TRAFFIC_PHASE_SECTION_TAG = b"TLPHASE1"


def _is_lane(road_type: int) -> bool:
    return 0 <= road_type <= 9


def _read(fmt, f):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _read_f_array(f, n):
    return struct.unpack(f"<{n}f", f.read(4 * n)) if n else ()


def _read_i_array(f, n):
    return struct.unpack(f"<{n}i", f.read(4 * n)) if n else ()


def read_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        num_agents, num_roads, num_traffic, num_objects = _read("<iiii", f)

        agents = []
        for i in range(num_agents):
            agent_id, agent_type = _read("<ii", f)
            assert agent_id == i, f"agent id {agent_id} != idx {i}"
            (T,) = _read("<i", f)
            cols = {k: _read_f_array(f, T) for k in ("x", "y", "z", "h", "vx", "vy", "len", "wid", "hgt")}
            cols["valid"] = _read_i_array(f, T)
            (route_len,) = _read("<i", f)
            route = _read_i_array(f, route_len)
            (route_gt_len,) = _read("<i", f)
            goal = _read("<fff", f)
            (mark_as_expert,) = _read("<i", f)
            agents.append(
                {
                    "id": agent_id,
                    "type": agent_type,
                    "T": T,
                    "cols": cols,
                    "route": route,
                    "route_gt_len": route_gt_len,
                    "goal": goal,
                    "mark_as_expert": mark_as_expert,
                }
            )

        roads = []
        for i in range(num_roads):
            road_id, road_type = _read("<ii", f)
            assert road_id == i, f"road id {road_id} != idx {i}"
            (S,) = _read("<i", f)
            road = {
                "id": road_id,
                "type": road_type,
                "S": S,
                "x": _read_f_array(f, S),
                "y": _read_f_array(f, S),
                "z": _read_f_array(f, S),
                "headings": _read_f_array(f, S),
            }
            if _is_lane(road_type):
                (ne,) = _read("<i", f)
                road["entry_lanes"] = _read_i_array(f, ne)
                (nx,) = _read("<i", f)
                road["exit_lanes"] = _read_i_array(f, nx)
                (road["speed_limit"],) = _read("<f", f)
                (road["length"],) = _read("<f", f)
                road["cum_lengths"] = _read_f_array(f, S)
            roads.append(road)

        traffic = []
        for i in range(num_traffic):
            traffic_id, traffic_type = _read("<ii", f)
            assert traffic_id == i, f"traffic id {traffic_id} != idx {i}"
            stop_line = _read("<6f", f)
            (heading,) = _read("<f", f)
            (state_len,) = _read("<i", f)
            states = _read_i_array(f, state_len)
            (ncl,) = _read("<i", f)
            ctrl_lanes = _read_i_array(f, ncl)
            traffic.append(
                {
                    "id": traffic_id,
                    "type": traffic_type,
                    "stop_line": stop_line,
                    "heading": heading,
                    "states": states,
                    "controlled_lanes": ctrl_lanes,
                }
            )

        objects = []
        for i in range(num_objects):
            obj_id, obj_type, T = _read("<iii", f)
            # 9 float arrays + 1 int array of length T, all concatenated.
            raw = f.read((9 * 4 + 4) * T)
            objects.append({"id": obj_id, "type": obj_type, "T": T, "raw": raw})

        (n_lanes_g,) = _read("<i", f)
        lane_ids = _read_i_array(f, n_lanes_g)
        distances = _read_f_array(f, n_lanes_g * n_lanes_g)

        scenario_id = f.read(128)
        dataset_name = f.read(32)
        (log_length,) = _read("<i", f)
        (log_dt,) = _read("<f", f)
        (n_ooi,) = _read("<i", f)
        objects_of_interest = _read_i_array(f, n_ooi)
        (n_ttp,) = _read("<i", f)
        tracks_to_predict = _read_i_array(f, n_ttp)

        phase_tag = f.read(len(TRAFFIC_PHASE_SECTION_TAG))
        has_phase_section = phase_tag == TRAFFIC_PHASE_SECTION_TAG
        assert has_phase_section or phase_tag == b"", f"unexpected bytes after metadata in {path}"
        for t in traffic:
            t["junction_id"], t["phase_idx"] = _read("<ii", f) if has_phase_section else (-1, -1)

        trailing = f.read()
        assert not trailing, f"{len(trailing)} unparsed trailing bytes in {path}"

    return {
        "agents": agents,
        "roads": roads,
        "traffic": traffic,
        "objects": objects,
        "lane_graph": {"n": n_lanes_g, "lane_ids": lane_ids, "distances": distances},
        "scenario_id": scenario_id,
        "dataset_name": dataset_name,
        "log_length": log_length,
        "log_dt": log_dt,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
        "has_phase_section": has_phase_section,
    }


def mirror(data: dict) -> dict:
    """Reflect across the x-axis: negate y for every position, vy for every
    velocity, heading → -heading. Lane-graph distances, segment lengths,
    cumulative lengths, and all int IDs are reflection-invariant."""
    for a in data["agents"]:
        c = a["cols"]
        c["y"] = tuple(-v for v in c["y"])
        c["vy"] = tuple(-v for v in c["vy"])
        c["h"] = tuple(-v for v in c["h"])
        gx, gy, gz = a["goal"]
        a["goal"] = (gx, -gy, gz)
    for r in data["roads"]:
        r["y"] = tuple(-v for v in r["y"])
        r["headings"] = tuple(-v for v in r["headings"])
    for t in data["traffic"]:
        x1, y1, z1, x2, y2, z2 = t["stop_line"]
        t["stop_line"] = (x1, -y1, z1, x2, -y2, z2)
        t["heading"] = -t["heading"]
    for o in data["objects"]:
        # Object trajectory raw layout: x[T] y[T] z[T] h[T] vx[T] vy[T] len[T]
        # wid[T] hgt[T] valid[T]. Negate y, h, vy in place.
        T = o["T"]
        if T == 0:
            continue
        arr = list(struct.unpack(f"<{9 * T}f{T}i", o["raw"]))
        for k in range(T):
            arr[T + k] = -arr[T + k]  # y
            arr[3 * T + k] = -arr[3 * T + k]  # heading
            arr[5 * T + k] = -arr[5 * T + k]  # vy
        o["raw"] = struct.pack(f"<{9 * T}f{T}i", *arr)
    return data


def write_bin(data: dict, path: Path):
    with open(path, "wb") as f:
        agents = data["agents"]
        roads = data["roads"]
        traffic = data["traffic"]
        objects = data["objects"]
        f.write(struct.pack("<iiii", len(agents), len(roads), len(traffic), len(objects)))

        for a in agents:
            f.write(struct.pack("<ii", a["id"], a["type"]))
            T = a["T"]
            f.write(struct.pack("<i", T))
            c = a["cols"]
            for k in ("x", "y", "z", "h", "vx", "vy", "len", "wid", "hgt"):
                if T:
                    f.write(struct.pack(f"<{T}f", *c[k]))
            if T:
                f.write(struct.pack(f"<{T}i", *c["valid"]))
            f.write(struct.pack("<i", len(a["route"])))
            if a["route"]:
                f.write(struct.pack(f"<{len(a['route'])}i", *a["route"]))
            f.write(struct.pack("<i", a["route_gt_len"]))
            f.write(struct.pack("<fff", *a["goal"]))
            f.write(struct.pack("<i", a["mark_as_expert"]))

        for r in roads:
            f.write(struct.pack("<ii", r["id"], r["type"]))
            S = r["S"]
            f.write(struct.pack("<i", S))
            for k in ("x", "y", "z", "headings"):
                if S:
                    f.write(struct.pack(f"<{S}f", *r[k]))
            if _is_lane(r["type"]):
                f.write(struct.pack("<i", len(r["entry_lanes"])))
                if r["entry_lanes"]:
                    f.write(struct.pack(f"<{len(r['entry_lanes'])}i", *r["entry_lanes"]))
                f.write(struct.pack("<i", len(r["exit_lanes"])))
                if r["exit_lanes"]:
                    f.write(struct.pack(f"<{len(r['exit_lanes'])}i", *r["exit_lanes"]))
                f.write(struct.pack("<f", r["speed_limit"]))
                f.write(struct.pack("<f", r["length"]))
                if S:
                    f.write(struct.pack(f"<{S}f", *r["cum_lengths"]))

        for t in traffic:
            f.write(struct.pack("<ii", t["id"], t["type"]))
            f.write(struct.pack("<6f", *t["stop_line"]))
            f.write(struct.pack("<f", t["heading"]))
            f.write(struct.pack("<i", len(t["states"])))
            if t["states"]:
                f.write(struct.pack(f"<{len(t['states'])}i", *t["states"]))
            f.write(struct.pack("<i", len(t["controlled_lanes"])))
            if t["controlled_lanes"]:
                f.write(struct.pack(f"<{len(t['controlled_lanes'])}i", *t["controlled_lanes"]))

        for o in objects:
            f.write(struct.pack("<iii", o["id"], o["type"], o["T"]))
            f.write(o["raw"])

        lg = data["lane_graph"]
        f.write(struct.pack("<i", lg["n"]))
        if lg["n"] > 0:
            f.write(struct.pack(f"<{lg['n']}i", *lg["lane_ids"]))
            f.write(struct.pack(f"<{lg['n'] ** 2}f", *lg["distances"]))

        f.write(data["scenario_id"])
        f.write(data["dataset_name"])
        f.write(struct.pack("<i", data["log_length"]))
        f.write(struct.pack("<f", data["log_dt"]))
        f.write(struct.pack("<i", len(data["objects_of_interest"])))
        if data["objects_of_interest"]:
            f.write(struct.pack(f"<{len(data['objects_of_interest'])}i", *data["objects_of_interest"]))
        f.write(struct.pack("<i", len(data["tracks_to_predict"])))
        if data["tracks_to_predict"]:
            f.write(struct.pack(f"<{len(data['tracks_to_predict'])}i", *data["tracks_to_predict"]))
        if data["has_phase_section"]:
            f.write(TRAFFIC_PHASE_SECTION_TAG)
            for t in traffic:
                f.write(struct.pack("<ii", t["junction_id"], t["phase_idx"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory of *.bin files or a single .bin path")
    parser.add_argument("--output", required=True, help="Output directory or output .bin path")
    parser.add_argument(
        "--verify-roundtrip",
        action="store_true",
        help="Read the input, write it back unmodified, and assert bytes match. Skips mirroring.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if src.is_file():
        pairs = [(src, dst)]
        dst.parent.mkdir(parents=True, exist_ok=True)
    else:
        bins = sorted(src.glob("*.bin"))
        dst.mkdir(parents=True, exist_ok=True)
        pairs = [(b, dst / b.name) for b in bins]

    for src_path, out_path in pairs:
        data = read_bin(src_path)
        if not args.verify_roundtrip:
            mirror(data)
        write_bin(data, out_path)
        if args.verify_roundtrip:
            with open(src_path, "rb") as f1, open(out_path, "rb") as f2:
                if f1.read() != f2.read():
                    raise AssertionError(f"round-trip mismatch on {src_path.name}")
            print(f"  ✓ round-trip {src_path.name}")
        else:
            print(f"  {src_path.name} → {out_path}")

    print(f"\n{'verified' if args.verify_roundtrip else 'mirrored'} {len(pairs)} file(s).")


if __name__ == "__main__":
    main()
