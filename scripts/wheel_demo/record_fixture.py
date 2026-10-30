"""Record a LABELLED G920 trace to use as a test fixture.

    python scripts/wheel_demo/record_fixture.py --out tests/fixtures/g920

Walks through named phases and writes the raw reports plus a JSON index of each phase's frame
range, so tests can assert against a known control instead of hunting through an unlabelled blob.

Must run in the foreground: macOS attributes Input Monitoring to the responsible process.
"""

import argparse
import json
import os
import sys
import time

from pufferlib.wheel_demo.input.g920 import G920_REPORT_LENGTH

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_g920 import collect, open_axis_device  # noqa: E402

# (phase name, seconds, what the operator should do)
PHASES = (
    ("steering_sweep", 12.0, "Wheel FULL LEFT, FULL RIGHT, centre -- twice. Feet OFF the pedals."),
    ("throttle", 8.0, "Throttle: floor it and fully release, three times."),
    ("brake", 8.0, "Brake: press as HARD as you can and fully release, three times."),
    ("clutch", 8.0, "Clutch: floor it and fully release, three times."),
    ("paddle_up", 8.0, "RIGHT paddle: pull and release, five separate times."),
    ("paddle_down", 8.0, "LEFT paddle: pull and release, five separate times."),
    ("buttons", 14.0, "Press every other button on the wheel, ONE AT A TIME, pausing between each."),
    ("combined", 18.0, "Drive it: steer while using throttle and brake, with a few paddle shifts."),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory to write fixture.bin and fixture_index.json into")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = open_axis_device()
    reports, index = [], {}
    try:
        for name, seconds, instruction in PHASES:
            input(f"\n>>> [{name}] {instruction}\n    Enter to start a {seconds:.0f}s capture ...")
            start = len(reports)
            captured = collect(device, seconds)
            reports += captured
            index[name] = {"start_frame": start, "end_frame": len(reports), "frames": len(captured)}
            status = "OK" if captured else "NO DATA -- keep the control moving during the capture"
            print(f"    {len(captured):5d} frames  {status}")
    finally:
        device.close()

    blob_path = os.path.join(args.out, "fixture.bin")
    with open(blob_path, "wb") as handle:
        handle.write(b"".join(reports))
    index_path = os.path.join(args.out, "fixture_index.json")
    with open(index_path, "w") as handle:
        json.dump(
            {"report_length": G920_REPORT_LENGTH, "total_frames": len(reports), "phases": index}, handle, indent=2
        )

    print(f"\nwrote {len(reports)} frames -> {blob_path}")
    print(f"wrote phase index    -> {index_path}")
    empty = [name for name, info in index.items() if info["frames"] == 0]
    if empty:
        print(f"WARNING: these phases captured nothing: {empty}")


if __name__ == "__main__":
    main()
