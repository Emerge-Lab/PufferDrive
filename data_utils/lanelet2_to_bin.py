"""Lanelet2-to-binary command line tool.

- Convert and validate a Lanelet2 OSM map.
- Write a PufferDrive 3.0 binary and optional JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


if __package__:
    from .lanelet2_conversion import convert_map
    from .lanelet2_validation import validate_bin
    from .mirror_map_bin import write_bin
else:
    from lanelet2_conversion import convert_map
    from lanelet2_validation import validate_bin
    from mirror_map_bin import write_bin


def parse_args():
    """Parse conversion paths and optional map-processing settings."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="clip output to this local-metre window",
    )
    parser.add_argument("--link-tolerance-m", type=float, default=0.0)
    parser.add_argument("--virtual-as-edge", action="store_true")
    parser.add_argument("--validation-report", type=Path)
    return parser.parse_args()


def _validate_args(args):
    """Reject invalid crop and link-tolerance values."""

    if args.crop is not None and (args.crop[0] >= args.crop[2] or args.crop[1] >= args.crop[3]):
        raise ValueError("crop must satisfy XMIN < XMAX and YMIN < YMAX")
    if args.link_tolerance_m < 0.0:
        raise ValueError("link tolerance must be non-negative")


def main():
    """Run conversion, binary validation, and report output."""

    args = parse_args()
    _validate_args(args)
    map_data, projection = convert_map(
        args.input,
        crop=args.crop,
        link_tolerance_m=args.link_tolerance_m,
        virtual_as_edge=args.virtual_as_edge,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_bin(map_data, args.output)
    report = validate_bin(args.output)
    report["projection"] = {
        "crs": f"EPSG:{projection.epsg}",
        "origin_easting": projection.origin_easting,
        "origin_northing": projection.origin_northing,
    }
    if args.validation_report is not None:
        args.validation_report.parent.mkdir(parents=True, exist_ok=True)
        args.validation_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
