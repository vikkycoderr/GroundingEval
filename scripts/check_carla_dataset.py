#!/usr/bin/env python3
"""Inspect a CARLA dataset folder and summarize available modalities."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SIMULATION_PATTERN = re.compile(r"^simulation_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect CARLA simulations and report available modalities."
    )
    parser.add_argument(
        "--carla-root",
        required=True,
        help="Path to the data/carla_dataset directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of simulations to print. Defaults to 10.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to save the full summary as pretty-printed JSON.",
    )
    return parser.parse_args()


def simulation_sort_key(path: Path) -> tuple[int, str]:
    match = SIMULATION_PATTERN.match(path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


def count_files(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.glob(pattern) if path.is_file())


def sorted_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted((path for path in directory.glob(pattern) if path.is_file()), key=lambda p: p.name)


def find_simulations(carla_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in carla_root.rglob("simulation_*")
            if path.is_dir() and SIMULATION_PATTERN.match(path.name)
        ),
        key=simulation_sort_key,
    )


def summarize_simulation(simulation_dir: Path) -> dict[str, Any]:
    frame_files = sorted_files(simulation_dir / "frames", "*.png")

    return {
        "name": simulation_dir.name,
        "path": str(simulation_dir),
        "frames_png_count": len(frame_files),
        "json_count": count_files(simulation_dir / "json", "*.json"),
        "semseg_png_count": count_files(simulation_dir / "semseg", "*.png"),
        "lidar_npy_count": count_files(simulation_dir / "lidar", "*.npy"),
        "video_mp4_count": count_files(simulation_dir, "*.mp4"),
        "first_frame_name": frame_files[0].name if frame_files else None,
        "last_frame_name": frame_files[-1].name if frame_files else None,
    }


def build_summary(carla_root: Path) -> dict[str, Any]:
    simulations = find_simulations(carla_root)
    return {
        "carla_root": str(carla_root),
        "simulation_count": len(simulations),
        "simulations": [summarize_simulation(simulation) for simulation in simulations],
    }


def print_summary(summary: dict[str, Any], limit: int) -> None:
    simulations = summary["simulations"]
    listed = simulations[: max(limit, 0)]

    print(f"CARLA root: {summary['carla_root']}")
    print(f"Simulations found: {summary['simulation_count']}")
    print(f"Simulations listed: {len(listed)}")

    for simulation in listed:
        print()
        print(f"- {simulation['name']}")
        print(f"  frames PNG: {simulation['frames_png_count']}")
        print(f"  json files: {simulation['json_count']}")
        print(f"  semseg PNG: {simulation['semseg_png_count']}")
        print(f"  lidar NPY: {simulation['lidar_npy_count']}")
        print(f"  MP4 videos: {simulation['video_mp4_count']}")
        print(f"  first frame: {simulation['first_frame_name']}")
        print(f"  last frame: {simulation['last_frame_name']}")


def write_json_summary(summary: dict[str, Any], json_out: str) -> None:
    output_path = Path(json_out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    carla_root = Path(args.carla_root).expanduser()

    if not carla_root.is_dir():
        print(f"Error: --carla-root does not exist or is not a directory: {carla_root}", file=sys.stderr)
        return 1

    summary = build_summary(carla_root)
    print_summary(summary, args.limit)

    if args.json_out:
        write_json_summary(summary, args.json_out)
        print()
        print(f"Saved JSON summary to: {Path(args.json_out).expanduser()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
