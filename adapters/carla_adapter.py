"""Utilities for inspecting and sampling one CARLA simulation folder.

The adapter only discovers paths and counts. It deliberately avoids loading
large image, video, semantic segmentation, or lidar arrays into memory.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


_NATURAL_PART_RE = re.compile(r"(\d+)")


def natural_sort_key(path: Path) -> list[int | str]:
    """Return a deterministic sort key that treats embedded digits naturally."""

    parts: list[int | str] = []
    for part in _NATURAL_PART_RE.split(path.name):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part.casefold())
    return parts


def _sorted_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.glob(pattern) if path.is_file()),
        key=natural_sort_key,
    )


def _path_or_none(directory: Path) -> str | None:
    return str(directory) if directory.is_dir() else None


def _available_modalities(
    frames: list[Path],
    json_files: list[Path],
    semseg_files: list[Path],
    lidar_files: list[Path],
    video_files: list[Path],
) -> list[str]:
    modalities: list[str] = []
    if frames:
        modalities.append("frames")
    if json_files:
        modalities.append("json")
    if semseg_files:
        modalities.append("semseg")
    if lidar_files:
        modalities.append("lidar")
    if video_files:
        modalities.append("video")
    return modalities


def load_simulation(simulation_dir: str | Path) -> dict[str, Any]:
    """Inspect one CARLA simulation directory and return normalized metadata.

    The returned dictionary contains string paths so it can be serialized
    directly into scenario metadata without custom JSON handling.
    """

    sim_dir = Path(simulation_dir).expanduser().resolve()
    if not sim_dir.is_dir():
        raise FileNotFoundError(f"CARLA simulation directory does not exist: {sim_dir}")

    frames_dir = sim_dir / "frames"
    json_dir = sim_dir / "json"
    semseg_dir = sim_dir / "semseg"
    lidar_dir = sim_dir / "lidar"

    frames = _sorted_files(frames_dir, "*.png")
    json_files = _sorted_files(json_dir, "*.json")
    semseg_files = _sorted_files(semseg_dir, "*.png")
    lidar_files = _sorted_files(lidar_dir, "*.npy")
    video_files = _sorted_files(sim_dir, "*.mp4")

    return {
        "simulation_name": sim_dir.name,
        "simulation_dir": str(sim_dir),
        "frames_dir": _path_or_none(frames_dir),
        "frames": [str(path) for path in frames],
        "total_frames": len(frames),
        "json_dir": _path_or_none(json_dir),
        "json_files": [str(path) for path in json_files],
        "semseg_dir": _path_or_none(semseg_dir),
        "semseg_files": [str(path) for path in semseg_files],
        "lidar_dir": _path_or_none(lidar_dir),
        "lidar_files": [str(path) for path in lidar_files],
        "video_files": [str(path) for path in video_files],
        "available_modalities": _available_modalities(
            frames=frames,
            json_files=json_files,
            semseg_files=semseg_files,
            lidar_files=lidar_files,
            video_files=video_files,
        ),
    }


def estimate_sample_count(
    *,
    total_items: int,
    duration_seconds: float | None,
    sample_every_seconds: float,
    min_frames: int,
    max_frames: int,
    num_frames: int | None = None,
) -> int:
    """Compute a bounded representative sample count for frames or video."""

    if total_items <= 0:
        return 0

    if num_frames is not None:
        if num_frames <= 0:
            raise ValueError("--num-frames must be greater than 0 when provided.")
        return min(num_frames, total_items)

    if sample_every_seconds <= 0:
        raise ValueError("--sample-every-seconds must be greater than 0.")
    if min_frames <= 0:
        raise ValueError("--min-frames must be greater than 0.")
    if max_frames <= 0:
        raise ValueError("--max-frames must be greater than 0.")
    if min_frames > max_frames:
        raise ValueError("--min-frames cannot be greater than --max-frames.")

    duration = max(float(duration_seconds or 0.0), 0.0)
    estimated = max(1, math.ceil(duration / sample_every_seconds))
    clamped = max(min_frames, min(estimated, max_frames))
    return min(clamped, total_items)


def sample_evenly(items: list[Any], num_samples: int) -> list[Any]:
    """Sample items evenly from first to last without duplicates."""

    if num_samples <= 0 or not items:
        return []

    if num_samples >= len(items):
        return list(items)

    if num_samples == 1:
        return [items[0]]

    last_index = len(items) - 1
    selected_indexes: list[int] = []

    for sample_index in range(num_samples):
        raw_index = round(sample_index * last_index / (num_samples - 1))
        index = int(raw_index)
        if selected_indexes and index <= selected_indexes[-1]:
            index = selected_indexes[-1] + 1
        selected_indexes.append(min(index, last_index))

    # The monotonic adjustment above is defensive; keep the contract explicit.
    unique_indexes = sorted(dict.fromkeys(selected_indexes))
    if len(unique_indexes) < num_samples:
        for index in range(len(items)):
            if index not in unique_indexes:
                unique_indexes.append(index)
            if len(unique_indexes) == num_samples:
                break
        unique_indexes.sort()

    return [items[index] for index in unique_indexes[:num_samples]]


def sample_representative_frames(
    frame_paths: list[str | Path],
    *,
    duration_seconds: float | None,
    sample_every_seconds: float = 5,
    min_frames: int = 8,
    max_frames: int = 32,
    num_frames: int | None = None,
) -> list[str]:
    """Return representative frame paths sampled evenly across the sequence."""

    ordered_paths = [str(Path(path)) for path in frame_paths]
    sample_count = estimate_sample_count(
        total_items=len(ordered_paths),
        duration_seconds=duration_seconds,
        sample_every_seconds=sample_every_seconds,
        min_frames=min_frames,
        max_frames=max_frames,
        num_frames=num_frames,
    )
    return sample_evenly(ordered_paths, sample_count)
