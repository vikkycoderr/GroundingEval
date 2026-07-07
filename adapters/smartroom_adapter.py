"""Utilities for inspecting one local Smart Room recording folder.

The adapter is intentionally lightweight: it discovers paths, counts, and
available modalities without loading video, frame, or sensor data into memory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_NATURAL_PART_RE = re.compile(r"(\d+)")
_VIDEO_PATTERNS = ("*.mp4", "*.avi", "*.mov")
_FRAME_PATTERNS = ("*.png", "*.jpg", "*.jpeg")


def natural_sort_key(path: Path) -> list[int | str]:
    """Return a deterministic sort key that treats embedded digits naturally."""

    parts: list[int | str] = []
    for part in _NATURAL_PART_RE.split(path.name):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part.casefold())
    return parts


def _sorted_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []

    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in directory.glob(pattern) if path.is_file())
    return sorted(files, key=natural_sort_key)


def _path_or_none(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _available_modalities(
    video_files: list[Path],
    frame_files: list[Path],
    sensors_dir: Path,
    metadata_file: Path,
) -> list[str]:
    modalities: list[str] = []
    if video_files:
        modalities.append("video")
    if frame_files:
        modalities.append("frames")
    if sensors_dir.is_dir():
        modalities.append("sensors")
    if metadata_file.is_file():
        modalities.append("metadata")
    return modalities


def load_recording(recording_dir: str | Path) -> dict[str, Any]:
    """Inspect one local Smart Room recording and return normalized metadata.

    Expected local layout:

    ```text
    data/smartroom/recording_001/
      camera_main.mp4
      frames/
      sensors/
      metadata.json
    ```
    """

    rec_dir = Path(recording_dir).expanduser().resolve()
    if not rec_dir.is_dir():
        raise FileNotFoundError(f"Smart Room recording directory does not exist: {rec_dir}")

    frames_dir = rec_dir / "frames"
    sensors_dir = rec_dir / "sensors"
    metadata_file = rec_dir / "metadata.json"

    video_files = _sorted_files(rec_dir, _VIDEO_PATTERNS)
    frame_files: list[Path] = []
    for pattern in _FRAME_PATTERNS:
        frame_files.extend(path for path in frames_dir.glob(pattern) if path.is_file())
    frame_files = sorted(frame_files, key=natural_sort_key)

    sensor_files = (
        sorted((path for path in sensors_dir.rglob("*") if path.is_file()), key=natural_sort_key)
        if sensors_dir.is_dir()
        else []
    )

    return {
        "recording_name": rec_dir.name,
        "recording_dir": str(rec_dir),
        "video_files": [str(path) for path in video_files],
        "frames_dir": _path_or_none(frames_dir),
        "frames": [str(path) for path in frame_files],
        "total_frames": len(frame_files),
        "sensors_dir": _path_or_none(sensors_dir),
        "sensor_files": [str(path) for path in sensor_files],
        "total_sensor_files": len(sensor_files),
        "metadata_file": str(metadata_file) if metadata_file.is_file() else None,
        "available_modalities": _available_modalities(
            video_files=video_files,
            frame_files=frame_files,
            sensors_dir=sensors_dir,
            metadata_file=metadata_file,
        ),
    }
