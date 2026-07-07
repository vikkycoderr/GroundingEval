#!/usr/bin/env python3
"""Run InternVL3-9B on one CARLA simulation using video-first input."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.carla_adapter import (  # noqa: E402
    estimate_sample_count,
    load_simulation,
    sample_evenly,
    sample_representative_frames,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448
MODEL_NAME = "InternVL3-9B"

DETAILED_ANALYSIS_PROMPT = """<image>
Analyze this CARLA driving scene in as much detail as possible.

Describe:
1. Overall road layout and driving environment.
2. Number, type, and approximate position of every visible vehicle.
3. Presence or absence of pedestrians.
4. Sidewalks, lanes, road markings, crosswalks, intersections, traffic lights, traffic signs, barriers, buildings, vegetation, and other environmental features.
5. Weather, lighting conditions, shadows, visibility, and any occlusions.
6. Potential hazards or safety-relevant observations.
7. Explain anything that cannot be confidently determined from the image/sequence instead of guessing.
8. Mention any additional observations that may be useful for understanding the driving scene.

Be extremely detailed, objective, and avoid hallucinating objects that are not visible."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InternVL3-9B on one CARLA simulation folder."
    )
    parser.add_argument(
        "--sim-dir",
        required=True,
        help="Path to one CARLA simulation folder, such as data/carla/simulation_45. Absolute paths are also supported.",
    )
    parser.add_argument(
        "--scene-id",
        help="Scenario ID to use for output. Defaults to the simulation folder name.",
    )
    parser.add_argument(
        "--source-type",
        default="carla",
        help='Source dataset type. Only "carla" is supported for now.',
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "video", "frames"),
        default="auto",
        help="Input selection mode: auto prefers MP4 video, video requires MP4, frames ignores MP4.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to the InternVL3-9B model directory.",
    )
    parser.add_argument(
        "--output-root",
        default="scenarios/carla",
        help="Directory where per-scene scenario outputs will be written.",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=5,
        help="Target spacing between representative samples when --num-frames is not set.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="Minimum number of frames to sample when enough frames are available.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of frames to sample when --num-frames is not set.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10,
        help="FPS used only to estimate duration from frame count when video metadata is unavailable.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Optional exact number of representative frames to sample, capped by available unique frames.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum new tokens generated for each InternVL3 analysis prompt.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_args(args: argparse.Namespace) -> None:
    if args.source_type != "carla":
        fail(f'Unsupported --source-type "{args.source_type}". Only "carla" is supported.')
    if args.sample_every_seconds <= 0:
        fail("--sample-every-seconds must be greater than 0.")
    if args.min_frames <= 0:
        fail("--min-frames must be greater than 0.")
    if args.max_frames <= 0:
        fail("--max-frames must be greater than 0.")
    if args.min_frames > args.max_frames:
        fail("--min-frames cannot be greater than --max-frames.")
    if args.fps <= 0:
        fail("--fps must be greater than 0.")
    if args.num_frames is not None and args.num_frames <= 0:
        fail("--num-frames must be greater than 0 when provided.")
    if args.max_new_tokens <= 0:
        fail("--max-new-tokens must be greater than 0.")


def decide_input_mode(requested_mode: str, video_files: list[str]) -> str:
    if requested_mode == "auto":
        return "video" if video_files else "frames"
    if requested_mode == "video" and not video_files:
        fail("Video mode requested, but no MP4 file was found in the simulation root.")
    return requested_mode


def validate_runtime(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        fail(f"Invalid model path. Path does not exist: {model_path}")
    if not torch.cuda.is_available():
        fail("CUDA unavailable. InternVL3-9B should be run on a GPU machine.")


def load_internvl3(model_path: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda",
    ).eval()
    return model, tokenizer


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    pixel_values = torch.tensor(list(image.getdata()), dtype=torch.float32)
    pixel_values = pixel_values.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1)
    pixel_values = pixel_values / 255.0

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    return pixel_values.to(dtype=torch.bfloat16, device="cuda")


def load_frame_tensor(frame_path: str | Path) -> torch.Tensor:
    with Image.open(frame_path) as image:
        return preprocess_image(image)


def get_video_metadata(video_path: str | Path) -> dict[str, float | int | None]:
    try:
        import cv2
    except ImportError:
        return {"fps": None, "frame_count": None, "duration_seconds": None}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return {"fps": None, "frame_count": None, "duration_seconds": None}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else None
    return {
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "duration_seconds": duration_seconds,
    }


def sample_video_as_tensors(
    video_path: str | Path,
    *,
    sample_every_seconds: float,
    min_frames: int,
    max_frames: int,
    num_frames: int | None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float | int | None]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video mode but could not be imported.") from exc

    video_metadata = get_video_metadata(video_path)
    frame_count = int(video_metadata["frame_count"] or 0)
    fps = float(video_metadata["fps"] or 0.0)
    duration_seconds = video_metadata["duration_seconds"]

    if frame_count <= 0 or fps <= 0:
        fail(f"Could not read usable video metadata from: {video_path}")

    sample_count = estimate_sample_count(
        total_items=frame_count,
        duration_seconds=float(duration_seconds or 0.0),
        sample_every_seconds=sample_every_seconds,
        min_frames=min_frames,
        max_frames=max_frames,
        num_frames=num_frames,
    )
    frame_indexes = sample_evenly(list(range(frame_count)), sample_count)

    capture = cv2.VideoCapture(str(video_path))
    tensors: list[torch.Tensor] = []
    sampled_frames: list[dict[str, Any]] = []

    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        tensors.append(preprocess_image(image))
        sampled_frames.append(
            {
                "frame_index": int(frame_index),
                "time_seconds": round(float(frame_index) / fps, 3),
            }
        )

    capture.release()

    if not tensors:
        fail(f"No frames could be sampled from video: {video_path}")

    return torch.stack(tensors), sampled_frames, video_metadata


def sequence_prompt(frame_count: int) -> str:
    frame_tokens = "".join(f"Frame{i + 1}: <image>\n" for i in range(frame_count))
    base_prompt = DETAILED_ANALYSIS_PROMPT.replace("<image>\n", "")
    return (
        f"{frame_tokens}\n"
        "Analyze these sampled video frames in temporal order as one CARLA driving sequence.\n\n"
        f"{base_prompt}"
    ).strip()


def run_single_frame_analysis(
    *,
    model: Any,
    tokenizer: Any,
    frame_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    pixel_values = load_frame_tensor(frame_path).unsqueeze(0)
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    with torch.no_grad():
        return model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            history=None,
            return_history=False,
        )


def run_video_sequence_analysis(
    *,
    model: Any,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
) -> str:
    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    num_patches_list = [1] * int(pixel_values.shape[0])
    with torch.no_grad():
        return model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )


def copy_selected_file(source_path: str | Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    destination = output_dir / source.name
    shutil.copy2(source, destination)
    return destination


def copy_sampled_frames(sampled_frames: list[str], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []

    for sequence_index, frame_path in enumerate(sampled_frames, start=1):
        source = Path(frame_path)
        destination = output_dir / f"{sequence_index:04d}_{source.name}"
        shutil.copy2(source, destination)
        copied.append(
            {
                "frame_index": parse_frame_index(source),
                "original_path": str(source),
                "copied_path": str(destination),
            }
        )

    return copied


def parse_frame_index(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_scenario_payload(
    *,
    args: argparse.Namespace,
    scene_id: str,
    simulation: dict[str, Any],
    input_mode_used: str,
    used_frames: list[dict[str, Any]],
    video_used: str | None,
    video_sampled_to_frames: bool,
) -> dict[str, Any]:
    frame_selection_mode = "video" if input_mode_used == "video" else "sampled"
    if input_mode_used == "frames" and len(used_frames) == simulation["total_frames"]:
        frame_selection_mode = "all"

    return {
        "scene_id": scene_id,
        "source_type": "carla",
        "source_sim_dir": simulation["simulation_dir"],
        "simulation_name": simulation["simulation_name"],
        "modality": "video" if input_mode_used == "video" else "multi_frame",
        "input_mode_requested": args.input_mode,
        "input_mode_used": input_mode_used,
        "camera_view": "front_ego",
        "total_frames_available": simulation["total_frames"],
        "frames_used_count": len(used_frames),
        "frame_selection_mode": frame_selection_mode,
        "sample_every_seconds": args.sample_every_seconds,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "fps_assumed_if_needed": args.fps,
        "used_frames": used_frames,
        "video_used": video_used,
        "video_sampled_to_frames": video_sampled_to_frames,
        "available_modalities": simulation["available_modalities"],
        "description": (
            "CARLA driving simulation analyzed with InternVL3-9B using "
            f"{input_mode_used} input."
        ),
    }


def run_frame_mode(
    *,
    args: argparse.Namespace,
    scene_id: str,
    simulation: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], Path]:
    duration_seconds = simulation["total_frames"] / args.fps
    sampled_frames = sample_representative_frames(
        simulation["frames"],
        duration_seconds=duration_seconds,
        sample_every_seconds=args.sample_every_seconds,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        num_frames=args.num_frames,
    )
    print(f"Frames sampled: {len(sampled_frames)}")

    copied_frames = copy_sampled_frames(sampled_frames, output_dir / "frames_sampled")
    analyses: list[dict[str, Any]] = []

    for sequence_index, frame_info in enumerate(copied_frames, start=1):
        print(f"Analyzing sampled frame {sequence_index}/{len(copied_frames)}: {frame_info['copied_path']}")
        model_analysis = run_single_frame_analysis(
            model=model,
            tokenizer=tokenizer,
            frame_path=frame_info["copied_path"],
            prompt=DETAILED_ANALYSIS_PROMPT,
            max_new_tokens=args.max_new_tokens,
        )
        analyses.append(
            {
                "analysis_id": f"{scene_id}_frame_{sequence_index:04d}",
                "input_type": "frame",
                "original_input_path": frame_info["original_path"],
                "copied_input_path": frame_info["copied_path"],
                "frame_index": frame_info["frame_index"],
                "prompt": DETAILED_ANALYSIS_PROMPT,
                "model_analysis": model_analysis,
            }
        )

    answer_path = output_dir / "model_answers" / "internvl3_9b_frame_analysis.json"
    answer_payload = {
        "scene_id": scene_id,
        "model_name": MODEL_NAME,
        "model_path": str(Path(args.model_path).expanduser()),
        "source_sim_dir": simulation["simulation_dir"],
        "input_mode_requested": args.input_mode,
        "input_mode_used": "frames",
        "analyses": analyses,
    }
    write_json(answer_path, answer_payload)
    return copied_frames, answer_path


def run_video_mode(
    *,
    args: argparse.Namespace,
    scene_id: str,
    simulation: dict[str, Any],
    output_dir: Path,
    model: Any,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], str, Path]:
    selected_video = simulation["video_files"][0]
    copied_video = copy_selected_file(selected_video, output_dir / "video_sample")
    pixel_values, used_frames, video_metadata = sample_video_as_tensors(
        selected_video,
        sample_every_seconds=args.sample_every_seconds,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        num_frames=args.num_frames,
    )
    print(f"Video sampled to frames internally: {len(used_frames)}")

    prompt = sequence_prompt(len(used_frames))
    model_analysis = run_video_sequence_analysis(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
    )

    analyses = [
        {
            "analysis_id": f"{scene_id}_video_0001",
            "input_type": "video",
            "original_input_path": selected_video,
            "copied_input_path": str(copied_video),
            "frame_index": None,
            "prompt": prompt,
            "model_analysis": model_analysis,
            "video_metadata": video_metadata,
            "sampled_video_frames": used_frames,
        }
    ]

    answer_path = output_dir / "model_answers" / "internvl3_9b_video_analysis.json"
    answer_payload = {
        "scene_id": scene_id,
        "model_name": MODEL_NAME,
        "model_path": str(Path(args.model_path).expanduser()),
        "source_sim_dir": simulation["simulation_dir"],
        "input_mode_requested": args.input_mode,
        "input_mode_used": "video",
        "analyses": analyses,
    }
    write_json(answer_path, answer_payload)
    return used_frames, str(copied_video), answer_path


def print_example_commands() -> None:
    print()
    print("Example commands:")
    print()
    print("Auto mode for simulation_45:")
    print(
        "python3 scripts/run_vlm_on_simulation.py "
        "--sim-dir data/carla/simulation_45 "
        "--scene-id carla_simulation_45 "
        "--input-mode auto "
        "--model-path /home/native/internvl3/InternVL/pretrained/InternVL3-9B"
    )
    print()
    print("Force frames for simulation_45:")
    print(
        "python3 scripts/run_vlm_on_simulation.py "
        "--sim-dir data/carla/simulation_45 "
        "--scene-id carla_simulation_45_frames "
        "--input-mode frames "
        "--model-path /home/native/internvl3/InternVL/pretrained/InternVL3-9B"
    )
    print()
    print("Force video for simulation_45:")
    print(
        "python3 scripts/run_vlm_on_simulation.py "
        "--sim-dir data/carla/simulation_45 "
        "--scene-id carla_simulation_45_video "
        "--input-mode video "
        "--model-path /home/native/internvl3/InternVL/pretrained/InternVL3-9B"
    )


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        simulation = load_simulation(args.sim_dir)

        scene_id = args.scene_id or simulation["simulation_name"]
        input_mode_used = decide_input_mode(args.input_mode, simulation["video_files"])

        if not simulation["frames"] and not simulation["video_files"]:
            fail("Simulation contains no PNG frames and no MP4 video.")

        output_dir = Path(args.output_root).expanduser() / scene_id

        print(f"Simulation being processed: {simulation['simulation_dir']}")
        print(f"Total frames discovered: {simulation['total_frames']}")
        print(f"Videos discovered: {len(simulation['video_files'])}")
        print(f"Requested input mode: {args.input_mode}")
        print(f"Actual input mode used: {input_mode_used}")
        print(
            "Sampling settings: "
            f"sample_every_seconds={args.sample_every_seconds}, "
            f"min_frames={args.min_frames}, max_frames={args.max_frames}, "
            f"fps={args.fps}, num_frames={args.num_frames}"
        )
        print(f"Output directory: {output_dir}")

        validate_runtime(args)
        print(f"Loading {MODEL_NAME} from: {Path(args.model_path).expanduser()}")
        model, tokenizer = load_internvl3(args.model_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model_answers").mkdir(parents=True, exist_ok=True)

        video_used: str | None = None
        video_sampled_to_frames = False

        if input_mode_used == "video":
            used_frames, video_used, answer_path = run_video_mode(
                args=args,
                scene_id=scene_id,
                simulation=simulation,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
            )
            video_sampled_to_frames = True
        else:
            if not simulation["frames"]:
                fail("Frame mode requested or selected, but no PNG frames were found.")
            used_frames, answer_path = run_frame_mode(
                args=args,
                scene_id=scene_id,
                simulation=simulation,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
            )

        scenario_payload = build_scenario_payload(
            args=args,
            scene_id=scene_id,
            simulation=simulation,
            input_mode_used=input_mode_used,
            used_frames=used_frames,
            video_used=video_used,
            video_sampled_to_frames=video_sampled_to_frames,
        )
        write_json(output_dir / "scenario.json", scenario_payload)

        print(f"Model answer JSON path: {answer_path}")
        print_example_commands()
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
