#!/usr/bin/env python3
"""Expand several horizontal views to the right, then return to the input view."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_WORLD_ROOT = Path("/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev")
DEFAULT_VIDEO_MODEL = Path(
    "/home/z00566689/dev/mnt/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_WORLD_ROOT / "image.jpg")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_WORLD_ROOT / "prompt.txt")
    parser.add_argument("--da3-weights", type=Path, default=DEFAULT_WORLD_ROOT / "DA3.pt")
    parser.add_argument("--video-model", type=Path, default=DEFAULT_VIDEO_MODEL)
    parser.add_argument(
        "--video-lora",
        type=Path,
        default=DEFAULT_VIDEO_MODEL / "33000_lora.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WORLD_ROOT / "output/translation_right_roundtrip",
    )
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument(
        "--shift-ratio",
        type=float,
        default=0.08,
        help="Camera translation per segment divided by robust center depth.",
    )
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--max-steps-this-run",
        type=int,
        help="Run at most this many new video segments, then pause safely.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("image", "prompt", "da3_weights", "video_lora"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if not args.video_model.is_dir():
        raise FileNotFoundError(f"--video-model does not exist: {args.video_model}")
    if args.segments < 1:
        raise ValueError("--segments must be positive")
    if args.shift_ratio <= 0:
        raise ValueError("--shift-ratio must be positive")
    if (args.frames - 1) % 4:
        raise ValueError("--frames must satisfy (frames - 1) % 4 == 0")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.max_steps_this_run is not None and args.max_steps_this_run < 1:
        raise ValueError("--max-steps-this-run must be positive")


def run(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_video_frame(video: Path, frame_index: int, destination: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Could not save memory image: {destination}")


def read_video_frames(path: Path) -> list:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No readable frames in {path}")
    return frames


def build_preview(output_dir: Path, segments: int, fps: int) -> Path:
    assembled = []
    for index in range(1, segments + 1):
        frames = read_video_frames(output_dir / f"out_{index:02d}" / "generated.mp4")
        selected = frames[: len(frames) // 2 + 1]
        assembled.extend(selected if not assembled else selected[1:])
    for index in range(segments, 0, -1):
        frames = read_video_frames(
            output_dir / f"back_{index:02d}_to_{index - 1:02d}" / "generated.mp4"
        )
        assembled.extend(frames[1:])

    height, width = assembled[0].shape[:2]
    output = output_dir / "translation_roundtrip.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create preview: {output}")
    for frame in assembled:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    return output


def generation_command(
    args: argparse.Namespace,
    image: Path,
    end_image: Path,
    geometry_dir: Path,
    output: Path,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "generate_with_videox_fun.py"),
        "--image",
        str(image),
        "--end-image",
        str(end_image),
        "--prompt",
        str(args.prompt),
        "--control-video",
        str(geometry_dir / "gs_render.mp4"),
        "--control-mask",
        str(geometry_dir / "mask.mp4"),
        "--model-path",
        str(args.video_model),
        "--lora-path",
        str(args.video_lora),
        "--output",
        str(output),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--lora-weight",
        str(args.lora_weight),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if args.height is not None:
        command.extend(["--height", str(args.height), "--width", str(args.width)])
    return command


def geometry_is_complete(root: Path, segments: int) -> bool:
    directories = [root / f"out_{index:02d}" for index in range(1, segments + 1)]
    directories.extend(
        root / f"back_{index:02d}_to_{index - 1:02d}"
        for index in range(segments, 0, -1)
    )
    required = ("gs_render.mp4", "mask.mp4", "camera_trajectory.npz", "metadata.json")
    return (root / "metadata.json").is_file() and all(
        all((directory / name).is_file() for name in required)
        for directory in directories
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = args.output_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "translation_state.json"
    geometry_root = args.output_dir / "geometry"
    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    config = {
        "image": str(args.image.resolve()),
        "da3_weights": str(args.da3_weights.resolve()),
        "segments": args.segments,
        "shift_ratio": args.shift_ratio,
        "direction": args.direction,
        "frames": args.frames,
        "fps": args.fps,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Existing state has {key}={state.get(key)!r}, expected {value!r}. "
                    "Use a new --output-dir for a different translation configuration."
                )
    else:
        original_suffix = args.image.suffix.lower() or ".png"
        original_memory = memory_dir / f"view_00{original_suffix}"
        original_memory.write_bytes(args.image.read_bytes())
        state = {
            **config,
            "original_image": str(args.image.resolve()),
            "completed_steps": 0,
            "memory_images": [str(original_memory.resolve())],
            "step_outputs": [],
            "status": "running",
        }
        atomic_write_json(state_path, state)

    if not geometry_is_complete(geometry_root, args.segments):
        geometry_command = [
            sys.executable,
            str(SCRIPT_DIR / "render_single_image_translation.py"),
            "--image",
            str(args.image),
            "--prompt",
            str(args.prompt),
            "--weights",
            str(args.da3_weights),
            "--output-dir",
            str(geometry_root),
            "--segments",
            str(args.segments),
            "--shift-ratio",
            str(args.shift_ratio),
            "--direction",
            args.direction,
            "--frames",
            str(args.frames),
            "--fps",
            str(args.fps),
            "--device",
            args.device,
        ]
        run(geometry_command, env)

    memory_images = [Path(path) for path in state["memory_images"]]
    for path in memory_images:
        if not path.is_file():
            raise FileNotFoundError(f"Remembered view is missing: {path}")

    total_steps = args.segments * 2
    first_step = int(state["completed_steps"]) + 1
    if first_step > total_steps:
        preview = args.output_dir / "translation_roundtrip.mp4"
        if not preview.is_file():
            preview = build_preview(args.output_dir, args.segments, args.fps)
        print(f"Translation round trip already complete: {preview}")
        return
    last_step = total_steps
    if args.max_steps_this_run is not None:
        last_step = min(total_steps, first_step + args.max_steps_this_run - 1)

    for task_index in range(first_step, last_step + 1):
        if task_index <= args.segments:
            level = task_index
            step_dir = args.output_dir / f"out_{level:02d}"
            geometry_dir = geometry_root / f"out_{level:02d}"
            start_image = memory_images[level - 1]
            end_image = start_image
            phase = "outbound"
        else:
            return_index = task_index - args.segments - 1
            level = args.segments - return_index
            step_dir = args.output_dir / f"back_{level:02d}_to_{level - 1:02d}"
            geometry_dir = geometry_root / f"back_{level:02d}_to_{level - 1:02d}"
            start_image = memory_images[level]
            end_image = memory_images[level - 1]
            phase = "return"

        output_video = step_dir / "generated.mp4"
        step_dir.mkdir(parents=True, exist_ok=True)
        run(
            generation_command(
                args,
                start_image,
                end_image,
                geometry_dir,
                output_video,
                args.seed + task_index - 1,
            ),
            env,
        )

        new_memory = None
        if phase == "outbound":
            new_memory = memory_dir / f"view_{level:02d}.png"
            save_video_frame(output_video, args.frames // 2, new_memory)
            memory_images.append(new_memory.resolve())
        state["completed_steps"] = task_index
        state["memory_images"] = [str(path) for path in memory_images]
        state["step_outputs"].append(
            {
                "task": task_index,
                "phase": phase,
                "level": level,
                "start_image": str(start_image),
                "end_image": str(end_image),
                "geometry": str(geometry_dir),
                "generated_video": str(output_video.resolve()),
                "new_memory_image": (
                    None if new_memory is None else str(new_memory.resolve())
                ),
            }
        )
        state["status"] = "complete" if task_index == total_steps else "running"
        atomic_write_json(state_path, state)
        print(f"Completed {phase} segment {task_index}/{total_steps}")

    if int(state["completed_steps"]) == total_steps:
        preview = build_preview(args.output_dir, args.segments, args.fps)
        print(f"Translation round trip complete: {preview}")
    else:
        print(f"Paused after task {state['completed_steps']}; rerun the same command to resume")


if __name__ == "__main__":
    main()
