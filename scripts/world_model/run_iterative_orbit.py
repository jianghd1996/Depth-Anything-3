#!/usr/bin/env python3
"""Generate absolute one-way orbit views, always starting from the real input."""

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
        default=DEFAULT_VIDEO_MODEL / "12000_lora.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WORLD_ROOT / "output/orbit_absolute_step10",
    )
    parser.add_argument("--degrees-per-step", type=int, default=10)
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument(
        "--center-crop",
        type=float,
        default=0.4,
        help="Centered image fraction used to estimate the orbit target in each DA3 view.",
    )
    parser.add_argument(
        "--pivot-depth-scale",
        type=float,
        default=0.9,
        help="Scale estimated orbit-target depth; values below 1 move it toward the camera.",
    )
    parser.add_argument("--max-da3-views", type=int, default=8)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8, help="VideoX-Fun denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--max-steps-this-run",
        type=int,
        help="Stop after this many new steps; rerun the same command to resume.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("image", "prompt", "da3_weights", "video_lora"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if not args.video_model.is_dir():
        raise FileNotFoundError(f"--video-model does not exist: {args.video_model}")
    if args.degrees_per_step <= 0 or 360 % args.degrees_per_step:
        raise ValueError("--degrees-per-step must be a positive integer divisor of 360")
    if args.max_da3_views < 2:
        raise ValueError("--max-da3-views must be at least 2")
    if not 0 < args.center_crop <= 1:
        raise ValueError("--center-crop must be in (0, 1]")
    if args.pivot_depth_scale <= 0:
        raise ValueError("--pivot-depth-scale must be positive")
    if args.max_steps_this_run is not None and args.max_steps_this_run < 1:
        raise ValueError("--max-steps-this-run must be positive")
    if (args.frames - 1) % 4:
        raise ValueError("--frames must satisfy (frames - 1) % 4 == 0")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")


def run(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def select_da3_views(memory_images: list[Path], maximum: int) -> list[Path]:
    """Uniformly sample history while preserving the original and latest views."""
    if len(memory_images) <= maximum:
        return memory_images.copy()
    count = len(memory_images)
    indices = [round(i * (count - 1) / (maximum - 1)) for i in range(maximum)]
    selected = [memory_images[index] for index in dict.fromkeys(indices)]
    latest = memory_images[-1]
    selected = [path for path in selected if path != latest]
    selected.append(latest)
    return selected


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


def build_orbit_preview(step_dirs: list[Path], output: Path, fps: int) -> None:
    """Build a contact-sheet video from the final absolute view of each segment."""
    assembled = []
    hold_frames = max(1, fps // 2)
    for step_dir in step_dirs:
        frames = read_video_frames(step_dir / "generated.mp4")
        assembled.extend([frames[-1]] * hold_frames)

    height, width = assembled[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create orbit preview: {output}")
    for frame in assembled:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()
    validate_args(args)
    total_steps = 360 // args.degrees_per_step
    state_path = args.output_dir / "orbit_state.json"
    memory_dir = args.output_dir / "memory/images"
    original_suffix = args.image.suffix.lower() or ".png"
    original_memory_image = memory_dir / f"view_000{original_suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "trajectory_strategy": "absolute-one-way-from-original",
            "degrees_per_step": args.degrees_per_step,
            "direction": args.direction,
            "center_crop": args.center_crop,
            "pivot_depth_scale": args.pivot_depth_scale,
            "total_steps": total_steps,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Existing orbit state has {key}={state.get(key)!r}, expected {value!r}. "
                    "Use a different --output-dir for a different orbit configuration."
                )
    else:
        original_memory_image.write_bytes(args.image.read_bytes())
        state = {
            "original_image": str(args.image.resolve()),
            "trajectory_strategy": "absolute-one-way-from-original",
            "degrees_per_step": args.degrees_per_step,
            "direction": args.direction,
            "center_crop": args.center_crop,
            "pivot_depth_scale": args.pivot_depth_scale,
            "total_steps": total_steps,
            "completed_steps": 0,
            "memory_images": [str(original_memory_image.resolve())],
            "step_outputs": [],
            "status": "running",
        }
        atomic_write_json(state_path, state)

    memory_images = [Path(path) for path in state["memory_images"]]
    for path in memory_images:
        if not path.is_file():
            raise FileNotFoundError(f"Remembered view is missing: {path}")
    original_constraint = memory_images[0]
    first_step = int(state["completed_steps"]) + 1
    if first_step > total_steps:
        preview = args.output_dir / "orbit_absolute_keyframes.mp4"
        if not preview.is_file():
            step_dirs = [
                args.output_dir
                / f"step_{index:02d}_{index * args.degrees_per_step:03d}deg"
                for index in range(1, total_steps + 1)
            ]
            build_orbit_preview(step_dirs, preview, args.fps)
        print(f"Orbit already complete: {preview}")
        return

    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    run_limit = total_steps
    if args.max_steps_this_run is not None:
        run_limit = min(total_steps, first_step + args.max_steps_this_run - 1)

    for step_index in range(first_step, run_limit + 1):
        cumulative_degrees = step_index * args.degrees_per_step
        is_closure = step_index == total_steps
        step_dir = args.output_dir / f"step_{step_index:02d}_{cumulative_degrees:03d}deg"
        geometry_dir = step_dir / "geometry"
        generated_video = step_dir / "generated.mp4"
        selected_images = select_da3_views(memory_images, args.max_da3_views)
        # render_single_image_orbit uses the last input as the trajectory source.
        # Keep generated views as optional geometry evidence, but always start the
        # planned camera path from the original real image.
        selected_images = [path for path in selected_images if path != original_constraint]
        selected_images.append(original_constraint)

        geometry_command = [
            sys.executable,
            str(SCRIPT_DIR / "render_single_image_orbit.py"),
            "--images",
            *[str(path) for path in selected_images],
            "--prompt",
            str(args.prompt),
            "--weights",
            str(args.da3_weights),
            "--output-dir",
            str(geometry_dir),
            "--model-name",
            "da3nested-giant-large",
            "--frames",
            str(args.frames),
            "--degrees",
            str(cumulative_degrees),
            "--center-crop",
            str(args.center_crop),
            "--pivot-depth-scale",
            str(args.pivot_depth_scale),
            "--direction",
            args.direction,
            "--trajectory-mode",
            "one-way",
            "--fps",
            str(args.fps),
            "--device",
            args.device,
        ]
        run(geometry_command, env)

        generation_command = [
            sys.executable,
            str(SCRIPT_DIR / "generate_with_videox_fun.py"),
            "--image",
            str(original_constraint),
            "--reference-image",
            str(original_constraint),
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
            str(generated_video),
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
            str(args.seed + step_index - 1),
            "--device",
            args.device,
        ]
        if is_closure:
            generation_command.extend(["--end-image", str(original_constraint)])
        else:
            generation_command.append("--no-end-image")
        if args.height is not None:
            generation_command.extend(
                ["--height", str(args.height), "--width", str(args.width)]
            )
        run(generation_command, env)

        new_memory_image = None
        if not is_closure:
            new_memory_image = memory_dir / f"view_{cumulative_degrees:03d}.png"
            save_video_frame(generated_video, args.frames - 1, new_memory_image)
            memory_images.append(new_memory_image.resolve())

        step_record = {
            "step": step_index,
            "cumulative_degrees": cumulative_degrees,
            "is_closure": is_closure,
            "trajectory_mode": "absolute-one-way-from-original",
            "center_crop": args.center_crop,
            "pivot_depth_scale": args.pivot_depth_scale,
            "start_image": str(original_constraint),
            "end_image": str(original_constraint) if is_closure else None,
            "reference_image": str(original_constraint),
            "da3_images": [str(path) for path in selected_images],
            "generated_video": str(generated_video.resolve()),
            "new_memory_image": (
                None if new_memory_image is None else str(new_memory_image.resolve())
            ),
        }
        state["completed_steps"] = step_index
        state["memory_images"] = [str(path) for path in memory_images]
        state["step_outputs"].append(step_record)
        state["status"] = "complete" if is_closure else "running"
        atomic_write_json(state_path, state)
        print(f"Completed orbit step {step_index}/{total_steps}: {cumulative_degrees}°")

    if int(state["completed_steps"]) == total_steps:
        step_dirs = [
            args.output_dir
            / f"step_{index:02d}_{index * args.degrees_per_step:03d}deg"
            for index in range(1, total_steps + 1)
        ]
        preview = args.output_dir / "orbit_absolute_keyframes.mp4"
        build_orbit_preview(step_dirs, preview, args.fps)
        print(f"Full orbit complete: {preview}")
    else:
        print(f"Paused after step {state['completed_steps']}; rerun the same command to resume")


if __name__ == "__main__":
    main()
