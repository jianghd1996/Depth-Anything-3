#!/usr/bin/env python3
"""Run one DA3 geometry-expansion step followed by VideoX-Fun generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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
        default=DEFAULT_WORLD_ROOT / "output/world_model_step",
    )
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument(
        "--center-crop",
        type=float,
        default=0.4,
        help="Centered image fraction used to estimate the orbit target depth.",
    )
    parser.add_argument(
        "--pivot-depth-scale",
        type=float,
        default=1.0,
        help="Scale orbit-target depth; use a value below 1 when the target is too deep.",
    )
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
        "--skip-geometry",
        action="store_true",
        help="Reuse geometry/gs_render.mp4 and geometry/mask.mp4 from an earlier run.",
    )
    parser.add_argument(
        "--save-peak-frame-to-memory",
        type=Path,
        metavar="MEMORY_DIR",
        help="Opt in to saving the generated maximum-angle middle frame under MEMORY_DIR/images.",
    )
    return parser.parse_args()


def run(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


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


def main() -> None:
    args = parse_args()
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.degrees <= 0:
        raise ValueError("--degrees must be positive")
    if not 0 < args.center_crop <= 1:
        raise ValueError("--center-crop must be in (0, 1]")
    if args.pivot_depth_scale <= 0:
        raise ValueError("--pivot-depth-scale must be positive")

    geometry_dir = args.output_dir / "geometry"
    generated_video = args.output_dir / "generated.mp4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    if not args.skip_geometry:
        geometry_command = [
            sys.executable,
            str(SCRIPT_DIR / "render_single_image_orbit.py"),
            "--image",
            str(args.image),
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
            str(args.degrees),
            "--center-crop",
            str(args.center_crop),
            "--pivot-depth-scale",
            str(args.pivot_depth_scale),
            "--direction",
            args.direction,
            "--trajectory-mode",
            "round-trip",
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
        str(args.image),
        "--reference-image",
        str(args.image),
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
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.height is not None:
        generation_command.extend(["--height", str(args.height), "--width", str(args.width)])
    run(generation_command, env)

    memory_image = None
    if args.save_peak_frame_to_memory is not None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        memory_image = (
            args.save_peak_frame_to_memory
            / "images"
            / f"generated_{args.direction}_{args.degrees:g}deg_{timestamp}.png"
        )
        save_video_frame(generated_video, args.frames // 2, memory_image)

    manifest = {
        "image": str(args.image),
        "reference_image": str(args.image),
        "prompt": str(args.prompt),
        "geometry_dir": str(geometry_dir),
        "generated_video": str(generated_video),
        "degrees": args.degrees,
        "center_crop": args.center_crop,
        "pivot_depth_scale": args.pivot_depth_scale,
        "direction": args.direction,
        "trajectory_mode": "round-trip",
        "known_endpoint_frames": [0, args.frames - 1],
        "peak_view_frame": args.frames // 2,
        "saved_memory_image": None if memory_image is None else str(memory_image),
    }
    (args.output_dir / "step_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"World-model step complete: {generated_video}")


if __name__ == "__main__":
    main()
