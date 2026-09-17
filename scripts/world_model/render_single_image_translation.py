#!/usr/bin/env python3
"""Run DA3 once and render fixed-geometry horizontal translation segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.world_model import (
    estimate_orbit_pivot,
    generate_translation_trajectory,
    render_orbit,
)


DEFAULT_WORLD_ROOT = Path("/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_WORLD_ROOT / "image.jpg")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_WORLD_ROOT / "prompt.txt")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WORLD_ROOT / "DA3.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WORLD_ROOT / "output/translation_right_roundtrip/geometry",
    )
    parser.add_argument("--model-name", default="da3nested-giant-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument(
        "--shift-ratio",
        type=float,
        default=0.08,
        help="Horizontal shift per segment as a fraction of robust scene depth.",
    )
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--center-crop", type=float, default=0.4)
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--output-width", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("image", "prompt", "weights"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name} does not exist: {path}")
    if args.segments < 1:
        raise ValueError("--segments must be positive")
    if args.shift_ratio <= 0:
        raise ValueError("--shift-ratio must be positive")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if (args.output_height is None) != (args.output_width is None):
        raise ValueError("--output-height and --output-width must be provided together")


def load_checkpoint_weights(model: DepthAnything3, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint).__name__}")
    state_dict = checkpoint
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            state_dict = candidate
            print(f"Loading weights from checkpoint[{key!r}]")
            break
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_gs = [key for key in missing if ".gs_head." in key]
    if missing_gs:
        raise RuntimeError(
            "Checkpoint is missing the Gaussian head; first missing key: "
            + missing_gs[0]
        )
    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unexpected checkpoint keys")


def write_mp4(frames: np.ndarray, path: Path, fps: int, crf: int = 18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = mpy.ImageSequenceClip(list(frames), fps=fps)
    try:
        clip.write_videofile(
            str(path),
            codec="libx264",
            audio=False,
            fps=fps,
            ffmpeg_params=["-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"],
        )
    finally:
        clip.close()


def robust_reference_depth(prediction, center_crop: float) -> tuple[torch.Tensor, float]:
    """Reuse the pivot filter, but keep only its camera-space depth."""
    pivot_world = estimate_orbit_pivot(
        depth=prediction.depth[0],
        intrinsics=prediction.intrinsics[0],
        extrinsics=prediction.extrinsics[0],
        confidence=None if prediction.conf is None else prediction.conf[0],
        sky=None if prediction.sky is None else prediction.sky[0],
        center_crop=center_crop,
    )
    w2c = torch.as_tensor(as_homogeneous(prediction.extrinsics[0]), dtype=torch.float32)
    pivot_h = torch.cat([pivot_world.cpu(), torch.ones(1)])
    depth = float((w2c @ pivot_h)[2])
    if not np.isfinite(depth) or depth <= 0:
        raise RuntimeError(f"Invalid robust reference depth: {depth}")
    return pivot_world, depth


def save_segment(
    prediction,
    trajectory: torch.Tensor,
    output_dir: Path,
    args: argparse.Namespace,
    metadata: dict,
) -> None:
    output_hw = None
    if args.output_height is not None:
        output_hw = (args.output_height, args.output_width)
    rgb, depth, valid_mask = render_orbit(
        prediction,
        trajectory,
        output_hw=output_hw,
        chunk_size=args.chunk_size,
        alpha_threshold=args.alpha_threshold,
        source_view_index=0,
    )
    rgb_u8 = rgb.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    mask_u8 = valid_mask.byte().mul(255).unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()
    write_mp4(rgb_u8, output_dir / "gs_render.mp4", args.fps)
    write_mp4(mask_u8, output_dir / "mask.mp4", args.fps, crf=0)
    Image.fromarray(rgb_u8[0]).save(output_dir / "frame_000.png")
    Image.fromarray(rgb_u8[-1]).save(output_dir / f"frame_{args.frames - 1:03d}.png")
    np.savez_compressed(
        output_dir / "camera_trajectory.npz",
        extrinsics=trajectory.cpu().numpy(),
        rendered_depth=depth.float().cpu().numpy(),
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model_name} from {args.weights}")
    model = DepthAnything3(model_name=args.model_name)
    load_checkpoint_weights(model, args.weights)
    model = model.to(torch.device(args.device)).eval()
    prediction = model.inference(
        [str(args.image)],
        infer_gs=True,
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
    )
    if prediction.gaussians is None:
        raise RuntimeError("DA3 inference completed without Gaussian output")

    pivot_world, reference_depth = robust_reference_depth(prediction, args.center_crop)
    step_distance = reference_depth * args.shift_ratio
    common = {
        "image": str(args.image),
        "weights": str(args.weights),
        "direction": args.direction,
        "segments": args.segments,
        "shift_ratio": args.shift_ratio,
        "reference_depth": reference_depth,
        "step_distance": step_distance,
        "pivot_world": pivot_world.cpu().tolist(),
        "fixed_single_image_geometry": True,
        "mask_convention": "white (255) = rendered geometry; black (0) = missing",
    }

    for index in range(1, args.segments + 1):
        start_distance = (index - 1) * step_distance
        end_distance = index * step_distance
        trajectory = generate_translation_trajectory(
            prediction.extrinsics[0],
            start_distance=start_distance,
            end_distance=end_distance,
            num_frames=args.frames,
            direction=args.direction,
            return_to_start=True,
        )
        save_segment(
            prediction,
            trajectory,
            args.output_dir / f"out_{index:02d}",
            args,
            {
                **common,
                "phase": "outbound",
                "segment": index,
                "start_distance": start_distance,
                "peak_distance": end_distance,
                "trajectory_mode": "round-trip",
                "peak_view_frame": args.frames // 2,
            },
        )

    for index in range(args.segments, 0, -1):
        start_distance = index * step_distance
        end_distance = (index - 1) * step_distance
        trajectory = generate_translation_trajectory(
            prediction.extrinsics[0],
            start_distance=start_distance,
            end_distance=end_distance,
            num_frames=args.frames,
            direction=args.direction,
            return_to_start=False,
        )
        save_segment(
            prediction,
            trajectory,
            args.output_dir / f"back_{index:02d}_to_{index - 1:02d}",
            args,
            {
                **common,
                "phase": "return",
                "segment": index,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "trajectory_mode": "one-way",
                "peak_view_frame": args.frames - 1,
            },
        )

    (args.output_dir / "metadata.json").write_text(
        json.dumps(common, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered {args.segments} outbound and {args.segments} return segments")


if __name__ == "__main__":
    main()
