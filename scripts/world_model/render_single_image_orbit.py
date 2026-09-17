#!/usr/bin/env python3
"""Reconstruct one image with DA3 and render an 81-frame 90-degree orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_ply
from depth_anything_3.world_model import (
    estimate_orbit_pivot,
    generate_orbit_trajectory,
    render_orbit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/image.jpg"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/prompt.txt"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/DA3.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output/task1_orbit_roundtrip_10"
        ),
    )
    parser.add_argument(
        "--model-name",
        default="da3nested-giant-large",
        choices=("da3-giant", "da3nested-giant-large"),
        help=(
            "DA3 architecture matching the local checkpoint. The default matches DA3.pt, "
            "which was saved from da3nested-giant-large."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument(
        "--trajectory-mode",
        choices=("round-trip", "one-way"),
        default="round-trip",
        help=(
            "round-trip rotates to the requested angle and returns to the input view, "
            "so the known image can constrain both endpoint frames."
        ),
    )
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--center-crop", type=float, default=0.4)
    parser.add_argument(
        "--pivot-depth-scale",
        type=float,
        default=1.0,
        help="Move the estimated pivot along the camera ray; >1 puts it behind the visible surface.",
    )
    parser.add_argument("--alpha-threshold", type=float, default=0.01)
    parser.add_argument("--output-height", type=int)
    parser.add_argument("--output-width", type=int)
    parser.add_argument("--no-save-ply", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("image", "prompt", "weights"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name} does not exist: {path}")
    if (args.output_height is None) != (args.output_width is None):
        raise ValueError("--output-height and --output-width must be provided together")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")


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


def load_checkpoint_weights(
    model: DepthAnything3, checkpoint_path: Path
) -> tuple[list[str], list[str]]:
    """Load DA3 training checkpoints without rewriting model parameter names.

    The project checkpoint is saved as ``{"model": model.state_dict(), ...}``.
    Loading that nested state dict directly mirrors the training/inference code
    that produced DA3.pt and, importantly, preserves the nested model's
    ``model.da3.*`` and ``model.da3_metric.*`` parameter names.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a state-dict checkpoint, got {type(checkpoint).__name__}"
        )

    state_dict = checkpoint
    for container_key in ("model", "state_dict", "model_state_dict"):
        candidate = checkpoint.get(container_key)
        if isinstance(candidate, dict) and candidate:
            state_dict = candidate
            print(f"Loading weights from checkpoint[{container_key!r}]")
            break

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text(encoding="utf-8").strip()

    print(f"Loading {args.model_name} from {args.weights}")
    model = DepthAnything3(model_name=args.model_name)
    missing, unexpected = load_checkpoint_weights(model, args.weights)
    missing_gs = [key for key in missing if ".gs_head." in key]
    if missing_gs:
        sample = "\n  ".join(missing_gs[:8])
        raise RuntimeError(
            "The checkpoint does not contain the Gaussian head required for orbit rendering. "
            f"Check --model-name and --weights. Missing keys include:\n  {sample}"
        )
    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unexpected checkpoint keys")

    device = torch.device(args.device)
    model = model.to(device).eval()
    prediction = model.inference(
        [str(args.image)],
        infer_gs=True,
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
    )
    if prediction.gaussians is None:
        raise RuntimeError("DA3 inference completed without Gaussian output")

    pivot = estimate_orbit_pivot(
        depth=prediction.depth[0],
        intrinsics=prediction.intrinsics[0],
        extrinsics=prediction.extrinsics[0],
        confidence=None if prediction.conf is None else prediction.conf[0],
        sky=None if prediction.sky is None else prediction.sky[0],
        center_crop=args.center_crop,
        depth_scale=args.pivot_depth_scale,
    )
    trajectory = generate_orbit_trajectory(
        prediction.extrinsics[0],
        pivot,
        num_frames=args.frames,
        degrees=args.degrees,
        direction=args.direction,
        return_to_start=args.trajectory_mode == "round-trip",
    )
    output_hw = None
    if args.output_height is not None:
        output_hw = (args.output_height, args.output_width)
    rgb, depth, valid_mask = render_orbit(
        prediction,
        trajectory,
        output_hw=output_hw,
        chunk_size=args.chunk_size,
        alpha_threshold=args.alpha_threshold,
    )

    rgb_u8 = (
        rgb.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    )
    mask_u8 = valid_mask.byte().mul(255).unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()
    write_mp4(rgb_u8, args.output_dir / "gs_render.mp4", args.fps)
    write_mp4(mask_u8, args.output_dir / "mask.mp4", args.fps, crf=0)

    Image.fromarray(rgb_u8[0]).save(args.output_dir / "frame_000.png")
    Image.fromarray(rgb_u8[-1]).save(args.output_dir / f"frame_{args.frames - 1:03d}.png")
    Image.fromarray(mask_u8[-1]).save(args.output_dir / f"mask_{args.frames - 1:03d}.png")

    if not args.no_save_ply:
        export_to_gs_ply(prediction, str(args.output_dir))

    render_height, render_width = int(rgb.shape[-2]), int(rgb.shape[-1])
    input_height, input_width = prediction.depth.shape[-2:]
    render_intrinsics = prediction.intrinsics[0].copy()
    render_intrinsics[0] *= render_width / input_width
    render_intrinsics[1] *= render_height / input_height
    np.savez_compressed(
        args.output_dir / "camera_trajectory.npz",
        extrinsics=trajectory.cpu().numpy(),
        intrinsics=np.repeat(render_intrinsics[None], args.frames, axis=0),
        pivot_world=pivot.cpu().numpy(),
        rendered_depth=depth.float().cpu().numpy(),
    )
    metadata = {
        "image": str(args.image),
        "prompt_file": str(args.prompt),
        "prompt": prompt,
        "weights": str(args.weights),
        "model_name": args.model_name,
        "frames": args.frames,
        "degrees": args.degrees,
        "trajectory_mode": args.trajectory_mode,
        "peak_view_frame": args.frames // 2 if args.trajectory_mode == "round-trip" else args.frames - 1,
        "direction": args.direction,
        "fps": args.fps,
        "process_resolution": args.process_res,
        "render_resolution": [render_height, render_width],
        "pivot_world": pivot.cpu().tolist(),
        "pivot_depth_scale": args.pivot_depth_scale,
        "alpha_threshold": args.alpha_threshold,
        "mask_convention": "white (255) = rendered/valid geometry; black (0) = missing geometry",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done. Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
