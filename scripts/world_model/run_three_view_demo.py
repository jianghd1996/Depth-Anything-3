#!/usr/bin/env python3
"""Render two cinematic three-view moves, generate endpoint-conditioned clips, and join them."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous
from depth_anything_3.world_model import estimate_orbit_pivot, render_orbit
from render_single_image_orbit import load_checkpoint_weights, write_mp4


def smooth_progress(t):
    """Quintic easing: near-zero speed at endpoints, faster in the middle."""
    return t * t * t * (10 + t * (-15 + 6 * t))


def plan_segment(start, end, pivot, frames: int, dolly: float, *, level_dolly: bool = False) -> torch.Tensor:
    """Smooth endpoint interpolation with a center-distance dolly at the halfway point.

    Camera rotations are interpolated in camera-to-world coordinates. The dolly
    displacement follows the camera-to-pivot ray, in units of starting distance.
    For level_dolly, its extra displacement is projected onto the camera horizontal plane.
    """
    a = affine_inverse(as_homogeneous(torch.as_tensor(start, dtype=torch.float64))).numpy()
    b = affine_inverse(as_homogeneous(torch.as_tensor(end, dtype=torch.float64))).numpy()
    p = np.asarray(pivot, dtype=np.float64)
    ca, cb = a[:3, 3], b[:3, 3]
    radius = np.linalg.norm(p - ca)
    if radius < 1e-6:
        raise ValueError("Target pivot is too close to the starting camera")
    # Negative dolly pushes toward the target; positive dolly pulls back.
    # Keep the push in front of the estimated target surface.
    if dolly < 0 and -dolly >= 0.8:
        raise ValueError("Push fraction must be less than 0.8 of target distance")
    # OpenCV camera Y points down. The mean photographed camera-up axis
    # defines the local horizontal plane without assuming a global Z-up scene.
    up = -(a[:3, 1] + b[:3, 1])
    if np.linalg.norm(up) < 1e-6:
        up = -a[:3, 1]
    up /= np.linalg.norm(up)
    rotations = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    slerp = Slerp([0, 1], rotations)
    positions = []
    progress = np.linspace(0, 1, frames)
    for t in progress:
        ease = smooth_progress(t)
        base = (1 - ease) * ca + ease * cb
        # Sin² has zero value and derivative at both photographed endpoints.
        envelope = np.sin(np.pi * ease) ** 2
        direction = (base - p) / max(np.linalg.norm(base - p), 1e-6)
        if level_dolly:
            direction = direction - up * np.dot(direction, up)
            direction /= max(np.linalg.norm(direction), 1e-6)
        positions.append(base + direction * radius * dolly * envelope)
    c2w = np.repeat(np.eye(4)[None], frames, axis=0)
    c2w[:, :3, :3] = slerp(smooth_progress(progress)).as_matrix()
    c2w[:, :3, 3] = np.asarray(positions)
    c2w[0], c2w[-1] = a, b
    return affine_inverse(torch.from_numpy(c2w).float())


def plan_crane_segment(start, end, pivot, frames: int, lift_fraction: float) -> torch.Tensor:
    """Move left-to-right with a gentle raised camera midway; preserve both poses."""
    poses = plan_segment(start, end, pivot, frames, 0.0)
    c2w = affine_inverse(poses.double())
    start_c2w = affine_inverse(as_homogeneous(torch.as_tensor(start, dtype=torch.float64)))
    up = -start_c2w[:3, 1]
    up = up / up.norm().clamp_min(1e-8)
    radius = (start_c2w[:3, 3] - torch.as_tensor(pivot, dtype=torch.float64)).norm()
    t = torch.linspace(0, 1, frames, dtype=torch.float64)
    c2w[:, :3, 3] += (radius * lift_fraction * torch.sin(torch.pi * smooth_progress(t)).square())[:, None] * up
    return affine_inverse(c2w).float()


def stitch_clips(clips: list, fps: int, overlap_frames: int):
    """Fade color correction away from each seam, then crossfade the moving clips."""
    if overlap_frames < 1 or overlap_frames >= min(round(c.duration * fps) for c in clips) // 4:
        raise ValueError('Crossfade must be positive and shorter than a quarter clip')
    corrected = [clips[0]]
    for clip in clips[1:]:
        previous = corrected[-1]
        before = previous.get_frame(max(0, previous.duration - 1 / fps)).astype(np.float32)
        after = clip.get_frame(0).astype(np.float32)
        # Robust central-region means avoid the black geometry border and outliers.
        h, w = before.shape[:2]
        area = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
        target = before[area].mean(axis=(0, 1))
        source = after[area].mean(axis=(0, 1))
        gain = np.clip(target / np.maximum(source, 16), 0.85, 1.15)
        fade_duration = max(0.75, overlap_frames / fps * 2)
        adjusted = clip.fl(lambda gf, t, gain=gain, fade=fade_duration:
            np.clip(gf(t).astype(np.float32) *
                    (1 + (gain - 1) * max(0.0, 1 - t / fade)), 0, 255).astype(np.uint8),
            keep_duration=True)
        corrected.append(adjusted)
    return mpy.concatenate_videoclips(
        [corrected[0]] + [c.crossfadein(overlap_frames / fps) for c in corrected[1:]],
        method='compose', padding=-overlap_frames / fps)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('configs') / 'three_view_demo.json')
    parser.add_argument('--images-dir', type=Path)
    parser.add_argument('--left', type=Path)
    parser.add_argument('--middle', type=Path)
    parser.add_argument('--right', type=Path)
    parser.add_argument('--prompt', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model-name', default='da3nested-giant-large')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--process-res', type=int, default=504)
    parser.add_argument('--frames', type=int, default=81)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--push-fraction', type=float, default=0.18)
    parser.add_argument('--lift-fraction', type=float, default=0.10)
    parser.add_argument('--pull-fraction', type=float, default=0.20)
    parser.add_argument('--crossfade-frames', type=int, default=6)
    parser.add_argument('--center-crop', type=float, default=0.4)
    parser.add_argument('--pivot-depth-scale', type=float, default=1.0)
    parser.add_argument('--alpha-threshold', type=float, default=0.01)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--guidance-scale', type=float, default=6.0)
    parser.add_argument('--lora-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video-model', type=Path)
    parser.add_argument('--lora-path', type=Path)
    parser.add_argument('--render-only', action='store_true', help='Preview DA3 geometry without loading VideoX-Fun')
    config_option, _ = parser.parse_known_args()
    if not config_option.config.is_file():
        parser.error(f'Config file does not exist: {config_option.config}')
    settings = json.loads(config_option.config.read_text(encoding='utf-8'))
    if not isinstance(settings, dict):
        parser.error('Config must contain a JSON object')
    actions = {action.dest: action for action in parser._actions}
    for key, value in settings.items():
        if key not in actions or key in ('help', 'config'):
            parser.error(f'Unknown config key: {key}')
        if value is not None and actions[key].type is Path:
            settings[key] = Path(value)
    parser.set_defaults(**settings)
    args = parser.parse_args()
    for name in ('prompt', 'weights', 'output_dir'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} must be set in config or CLI')
    if args.images_dir is None and not all((args.left, args.middle, args.right)):
        parser.error('Set images_dir or all three explicit image paths')
    return args


def main():
    args = parse_args()
    explicit = [args.left, args.middle, args.right]
    if any(explicit):
        if not all(explicit):
            raise ValueError('Specify all of --left, --middle and --right together')
        images = explicit
    else:
        candidates = [p for p in args.images_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
        if len(candidates) != 3:
            raise ValueError(f'Expected exactly 3 images in {args.images_dir}; found {len(candidates)}. '
                             'Pass --left, --middle and --right for an explicit selection.')
        by_prefix = {prefix: [p for p in candidates if p.name.startswith(prefix + '_')]
                     for prefix in ('0', '1', '2')}
        if any(len(matches) != 1 for matches in by_prefix.values()):
            raise ValueError('Cannot identify unique 0_, 1_, 2_ images. Set left/middle/right explicitly.')
        images = [by_prefix['1'][0], by_prefix['0'][0], by_prefix['2'][0]]
    print('Input order: left={}, middle={}, right={}'.format(*images))
    for path in [*images, args.prompt, args.weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.frames < 5 or (args.frames - 1) % 4:
        raise ValueError('Frames must be at least 5 and satisfy (frames - 1) % 4 == 0')
    if args.fps < 1 or args.chunk_size < 1:
        raise ValueError('fps and chunk-size must be positive')
    if not 0 <= args.push_fraction < 0.8 or args.pull_fraction < 0 or args.lift_fraction < 0:
        raise ValueError('push-fraction must be in [0, 0.8); pull-fraction and lift-fraction must be nonnegative')
    if (args.height is None) != (args.width is None):
        raise ValueError('Specify height and width together')
    if args.height is not None and (args.height % 32 or args.width % 32):
        raise ValueError('Output dimensions must be multiples of 32')
    if not args.render_only:
        for path in [args.video_model, args.lora_path]:
            if path is None or not path.exists():
                raise FileNotFoundError(f'Video model or LoRA missing: {path}; use --render-only for previews')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = DepthAnything3(model_name=args.model_name)
    missing, unexpected = load_checkpoint_weights(model, args.weights)
    if any('.gs_head.' in key for key in missing):
        raise RuntimeError('DA3 checkpoint is missing Gaussian-head weights')
    if unexpected:
        print(f'Ignored {len(unexpected)} unexpected checkpoint keys')
    model = model.to(args.device).eval()
    prediction = model.inference([str(p) for p in images], infer_gs=True,
        process_res=args.process_res, process_res_method='upper_bound_resize')
    if prediction.gaussians is None:
        raise RuntimeError('DA3 produced no Gaussian reconstruction')
    pivots = [estimate_orbit_pivot(prediction.depth[i], prediction.intrinsics[i],
        prediction.extrinsics[i],
        confidence=None if prediction.conf is None else prediction.conf[i],
        sky=None if prediction.sky is None else prediction.sky[i],
        center_crop=args.center_crop, depth_scale=args.pivot_depth_scale)
        for i in range(3)]
    pivot = torch.stack(pivots).median(dim=0).values
    specs = [('left_middle_push', 0, 1, 'push'),
             ('middle_right_lift', 1, 2, 'lift'),
             ('right_middle_pull', 2, 1, 'pull')]
    metadata = {'images': [str(x) for x in images], 'pivot_world': pivot.tolist(),
                'frames_per_clip': args.frames, 'fps': args.fps, 'segments': []}
    output_hw = (args.height, args.width) if args.height else None
    for name, start, end, trajectory_type in specs:
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        if trajectory_type == 'push':
            poses = plan_segment(prediction.extrinsics[start], prediction.extrinsics[end],
                                 pivot, args.frames, -args.push_fraction)
        elif trajectory_type == 'lift':
            poses = plan_crane_segment(prediction.extrinsics[start], prediction.extrinsics[end],
                                       pivot, args.frames, args.lift_fraction)
        else:
            poses = plan_segment(prediction.extrinsics[start], prediction.extrinsics[end],
                                 pivot, args.frames, args.pull_fraction, level_dolly=True)
        rgb, depth, valid = render_orbit(prediction, poses, output_hw=output_hw,
            chunk_size=args.chunk_size, alpha_threshold=args.alpha_threshold,
            source_view_index=start)
        video = rgb.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
        mask = valid.byte().mul(255).unsqueeze(-1).expand(-1, -1, -1, 3).cpu().numpy()
        write_mp4(video, folder / 'gs_render.mp4', args.fps)
        write_mp4(mask, folder / 'mask.mp4', args.fps, crf=0)
        Image.fromarray(video[args.frames // 2]).save(folder / 'turning_point.png')
        np.savez_compressed(folder / 'camera_trajectory.npz', extrinsics=poses.numpy(),
            pivot_world=pivot.numpy(), rendered_depth=depth.float().cpu().numpy())
        metadata['segments'].append({'name': name, 'start': str(images[start]),
            'end': str(images[end]), 'trajectory_type': trajectory_type,
            'minimum_valid_fraction': float(valid.float().mean(dim=(1,2)).min())})
        if args.render_only:
            continue
        command = [sys.executable, str(Path(__file__).with_name('generate_with_videox_fun.py')),
            '--image', str(images[start]), '--end-image', str(images[end]),
            '--reference-image', str(images[end]), '--prompt', str(args.prompt),
            '--control-video', str(folder / 'gs_render.mp4'),
            '--control-mask', str(folder / 'mask.mp4'),
            '--model-path', str(args.video_model), '--lora-path', str(args.lora_path),
            '--output', str(folder / 'generated.mp4'), '--frames', str(args.frames),
            '--fps', str(args.fps), '--steps', str(args.steps),
            '--guidance-scale', str(args.guidance_scale),
            '--lora-weight', str(args.lora_weight), '--seed', str(args.seed + start),
            '--device', args.device]
        if output_hw:
            command += ['--height', str(args.height), '--width', str(args.width)]
        subprocess.run(command, check=True)
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.render_only:
        print('Geometry previews and trajectories saved to', args.output_dir)
        return
    clips = [mpy.VideoFileClip(str(args.output_dir / name / 'generated.mp4'))
             for name, *_ in specs]
    try:
        joined = stitch_clips(clips, args.fps, args.crossfade_frames)
        try:
            joined.write_videofile(str(args.output_dir / 'demo.mp4'), codec='libx264',
                audio=False, fps=args.fps, ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p'])
        finally:
            joined.close()
    finally:
        for clip in clips:
            clip.close()
    print('Demo saved to', args.output_dir / 'demo.mp4')


if __name__ == '__main__':
    main()
