#!/usr/bin/env python3
"""Render two spherical camera moves, generate endpoint-conditioned clips, and join them."""
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


def resample_constant_speed(c2w: np.ndarray, frames: int) -> torch.Tensor:
    """Sample camera positions at equal arc-length intervals, preserving endpoints."""
    centers = c2w[:, :3, 3]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    if cumulative[-1] < 1e-8:
        raise ValueError('DA3 camera positions coincide; cannot plan a moving shot')
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    distance = np.linspace(0, cumulative[-1], frames)
    sampled = np.repeat(np.eye(4)[None], frames, axis=0)
    for axis in range(3):
        sampled[:, axis, 3] = np.interp(distance, cumulative[keep], centers[keep, axis])
    rotations = Rotation.from_matrix(c2w[keep, :3, :3])
    sampled[:, :3, :3] = Slerp(cumulative[keep], rotations)(distance).as_matrix()
    sampled[0], sampled[-1] = c2w[0], c2w[-1]
    return affine_inverse(torch.from_numpy(sampled).float())


def plan_spherical_segment(start, end, pivot, frames: int) -> torch.Tensor:
    """Great-circle orbit about the target, with photographed endpoint poses."""
    a = affine_inverse(as_homogeneous(torch.as_tensor(start, dtype=torch.float64))).numpy()
    b = affine_inverse(as_homogeneous(torch.as_tensor(end, dtype=torch.float64))).numpy()
    center = np.asarray(pivot, dtype=np.float64)
    va, vb = a[:3, 3] - center, b[:3, 3] - center
    ra, rb = np.linalg.norm(va), np.linalg.norm(vb)
    if min(ra, rb) < 1e-6:
        raise ValueError('Camera coincides with the orbit pivot')
    ua, ub = va / ra, vb / rb
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    angle = np.arccos(dot)
    if angle > np.pi - 1e-5:
        raise ValueError('Opposite camera directions make the shortest spherical arc ambiguous')
    progress = np.linspace(0.0, 1.0, max(frames * 8, 64))
    if angle < 1e-6:
        directions = (1 - progress[:, None]) * ua + progress[:, None] * ub
    else:
        directions = (np.sin((1 - progress) * angle)[:, None] * ua
                      + np.sin(progress * angle)[:, None] * ub) / np.sin(angle)
    radii = (1 - progress) * ra + progress * rb
    c2w = np.repeat(np.eye(4)[None], len(progress), axis=0)
    c2w[:, :3, 3] = center + directions * radii[:, None]
    c2w[:, :3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))(progress).as_matrix()
    c2w[0], c2w[-1] = a, b
    return resample_constant_speed(c2w, frames)


def join_original_clips(paths: list[Path], output: Path) -> None:
    """Join encoded clips without overlap, dropped frames, filtering, or re-encoding."""
    manifest = output.parent / 'concat_inputs.txt'
    # ffmpeg concat demuxer needs absolute paths; quote escaping is for its file syntax.
    manifest.write_text(''.join(
        "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"
        for path in paths), encoding='utf-8')
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(manifest),
                    '-c', 'copy', str(output)], check=True)


def parse_args(argv=None):
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
    parser.add_argument('--center-crop', type=float, default=0.4)
    parser.add_argument('--pivot-depth-scale', type=float, default=1.0)
    parser.add_argument('--alpha-threshold', type=float, default=0.01)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--short-edge', type=int, default=1088)
    parser.add_argument('--long-edge', type=int, default=1920)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--guidance-scale', type=float, default=6.0)
    parser.add_argument('--lora-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video-model', type=Path)
    parser.add_argument('--lora-path', type=Path)
    parser.add_argument('--render-only', action='store_true', help='Preview DA3 geometry without loading VideoX-Fun')
    config_option, _ = parser.parse_known_args(argv)
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
    args = parser.parse_args(argv)
    for name in ('prompt', 'weights', 'output_dir'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} must be set in config or CLI')
    if args.images_dir is None and not all((args.left, args.middle, args.right)):
        parser.error('Set images_dir or all three explicit image paths')
    return args


def main(argv=None, *, da3_cache=None, video_cache=None):
    args = parse_args(argv)
    if video_cache is None and not args.render_only:
        video_cache = {}
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
    if (args.height is None) != (args.width is None):
        raise ValueError('Specify height and width together')
    if args.height is not None and (args.height % 32 or args.width % 32):
        raise ValueError('Output dimensions must be multiples of 32')
    if args.short_edge < 32 or args.long_edge < args.short_edge or args.short_edge % 32 or args.long_edge % 32:
        raise ValueError('short-edge and long-edge must be positive multiples of 32, long >= short')
    if args.height is not None:
        generation_hw = (args.height, args.width)
    else:
        with Image.open(images[0]) as source_image:
            image_width, image_height = source_image.size
        generation_hw = ((args.short_edge, args.long_edge) if image_width >= image_height
                         else (args.long_edge, args.short_edge))
    print(f'Generation resolution: {generation_hw[1]}x{generation_hw[0]} (width x height)')
    if not args.render_only:
        for path in [args.video_model, args.lora_path]:
            if path is None or not path.exists():
                raise FileNotFoundError(f'Video model or LoRA missing: {path}; use --render-only for previews')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    da3_key = (str(args.weights.resolve()), args.model_name)
    if da3_cache is not None and da3_cache.get('key') == da3_key:
        model = da3_cache['model']
    else:
        model = DepthAnything3(model_name=args.model_name)
        missing, unexpected = load_checkpoint_weights(model, args.weights)
        if any('.gs_head.' in key for key in missing):
            raise RuntimeError('DA3 checkpoint is missing Gaussian-head weights')
        if unexpected:
            print(f'Ignored {len(unexpected)} unexpected checkpoint keys')
        if da3_cache is not None:
            da3_cache.clear()
            da3_cache.update(key=da3_key, model=model)
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
    specs = [('left_middle_orbit', 0, 1, 'spherical_orbit'),
             ('middle_right_orbit', 1, 2, 'spherical_orbit')]
    metadata = {'images': [str(x) for x in images], 'pivot_world': pivot.tolist(),
                'frames_per_clip': args.frames, 'fps': args.fps,
                'generation_resolution_hw': list(generation_hw),
                'guidance_scale': args.guidance_scale, 'segments': []}
    # DA3 geometry stays at its reconstruction resolution. VideoX-Fun resizes
    # guidance to generation_hw, avoiding 81 high-res render frames in GPU memory.
    output_hw = None
    commands = []
    for name, start, end, trajectory_type in specs:
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        poses = plan_spherical_segment(prediction.extrinsics[start],
                                       prediction.extrinsics[end], pivot, args.frames)
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
        command += ['--height', str(generation_hw[0]), '--width', str(generation_hw[1])]
        commands.append(command)
    del prediction, model, rgb, depth, valid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for command in commands:
        if video_cache is None:
            subprocess.run(command, check=True)
        else:
            from generate_with_videox_fun import generate, parse_args as video_parse_args
            generate(video_parse_args(command[2:]), cache=video_cache)
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.render_only:
        print('Geometry previews and trajectories saved to', args.output_dir)
        return
    join_original_clips(
        [args.output_dir / name / 'generated.mp4' for name, *_ in specs],
        args.output_dir / 'demo.mp4',
    )
    print('Demo saved to', args.output_dir / 'demo.mp4')


if __name__ == '__main__':
    main()
