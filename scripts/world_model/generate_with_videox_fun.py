#!/usr/bin/env python3
"""Generate a high-quality view-expansion video from DA3 geometry guidance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VENDORED_VIDEOX_FUN_ROOT = REPO_ROOT / "third_party/VideoX-Fun"
DEFAULT_WORLD_ROOT = Path("/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev")
DEFAULT_MODEL_PATH = Path(
    "/home/z00566689/dev/mnt/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control"
)
DEFAULT_LORA_PATH = DEFAULT_MODEL_PATH / "12000_lora.safetensors"
REQUIRED_SPATIAL_MULTIPLE = 32
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋，残缺，变形，扭曲，杂乱的背景，闪烁，颜色漂移"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    geometry_dir = DEFAULT_WORLD_ROOT / "output/task1_orbit_roundtrip_5"
    parser.add_argument("--image", type=Path, default=DEFAULT_WORLD_ROOT / "image.jpg")
    parser.add_argument(
        "--end-image",
        type=Path,
        help="Optional known final frame. Defaults to --image unless --no-end-image is set.",
    )
    parser.add_argument(
        "--no-end-image",
        action="store_true",
        help="Condition only the first frame; do not constrain the generated final frame.",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help=(
            "Persistent appearance/identity reference passed through the transformer's "
            "ref_conv branch. Defaults to --image."
        ),
    )
    parser.add_argument(
        "--disable-reference-image",
        action="store_true",
        help="Disable ref_conv conditioning for an ablation run.",
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_WORLD_ROOT / "prompt.txt")
    parser.add_argument("--control-video", type=Path, default=geometry_dir / "gs_render.mp4")
    parser.add_argument("--control-mask", type=Path, default=geometry_dir / "mask.mp4")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "configs/wan_civitai_5b.yaml",
    )
    parser.add_argument("--output", type=Path, default=geometry_dir / "generated.mp4")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("image", "prompt", "control_video", "control_mask", "lora_path", "config"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.end_image is not None and not args.end_image.is_file():
        raise FileNotFoundError(f"--end-image does not exist: {args.end_image}")
    if args.no_end_image and args.end_image is not None:
        raise ValueError("--no-end-image and --end-image cannot be used together")
    if args.reference_image is not None and not args.reference_image.is_file():
        raise FileNotFoundError(
            f"--reference-image does not exist: {args.reference_image}"
        )
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")
    if (args.frames - 1) % 4:
        raise ValueError("--frames must satisfy (frames - 1) % 4 == 0; use 81 by default")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.height is not None and (
        args.height % REQUIRED_SPATIAL_MULTIPLE
        or args.width % REQUIRED_SPATIAL_MULTIPLE
    ):
        raise ValueError(
            "--height and --width must both be divisible by 32. Wan2.2-Fun-5B "
            "compresses space by 16 and then applies 2x2 transformer patches."
        )


def configure_videox_fun_import() -> None:
    package_dir = VENDORED_VIDEOX_FUN_ROOT / "videox_fun"
    if not package_dir.is_dir():
        raise ModuleNotFoundError(
            f"Vendored VideoX-Fun package is missing: {package_dir}. "
            "Pull the complete world-model-dev branch."
        )
    sys.path.insert(0, str(VENDORED_VIDEOX_FUN_ROOT))


def choose_output_size(image_path: Path) -> tuple[int, int]:
    """Return (height, width) using roughly-720p, model-aligned buckets."""
    with Image.open(image_path) as image:
        width, height = image.size
    ratio = min(width, height) / max(width, height)
    long_side = 960 if ratio > 0.70 else 1056 if ratio > 0.65 else 1280
    # 720 / 16 = 45 produces an odd latent side. The transformer's 2x2
    # patchification then truncates it to 44, so use the nearest 32-aligned side.
    short_side = 736
    if width <= height:
        return long_side, short_side
    return short_side, long_side


def video_info(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, frames


def load_missing_region_mask(
    path: Path, frames: int, height: int, width: int
) -> torch.Tensor:
    """Read DA3's white-valid/black-missing video as a 1=missing tensor."""
    cap = cv2.VideoCapture(str(path))
    masks: list[np.ndarray] = []
    while len(masks) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append((gray < 128).astype(np.float32))
    cap.release()
    if len(masks) != frames:
        raise ValueError(f"Control mask has {len(masks)} readable frames; expected {frames}")
    return torch.from_numpy(np.stack(masks)).unsqueeze(0).unsqueeze(0)


def expand_and_load_patch_embedding(transformer, state_dict: dict[str, torch.Tensor]) -> dict:
    patch_state = {
        key.removeprefix("patch_embedding."): value
        for key, value in state_dict.items()
        if key.startswith("patch_embedding.")
    }
    if "weight" not in patch_state:
        return state_dict

    old = transformer.patch_embedding
    target_in_channels = int(patch_state["weight"].shape[1])
    if old.in_channels != target_in_channels:
        expanded = torch.nn.Conv3d(
            target_in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        ).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            copied_channels = min(old.in_channels, target_in_channels)
            expanded.weight.zero_()
            expanded.weight[:, :copied_channels].copy_(old.weight[:, :copied_channels])
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        transformer.patch_embedding = expanded
        transformer.in_dim = target_in_channels
        print(f"Expanded patch_embedding: {old.in_channels} -> {target_in_channels} channels")

    missing, unexpected = transformer.patch_embedding.load_state_dict(patch_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"patch_embedding load mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {key: value for key, value in state_dict.items() if not key.startswith("patch_embedding.")}


def generate(args: argparse.Namespace, cache: dict | None = None) -> None:
    validate_args(args)
    configure_videox_fun_import()

    from videox_fun.models import (
        AutoencoderKLWan,
        AutoencoderKLWan3_8,
        Wan2_2Transformer3DModel,
        WanT5EncoderModel,
    )
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils.lora_utils import merge_lora
    from videox_fun.utils.utils import (
        filter_kwargs,
        get_image_latent,
        get_image_to_video_latent,
        get_video_to_video_latent,
        save_videos_grid,
    )

    import inspect

    if "control_mask" not in inspect.signature(Wan2_2FunControlPipeline.__call__).parameters:
        raise RuntimeError(
            "This VideoX-Fun checkout predates mask-aware Control inference. "
            "Use main commit 18b9b78 or newer."
        )

    config = OmegaConf.load(args.config)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    height, width = (
        (args.height, args.width)
        if args.height is not None
        else choose_output_size(args.image)
    )
    control_width, control_height, control_frames = video_info(args.control_video)
    _, _, mask_frames = video_info(args.control_mask)
    if control_frames < args.frames or mask_frames < args.frames:
        raise ValueError(
            f"Need {args.frames} guidance frames, got control={control_frames}, mask={mask_frames}"
        )
    print(
        f"Guidance: {control_width}x{control_height}, {args.frames} frames; "
        f"generation: {width}x{height}"
    )

    cache_key = (str(args.model_path.resolve()), str(args.lora_path.resolve()),
                 args.lora_weight, args.device, args.dtype, str(args.config.resolve()))
    if cache is not None and cache.get('key') == cache_key:
        pipeline, transformer, use_control_mask = cache['loaded']
    else:
        pipeline, transformer, use_control_mask = _load_pipeline(args, config, dtype)
        if cache is not None:
            cache.clear()
            cache.update(key=cache_key, loaded=(pipeline, transformer, use_control_mask))

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    start_image = Image.open(args.image).convert("RGB")
    end_image_path = (
        None if args.no_end_image else (args.end_image if args.end_image is not None else args.image)
    )
    end_image = None if end_image_path is None else Image.open(end_image_path).convert("RGB")
    inpaint_video, inpaint_mask, _ = get_image_to_video_latent(
        [start_image], None if end_image is None else [end_image],
        video_length=args.frames, sample_size=[height, width],
    )
    control_video, _, _, _ = get_video_to_video_latent(
        str(args.control_video), video_length=args.frames,
        sample_size=[height, width], fps=args.fps, ref_image=None,
    )
    control_mask = (load_missing_region_mask(args.control_mask, args.frames, height, width)
                    if use_control_mask else None)
    reference_image_path = args.reference_image if args.reference_image is not None else args.image
    reference_image = None
    if not args.disable_reference_image:
        if not transformer.config.get("add_ref_conv", False) or transformer.ref_conv is None:
            raise RuntimeError("Reference conditioning requires add_ref_conv; use --disable-reference-image for ablation")
        reference_image = get_image_latent(str(reference_image_path), sample_size=[height, width])
        print(f"Reference conditioning: {reference_image_path}")
    else:
        print("Reference conditioning disabled")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    boundary = config.transformer_additional_kwargs.get("boundary", 0.875)
    with torch.inference_mode():
        sample = pipeline(
            prompt, num_frames=args.frames, negative_prompt=args.negative_prompt,
            height=height, width=width, generator=generator,
            guidance_scale=args.guidance_scale, num_inference_steps=args.steps,
            video=inpaint_video, mask_video=inpaint_mask, control_video=control_video,
            control_mask=control_mask, ref_image=reference_image,
            boundary=boundary, shift=args.shift,
        ).videos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(sample.cpu(), str(args.output), fps=args.fps)
    metadata = {
        "image": str(args.image), "end_image": None if end_image_path is None else str(end_image_path),
        "reference_image": None if args.disable_reference_image else str(reference_image_path),
        "reference_branch": "disabled" if args.disable_reference_image else "vae_latent_ref_conv",
        "prompt_file": str(args.prompt), "control_video": str(args.control_video),
        "control_mask": str(args.control_mask), "model_path": str(args.model_path),
        "lora_path": str(args.lora_path), "videox_fun_root": str(VENDORED_VIDEOX_FUN_ROOT),
        "videox_fun_upstream_commit": "18b9b78d85b69edf483e9eeebaa057b39716aba1",
        "output": str(args.output), "frames": args.frames, "resolution": [height, width],
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "lora_weight": args.lora_weight, "seed": args.seed,
        "endpoint_constraint": "known start image only" if end_image_path is None else "known start and end images",
        "control_mask_used": use_control_mask,
        "control_mask_convention": "1 = missing DA3 geometry, 0 = valid geometry",
        "memory_mode": "model_full_load",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Generated video: {args.output}")


def _load_pipeline(args, config, dtype):
    from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                                   Wan2_2Transformer3DModel, WanT5EncoderModel)
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils.lora_utils import merge_lora
    from videox_fun.utils.utils import filter_kwargs
    transformer_subpath = config.transformer_additional_kwargs.get(
        "transformer_low_noise_model_subpath", "transformer"
    )
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        str(args.model_path / transformer_subpath),
        transformer_additional_kwargs=OmegaConf.to_container(
            config.transformer_additional_kwargs
        ),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    lora_state = load_file(str(args.lora_path), device="cpu")
    base_control_channels = int(transformer.patch_embedding.in_channels)
    patch_weight = lora_state.get("patch_embedding.weight")
    checkpoint_channels = (
        int(patch_weight.shape[1]) if patch_weight is not None else base_control_channels
    )
    if checkpoint_channels not in (base_control_channels, base_control_channels + 4):
        raise RuntimeError(
            f"Unsupported checkpoint patch channels: base={base_control_channels}, "
            f"checkpoint={checkpoint_channels}; expected base or base+4"
        )
    use_control_mask = checkpoint_channels == base_control_channels + 4
    print(f"Control-mask conditioning: {'enabled' if use_control_mask else 'disabled'}")
    lora_state = expand_and_load_patch_embedding(transformer, lora_state)

    vae_class = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config.vae_kwargs.get("vae_type", "AutoencoderKLWan")]
    vae = vae_class.from_pretrained(
        str(args.model_path / config.vae_kwargs.get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config.vae_kwargs),
    ).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path / config.text_encoder_kwargs.get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(args.model_path / config.text_encoder_kwargs.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config.text_encoder_kwargs),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config.scheduler_kwargs),
        )
    )
    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    pipeline = merge_lora(
        pipeline,
        str(args.lora_path),
        args.lora_weight,
        device=args.device,
        dtype=dtype,
        state_dict=lora_state,
    )
    pipeline.to(device=args.device)
    print("VideoX-Fun loaded in model_full_load mode")
    return pipeline, transformer, use_control_mask


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
