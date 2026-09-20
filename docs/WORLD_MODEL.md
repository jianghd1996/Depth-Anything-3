# DA3 + VideoX-Fun world-model development

## Task 1: conservative single-image view expansion

The first stage reconstructs DA3 Gaussians from one image, estimates a robust
orbit pivot from the center of the predicted geometry, and renders an 81-frame
round trip. The default trajectory is `0° -> 5° -> 0°`: frame 40 is the peak
novel view, while frames 0 and 80 use the same known input camera. This limits
the effect of an inaccurate single-view pivot and gives VideoX-Fun known first
and last frames. The prompt does not affect DA3 geometry prediction.

Install the Gaussian rasterizer in the active DA3 environment first. The
commands below use `PYTHONPATH` to expose this checkout's `src` directory, so
the repository itself does not need to be installed into the environment:

```bash
pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

# Required by DA3 to rotate Gaussian spherical-harmonic coefficients.
python -m pip install e3nn
```

The script defaults to the requested paths, so the basic run is:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 \
python scripts/world_model/render_single_image_orbit.py
```

Equivalent explicit invocation:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 \
python scripts/world_model/render_single_image_orbit.py \
  --image /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/image.jpg \
  --prompt /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/prompt.txt \
  --weights /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/DA3.pt \
  --output-dir /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output/task1_orbit_roundtrip_5 \
  --model-name da3nested-giant-large \
  --device cuda:0 \
  --frames 81 \
  --degrees 5 \
  --direction right \
  --trajectory-mode round-trip
```

Outputs:

- `gs_render.mp4`: 81-frame Gaussian RGB guidance video.
- `mask.mp4`: white is valid rendered geometry; black is missing geometry.
- `camera_trajectory.npz`: world-to-camera matrices, intrinsics, pivot, and depth.
- `gs_ply/0000.ply`: reconstructed Gaussian scene.
- `frame_000.png`, `frame_080.png`, `mask_080.png`: quick endpoint checks.
- `metadata.json`: paths, prompt, orbit settings, and mask convention.

Useful controls:

- Use `--direction left` for the opposite side of the object.
- Start with `--degrees 5` and increase it gradually only after the geometry is
  stable.
- If the estimated target lies too deep, try `--pivot-depth-scale 0.9`, then
  `0.8`. Keep each test in a separate `--output-dir` for direct comparison.
- Use `--trajectory-mode one-way` only when a known final view is available or
  endpoint image conditioning is not required.
- Decrease `--pivot-depth-scale` slightly (for example `0.9`) if the estimated
  orbit center is visibly behind the intended object.
- Set `--output-height` and `--output-width` together to render a chosen size.
- Reduce `--chunk-size` if Gaussian rendering runs out of GPU memory.
- The default `--model-name da3nested-giant-large` matches the provided
  `DA3.pt` training checkpoint. Use `--model-name da3-giant` only with a
  checkpoint saved from that single-branch architecture.

## Task 2: DA3 geometry followed by VideoX-Fun

The integrated runner performs one complete geometry-and-generation step:

1. DA3 reconstructs Gaussians and renders `geometry/gs_render.mp4` plus
   `geometry/mask.mp4` on the configurable small-angle round trip.
2. VideoX-Fun uses the same input image as both endpoint constraints, the GS
   video as control, and the mask's black pixels as missing geometry. The same
   real input image is also VAE-encoded and injected through the transformer's
   `ref_conv` branch as a persistent appearance and identity reference.
3. The mask-aware LoRA's `patch_embedding.*` weights are loaded separately
   after expanding the transformer Conv3d by four channels. The remaining LoRA
   weights are merged at weight 1.0.

The required VideoX-Fun inference subset is vendored under
`third_party/VideoX-Fun` at upstream commit `18b9b78`. It includes the
control-mask device fix and is imported directly by the runner; no separate
VideoX-Fun checkout, installation, or source-path argument is required.

Run the full step with one GPU:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 \
python scripts/world_model/run_world_model_step.py
```

The model paths default to:

```text
/home/z00566689/dev/mnt/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control
/home/z00566689/dev/mnt/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control/33000_lora.safetensors
```

Generation defaults are 81 frames, 8 inference steps, CFG 6.0, LoRA weight
1.0, and full GPU loading. Output resolution uses approximately 720p,
32-aligned short-side buckets (`960x736`, `1056x736`, or `1280x736` for
landscape inputs, transposed for portrait). Exact 720 is not valid here because
the 16x VAE compression produces 45 latent pixels, which cannot be preserved by
the transformer's 2x2 spatial patches. Override both dimensions together when
needed; each dimension must be divisible by 32.

Outputs under `output/world_model_step`:

- `geometry/gs_render.mp4` and `geometry/mask.mp4`: DA3 guidance.
- `geometry/camera_trajectory.npz`: round-trip cameras and rendered depth.
- `generated.mp4`: VideoX-Fun result.
- `generated.json`: exact generation parameters and model paths.
- `step_manifest.json`: stage linkage, endpoint frames, and peak-view frame.

To rerun only VideoX-Fun after geometry succeeds, add `--skip-geometry`. To
selectively add the maximum-angle generated frame to a memory directory, add:

```bash
--save-peak-frame-to-memory /path/to/memory
```

## Task 3: iterative 360-degree orbit

`run_iterative_orbit.py` expands the memory in 10-degree increments and is
safe to resume. With the default configuration it performs 36 steps:

- Steps 1-35 render `0° -> 10° -> 0°` relative to the latest remembered view.
  VideoX-Fun receives that known view as both endpoint images, and generated
  frame 40 is stored as the next 10-degree memory view.
- Step 36 renders a one-way 350° -> 360° closure. Its start constraint is the
  latest 350-degree memory view and its end constraint is the original input
  image, so the loop is explicitly closed rather than extrapolated blindly.
- DA3 jointly reconstructs from remembered images on every step. To bound Giant
  model memory, at most eight views are sampled uniformly by default, always
  including the original and latest views. Set `--max-da3-views` to change it.
- VideoX-Fun always receives the original real input through `ref_image`, even
  when a generated view is used as the current segment's endpoint image. This
  prevents the reference identity and color from becoming fully autoregressive.

Start or resume the complete orbit from the repository root:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 \
python scripts/world_model/run_iterative_orbit.py
```

For an initial three-step test before committing to all 36 generations:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 \
python scripts/world_model/run_iterative_orbit.py --max-steps-this-run 3
```

Run the same command again without changing `--output-dir`, direction, or step
angle to resume from `orbit_state.json`. Each step has its own geometry and
generated video directory. On completion, outbound halves are assembled into
`orbit_360.mp4`; the individual round-trip videos remain available for quality
inspection.
