# DA3 + VideoX-Fun world-model development

## Task 1: conservative single-image view expansion

The first stage reconstructs DA3 Gaussians from one image, estimates a robust
orbit pivot from the center of the predicted geometry, and renders an 81-frame
round trip. The default trajectory is `0° -> 10° -> 0°`: frame 40 is the peak
novel view, while frames 0 and 80 use the same known input camera. This limits
the effect of an inaccurate single-view pivot and gives VideoX-Fun known first
and last frames. The prompt does not affect DA3 geometry prediction.

Install the Gaussian rasterizer in the active DA3 environment first. The
commands below use `PYTHONPATH` to expose this checkout's `src` directory, so
the repository itself does not need to be installed into the environment:

```bash
pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
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
  --output-dir /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output/task1_orbit_roundtrip_10 \
  --model-name da3nested-giant-large \
  --device cuda:0 \
  --frames 81 \
  --degrees 10 \
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
- Use `--degrees 5` for a more conservative first expansion. Increase it only
  after additional remembered views make geometry and pose more stable.
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
   `geometry/mask.mp4` on the 10-degree round trip.
2. VideoX-Fun uses the same input image as both endpoint constraints, the GS
   video as control, and the mask's black pixels as missing geometry.
3. The mask-aware LoRA's `patch_embedding.*` weights are loaded separately
   after expanding the transformer Conv3d by four channels. The remaining LoRA
   weights are merged at weight 1.0.

The wrapper imports the `videox_fun` package from an existing checkout. Use the
VideoX-Fun `main` branch at commit `18b9b78` or newer because it contains the
control-mask device fix. Pass that checkout with `--videox-fun-root`, set
`VIDEOX_FUN_ROOT`, or place it next to this repository as `../VideoX-Fun`.

Run the full step with one GPU:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 \
python scripts/world_model/run_world_model_step.py \
  --videox-fun-root /path/to/VideoX-Fun
```

The model paths default to:

```text
/home/z00566689/dev/mnt/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control
/home/z00566689/dev/mnt/SingleRecon/Cloud_Models/Wan2.2-Fun-5B-Control/33000_lora.safetensors
```

Generation defaults are 81 frames, 8 inference steps, CFG 6.0, LoRA weight
1.0, and full GPU loading. Output resolution is selected from the established
1088-short-side aspect-ratio buckets. Override both dimensions together when
needed, for example `--height 1088 --width 1440`.

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
