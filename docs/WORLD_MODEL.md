# DA3 + VideoX-Fun world-model development

## Task 1: single-image 90-degree orbit

The first stage reconstructs DA3 Gaussians from one image, estimates a robust
orbit pivot from the center of the predicted geometry, and renders 81 views over
a 90-degree horizontal orbit. The text prompt is preserved in `metadata.json`
for the later VideoX-Fun stage; it does not affect DA3 geometry prediction.

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
  --output-dir /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output/task1_orbit_90 \
  --model-name da3-giant \
  --device cuda:0 \
  --frames 81 \
  --degrees 90 \
  --direction right
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
- Increase `--pivot-depth-scale` slightly (for example `1.05`) if the visible
  surface lies noticeably in front of the desired object center.
- Set `--output-height` and `--output-width` together to render a chosen size.
- Reduce `--chunk-size` if Gaussian rendering runs out of GPU memory.
- Use `--model-name da3nested-giant-large` only when `DA3.pt` is a nested-model
  checkpoint; the default matches the standard Gaussian-capable DA3 checkpoint.
