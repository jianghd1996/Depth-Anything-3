# Three-view cinematic demo

Run from the repository root. Supply three photographs of the same target in left-to-right camera order and one scene prompt. DA3 jointly estimates their cameras and Gaussian geometry. The first 81-frame segment moves from left toward middle while pushing toward the target; the second moves from middle to right while pulling away. VideoX-Fun uses the photographs as each segment's actual start/end image constraints. The duplicate middle frame is removed when joining the clips.

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 python scripts/world_model/run_three_view_demo.py \
  --left /path/to/left.jpg --middle /path/to/middle.jpg --right /path/to/right.jpg \
  --prompt /path/to/prompt.txt --weights /path/to/DA3.pt \
  --video-model /path/to/Wan2.2-Fun-5B-Control \
  --lora-path /path/to/12000_lora.safetensors \
  --output-dir output/three_view_demo --device cuda:0 \
  --height 704 --width 1280
```

Start with `--render-only` (omit the video model and LoRA options) to review `gs_render.mp4`, `mask.mp4`, and `turning_point.png` in each segment folder. Black mask pixels have no Gaussian coverage. The script writes camera poses and `metadata.json` for diagnostics. Tune `--push-fraction` and `--pull-fraction` (default 0.18 of the left camera to target distance), `--pivot-depth-scale`, or `--center-crop` if the inferred target center is wrong. A push under 0.8 is enforced to avoid crossing the estimated surface.

DA3 camera estimates across three images must be consistent for a meaningful trajectory. The endpoint photographs constrain the video model, but exact pixel equality at the ends is not guaranteed by the model. The last output is `demo.mp4`; the two generated clips remain available for inspection. Set `--height` and `--width` to multiples of 32 and match the aspect ratio of the input photos.

The 12000-step LoRA is assumed to predate control-mask training. The inference adapter inspects checkpoint patch-input channels and skips `control_mask` for the original layout; the DA3 mask preview is still saved for diagnostics. If the provided checkpoint has the expanded 4-channel mask layout, the adapter enables it automatically.
