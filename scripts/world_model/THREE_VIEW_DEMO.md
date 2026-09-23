# Three-view cinematic demo

All known paths and inference settings are in [`configs/three_view_demo.json`](configs/three_view_demo.json). The prompt path uses the earlier `WorldModel-dev/prompt.txt` location; change it in the config if this demo uses another prompt. Run from the repository root:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 python scripts/world_model/run_three_view_demo.py
```

Start with `--render-only` to review `gs_render.mp4`, `mask.mp4`, and `turning_point.png` in each segment folder. Black mask pixels have no Gaussian coverage. The script writes camera poses and `metadata.json` for diagnostics. Tune `--push-fraction` and `--pull-fraction` (default 0.18 of the left camera to target distance), `--pivot-depth-scale`, or `--center-crop` if the inferred target center is wrong. A push under 0.8 is enforced to avoid crossing the estimated surface.

DA3 camera estimates across three images must be consistent for a meaningful trajectory. The endpoint photographs constrain the video model, but exact pixel equality at the ends is not guaranteed by the model. The last output is `demo.mp4`; the two generated clips remain available for inspection. Set `--height` and `--width` to multiples of 32 and match the aspect ratio of the input photos.

The 12000-step LoRA is assumed to predate control-mask training. The inference adapter inspects checkpoint patch-input channels and skips `control_mask` for the original layout; the DA3 mask preview is still saved for diagnostics. If the provided checkpoint has the expanded 4-channel mask layout, the adapter enables it automatically.

With `--images-dir`, the script accepts exactly three JPG/PNG/WebP images and maps sorted filenames to left, middle, right. Check the printed mapping before rendering. If filenames do not sort in camera order, pass all three `--left`, `--middle`, `--right` paths explicitly.

To use a different config, pass `--config /path/to/config.json`. Command-line options override config values. Set `left`, `middle`, and `right` to explicit image paths in the config if filename sorting does not match the camera order.
