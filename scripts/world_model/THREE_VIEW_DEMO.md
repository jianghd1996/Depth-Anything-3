# Three-view cinematic demo

All known paths and inference settings are in [`configs/three_view_demo.json`](configs/three_view_demo.json). The prompt path uses the earlier `WorldModel-dev/prompt.txt` location; change it in the config if this demo uses another prompt. Run from the repository root:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=5 python scripts/world_model/run_three_view_demo.py
```

The output contains three 81-frame geometry clips: `middle_left_push`, `left_right_lift`, and `right_middle_pull`. The first pushes toward the subject and settles at the left photograph; the second raises the camera slightly while moving left to right; the third pulls away and settles at the middle photograph. Each video generation call uses its segment's destination photograph as both the last-frame constraint and appearance reference. At both joins, six frames crossfade and a restrained color gain tapers back to neutral over approximately 0.75 seconds. The generated clips and final `demo.mp4` are saved separately.

Start with `--render-only` to inspect geometry and masks. Tune `push_fraction`, `lift_fraction`, `pull_fraction`, and `crossfade_frames` in the JSON config.

DA3 camera estimates across three images must be consistent for a meaningful trajectory. The endpoint photographs constrain the video model, but exact pixel equality at the ends is not guaranteed by the model. The last output is `demo.mp4`; the three generated clips remain available for inspection. Set `--height` and `--width` to multiples of 32 and match the aspect ratio of the input photos.

The 12000-step LoRA is assumed to predate control-mask training. The inference adapter inspects checkpoint patch-input channels and skips `control_mask` for the original layout; the DA3 mask preview is still saved for diagnostics. If the provided checkpoint has the expanded 4-channel mask layout, the adapter enables it automatically.

The supplied three images are explicitly mapped in the config: `1_` is left, `0_` is middle, and `2_` is right. Without explicit image paths, directory mode requires unique `0_`, `1_`, `2_` prefixes and uses the same mapping. It never guesses camera direction from image content.

To use a different config, pass `--config /path/to/config.json`. Command-line options override config values. Set `left`, `middle`, and `right` to explicit image paths in the config if filename sorting does not match the camera order.

