# Three-view cinematic demo

All known paths and inference settings are in [`configs/three_view_demo.json`](configs/three_view_demo.json). The prompt path uses the earlier `WorldModel-dev/prompt.txt` location; change it in the config if this demo uses another prompt. Run from the repository root:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=3 python scripts/world_model/run_three_view_demo.py
```

The output contains three 81-frame geometry clips: `left_middle_push`, `middle_right_lift`, and `right_middle_pull`. The first moves left to middle with a push toward the subject; the second moves middle to right with a modest camera rise; the third moves right back to middle with a stronger pullback (0.60 of the subject distance). The pullback displacement stays in the camera horizontal plane; any remaining height change comes from the photographed right and middle camera poses. Camera positions are resampled by path length so successive frames travel nearly equal distances; push, lift, and pull amplitudes still peak midway. Each video generation call uses its segment's destination photograph as both the last-frame constraint and appearance reference. The three generated MP4s are concatenated directly with ffmpeg stream copy: no transition, color adjustment, frame removal, or re-encoding. The generated clips and final `demo.mp4` are saved separately.

Start with `--render-only` to inspect geometry and masks. Tune `push_fraction`, `lift_fraction`, `pull_fraction`, and `yaw_degrees` in the JSON config.

DA3 camera estimates across three images must be consistent for a meaningful trajectory. The endpoint photographs constrain the video model, but exact pixel equality at the ends is not guaranteed by the model. The last output is `demo.mp4`; the three generated clips remain available for inspection. Set `--height` and `--width` to multiples of 32 and match the aspect ratio of the input photos.

The 12000-step LoRA is assumed to predate control-mask training. The inference adapter inspects checkpoint patch-input channels and skips `control_mask` for the original layout; the DA3 mask preview is still saved for diagnostics. If the provided checkpoint has the expanded 4-channel mask layout, the adapter enables it automatically.

The config reads the current images in `3image` on each run: `1_` is left, `0_` is middle, and `2_` is right. Directory mode requires exactly three images with unique `0_`, `1_`, `2_` prefixes. It never guesses camera direction from image content.

To use a different config, pass `--config /path/to/config.json`. Command-line options override config values. Set `left`, `middle`, and `right` to explicit image paths in the config if filename sorting does not match the camera order.


The optional `yaw_degrees` setting defaults to 18°. Camera yaw is strongest near the middle of each segment and zero at both photographed endpoints, so the supplied views remain fixed. Large yaw can reveal geometry with weak DA3 coverage; check the per-segment `mask.mp4`.

Default generation is 1920×1088 (landscape) or 1088×1920 (portrait), selected from the first image; the config uses `guidance_scale=0`. The vendored pipeline enables classifier-free guidance only when the scale exceeds 1, so 0 runs one conditional denoising pass without the unconditional branch. DA3 uses `process_res=840`, and geometry remains at reconstruction resolution and is resized for 1080P inference.
