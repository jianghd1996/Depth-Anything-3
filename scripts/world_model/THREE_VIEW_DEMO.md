# Three-view spherical demo

Known model paths and inference settings are in [`configs/three_view_demo.json`](configs/three_view_demo.json). From the repository root, run the outdoor dragon case on GPU 3:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=3 python scripts/world_model/run_three_view_demo.py --images-dir /home/z00566689/dev/mnt/jiang_test/B004_物体/3张_outdoor_dragon/image --output-dir /home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output/outdoor_dragon_orbit
```

The directory must contain exactly one image each with prefixes `1_` (left), `0_` (middle), and `2_` (right). Check the printed mapping when it starts.

The script jointly estimates three views with DA3, then renders two 81-frame segments: `left_middle_orbit` and `middle_right_orbit`. Camera positions follow the shortest spherical arc around the estimated target. Radius changes linearly if the photographed cameras are at different distances, and poses are sampled at nearly constant travel distance per frame. The camera tracks the same target point during each arc, with a smooth transition to the exact photographed rotations at both endpoints. Each clip uses its destination photograph as its final frame constraint and appearance reference. The clips are joined with ffmpeg stream copy, preserving every encoded frame. The final file is `demo.mp4`.

To run every case under `B004_物体`, from the repository root:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=3 python scripts/world_model/run_three_view_batch.py
```

Each batch run creates a new timestamped output directory under `WorldModel-dev/output`; case folders and `batch_summary.json` live inside it. The batch varies the random seed per case, while retaining the same centered orbit and LoRA weight `0.55`.

For a geometry preview, add `--render-only`. The DA3 and VideoX-Fun models are loaded once per process and remain on GPU during video generation and batch runs. This requires enough VRAM to hold both models alongside inference activations. There is no actual model run in this development environment; inspect `gs_render.mp4` and `mask.mp4` on the server before judging quality.

The config uses `12000_lora.safetensors` at weight `0.55`, eight inference steps, `guidance_scale=0`, 1088×1920 or 1920×1088 output (depending on orientation), and DA3 `process_res=840`. Its prompt path points to `WorldModel-dev/prompt.txt`. CLI options override config values. The checkpoint patch input channels decide whether control-mask conditioning is enabled; DA3 mask previews are always saved.
