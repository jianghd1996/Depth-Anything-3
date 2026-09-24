# Batch three-view demo

Each immediate subfolder of `/home/z00566689/dev/mnt/jiang_test/B004_物体` with an `image` directory is a case. Its image directory must contain exactly three images named with unique `0_`, `1_`, and `2_` prefixes. The mapping remains `1_` = left, `0_` = middle, `2_` = right. Cases run sequentially on the selected GPU.

From the repository root:

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=3 python scripts/world_model/run_three_view_batch.py
```

Each run creates a new, timestamped directory: `/home/z00566689/dev/mnt/jiang_dev/WorldModel-dev/output/B004_物体_YYYYmmdd_HHMMSS_microseconds/<case>/`. This keeps repeated runs and similarly named files separate. Existing directories are never overwritten. Each case folder contains `demo.mp4`, three clip folders, and a per-case prompt. Batch status and sampled parameters are in `batch_summary.json` at the run root. Failed cases are recorded and the batch continues.

The runner uses `<case>/prompt.txt` if present. Otherwise, it writes an object-preserving prompt derived from the case folder name into the output case folder. Per-case randomness is reproducible from `--batch-seed 42`: push 0.14–0.23, lift 0.07–0.13, pull 0.52–0.68, and yaw 14–24 degrees. To preview just one case's geometry, append `--render-only --limit 1`. Change `--batch-root` or `--output-root` for other datasets.

Generation uses `guidance_scale=0` and 1080P-class output per case: 1920×1088 pixels for landscape or 1088×1920 for portrait. The DA3 geometry guide is rendered at its reconstructed resolution and resized for VideoX-Fun, limiting GPU memory use.
