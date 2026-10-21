"""Legacy-compatible DA3 runner for the reconstruction pipeline.

Exports original-view Gaussian video and no-track COLMAP initialization.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3


class run_DA3:
    """Keep the original run_DA3(model_path, device).infer(args) entry point."""

    def __init__(self, model_path, device=torch.device("cuda")):
        self.model = DepthAnything3(model_name="da3nested-giant-large")
        state_dicts = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dicts["model"], strict=False)
        del state_dicts
        self.model.to("cpu")
        self.device = torch.device(device)

    def infer(self, args):
        colmap_path = args.colmap_path
        img_path = args.data_path
        gs_render_vid_path = args.gs_render_vid_path
        extra_view = args.extra_view  # Reserved for legacy sparse reconstruction.

        images = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
        if not images:
            images = sorted(glob.glob(os.path.join(img_path, "*.png")))
        if not images:
            raise FileNotFoundError(f"No JPG or PNG images in {img_path}")

        video_path = Path(gs_render_vid_path)
        export_root = video_path.parent
        export_root.mkdir(parents=True, exist_ok=True)
        if video_path.exists() and video_path.is_dir():
            raise IsADirectoryError(f"Expected MP4 file, found directory: {video_path}")
        self.model.to(device=self.device)
        try:
            prediction = self.model.inference(
                images,
                export_dir=str(export_root),
                export_kwargs={"gs_video": {
                    "trj_mode": "original",
                    "vis_depth": None,
                    "output_name": "da3_original_views",
                }},
                infer_gs=True,
                export_format="gs_video",
                use_ray_pose=True,
                process_res=840,
            )

            rendered = export_root / "gs_video" / "da3_original_views.mp4"
            if not rendered.is_file():
                raise FileNotFoundError(f"Missing rendered GS video: {rendered}")
            rendered.replace(video_path)
            print(f"[DA3 GS VIDEO] {video_path}", flush=True)

            import numpy as np
            sf = prediction.scale_factor
            ext = np.asarray(prediction.extrinsics)
            centers = -np.einsum("nij,nj->ni", np.swapaxes(ext[:, :3, :3], 1, 2), ext[:, :3, 3])
            span = float(np.linalg.norm(np.percentile(centers, 95, axis=0) - np.percentile(centers, 5, axis=0)))
            print(
                f"[DA3 SCALE DIAG] is_metric={prediction.is_metric} "
                f"scale_factor={sf} inv_scale_factor={1.0 / float(sf) if sf is not None and float(sf) != 0 else None} "
                f"camera_center_p05_p95_diagonal={span:.9g} "
                f"depth_median={float(np.nanmedian(prediction.depth)):.9g} "
                f"gs_means_shape={tuple(prediction.gaussians.means.shape) if prediction.gaussians is not None else None}",
                flush=True,
            )
            from depth_anything_3.utils.export.colmap_init import save_sparserecon
            return save_sparserecon(prediction, images, colmap_path)
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.model.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
