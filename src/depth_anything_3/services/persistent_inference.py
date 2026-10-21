"""Persistent, profiled DA3 inference for multi-case reconstruction.

This wrapper retains the upstream DepthAnything3 inference/export semantics.
It intentionally does not implement GPU offload or quantization.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3


def _synchronize(model: DepthAnything3) -> None:
    device = model._get_model_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PersistentDA3:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        preprocess_workers: int = 8,
        profile: bool = True,
    ) -> None:
        self.device = device
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.preprocess_workers = preprocess_workers
        self.profile = profile
        start = time.perf_counter()
        self.model = DepthAnything3.from_pretrained(model_path).to(device).eval()
        self.load_seconds = time.perf_counter() - start
        print(f"[DA3 TIMING] model_load={self.load_seconds:.3f}s", flush=True)

    def infer(
        self,
        images: list[str],
        *,
        export_dir: str | None = None,
        export_format: str = "colmap",
        timing_path: str | None = None,
        **kwargs,
    ):
        timings = {}
        start = time.perf_counter()
        # Preprocessing uses the upstream image transform and coordinate conventions.
        preprocess_start = time.perf_counter()
        imgs_cpu, extrinsics, intrinsics = self.model.input_processor(
            images,
            kwargs.pop("extrinsics", None),
            kwargs.pop("intrinsics", None),
            self.process_res,
            self.process_res_method,
            num_workers=self.preprocess_workers,
        )
        timings["preprocess"] = time.perf_counter() - preprocess_start

        transfer_start = time.perf_counter()
        imgs, ex_t, in_t = self.model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        ex_t_norm = self.model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        _synchronize(self.model)
        timings["transfer"] = time.perf_counter() - transfer_start

        forward_start = time.perf_counter()
        with torch.inference_mode():
            raw = self.model.forward(
                imgs, ex_t_norm, in_t,
                [], kwargs.pop("infer_gs", False),
                kwargs.pop("use_ray_pose", False),
                kwargs.pop("ref_view_strategy", "saddle_balanced"),
            )
        _synchronize(self.model)
        timings["forward"] = time.perf_counter() - forward_start

        post_start = time.perf_counter()
        prediction = self.model._convert_to_prediction(raw)
        prediction = self.model._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction,
            kwargs.pop("align_to_input_ext_scale", True),
        )
        prediction = self.model._add_processed_images(prediction, imgs_cpu)
        timings["postprocess"] = time.perf_counter() - post_start

        if export_dir is not None:
            export_start = time.perf_counter()
            export_kwargs = dict(kwargs.pop("export_kwargs", {}))
            if "colmap" in export_format:
                export_kwargs["colmap"] = {
                    **export_kwargs.get("colmap", {}),
                    "image_paths": images,
                    "process_res_method": self.process_res_method,
                    "conf_thresh_percentile": kwargs.pop("conf_thresh_percentile", 40.0),
                }
            self.model._export_results(prediction, export_format, export_dir, **export_kwargs)
            timings["export"] = time.perf_counter() - export_start

        timings["total"] = time.perf_counter() - start
        if self.profile:
            print("[DA3 TIMING] " + " ".join(f"{k}={v:.3f}s" for k, v in timings.items()), flush=True)
        if timing_path is not None:
            path = Path(timing_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        return prediction
