"""Single-image orbit planning and Gaussian rendering utilities."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

from depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def estimate_orbit_pivot(
    depth: np.ndarray | torch.Tensor,
    intrinsics: np.ndarray | torch.Tensor,
    extrinsics: np.ndarray | torch.Tensor,
    confidence: np.ndarray | torch.Tensor | None = None,
    sky: np.ndarray | torch.Tensor | None = None,
    center_crop: float = 0.4,
    depth_scale: float = 1.0,
) -> torch.Tensor:
    """Estimate a robust world-space orbit pivot from a centered image region.

    DA3 cannot infer the hidden center of a single-view object.  The median 3D
    point in the center crop is therefore used as a stable, deterministic pivot.
    ``depth_scale`` can move the pivot farther along the camera ray when the
    visible surface is known to lie in front of the object's actual center.
    """

    if not 0.0 < center_crop <= 1.0:
        raise ValueError(f"center_crop must be in (0, 1], got {center_crop}")
    if depth_scale <= 0.0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")

    depth_t = torch.as_tensor(depth, dtype=torch.float32)
    intrinsic_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=depth_t.device)
    extrinsic_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=depth_t.device)
    if depth_t.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {tuple(depth_t.shape)}")

    height, width = depth_t.shape
    crop_h = max(1, round(height * center_crop))
    crop_w = max(1, round(width * center_crop))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    y1, x1 = y0 + crop_h, x0 + crop_w

    valid = torch.isfinite(depth_t) & (depth_t > 0)
    crop_mask = torch.zeros_like(valid)
    crop_mask[y0:y1, x0:x1] = True
    valid &= crop_mask

    if sky is not None:
        valid &= ~torch.as_tensor(sky, dtype=torch.bool, device=valid.device)

    if confidence is not None:
        confidence_t = torch.as_tensor(confidence, dtype=torch.float32, device=valid.device)
        finite_conf = confidence_t[valid & torch.isfinite(confidence_t)]
        if finite_conf.numel() >= 16:
            valid &= confidence_t >= torch.quantile(finite_conf, 0.5)

    if valid.sum() < 16:
        valid = torch.isfinite(depth_t) & (depth_t > 0)
        if sky is not None:
            valid &= ~torch.as_tensor(sky, dtype=torch.bool, device=valid.device)
    if valid.sum() == 0:
        raise ValueError("DA3 produced no finite positive depth from which to estimate an orbit pivot")

    ys, xs = torch.where(valid)
    zs = depth_t[valid]
    fx, fy = intrinsic_t[0, 0], intrinsic_t[1, 1]
    cx, cy = intrinsic_t[0, 2], intrinsic_t[1, 2]
    points_camera = torch.stack(
        [
            (xs.float() - cx) / fx * zs,
            (ys.float() - cy) / fy * zs,
            zs,
        ],
        dim=-1,
    )
    pivot_camera = points_camera.median(dim=0).values
    pivot_camera[2] *= depth_scale

    c2w = affine_inverse(as_homogeneous(extrinsic_t))
    pivot_world = c2w[:3, :3] @ pivot_camera + c2w[:3, 3]
    return pivot_world


def generate_orbit_trajectory(
    start_extrinsic: np.ndarray | torch.Tensor,
    pivot_world: np.ndarray | torch.Tensor,
    num_frames: int = 81,
    degrees: float = 90.0,
    direction: Literal["left", "right"] = "right",
    ease: bool = True,
) -> torch.Tensor:
    """Generate an OpenCV world-to-camera orbit that preserves initial framing.

    The camera center and its orientation are rotated by the same world-space
    transform around the camera's initial up axis.  Consequently frame zero is
    exactly the DA3 input camera and there is no initial look-at snap.
    """

    if num_frames < 2:
        raise ValueError(f"num_frames must be at least 2, got {num_frames}")
    if direction not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    start_w2c = torch.as_tensor(start_extrinsic, dtype=torch.float32)
    start_c2w = affine_inverse(as_homogeneous(start_w2c))
    pivot = torch.as_tensor(pivot_world, dtype=torch.float32, device=start_c2w.device)

    progress = torch.linspace(0.0, 1.0, num_frames, device=start_c2w.device)
    if ease:
        progress = 0.5 - 0.5 * torch.cos(progress * math.pi)
    sign = 1.0 if direction == "right" else -1.0
    angles = progress * math.radians(degrees) * sign

    # OpenCV camera Y points down, hence negative Y is the camera/world up axis.
    axis = _normalize(-start_c2w[:3, 1])
    kx, ky, kz = axis
    zeros = torch.zeros((), dtype=start_c2w.dtype, device=start_c2w.device)
    skew = torch.stack(
        [
            torch.stack([zeros, -kz, ky]),
            torch.stack([kz, zeros, -kx]),
            torch.stack([-ky, kx, zeros]),
        ]
    )
    identity = torch.eye(3, dtype=start_c2w.dtype, device=start_c2w.device)
    outer = axis[:, None] @ axis[None, :]
    rotations = (
        torch.cos(angles)[:, None, None] * identity
        + (1.0 - torch.cos(angles))[:, None, None] * outer
        + torch.sin(angles)[:, None, None] * skew
    )

    camera_center = start_c2w[:3, 3]
    offsets = torch.einsum("tij,j->ti", rotations, camera_center - pivot)
    c2ws = torch.eye(4, dtype=start_c2w.dtype, device=start_c2w.device).repeat(
        num_frames, 1, 1
    )
    c2ws[:, :3, :3] = rotations @ start_c2w[:3, :3]
    c2ws[:, :3, 3] = pivot + offsets
    return affine_inverse(c2ws)


@torch.inference_mode()
def render_orbit(
    prediction: Prediction,
    target_extrinsics: torch.Tensor,
    output_hw: tuple[int, int] | None = None,
    chunk_size: int = 4,
    alpha_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render RGB, depth and a binary valid-geometry mask for an orbit."""

    if prediction.gaussians is None:
        raise ValueError("Prediction has no Gaussians; call inference(..., infer_gs=True)")
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must be in [0, 1]")

    gaussians = prediction.gaussians
    device = gaussians.means.device
    target_extrinsics = target_extrinsics.to(device=device, dtype=gaussians.means.dtype)
    # Nested DA3 reports metric cameras/depth after inference, while its Gaussians
    # remain in the any-view branch's pre-alignment coordinate scale.
    if prediction.is_metric and prediction.scale_factor is not None:
        target_extrinsics = target_extrinsics.clone()
        target_extrinsics[:, :3, 3] /= prediction.scale_factor
    frame_count = target_extrinsics.shape[0]
    height, width = output_hw or tuple(prediction.depth.shape[-2:])

    intrinsic = torch.as_tensor(
        prediction.intrinsics[0], dtype=gaussians.means.dtype, device=device
    )
    input_height, input_width = prediction.depth.shape[-2:]
    intrinsic = intrinsic.clone()
    intrinsic[0] *= width / input_width
    intrinsic[1] *= height / input_height
    intrinsic_norm = intrinsic.clone()
    intrinsic_norm[0] /= width
    intrinsic_norm[1] /= height

    colors, depths, alphas = [], [], []
    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        view_count = end - start
        color, depth, alpha = render_3dgs(
            extrinsics=target_extrinsics[start:end],
            intrinsics=intrinsic_norm.unsqueeze(0).expand(view_count, -1, -1),
            image_shape=(height, width),
            gaussian=gaussians,
            num_view=view_count,
            color_mode="RGB+ED",
            use_sh=True,
            return_alpha=True,
        )
        colors.append(color)
        depths.append(depth)
        alphas.append(alpha)

    rgb = torch.cat(colors, dim=0)
    depth = torch.cat(depths, dim=0)
    if prediction.is_metric and prediction.scale_factor is not None:
        depth = depth * prediction.scale_factor
    valid_mask = torch.cat(alphas, dim=0) >= alpha_threshold
    return rgb, depth, valid_mask
