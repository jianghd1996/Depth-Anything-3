"""Export DA3 predictions as COLMAP initialization (no feature tracks / BA)."""
from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors, _filter_and_downsample


def save_sparserecon(prediction, image_paths, output_dir, *,
                     conf_percentile=40.0, max_points=1000000):
    """Save input-view poses and Gaussian-head colored points to COLMAP.

    No novel views, 2D observations, or tracks are generated. Suitable as
    initialization for Gaussian reconstruction, not bundle adjustment.
    """
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError("DA3 did not predict camera poses/intrinsics")
    if prediction.depth is None or prediction.processed_images is None:
        raise ValueError("DA3 depth and processed images are required for points")
    sf = prediction.scale_factor
    print(
        f"[DA3 COLMAP SCALE] is_metric={prediction.is_metric} "
        f"scale_factor={sf} "
        f"inverse_scale={1.0 / float(sf) if sf is not None and float(sf) != 0 else None} "
        f"export_pose_scale=1.0 (unchanged) "
        f"depth_median={float(np.nanmedian(prediction.depth)):.9g}",
        flush=True,
    )
    n, h, w = prediction.depth.shape
    if len(image_paths) != n:
        raise ValueError(f"Expected {n} image paths, got {len(image_paths)}")
    # Legacy initializer: use Gaussian-head means, not depth backprojection.
    if prediction.gaussians is None:
        raise ValueError("Gaussian Head output required for COLMAP initialization")
    means = prediction.gaussians.means
    if hasattr(means, "detach"):
        means = means.detach().cpu().numpy()
    means = np.asarray(means).reshape(n, h, w, 3)
    conf = np.asarray(prediction.conf)
    threshold = float(np.percentile(conf, 20))
    mask = conf > threshold
    points = means[mask].astype(np.float32, copy=False)
    colors = np.asarray(prediction.processed_images)[mask].astype(np.uint8, copy=False)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) > max_points:
        idxs = np.random.choice(len(points), size=max_points, replace=False)
        points, colors = points[idxs], colors[idxs]
    print(f"[DA3 COLMAP SOURCE] gaussian_means conf_p20={threshold:.7g} points={len(points)}", flush=True)
    recon = pycolmap.Reconstruction()
    for xyz, rgb in zip(points, colors):
        recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb.astype(np.uint8))
    for idx, path in enumerate(image_paths):
        with Image.open(path) as im:
            orig_w, orig_h = im.size
        k = np.array(prediction.intrinsics[idx], dtype=np.float64, copy=True)
        k[0, :] *= orig_w / w
        k[1, :] *= orig_h / h
        k[2, 2] = 1.0
        cam = pycolmap.Camera(
            model="PINHOLE", width=orig_w, height=orig_h,
            params=[k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
            camera_id=idx + 1)
        recon.add_camera(cam)
        ext = np.array(prediction.extrinsics[idx], dtype=np.float64, copy=True)
        if prediction.is_metric and prediction.scale_factor is not None:
            ext[:3, 3] /= float(prediction.scale_factor)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(ext[:3, :3]), ext[:3, 3])
        image = pycolmap.Image(
            id=idx + 1, name=Path(path).name,
            camera_id=cam.camera_id, cam_from_world=pose)
        image.registered = True
        recon.add_image(image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recon.write(str(output_dir))
    recon.write_text(str(output_dir))
    import trimesh
    trimesh.PointCloud(points, colors=colors).export(str(output_dir / "points3D.ply"))
    print(f"[DA3 COLMAP] {n} images, {len(points)} points -> {output_dir}", flush=True)
    return recon
