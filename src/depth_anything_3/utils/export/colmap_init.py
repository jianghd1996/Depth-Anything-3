"""Export DA3 predictions as COLMAP initialization (no feature tracks / BA)."""
from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors, _filter_and_downsample


def save_sparserecon(prediction, image_paths, output_dir, *,
                     conf_percentile=40.0, max_points=300000):
    """Save input-view poses and depth-derived colored points to COLMAP.

    No novel views, 2D observations, or tracks are generated. Suitable as
    initialization for Gaussian reconstruction, not bundle adjustment.
    """
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError("DA3 did not predict camera poses/intrinsics")
    if prediction.depth is None or prediction.processed_images is None:
        raise ValueError("DA3 depth and processed images are required for points")
    n, h, w = prediction.depth.shape
    if len(image_paths) != n:
        raise ValueError(f"Expected {n} image paths, got {len(image_paths)}")
    conf = prediction.conf
    threshold = float(np.percentile(conf, conf_percentile)) if conf is not None else 0.0
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth, prediction.intrinsics, prediction.extrinsics,
        prediction.processed_images, conf, threshold)
    points, colors = _filter_and_downsample(points, colors, max_points)
    recon = pycolmap.Reconstruction()
    for xyz, rgb in zip(points, colors):
        recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb.astype(np.uint8))
    for idx, path in enumerate(image_paths):
        with Image.open(path) as im:
            orig_w, orig_h = im.size
        k = np.array(prediction.intrinsics[idx], dtype=np.float64, copy=True)
        k[0, :] *= orig_w / w
        k[1, :] *= orig_h / h
        cam = pycolmap.Camera(
            model="PINHOLE", width=orig_w, height=orig_h,
            params=[k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
            camera_id=idx + 1)
        recon.add_camera(cam)
        ext = np.asarray(prediction.extrinsics[idx], dtype=np.float64)
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(ext[:3, :3]), ext[:3, 3])
        image = pycolmap.Image(
            id=idx + 1, name=Path(path).name,
            camera_id=cam.camera_id, cam_from_world=pose)
        image.registered = True
        recon.add_image(image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recon.write(str(output_dir))
    print(f"[DA3 COLMAP] {n} images, {len(points)} points -> {output_dir}", flush=True)
    return recon
