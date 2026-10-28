"""Unit and crop rules for the isolated TNT virtual-scan protocol."""
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = 'virtual_scan_scale_corrected_v1'


def similarity_scale(transform, *, rtol=1e-6):
    """Return GT units per source unit, rejecting non-similarity transforms."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected a finite 4x4 similarity transform')
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-9):
        raise ValueError('Invalid homogeneous transform row')
    linear = transform[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(singular_values.mean())
    if scale <= 0 or not np.allclose(singular_values, scale, rtol=rtol, atol=0):
        raise ValueError('Alignment is singular or has anisotropic scale/shear')
    if np.linalg.det(linear) <= 0:
        raise ValueError('Expected a proper rotation with positive similarity scale')
    return scale


def source_voxel_size(tau, transform):
    """A tau/4 GT-space spacing, expressed in original COLMAP coordinates."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('Distance threshold must be positive and finite')
    return float(tau) / (4.0 * similarity_scale(transform))


def final_metric_clouds(predicted_gt, ground_truth, crop_volume, tau):
    """Both inputs are already in GT space; never apply alignment a second time."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('Distance threshold must be positive and finite')
    source = crop_volume.crop_point_cloud(predicted_gt)
    target = crop_volume.crop_point_cloud(ground_truth)
    counts = {
        'prediction_before_final_crop': len(predicted_gt.points),
        'ground_truth_before_final_crop': len(ground_truth.points),
        'prediction_after_final_crop': len(source.points),
        'ground_truth_after_final_crop': len(target.points),
    }
    if not len(source.points) or not len(target.points):
        raise ValueError('Final evaluation crop has an empty prediction or ground truth')
    source = source.voxel_down_sample(voxel_size=tau / 2.0)
    target = target.voxel_down_sample(voxel_size=tau / 2.0)
    counts.update(prediction_final=len(source.points), ground_truth_final=len(target.points))
    return source, target, counts


def write_metadata(out_dir, metadata):
    path = Path(out_dir) / 'evaluation_metadata.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
