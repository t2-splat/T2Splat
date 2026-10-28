"""Small Open3D checks; no dataset evaluation, GPU work, or training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d

from protocol import PROTOCOL, similarity_scale, source_voxel_size, final_metric_clouds, write_metadata


def cloud(points):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return result


def ordered(points):
    points = np.asarray(points)
    return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


def transform(scale, translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = translation
    return matrix


def crop_box():
    volume = o3d.visualization.SelectionPolygonVolume()
    volume.orthogonal_axis = 'Z'
    volume.axis_min = -1.0
    volume.axis_max = 1.0
    volume.bounding_polygon = o3d.utility.Vector3dVector(
        np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]]))
    return volume


class ProtocolTests(unittest.TestCase):
    def test_scale_rotation_and_translation(self):
        angle = .47
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for scale in [.1, .8829052962333452, 1., 4.138549987926627, 9.43889284283514]:
            matrix = transform(scale, [2, -3, 7]); matrix[:3, :3] = scale * rotation
            self.assertAlmostEqual(similarity_scale(matrix), scale)
            self.assertAlmostEqual(source_voxel_size(.025, matrix) * scale, .025 / 4)

    def test_invalid_transforms_and_thresholds(self):
        invalid = [np.eye(3), transform(0), transform(-1), np.eye(4) * np.nan]
        shear = np.eye(4); shear[0, 1] = .1; invalid.append(shear)
        anisotropic = np.eye(4); anisotropic[2, 2] = 1.01; invalid.append(anisotropic)
        projective = np.eye(4); projective[3, 0] = .01; invalid.append(projective)
        for matrix in invalid:
            with self.assertRaises(ValueError): similarity_scale(matrix)
        for tau in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): source_voxel_size(tau, np.eye(4))

    def test_corresponding_scaled_surfaces_have_same_gt_samples(self):
        # Equivalent planar surfaces represented at different COLMAP scales.
        # Scale+translation preserves voxel axes, so corrected sampling must agree.
        # Random positions avoid ambiguous exact voxel-boundary points, whose
        # floating-point tie-breaking may differ under a change of coordinates.
        xy = np.random.default_rng(17).uniform([-.47, -.36], [.53, .41], size=(8000, 2))
        gt = np.column_stack([xy, np.full(len(xy), .2)])
        tau = .08
        expected = ordered(cloud(gt).voxel_down_sample(tau / 4).points)
        for scale in [.3, 1., 4.138549987926627, 9.43889284283514]:
            matrix = transform(scale, [.123, -.234, .345])
            source = (gt - matrix[:3, 3]) / scale
            sampled = cloud(source).voxel_down_sample(source_voxel_size(tau, matrix))
            sampled.transform(matrix)
            self.assertEqual(len(sampled.points), len(expected))
            expected_cloud = cloud(expected)
            np.testing.assert_allclose(sampled.compute_point_cloud_distance(expected_cloud), 0, atol=1e-12)
            np.testing.assert_allclose(expected_cloud.compute_point_cloud_distance(sampled), 0, atol=1e-12)
        # The previous fixed source-unit voxel is observably scale dependent.
        matrix = transform(9.43889284283514)
        old = cloud(gt / 9.43889284283514).voxel_down_sample(tau / 4)
        self.assertLess(len(old.points), len(expected) // 4)

    def test_both_gt_and_prediction_are_cropped_without_mutation(self):
        inside = np.array([[-.4, -.2, .2], [.1, .3, .4], [.5, -.3, -.2]])
        predicted = cloud(np.vstack([inside, [[3, 0, 0]]]))
        target = cloud(np.vstack([inside, [[-4, 0, 0], [0, 0, 2]]]))
        before_pred = np.asarray(predicted.points).copy()
        before_gt = np.asarray(target.points).copy()
        source, gt, counts = final_metric_clouds(predicted, target, crop_box(), .01)
        self.assertEqual(counts, {
            'prediction_before_final_crop': 4, 'ground_truth_before_final_crop': 5,
            'prediction_after_final_crop': 3, 'ground_truth_after_final_crop': 3,
            'prediction_final': 3, 'ground_truth_final': 3})
        np.testing.assert_allclose(ordered(source.points), ordered(inside))
        np.testing.assert_allclose(ordered(gt.points), ordered(inside))
        np.testing.assert_allclose(source.compute_point_cloud_distance(gt), 0)
        np.testing.assert_allclose(gt.compute_point_cloud_distance(source), 0)
        np.testing.assert_array_equal(predicted.points, before_pred)
        np.testing.assert_array_equal(target.points, before_gt)

    def test_final_crop_does_not_reapply_alignment(self):
        matrix = transform(3, [.2, .1, -.1])
        source = cloud([[-.1, .1, .1], [.2, -.1, .2]])
        already_aligned = copy.deepcopy(source).transform(matrix)
        result, target, _ = final_metric_clouds(already_aligned, already_aligned, crop_box(), .01)
        np.testing.assert_allclose(ordered(result.points), ordered(already_aligned.points))
        np.testing.assert_allclose(result.compute_point_cloud_distance(target), 0)

    def test_icp_composes_original_cloud_alignment_once(self):
        from virtual_scan_sampling_eval import compute_trajectory_transform_from_pc
        points = np.random.default_rng(23).uniform(-.25, .25, size=(500, 3))
        source = cloud(points)
        initial = transform(1.7, [.05, -.08, .12])
        target = copy.deepcopy(source).transform(initial)
        audit = {}
        final = compute_trajectory_transform_from_pc(
            source, initial, target, crop_box(), .01, n_icp_iterations=3, audit=audit)
        np.testing.assert_allclose(final, initial, atol=1e-10)
        np.testing.assert_array_equal(source.points, points)
        self.assertEqual([row['stage'] for row in audit['icp_stages']], ['coarse', 'fine', 'uniform'])
        self.assertAlmostEqual(audit['final_similarity_scale'], 1.7)
        aligned = copy.deepcopy(source).transform(final)
        np.testing.assert_allclose(aligned.compute_point_cloud_distance(target), 0, atol=1e-10)
        np.testing.assert_allclose(audit['final_alignment'], final)

    def test_empty_crop_fails(self):
        with self.assertRaises(ValueError):
            final_metric_clouds(cloud([[3, 0, 0]]), cloud([[0, 0, 0]]), crop_box(), .01)

    def test_metadata_is_strict_json_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            value = {'protocol': PROTOCOL, 'initial_alignment': np.eye(4).tolist(), 'status': 'complete'}
            write_metadata(directory, value)
            self.assertEqual(json.loads((Path(directory) / 'evaluation_metadata.json').read_text()), value)
            self.assertFalse((Path(directory) / 'evaluation_metadata.json.tmp').exists())
            with self.assertRaises(ValueError): write_metadata(directory, {'bad': float('nan')})
            self.assertEqual(json.loads((Path(directory) / 'evaluation_metadata.json').read_text()), value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
