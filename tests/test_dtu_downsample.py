"""The memory optimization must preserve the exact DTU evaluation point set."""
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from eval_dtu.radius_downsample import greedy_radius_mask


def original_mask(points, radius):
    engine = NearestNeighbors(n_neighbors=1, radius=radius, algorithm='kd_tree', n_jobs=1).fit(points)
    neighborhoods = engine.radius_neighbors(points, radius=radius, return_distance=False)
    mask = np.ones(len(points), dtype=np.bool_)
    for i, neighbors in enumerate(neighborhoods):
        if mask[i]:
            mask[neighbors] = False
            mask[i] = True
    return mask


class DTUDownsampleTests(unittest.TestCase):
    def test_matches_original_for_shuffled_clouds_and_dense_overlaps(self):
        rng = np.random.default_rng(41)
        clouds = [rng.normal(size=(1500, 3)), rng.normal(scale=.04, size=(900, 3)),
                  np.repeat(rng.normal(size=(30, 3)), 30, axis=0)]
        for cloud in clouds:
            rng.shuffle(cloud, axis=0)
            expected = original_mask(cloud, .2)
            for batch in (7, 113, 8192):
                with self.subTest(n=len(cloud), batch=batch):
                    actual = greedy_radius_mask(cloud, .2, workers=1, batch_size=batch)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(cloud[actual], cloud[expected])

    def test_boundary_points_chains_and_in_batch_suppression(self):
        values = np.array([0., .125, .25, .375, .5, .5, np.nextafter(.75, 0), .75, np.nextafter(.75, 1), 1.])
        cloud = np.column_stack((values, np.zeros((len(values), 2))))
        for order in (np.arange(len(cloud)), np.arange(len(cloud))[::-1]):
            for batch in (1, 2, 4, 20):
                points = cloud[order]
                np.testing.assert_array_equal(greedy_radius_mask(points, .25, workers=1, batch_size=batch), original_mask(points, .25))

    def test_queries_are_bounded_and_already_suppressed_points_are_skipped(self):
        points = np.zeros((400, 3));calls=[]
        original = NearestNeighbors.radius_neighbors
        def observed(engine, X, **kwargs):
            calls.append(len(X));return original(engine, X, **kwargs)
        with patch.object(NearestNeighbors, 'radius_neighbors', observed):
            result = greedy_radius_mask(points, .2, workers=1, batch_size=13)
        self.assertEqual(calls, [13])
        self.assertEqual(np.flatnonzero(result).tolist(), [0])

    def test_empty_and_invalid_batch(self):
        self.assertEqual(len(greedy_radius_mask(np.empty((0, 3)), .2)), 0)
        with self.assertRaises(ValueError):greedy_radius_mask(np.zeros((1, 3)), .2, batch_size=0)


if __name__ == '__main__':unittest.main()
