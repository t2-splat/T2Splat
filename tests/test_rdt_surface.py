"""Topology guarantees, preservation cases, retries and input validation."""
import unittest
import numpy as np
from utils.rdt_surface import select_manifold_faces, score_surface_candidates
from utils.mesh_topology import _surface_link


def assert_surface(test, faces):
    if len(faces):
        canonical = np.sort(faces, axis=1)
        test.assertEqual(len(np.unique(canonical, axis=0)), len(faces))
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        test.assertTrue((counts <= 2).all())
        for v in np.unique(faces):
            test.assertIsNotNone(_surface_link(faces[np.any(faces == v, axis=1)], v))
        directed = {tuple(e) for e in edges}
        for e in unique[counts == 2]:
            test.assertIn(tuple(e), directed)
            test.assertIn(tuple(e[::-1]), directed)


class SurfaceTests(unittest.TestCase):
    def select(self, faces, scores=None):
        faces = np.asarray(faces, np.int32).reshape(-1, 3)
        out, ids, info = select_manifold_faces(faces, int(faces.max())+1 if faces.size else 0, scores)
        assert_surface(self, out)
        np.testing.assert_array_equal(np.sort(out, axis=1), np.sort(faces[ids], axis=1))
        return out, ids, info

    def test_tetrahedron_orientation(self):
        f = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        self.assertEqual(len(self.select(f)[0]), 4)

    def test_disk_preserved(self):
        f = [[0, i+1, (i+1) % 12+1] for i in range(12)]
        for seed in range(8):
            self.assertEqual(len(self.select(f, np.random.default_rng(seed).random(12))[0]), 12)

    def test_deferred_face_retried(self):
        f = [[0, 1, 2], [0, 3, 4], [0, 2, 3]]
        out, _, info = self.select(f)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(set(ids for ids in out.ravel())), 5)

    def test_grid_full_coverage(self):
        n = 8
        f = []
        for y in range(n-1):
            for x in range(n-1):
                a = y*n+x
                f.extend([[a, a+1, a+n], [a+1, a+n+1, a+n]])
        for seed in range(5):
            out, _, _ = self.select(f, np.random.default_rng(seed).random(len(f)))
            self.assertEqual(len(out), len(f))

    def test_branching_edge(self):
        self.assertEqual(len(self.select([[0, 1, 2], [1, 0, 3], [0, 1, 4]])[0]), 2)

    def test_bowtie(self):
        self.assertEqual(len(self.select([[0, 1, 2], [0, 3, 4]])[0]), 1)

    def test_duplicate_reverse(self):
        self.assertEqual(len(self.select([[0, 1, 2], [2, 1, 0]])[0]), 1)

    def test_disconnected_surfaces(self):
        self.assertEqual(len(self.select([[0, 1, 2], [3, 4, 5]])[0]), 2)

    def test_ineligible(self):
        f = [[0, 1, 2], [3, 4, 5], [6, 6, 7], [8, 9, 10]]
        self.assertEqual(len(self.select(f, np.array([1., np.nan, 1., -np.inf]))[0]), 1)
        self.assertEqual(len(self.select([])[0]), 0)

    def test_random_soup_deterministic(self):
        rng = np.random.default_rng(40)
        f = np.stack([rng.choice(60, 3, replace=False) for _ in range(1000)])
        a = self.select(f)
        b = self.select(f)
        np.testing.assert_array_equal(a[0], b[0])

    def test_invalid_input(self):
        for f in (np.array([[0, 1, 4]]), np.array([[-1, 1, 2]]), np.zeros((1, 3), float)):
            with self.assertRaises(ValueError): select_manifold_faces(f, 3)
        with self.assertRaises(ValueError): select_manifold_faces(np.array([[0, 1, 2]]), 3, [1, 2])

    def test_scoring_distance_and_geometry(self):
        v = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,2], [1,0,2], [0,1,2]], np.float32)
        before = v.copy()
        f = np.array([[0,1,2], [3,4,5], [0,0,1]], np.int32)
        scores, stats = score_surface_candidates(v, f[:1], f, chunk_size=1)
        self.assertTrue(np.isfinite(scores[0]))
        self.assertTrue(np.isneginf(scores[1:]).all())
        np.testing.assert_array_equal(v, before)

if __name__ == '__main__': unittest.main()
