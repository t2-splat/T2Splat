# Test organization maintained by zhtang2004@gmail.com.
"""Regression and invariant checks for training topology and mesh losses.

Run: python -m unittest -v tests.test_training_topology
"""
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utils.loss_utils import mesh_regularization_losses
from utils.mesh_topology import CollapseSafety, TopologyMonitor, mesh_topology_metrics

spec = importlib.util.spec_from_file_location(
    "heartbeat_topology_test_module", Path(__file__).resolve().parents[1] / "scene/heartbeat.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)


def grid(n=5):
    xyz = np.array([[x, y, 0] for y in range(n) for x in range(n)], dtype=np.float32)
    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x
            faces.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]])
    return xyz, np.asarray(faces, dtype=np.int64)


def attributes(xyz, faces):
    n = len(xyz)
    return [xyz.copy(), np.zeros((n, 1, 3), np.float32),
            np.zeros((n, 15, 3), np.float32), np.zeros((n, 1), np.float32), faces.copy()]


TETRA_X = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32)
TETRA_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)


class TopologyTests(unittest.TestCase):
    def assert_surface(self, xyz, faces, euler, boundary=None):
        m = mesh_topology_metrics(xyz, faces)
        for field in ("nonmanifold_edges", "nonmanifold_vertices", "duplicate_faces",
                      "invalid_index_faces", "repeated_index_faces", "zero_area_faces",
                      "nonfinite_vertices", "nonfinite_faces", "winding_conflict_edges",
                      "unused_vertices"):
            self.assertEqual(m[field], 0, (field, m))
        self.assertEqual(m["face_components"], 1)
        self.assertEqual(len(xyz) - m["edges"] + len(faces), euler)
        if boundary is not None:
            self.assertEqual(m["boundary_edges"], boundary)
        return m

    def test_tetrahedron_collapse_rejects_duplicate_faces(self):
        for qem in (False, True):
            with self.subTest(qem=qem):
                result = hb.batch_edge_collapse(
                    *attributes(TETRA_X, TETRA_F), edges=np.array([[0, 1]]),
                    opacity_floor=0, collapse_ratio=.25, use_qem=qem, enable_safety=True)
                self.assertEqual(len(result[6]), 0)
                np.testing.assert_array_equal(result[4], TETRA_F)

    def test_boundary_chord_and_vanishing_component_rejected(self):
        xyz, faces = grid(2)
        guard = CollapseSafety(xyz, faces)
        self.assertIsNone(guard.check(1, 2, (xyz[1] + xyz[2]) / 2, set()))
        guard = CollapseSafety(xyz[:3], np.array([[0, 1, 2]]))
        self.assertIsNone(guard.check(0, 1, (xyz[0] + xyz[1]) / 2, set()))

    def test_actual_placement_rejects_flips_degeneracy_and_remote_minima(self):
        xyz, faces = grid(4)
        guard = CollapseSafety(xyz, faces)
        good = (xyz[5] + xyz[6]) / 2
        self.assertIsNotNone(guard.check(5, 6, good, set()))
        for position in ([1.5, 2., 0.], [1.5, 2.5, 0.], [100., 100., 0.], [np.nan, 1, 0]):
            with self.subTest(position=position):
                self.assertIsNone(guard.check(5, 6, np.asarray(position), set()))

    def test_safe_batch_uses_disjoint_neighborhoods(self):
        xyz, faces = grid(8)
        edges = hb.extract_edges(faces)
        positions = xyz[edges].mean(axis=1)
        selected = hb._find_independent_set_greedy(
            edges, np.arange(len(edges)), np.ones(len(edges), bool), len(xyz),
            xyz=xyz, indices_np=faces, enable_safety=True, candidate_positions=positions)
        self.assertGreater(len(selected), 1)
        guard, used = CollapseSafety(xyz, faces), set()
        for edge_id in selected:
            u, v = map(int, edges[edge_id])
            affected = guard.check(u, v, positions[edge_id], used)
            self.assertIsNotNone(affected)
            used.update(affected)

    def test_all_split_templates_preserve_area_winding_and_shared_edges(self):
        # Arbitrary subsets exercise 1, 2 and 3 split edges, including all rotations.
        edges = hb.extract_edges(TETRA_F)
        for ids in ([0], [0, 1], [0, 2], [1, 2], [0, 1, 2], list(range(len(edges)))):
            with self.subTest(ids=ids):
                selected = edges[ids]
                xyz = np.concatenate([TETRA_X, TETRA_X[selected].mean(axis=1)])
                faces = hb._subdivide_faces(TETRA_F, selected, len(TETRA_X))
                self.assert_surface(xyz, faces, euler=2, boundary=0)
                old_tri, new_tri = TETRA_X[TETRA_F], xyz[faces]
                def normals(tri):
                    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                old_n, new_n = normals(old_tri), normals(new_tri)
                self.assertAlmostEqual(np.linalg.norm(old_n, axis=1).sum(),
                                       np.linalg.norm(new_n, axis=1).sum(), places=4)
                self.assertTrue(np.all(np.max(new_n @ old_n.T, axis=1) > 0))

    def test_repeated_collapse_split_preserves_disk_topology(self):
        xyz, faces = grid(10)
        data = attributes(xyz, faces)
        total_collapsed = 0
        for i in range(4):
            collapsed = hb.batch_edge_collapse(
                *data, edges=hb.extract_edges(data[4]), opacity_floor=0,
                collapse_ratio=.08, enable_safety=True, use_qem=True)
            total_collapsed += len(collapsed[6])
            data = list(collapsed[:5])
            self.assert_surface(data[0], data[4], euler=1)
            result = hb.batch_vertex_split(
                *data, importance_np=np.random.default_rng(i).random(len(data[0])),
                top_k_percent=60, enable_safety=True)
            self.assertGreater(result[5], 0)
            self.assertEqual(len(np.unique(result[6])), result[5])
            data = list(result[:5])
            self.assert_surface(data[0], data[4], euler=1)
        self.assertGreater(total_collapsed, 0)

    def test_detect_bow_tie_even_without_nonmanifold_edges(self):
        xyz, faces = grid(2)
        xyz = np.r_[xyz[:3], [[-1, 0, 0], [0, -1, 0]]]
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        metrics = mesh_topology_metrics(xyz, faces)
        self.assertEqual(metrics["nonmanifold_edges"], 0)
        self.assertEqual(metrics["nonmanifold_vertices"], 1)
        self.assertEqual(metrics["disconnected_fan_vertices"], 1)
        self.assertIsNone(CollapseSafety(xyz, faces).link(0))

    def test_safe_split_does_not_multiply_duplicate_faces(self):
        faces = np.array([[0, 1, 2], [0, 2, 1]])
        result = hb.batch_vertex_split(
            *attributes(TETRA_X[:3], faces), importance_np=np.ones(3),
            top_k_percent=100, enable_safety=True)
        self.assertEqual(result[5], 0)
        np.testing.assert_array_equal(result[4], faces)

    def test_safe_heartbeat_optimizer_and_event_order_cpu_cuda(self):
        from tests.test_heartbeat_momentum import MomentumTests
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                fixture = MomentumTests()
                fixture.device = device
                xyz, faces = grid(8)
                model = fixture.make_model(xyz=xyz, faces=faces)
                events = []
                def record(stage, xyz, faces):
                    self.assert_surface(xyz, faces, euler=1)
                    events.append(stage)
                for i in range(3):
                    hb.init_heartbeat_stats(model)
                    model._hb_grad_accum.fill_(1)
                    model._hb_obs_count.fill_(1)
                    hb.heartbeat_step(model, mode="both", collapse_ratio=.08,
                                      split_top_k=30, enable_safety=True,
                                      topology_callback=record)
                    fixture.assert_can_step(model, 7 + i)
                self.assertEqual(events, ["before_dte", "after_collapse", "after_split"] * 3)

    def test_metrics_distinguish_duplicate_winding_and_nonmanifold_edges(self):
        self.assert_surface(TETRA_X, TETRA_F, euler=2, boundary=0)
        f = np.r_[TETRA_F, TETRA_F[:1, ::-1]]
        m = mesh_topology_metrics(TETRA_X, f)
        self.assertEqual(m["duplicate_faces"], 1)
        self.assertEqual(m["nonmanifold_edges"], 3)
        self.assertEqual(m["nonmanifold_vertices"], 3)
        f = TETRA_F.copy()
        f[0] = f[0, ::-1]
        self.assertEqual(mesh_topology_metrics(TETRA_X, f)["winding_conflict_edges"], 3)

    def test_monitor_writes_finite_json_and_handles_invalid_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            import json
            monitor = TopologyMonitor(Path(tmp) / "metrics.jsonl")
            monitor.record(13, "after_split", TETRA_X, TETRA_F)
            m = monitor.record(14, "invalid", TETRA_X, [[-1, 1, 2], [1, 1, 2]])
            self.assertEqual(m["invalid_index_faces"], 1)
            self.assertEqual(m["repeated_index_faces"], 1)
            records = [json.loads(x) for x in monitor.path.read_text().splitlines()]
            self.assertEqual([r["iteration"] for r in records], [13, 14])


class LossTests(unittest.TestCase):
    device = "cpu"

    def tensors(self, xyz, faces):
        return (torch.tensor(xyz, device=self.device, dtype=torch.float64, requires_grad=True),
                torch.tensor(faces, device=self.device, dtype=torch.long).reshape(-1, 3))

    def test_disconnected_opposite_triangles_have_zero_normal_loss(self):
        v, f = self.tensors([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [3, 0, 0], [3, 1, 0], [4, 0, 0]], [[0, 1, 2], [3, 4, 5]])
        loss = mesh_regularization_losses(v, f, compute_smooth=False, compute_edge=False)["nc"]
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(v.grad, torch.zeros_like(v)))

    def test_true_edge_mean_and_gradient_match_independent_reference(self):
        v, f = self.tensors([[0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 3, 1]],
                            [[0, 1, 2], [1, 3, 2]])
        edges = sorted({tuple(sorted(pair)) for face in f.tolist()
                        for pair in itertools.combinations(face, 2)})
        expected = torch.stack([(v[a] - v[b]).square().sum() for a, b in edges]).mean()
        actual = mesh_regularization_losses(v, f, compute_smooth=False, compute_nc=False)["edge"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, v)[0], torch.autograd.grad(expected, v)[0])

    def test_only_two_incident_faces_contribute_and_gradients_are_finite(self):
        v, f = self.tensors([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[0, 1, 2], [1, 0, 3]])
        def loss(x):
            return mesh_regularization_losses(x, f, compute_smooth=False, compute_edge=False)["nc"]
        self.assertAlmostEqual(loss(v).item(), 1)
        self.assertTrue(torch.autograd.gradcheck(loss, (v,), eps=1e-6, atol=1e-5))
        v, f = self.tensors([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]],
                            [[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        loss = mesh_regularization_losses(v, f, compute_smooth=False, compute_edge=False)["nc"]
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(v.grad).all())

    def test_empty_mesh_enabled_losses_are_differentiable_zeros(self):
        v, f = self.tensors([[1, 2, 3]], [])
        losses = mesh_regularization_losses(v, f)
        sum(losses.values()).backward()
        self.assertTrue(torch.equal(v.grad, torch.zeros_like(v)))
        self.assertTrue(all(x.item() == 0 for x in losses.values()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaLossTests(LossTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
