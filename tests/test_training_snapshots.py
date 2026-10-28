"""Snapshot serialization must preserve geometry and training RNG streams."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from plyfile import PlyData
import torch

from utils.training_snapshots import save_training_snapshot


class SnapshotTests(unittest.TestCase):
    def make_scene(self, directory, device):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [8., 8., 8.]],
                                device=device, requires_grad=True)
        faces = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32, device=device)
        scene = SimpleNamespace(model_path=str(directory),
                                triangles=SimpleNamespace(vertices=vertices, _triangle_indices=faces), saved=[])
        def save(key):
            scene.saved.append(key)
            path = Path(directory) / "point_cloud" / f"iteration_{key}" / "point_cloud_state_dict.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"triangles_points": vertices, "_triangle_indices": faces}, path)
            return str(path)
        scene.save = save
        return scene

    def test_raw_snapshot_roundtrip_and_no_model_or_rng_changes(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                scene = self.make_scene(directory, device)
                vertices, faces = scene.triangles.vertices, scene.triangles._triangle_indices
                original_vertices, original_faces = vertices.detach().clone(), faces.clone()
                py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                gpu_rng = torch.cuda.get_rng_state() if device == "cuda" else None
                record = save_training_snapshot(scene, 11001, "before_rdt")
                self.assertIs(scene.triangles.vertices, vertices)
                self.assertTrue(torch.equal(vertices, original_vertices))
                self.assertTrue(torch.equal(faces, original_faces))
                self.assertIsNone(vertices.grad)
                self.assertEqual(random.getstate(), py_rng)
                np.testing.assert_array_equal(np.random.get_state()[1], np_rng[1])
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
                if gpu_rng is not None:
                    self.assertTrue(torch.equal(torch.cuda.get_rng_state(), gpu_rng))
                ply = PlyData.read(record["raw_mesh"])
                np.testing.assert_array_equal(np.stack([ply["vertex"][name] for name in ("x", "y", "z")], axis=1),
                                              vertices.detach().cpu().numpy())
                np.testing.assert_array_equal(np.stack(ply["face"]["vertex_indices"]), faces.cpu().numpy())
                loaded = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
                self.assertTrue(torch.equal(loaded["_triangle_indices"], faces.cpu()))
                self.assertEqual(scene.saved, ["11001_before_rdt"])
                self.assertEqual(record["position"], "before_iteration_update")
                self.assertFalse(record["contains_optimizer_state"])
                self.assertEqual(json.loads((Path(directory) / "training_snapshots.jsonl").read_text()), record)

    def test_final_snapshot_reuses_already_finalized_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(directory, "cpu")
            checkpoint = scene.save(30000)
            before = Path(checkpoint).read_bytes()
            record = save_training_snapshot(scene, 30000, "training_complete", checkpoint)
            self.assertEqual(scene.saved, [30000])
            self.assertEqual(Path(checkpoint).read_bytes(), before)
            self.assertEqual(record["stage"], "training_complete")

    def test_before_and_after_rdt_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = self.make_scene(directory, "cpu")
            before = save_training_snapshot(scene, 11001, "before_rdt")
            before_bytes = Path(before["checkpoint"]).read_bytes()
            scene.triangles._triangle_indices = scene.triangles._triangle_indices[:1]
            after = save_training_snapshot(scene, 11001, "after_rdt")
            self.assertNotEqual(before["checkpoint"], after["checkpoint"])
            self.assertEqual(Path(before["checkpoint"]).read_bytes(), before_bytes)
            self.assertEqual((before["faces"], after["faces"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
