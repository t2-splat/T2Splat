"""Check signed texture fusion, native mip pooling, UVs and standalone GLB IO."""
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.glb_export import (ExportScene, bake_tiles, bake_vertex_colors, camera_matrices,
                              export_checkpoint, face_mip, load_scene,
                              load_training_cameras, resolve_checkpoint, write_glb)
from utils.sh_utils import C1


def fixture_state(resolution=4, face_count=3):
    vertices = torch.tensor([[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]])
    textures = torch.zeros(face_count, resolution, resolution, 3)
    v, u = torch.meshgrid(torch.linspace(0, 1, resolution), torch.linspace(0, 1, resolution), indexing="ij")
    for i in range(face_count):
        textures[i] = torch.stack((.2 * u - .1, .2 * v - .1, u * 0 + .03 * i), -1)
    return {"triangles_points": vertices, "_triangle_indices": torch.tensor([[0, 1, 2]] * face_count),
            "features_dc": torch.zeros(3, 1, 3), "features_rest": torch.zeros(3, 15, 3),
            "active_sh_degree": 3, "face_textures": textures, "face_tex_res": resolution, "use_mip_texture": True}


def read_glb(path):
    # Independent binary reader: inspect actual serialized offsets and PNGs.
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    json_length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A and json_length % 4 == 0
    document = json.loads(data[20:20 + json_length])
    bin_length, kind = struct.unpack_from("<II", data, 20 + json_length)
    assert kind == 0x004E4942 and bin_length % 4 == 0
    binary = data[28 + json_length:]
    assert len(binary) == bin_length
    assert 0 <= bin_length - document["buffers"][0]["byteLength"] <= 3
    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= document["buffers"][0]["byteLength"]
    return document, binary


def accessor(document, binary, index):
    a = document["accessors"][index]
    v = document["bufferViews"][a["bufferView"]]
    columns = {"VEC2": 2, "VEC3": 3}[a["type"]]
    return np.frombuffer(binary, dtype="<f4", count=a["count"] * columns,
                         offset=v["byteOffset"]).reshape(-1, columns)


def png(document, binary, index):
    im = document["images"][index]
    view = document["bufferViews"][im["bufferView"]]
    assert im["mimeType"] == "image/png" and "uri" not in im
    start = view["byteOffset"]
    return np.array(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])))


class TextureFusionTests(unittest.TestCase):
    def test_signed_residual_and_clamp_order_match_analytic_surface(self):
        base = np.array([[[1.2, .2, .6], [1.2, .4, .6], [1.2, .2, .8]]], np.float32)
        texture = np.zeros((1, 4, 4, 3), np.float32)
        texture[..., 0] = -.4
        v, u = np.mgrid[:4, :4] / 3
        texture[0, ..., 1] = -.1 + .1 * u
        texture[0, ..., 2] = -.2 + .1 * v
        pixels, _ = bake_tiles(base, texture, 4)
        for y in range(4):
            for x in range(4 - y):
                expected = np.array([.8, .1 + .3 * x / 3, .4 + .3 * y / 3])
                np.testing.assert_allclose(pixels[0, y, x] / 255, expected, atol=1 / 255)

    def test_mip_pooling_matches_native_torch_for_r8_r16_and_odd_size(self):
        rng = np.random.default_rng(12)
        for resolution in (5, 8, 16):
            data = rng.normal(size=(2, resolution, resolution, 3)).astype(np.float32)
            expected = torch.from_numpy(data).permute(0, 3, 1, 2)
            level = 0
            while expected.shape[-1] > 1:
                level += 1
                expected = F.avg_pool2d(expected, 2)
                np.testing.assert_allclose(face_mip(data, level), expected.permute(0, 2, 3, 1).numpy(), atol=2e-7)

    def test_last_mip_is_constant_residual_but_preserves_base_gradient(self):
        base = np.array([[[.2, .2, .2], [.4, .2, .2], [.2, .4, .2]]], np.float32)
        texture = np.arange(4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3) / 1000 - .02
        pixels, _ = bake_tiles(base, texture, 4, mip_level=2)
        mean = texture.mean(axis=(1, 2))[0]
        for xy, corner in [((0, 0), 0), ((0, 3), 1), ((3, 0), 2)]:
            np.testing.assert_allclose(pixels[(0, *xy)] / 255, base[0, corner] + mean, atol=1 / 255)

    def test_one_texel_and_absent_texture(self):
        base = np.array([[[.2, .3, .4], [.4, .3, .4], [.2, .5, .4]]], np.float32)
        without, _ = bake_tiles(base, None, 2)
        with_texture, _ = bake_tiles(base, np.full((1, 1, 1, 3), .1, np.float32), 2)
        np.testing.assert_allclose((with_texture.astype(float) - without) / 255, .1, atol=1 / 255)


class GlbExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "point_cloud_state_dict.pt"
        self.state = fixture_state()
        torch.save(self.state, self.checkpoint)

    def test_complete_export_embeds_fused_pngs_uvs_and_geometry(self):
        output = self.root / "mesh.glb"
        report = export_checkpoint(self.root, output, color_mode="dc", atlas_size=8, chunk_faces=1)
        document, binary = read_glb(output)
        self.assertEqual(report["faces"], 3)
        self.assertEqual(report["atlas_count"], 3)
        self.assertEqual(document["samplers"][0]["minFilter"], 9987)
        for i, primitive in enumerate(document["meshes"][0]["primitives"]):
            positions = accessor(document, binary, primitive["attributes"]["POSITION"])
            uv = accessor(document, binary, primitive["attributes"]["TEXCOORD_0"])
            np.testing.assert_array_equal(positions, self.state["triangles_points"].numpy())
            np.testing.assert_allclose(uv, [[2.5 / 8, 2.5 / 8], [5.5 / 8, 2.5 / 8], [2.5 / 8, 5.5 / 8]])
            image = png(document, binary, i)
            for y in range(4):
                for x in range(4 - y):
                    expected = np.array([.4 + .2 * x / 3, .4 + .2 * y / 3, .5 + .03 * i])
                    np.testing.assert_allclose(image[y + 2, x + 2] / 255, expected, atol=1 / 255)
            np.testing.assert_array_equal(image[0, 2:6], image[2, 2:6])
            np.testing.assert_array_equal(image[2:6, 0], image[2:6, 2])
            material = document["materials"][primitive["material"]]
            self.assertIn("KHR_materials_unlit", material["extensions"])
            self.assertTrue(material["doubleSided"])
            self.assertEqual(material["alphaMode"], "OPAQUE")
        self.assertEqual(report["bytes"], output.stat().st_size)
        self.assertTrue(output.with_name(output.name + ".export.json").is_file())
        actual = torch.load(self.checkpoint, weights_only=True)
        for key, value in self.state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)

    def test_batching_does_not_change_asset_and_overwrite_is_explicit(self):
        scene = load_scene(self.checkpoint)
        colors, _ = bake_vertex_colors(scene, "dc")
        a, b = self.root / "a/mesh.glb", self.root / "b/mesh.glb"
        write_glb(scene, colors, a, atlas_size=32, chunk_faces=1)
        write_glb(scene, colors, b, atlas_size=32, chunk_faces=100)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        before = a.read_bytes()
        with self.assertRaises(FileExistsError):
            write_glb(scene, colors, a, atlas_size=32)
        self.assertEqual(before, a.read_bytes())

    def test_no_texture_and_zup(self):
        self.state.pop("face_textures")
        self.state["use_mip_texture"] = False
        torch.save(self.state, self.checkpoint)
        out = self.root / "mesh.glb"
        report = export_checkpoint(self.root, out, color_mode="dc", atlas_size=16, zup=True)
        document, binary = read_glb(out)
        self.assertFalse(report["has_face_textures"])
        self.assertEqual(document["samplers"][0]["minFilter"], 9729)
        positions = accessor(document, binary, 0)
        expected = self.state["triangles_points"].numpy()[:, [0, 2, 1]].copy()
        expected[:, 2] *= -1
        np.testing.assert_array_equal(positions[:3], expected)

    def test_r8_r16_exports_keep_face_resolution_and_selected_mip(self):
        for resolution in (8, 16):
            for level in (0, 1):
                with self.subTest(resolution=resolution, level=level):
                    state = fixture_state(resolution, face_count=1)
                    torch.save(state, self.checkpoint)
                    out = self.root / f"r{resolution}_lod{level}.glb"
                    report = export_checkpoint(self.root, out, color_mode="dc", atlas_size=32, mip_level=level)
                    document, binary = read_glb(out)
                    image = png(document, binary, 0)
                    self.assertEqual(report["tile_size"], resolution)
                    self.assertEqual(report["baked_mip_level"], level)
                    texture = state["face_textures"].permute(0, 3, 1, 2)
                    if level:
                        texture = F.avg_pool2d(texture, 2)
                    expected = (.5 + texture[0, :, 0, 0]).numpy()
                    np.testing.assert_allclose(image[2, 2] / 255, expected, atol=1 / 255)

    def test_invalid_texture_shape_and_nonfinite_values_fail(self):
        for texture in (torch.zeros(2, 4, 4, 3), torch.full((3, 4, 4, 3), float("nan"))):
            self.state["face_textures"] = texture
            torch.save(self.state, self.checkpoint)
            with self.assertRaises(ValueError):
                load_scene(self.checkpoint)

    def test_checkpoint_resolution(self):
        for iteration in (9, 10):
            path = self.root / "model/point_cloud" / f"iteration_{iteration}/point_cloud_state_dict.pt"
            path.parent.mkdir(parents=True)
            torch.save(self.state, path)
        checkpoint, root = resolve_checkpoint(self.root / "model")
        self.assertEqual(checkpoint.parent.name, "iteration_10")
        self.assertEqual(root, self.root / "model")
        self.assertEqual(resolve_checkpoint(self.root / "model", 9)[0].parent.name, "iteration_9")
        self.assertEqual(resolve_checkpoint(checkpoint)[0], checkpoint)


def camera(name="view"):
    return {"img_name": name, "width": 64, "height": 64, "fx": 40., "fy": 40.,
            "position": [0, 0, 0], "rotation": np.eye(3).tolist()}


class CameraTests(unittest.TestCase):
    def test_training_split_excludes_test_and_explicit_selection_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path, selection = Path(temp) / "cameras.json", Path(temp) / "train.txt"
            path.write_text(json.dumps([camera(f"{i:03d}") for i in reversed(range(10))]))
            selected = load_training_cameras(path)
            self.assertEqual([c["img_name"] for c in selected], [f"{i:03d}" for i in range(10) if i % 8])
            selection.write_text("009\n001\n")
            self.assertEqual([c["img_name"] for c in load_training_cameras(path, train_list=selection)], ["001", "009"])
            selection.write_text("missing\n")
            with self.assertRaises(ValueError):
                load_training_cameras(path, train_list=selection)

    def test_camera_projection_uses_saved_camera_to_world(self):
        matrix, center, size = camera_matrices(camera(), 32)
        self.assertEqual(size, (32, 32))
        np.testing.assert_array_equal(center, [0, 0, 0])
        clip = torch.tensor([[0., 0., 2., 1.], [.5, 0., 2., 1.]]) @ matrix
        np.testing.assert_allclose((clip[:, :2] / clip[:, 3:4]).numpy(), [[0, 0], [.3125, 0]], atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA visibility test")
    def test_visible_sh_and_occluded_vertex_fallback(self):
        front = np.array([[-.5, -.5, 2], [.5, -.5, 2], [0, .5, 2]], np.float32)
        vertices = np.concatenate((front, front * 1.5))
        rest = np.zeros((6, 3, 3), np.float32)
        rest[:, 1, 0] = .2
        scene = ExportScene(vertices, np.arange(6, dtype=np.int32).reshape(2, 3),
                            np.zeros((6, 1, 3), np.float32), rest, 1, None, False)
        actual, stats = bake_vertex_colors(scene, "visible-mean", [camera()], max_size=64)
        direction = front / np.linalg.norm(front, axis=-1, keepdims=True)
        expected = np.full((6, 3), .5, np.float32)
        expected[:3, 0] += C1 * .2 * direction[:, 2]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertEqual(stats["observed_vertices"], 3)


if __name__ == "__main__":
    unittest.main()
