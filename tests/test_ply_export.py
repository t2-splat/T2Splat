"""Verify ordinary PLY appearance baking through independently decoded assets."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from utils.glb_export import ExportScene, bake_vertex_colors, write_glb
from utils.ply_export import subdivision_grid, write_ply
from utils.sh_utils import C0, C1, C2


def scene_fixture(resolution=4, two_faces=False):
    vertices = np.array([[0., 0., 2.], [1., 0., 2.], [0., 1., 2.],
                         [-1., 0., 2.]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    if not two_faces:
        vertices, faces = vertices[:3], faces[:1]
    texture = (None if resolution is None else
               np.zeros((len(faces), resolution, resolution, 3), dtype=np.float32))
    return ExportScene(vertices, faces, np.zeros((len(vertices), 1, 3), np.float32),
                       np.zeros((len(vertices), 8, 3), np.float32), 2, texture,
                       texture is not None)


def read_ply(path):
    """Small independent binary PLY reader, without using the export helpers."""
    types = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B",
             "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
             "int": "i", "int32": "i", "uint": "I", "uint32": "I",
             "float": "f", "float32": "f", "double": "d", "float64": "d"}
    with Path(path).open("rb") as stream:
        assert stream.readline().strip() == b"ply"
        assert stream.readline().strip() == b"format binary_little_endian 1.0"
        elements = []
        while True:
            line = stream.readline()
            assert line, "Truncated PLY header"
            words = line.decode("ascii").strip().split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                break
            if words[0] == "element":
                elements.append((words[1], int(words[2]), []))
            elif words[0] == "property":
                elements[-1][2].append(words[1:])
            else:
                raise AssertionError(f"Unexpected header: {line!r}")

        def scalar(kind):
            fmt = "<" + types[kind]
            return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]

        result = {}
        for name, count, properties in elements:
            values = {prop[-1]: [] for prop in properties}
            for _ in range(count):
                for prop in properties:
                    if prop[0] == "list":
                        value = [scalar(prop[2]) for _ in range(scalar(prop[1]))]
                    else:
                        value = scalar(prop[0])
                    values[prop[-1]].append(value)
            result[name] = {key: np.asarray(value) for key, value in values.items()}
        assert stream.read() == b"", "Unexpected trailing PLY payload"
    vertex = result["vertex"]
    positions = np.column_stack([vertex[key] for key in ("x", "y", "z")])
    colors = np.column_stack([vertex[key] for key in ("red", "green", "blue")])
    face = result["face"]
    faces = face.get("vertex_indices", face.get("vertex_index"))
    assert faces is not None and faces.ndim == 2 and faces.shape[1] == 3
    return positions, colors, faces


class SubdivisionTests(unittest.TestCase):
    def test_regular_grid_preserves_area_winding_and_boundaries(self):
        for subdivisions in (1, 3, 7):
            with self.subTest(subdivisions=subdivisions):
                bary, faces = subdivision_grid(subdivisions)
                self.assertEqual(bary.shape, ((subdivisions + 1) * (subdivisions + 2) // 2, 3))
                self.assertEqual(faces.shape, (subdivisions ** 2, 3))
                np.testing.assert_allclose(bary.sum(axis=1), 1, atol=1e-6)
                self.assertGreaterEqual(bary.min(), -1e-7)
                self.assertGreaterEqual(faces.min(), 0)
                self.assertLess(faces.max(), len(bary))
                points = np.column_stack((bary[:, 1:], np.zeros(len(bary))))
                triangles = points[faces]
                normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0])
                np.testing.assert_allclose(normals[:, :2], 0, atol=1e-7)
                self.assertTrue((normals[:, 2] > 0).all())
                self.assertAlmostEqual(normals[:, 2].sum() / 2, .5, places=6)
                for corner in np.eye(3):
                    self.assertEqual(np.count_nonzero(np.all(np.isclose(bary, corner), axis=1)), 1)


class PlyExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_interior_miptex_samples_survive_in_standard_vertex_colors(self):
        scene = scene_fixture(4)
        scene.textures[0, 1, 1] = [.25, -.15, .05]
        output = self.root / "interior.ply"
        write_ply(scene, np.full((3, 3), .5, np.float32), output, texture_mode="subdivide")
        positions, colors, faces = read_ply(output)
        self.assertEqual(len(positions), 10)
        self.assertEqual(len(faces), 9)
        interior = np.all(np.isclose(positions, [1 / 3, 1 / 3, 2]), axis=1)
        self.assertEqual(interior.sum(), 1)
        np.testing.assert_allclose(colors[interior][0] / 255, [.75, .35, .55], atol=1 / 255)
        for corner in scene.vertices:
            found = np.all(np.isclose(positions, corner), axis=1)
            np.testing.assert_allclose(colors[found] / 255, .5, atol=1 / 255)

    def test_signed_residual_clips_after_fusion_and_uses_row_c_column_b(self):
        scene = scene_fixture(4)
        v, u = np.mgrid[:4, :4] / 3
        scene.textures[0, ..., 0] = -.4
        scene.textures[0, ..., 1] = -.1 + .1 * u
        scene.textures[0, ..., 2] = -.2 + .1 * v
        base = np.array([[1.2, .2, .6], [1.2, .4, .6], [1.2, .2, .8]], np.float32)
        output = self.root / "signed.ply"
        write_ply(scene, base, output, texture_mode="subdivide")
        positions, colors, _ = read_ply(output)
        b, c = positions[:, 0], positions[:, 1]
        expected = np.column_stack((np.full(len(b), .8), .1 + .3 * b, .4 + .3 * c))
        np.testing.assert_allclose(colors / 255, expected, atol=1 / 255)

    def test_face_seams_keep_distinct_colors_at_shared_source_vertices(self):
        scene = scene_fixture(4, two_faces=True)
        scene.textures[0] = .1
        scene.textures[1] = -.1
        output = self.root / "seams.ply"
        write_ply(scene, np.full((4, 3), .5, np.float32), output, texture_mode="subdivide")
        positions, colors, faces = read_ply(output)
        self.assertEqual(len(positions), 20)
        self.assertEqual(len(faces), 18)
        shared = np.all(np.isclose(positions, scene.vertices[0]), axis=1)
        self.assertEqual(shared.sum(), 2)
        np.testing.assert_allclose(np.sort(colors[shared, 0] / 255), [.4, .6], atol=1 / 255)
        self.assertEqual(len(set(faces[:9].flatten()) & set(faces[9:].flatten())), 0)

    def test_native_resolution_and_selected_mips_match_torch_pooling(self):
        rng = np.random.default_rng(4)
        for resolution in (4, 8):
            scene = scene_fixture(resolution)
            scene.textures[:] = rng.uniform(-.2, .2, size=scene.textures.shape)
            for level in range(int(np.log2(resolution)) + 1):
                with self.subTest(resolution=resolution, mip_level=level):
                    output = self.root / f"r{resolution}_mip{level}.ply"
                    write_ply(scene, np.full((3, 3), .5, np.float32), output,
                              texture_mode="subdivide", mip_level=level)
                    positions, colors, faces = read_ply(output)
                    s = resolution - 1
                    self.assertEqual(len(positions), (s + 1) * (s + 2) // 2)
                    self.assertEqual(len(faces), s * s)
                    texture = torch.from_numpy(scene.textures).permute(0, 3, 1, 2)
                    for _ in range(level):
                        texture = F.avg_pool2d(texture, 2)
                    grid = torch.tensor(2 * positions[:, :2] - 1, dtype=torch.float32)[None, None]
                    residual = F.grid_sample(texture, grid, align_corners=True)[0, :, 0].T.numpy()
                    np.testing.assert_allclose(colors / 255, .5 + residual, atol=1 / 255)

    def test_chunking_produces_identical_serialized_geometry_and_colors(self):
        scene = scene_fixture(8, two_faces=True)
        scene.textures[:] = np.random.default_rng(7).normal(0, .1, scene.textures.shape)
        base = np.full((4, 3), .5, np.float32)
        first, second = self.root / "a.ply", self.root / "b.ply"
        write_ply(scene, base, first, texture_mode="subdivide", chunk_faces=1)
        write_ply(scene, base, second, texture_mode="subdivide", chunk_faces=100)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_untextured_export_keeps_original_topology_and_transforms_zup(self):
        scene = scene_fixture(None, two_faces=True)
        base = np.array([[.2, .3, .4], [.4, .5, .6], [.6, .7, .8], [.8, .9, 1.]], np.float32)
        for zup in (False, True):
            output = self.root / f"plain_{zup}.ply"
            write_ply(scene, base, output, zup=zup)
            positions, colors, faces = read_ply(output)
            expected = scene.vertices.copy()
            if zup:
                expected = expected[:, [0, 2, 1]]
                expected[:, 2] *= -1
            np.testing.assert_array_equal(positions, expected)
            np.testing.assert_array_equal(faces, scene.faces)
            np.testing.assert_allclose(colors / 255, base, atol=1 / 255)
            old_triangles, new_triangles = scene.vertices[scene.faces], positions[faces]
            old_normals = np.cross(old_triangles[:, 1] - old_triangles[:, 0],
                                   old_triangles[:, 2] - old_triangles[:, 0])
            new_normals = np.cross(new_triangles[:, 1] - new_triangles[:, 0],
                                   new_triangles[:, 2] - new_triangles[:, 0])
            if zup:
                old_normals = old_normals[:, [0, 2, 1]]
                old_normals[:, 2] *= -1
            np.testing.assert_array_equal(new_normals, old_normals)

    def test_overwrite_requires_explicit_request(self):
        scene = scene_fixture(4)
        output = self.root / "existing.ply"
        colors = np.full((3, 3), .5, np.float32)
        write_ply(scene, colors, output)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_ply(scene, colors + .1, output)
        self.assertEqual(output.read_bytes(), before)
        write_ply(scene, colors + .1, output, overwrite=True)
        _, updated, _ = read_ply(output)
        np.testing.assert_allclose(updated / 255, .6, atol=1 / 255)

    def test_same_basename_ply_and_glb_keep_independent_reports(self):
        scene = scene_fixture(4, two_faces=True)
        scene.textures[0, 1, 1] = [.1, -.1, .05]
        colors = np.full((4, 3), .5, np.float32)
        ply, glb = self.root / "mesh.ply", self.root / "mesh.glb"
        ply_report = self.root / "mesh.ply.export.json"
        glb_report = self.root / "mesh.glb.export.json"
        report = write_ply(scene, colors, ply)
        positions, _, faces = read_ply(ply)
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        self.assertEqual(report["texture_mode"], "vertex")
        original = ply.read_bytes(), ply_report.read_bytes()

        write_glb(scene, colors, glb, atlas_size=16)

        self.assertEqual(original, (ply.read_bytes(), ply_report.read_bytes()))
        self.assertEqual(glb.read_bytes()[:4], b"glTF")
        self.assertEqual(json.loads(ply_report.read_text())["format"], "t2splat-static-ply/1")
        self.assertEqual(json.loads(glb_report.read_text())["format"], "t2splat-static-glb/1")
        self.assertFalse((self.root / "mesh.export.json").exists())

    def test_invalid_options_or_colors_fail_without_partial_output(self):
        scene = scene_fixture(4)
        colors = np.full((3, 3), .5, np.float32)
        invalid_options = [{"subdivisions": 0}, {"subdivisions": -1},
                           {"subdivisions": 1.5}, {"mip_level": -1},
                           {"mip_level": 3}, {"chunk_faces": 0}, {"chunk_faces": -1}]
        for i, options in enumerate(invalid_options):
            with self.subTest(options=options):
                output = self.root / f"bad_option_{i}.ply"
                with self.assertRaises((ValueError, TypeError)):
                    write_ply(scene, colors, output, **options)
                self.assertFalse(output.exists())
        for i, invalid in enumerate((colors[:2], np.full((3, 3), np.nan, np.float32))):
            with self.subTest(invalid_color=i):
                output = self.root / f"bad_color_{i}.ply"
                with self.assertRaises(ValueError):
                    write_ply(scene, invalid, output)
                self.assertFalse(output.exists())

    def test_camera_baking_includes_higher_order_sh_before_texture_fusion(self):
        scene = scene_fixture(4)
        scene.dc[:, 0, 0] = .2
        scene.rest[:, 1, 0] = .3   # SH coefficient 2: +C1 * z.
        scene.rest[:, 5, 1] = .4   # SH coefficient 6: C2[2] * (2zz - xx - yy).
        scene.rest[:, 7, 2] = .2   # SH coefficient 8: C2[4] * (xx - yy).
        scene.textures[:] = [-.05, .02, -.03]
        center = np.array([.2, -.3, 0], np.float32)
        actual, _ = bake_vertex_colors(scene, "camera", camera_center=center)
        direction = scene.vertices - center
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        x, y, z = direction.T
        expected = np.column_stack((.5 + C0 * .2 + C1 * .3 * z,
                                    .5 + C2[2] * .4 * (2 * z * z - x * x - y * y),
                                    .5 + C2[4] * .2 * (x * x - y * y)))
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        output = self.root / "sh_texture.ply"
        write_ply(scene, actual, output, texture_mode="subdivide")
        positions, colors, _ = read_ply(output)
        b, c = positions[:, 0], positions[:, 1]
        bary = np.column_stack((1 - b - c, b, c))
        np.testing.assert_allclose(colors / 255, bary @ expected + [-.05, .02, -.03], atol=1 / 255)


class PlyCliIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save_scene(self, scene):
        checkpoint = self.root / "point_cloud_state_dict.pt"
        state = {"triangles_points": torch.from_numpy(scene.vertices.copy()),
                 "_triangle_indices": torch.from_numpy(scene.faces.copy()),
                 "features_dc": torch.from_numpy(scene.dc.copy()),
                 "features_rest": torch.from_numpy(scene.rest.copy()),
                 "active_sh_degree": scene.degree, "use_mip_texture": scene.use_mip}
        if scene.textures is not None:
            state.update(face_textures=torch.from_numpy(scene.textures.copy()),
                         face_tex_res=scene.textures.shape[1])
        torch.save(state, checkpoint)
        return checkpoint

    def run_cli(self, *arguments):
        from create_ply import main
        with contextlib.redirect_stdout(io.StringIO()):
            return main([str(self.root), *map(str, arguments)])

    @staticmethod
    def tagged_scene(vertices, faces):
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        dc = np.arange(len(vertices) * 3, dtype=np.float32).reshape(-1, 1, 3) / 100
        rest = np.arange(len(vertices) * 8 * 3, dtype=np.float32).reshape(-1, 8, 3) / 1000
        textures = np.zeros((len(faces), 4, 4, 3), dtype=np.float32)
        for index in range(len(faces)):
            textures[index] = [.01 * (index + 1), -.02 * (index + 1), .03 * (index + 1)]
        return ExportScene(vertices, faces, dc, rest, 2, textures, True)

    def test_max_edge_filter_keeps_matching_face_texture_and_sh(self):
        from create_ply import prepare_bake_scene
        scene = self.tagged_scene(
            [[0, 0, 2], [4, 0, 2], [0, 4, 2],
             [5, 0, 2], [6, 0, 2], [5, 1, 2],
             [7, 0, 2], [8, 0, 2], [7, 1, 2]],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = prepare_bake_scene(scene, max_edge=2)
        np.testing.assert_array_equal(actual.vertices[actual.faces], scene.vertices[scene.faces[1:]])
        np.testing.assert_array_equal(actual.dc[actual.faces], scene.dc[scene.faces[1:]])
        np.testing.assert_array_equal(actual.rest[actual.faces], scene.rest[scene.faces[1:]])
        np.testing.assert_array_equal(actual.textures, scene.textures[1:])
        self.save_scene(scene)
        output = self.root / "filtered.ply"
        self.run_cli("--cpu", "--color-mode", "dc", "--max_edge", "2",
                     "--texture-mode", "subdivide", "--out", output)
        positions, colors, faces = read_ply(output)
        self.assertEqual(len(faces), 18)
        self.assertEqual(len(positions), 20)
        for index, source_face in enumerate(scene.faces[1:]):
            for source_vertex in source_face:
                found = np.all(np.isclose(positions[index * 10:(index + 1) * 10],
                                         scene.vertices[source_vertex]), axis=1)
                expected = .5 + C0 * scene.dc[source_vertex, 0] + scene.textures[index + 1, 0, 0]
                np.testing.assert_allclose(colors[index * 10:(index + 1) * 10][found] / 255,
                                           expected[None], atol=1 / 255)

    def test_cleaning_occurs_before_seam_splitting_and_preserves_sh_texture_mapping(self):
        from create_ply import prepare_bake_scene
        scene = self.tagged_scene(
            [[9, 0, 2], [10, 0, 2], [9, 1, 2],
             [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]],
            [[0, 1, 2], [3, 4, 5], [3, 5, 6]])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = prepare_bake_scene(scene, clean=True, keep_largest=True)
        np.testing.assert_array_equal(actual.vertices, scene.vertices[3:])
        np.testing.assert_array_equal(actual.faces, [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(actual.dc, scene.dc[3:])
        np.testing.assert_array_equal(actual.rest, scene.rest[3:])
        np.testing.assert_array_equal(actual.textures, scene.textures[1:])
        self.save_scene(scene)
        output = self.root / "cleaned.ply"
        report = self.run_cli("--cpu", "--camera", "0,0,0", "--clean", "--keep_largest",
                              "--texture-mode", "subdivide", "--out", output)
        positions, _, faces = read_ply(output)
        # Cleaning after seams split would turn the two-face component into two
        # one-face components and keep_largest would delete one of its faces.
        self.assertEqual(len(faces), 18)
        self.assertEqual(len(positions), 20)
        self.assertLessEqual(positions[:, 0].max(), 1)
        self.assertEqual(report["source_faces"], 2)
        self.assertEqual(report["source_vertices"], 4)

    def test_bowtie_split_source_ids_retain_vertex_attributes(self):
        from create_ply import _split_nonmanifold_vertex_fans
        scene = self.tagged_scene(
            [[0, 0, 2], [1, 0, 2], [0, 1, 2], [-1, 0, 2], [0, -1, 2]],
            [[0, 1, 2], [0, 3, 4]])
        vertices, faces, _, splits, sources = _split_nonmanifold_vertex_fans(
            scene.vertices, scene.faces, None, [0], return_sources=True)
        self.assertEqual(splits, 1)
        self.assertEqual(len(vertices), 6)
        np.testing.assert_array_equal(vertices[faces], scene.vertices[scene.faces])
        np.testing.assert_array_equal(scene.dc[sources][faces], scene.dc[scene.faces])
        np.testing.assert_array_equal(scene.rest[sources][faces], scene.rest[scene.faces])
        self.assertEqual(len(set(faces[0]) & set(faces[1])), 0)

    @unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "Open3D manifold repair")
    def test_manifold_bowtie_keeps_sh_and_original_face_textures(self):
        from create_ply import prepare_bake_scene
        scene = self.tagged_scene(
            [[0, 0, 2], [1, 0, 2], [0, 1, 2], [-1, 0, 2], [0, -1, 2]],
            [[0, 1, 2], [0, 3, 4]])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = prepare_bake_scene(scene, manifold=True)
        self.assertEqual(len(actual.vertices), 6)
        np.testing.assert_array_equal(actual.vertices[actual.faces], scene.vertices[scene.faces])
        np.testing.assert_array_equal(actual.dc[actual.faces], scene.dc[scene.faces])
        np.testing.assert_array_equal(actual.rest[actual.faces], scene.rest[scene.faces])
        np.testing.assert_array_equal(actual.textures, scene.textures)

    @unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "Open3D manifold repair")
    def test_nonmanifold_edge_removal_retains_original_face_texture_ids(self):
        from create_ply import prepare_bake_scene
        scene = self.tagged_scene(
            [[0, 0, 2], [1, 0, 2], [0, .1, 2], [0, -2, 2], [0, 0, 5]],
            [[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = prepare_bake_scene(scene, manifold=True)
        self.assertEqual(len(actual.faces), 2)
        # Open3D deletes the smallest triangle (source face 0).
        np.testing.assert_array_equal(actual.vertices[actual.faces], scene.vertices[scene.faces[1:]])
        np.testing.assert_array_equal(actual.dc[actual.faces], scene.dc[scene.faces[1:]])
        np.testing.assert_array_equal(actual.rest[actual.faces], scene.rest[scene.faces[1:]])
        np.testing.assert_array_equal(actual.textures, scene.textures[1:])

    def test_cpu_camera_defaults_to_full_sh_original_geometry_and_degree_override(self):
        scene = scene_fixture(4)
        scene.rest[:, 1, 0] = .4
        scene.rest[:, 5, 1] = .3
        scene.textures[:] = [-.03, .02, .04]
        self.save_scene(scene)
        full_output, dc_output = self.root / "full.ply", self.root / "dc.ply"
        report = self.run_cli("--cpu", "--camera", "0,0,0", "--out", full_output)
        self.run_cli("--cpu", "--camera", "0,0,0", "--degree", "0", "--out", dc_output)
        positions, full_colors, faces = read_ply(full_output)
        _, dc_colors, _ = read_ply(dc_output)
        direction = scene.vertices / np.linalg.norm(scene.vertices, axis=1, keepdims=True)
        x, y, z = direction.T
        expected = np.column_stack((.5 + C1 * .4 * z,
                                    .5 + C2[2] * .3 * (2 * z * z - x * x - y * y),
                                    np.full(3, .5)))
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        np.testing.assert_allclose(full_colors / 255, expected + [-.03, .02, .04], atol=1 / 255)
        np.testing.assert_allclose(dc_colors / 255, np.tile([.47, .52, .54], (3, 1)), atol=1 / 255)
        self.assertGreater(np.abs(full_colors.astype(int) - dc_colors).max(), 20)
        self.assertEqual(report["color_bake"]["sh_degree"], 2)
        self.assertEqual(report["color_bake"]["mode"], "camera")
        saved = json.loads(full_output.with_name(full_output.name + ".export.json").read_text())
        self.assertEqual(saved["color_bake"]["sh_degree"], 2)

    def test_geometry_only_skips_textures_and_preserves_face_count(self):
        scene = scene_fixture(8, two_faces=True)
        scene.textures[:] = .3
        scene.rest[:, 1] = .4
        self.save_scene(scene)
        output = self.root / "geometry.ply"
        self.run_cli("--cpu", "--geometry-only", "--out", output)
        positions, colors, faces = read_ply(output)
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        np.testing.assert_allclose(colors / 255, .5, atol=1 / 255)

    def test_missing_training_cameras_fails_explicitly_without_dc_fallback(self):
        self.save_scene(scene_fixture(4))
        output = self.root / "no_cameras.ply"
        with self.assertRaisesRegex(FileNotFoundError, "cameras.json"):
            self.run_cli("--out", output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".export.json").exists())

    def test_cli_overwrite_protects_existing_appearance_and_report(self):
        self.save_scene(scene_fixture(4))
        output = self.root / "protected.ply"
        self.run_cli("--cpu", "--camera", "0,0,0", "--out", output)
        before = output.read_bytes(), output.with_name(output.name + ".export.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_cli("--cpu", "--camera", "0,0,0", "--out", output)
        self.assertEqual(before, (output.read_bytes(), output.with_name(output.name + ".export.json").read_bytes()))
        self.run_cli("--cpu", "--camera", "0,0,0", "--out", output, "--overwrite")
        self.assertEqual(before, (output.read_bytes(), output.with_name(output.name + ".export.json").read_bytes()))


class PlyVertexBakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_interior_texels_affect_bake_without_adding_vertices_or_faces(self):
        scene = scene_fixture(4)
        scene.textures[0, 1, 1, 0] = .3
        output = self.root / "interior_vertex.ply"
        write_ply(scene, np.full((3, 3), .5, np.float32), output, texture_mode="vertex")
        positions, colors, faces = read_ply(output)
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        # Independent bilinear/least-squares reference on half-texel samples.
        uv = np.array([(column / 6, row / 6) for row in range(7)
                       for column in range(7 - row)], dtype=np.float32)
        bary = np.column_stack((1 - uv.sum(axis=1), uv))
        grid = torch.from_numpy(2 * uv - 1)[None, None]
        sampled = F.grid_sample(torch.from_numpy(scene.textures).permute(0, 3, 1, 2),
                                grid, align_corners=True)[0, :, 0].T.numpy()
        expected = .5 + np.linalg.lstsq(bary, sampled, rcond=None)[0]
        np.testing.assert_allclose(colors / 255, expected, atol=1 / 255)
        self.assertGreater(colors[:, 0].max(), 128)
        np.testing.assert_allclose(colors[:, 1:] / 255, .5, atol=1 / 255)

    def test_outside_diagonal_texel_bilinear_support_affects_vertex_fit(self):
        scene = scene_fixture(4)
        scene.textures[0, 2, 2, 0] = .8
        # This texel center lies outside u+v<=1, but its bilinear footprint
        # reaches valid interior points. Omitting it silently loses appearance.
        interior_uv = torch.tensor([[[[.45, .45]]]], dtype=torch.float32)
        sampled = F.grid_sample(torch.from_numpy(scene.textures).permute(0, 3, 1, 2),
                                2 * interior_uv - 1, align_corners=True)
        self.assertAlmostEqual(sampled[0, 0, 0, 0].item(), .098, places=6)
        output = self.root / "diagonal_support.ply"
        write_ply(scene, np.full((3, 3), .5, np.float32), output, texture_mode="vertex")
        positions, colors, faces = read_ply(output)
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        self.assertTrue(np.any(colors[:, 0] != 128), "Bilinear support must contribute to fitted colors")
        self.assertGreater(colors[:, 0].max(), 128)
        np.testing.assert_allclose(colors[:, 1:] / 255, .5, atol=1 / 255)

    def test_affine_texture_is_exactly_recovered_at_original_corners(self):
        scene = scene_fixture(8)
        v, u = np.mgrid[:8, :8] / 7
        scene.textures[0] = np.stack((.1 + .3 * u + .4 * v, -.1 + .2 * u - .3 * v,
                                      np.full_like(u, .05)), axis=-1)
        output = self.root / "affine.ply"
        write_ply(scene, np.full((3, 3), .4, np.float32), output, texture_mode="vertex")
        positions, colors, faces = read_ply(output)
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        expected = .4 + np.array([[.1, -.1, .05], [.4, .1, .05], [.5, -.4, .05]])
        np.testing.assert_allclose(colors / 255, expected, atol=1 / 255)

    def test_shared_vertex_residuals_are_area_weighted(self):
        scene = scene_fixture(4, two_faces=True)
        scene.vertices[3, 0] = -2  # Areas: first face .5, second face 1.
        scene.textures[0, ..., 0] = .1
        scene.textures[1, ..., 0] = -.2
        output = self.root / "area_weighted.ply"
        write_ply(scene, np.full((4, 3), .5, np.float32), output, texture_mode="vertex", chunk_faces=1)
        positions, colors, faces = read_ply(output)
        np.testing.assert_array_equal(positions, scene.vertices)
        np.testing.assert_array_equal(faces, scene.faces)
        np.testing.assert_allclose(colors[:, 0] / 255, [.4, .6, .4, .3], atol=1 / 255)
        np.testing.assert_allclose(colors[:, 1:] / 255, .5, atol=1 / 255)


if __name__ == "__main__":
    unittest.main()
