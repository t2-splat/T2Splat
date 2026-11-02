"""Check benchmark PLY loading without creating an OpenGL context."""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from tools.mesh_fps_benchmark import read_ply


_PROPERTIES = (
    ("float", "x"), ("float", "y"), ("float", "z"),
    ("uchar", "red"), ("uchar", "green"), ("uchar", "blue"),
)
_VERTICES = (
    dict(x=0., y=0., z=2., red=255, green=0, blue=20),
    dict(x=1., y=0., z=2., red=0, green=128, blue=40),
    dict(x=0., y=1., z=2., red=10, green=30, blue=255),
)


def ply_fixture(*, properties=_PROPERTIES, vertices=None,
                faces=((0, 1, 2),), encoding="binary_little_endian"):
    """Encode a small PLY independently of the reader and export helpers."""
    if vertices is None:
        vertices = _VERTICES
    lines = ["ply", f"format {encoding} 1.0", "comment benchmark test",
             f"element vertex {len(vertices)}"]
    lines.extend(f"property {kind} {name}" for kind, name in properties)
    lines.extend((f"element face {len(faces)}",
                  "property list uchar int vertex_indices", "end_header"))
    header = ("\n".join(lines) + "\n").encode("ascii")
    if encoding == "ascii":
        rows = [" ".join(str(vertex[name]) for _, name in properties)
                for vertex in vertices]
        rows.extend(" ".join(map(str, (len(face), *face))) for face in faces)
        return header + ("\n".join(rows) + "\n").encode("ascii"), len(header)

    order = ">" if encoding == "binary_big_endian" else "<"
    formats = {"float": "f", "double": "d", "uchar": "B", "int": "i"}
    vertex_format = order + "".join(formats[kind] for kind, _ in properties)
    records = [struct.pack(vertex_format,
                           *(vertex[name] for _, name in properties))
               for vertex in vertices]
    records.extend(struct.pack(order + "B" + "i" * len(face), len(face), *face)
                   for face in faces)
    return header + b"".join(records), len(header)


class BenchmarkPlyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "mesh.ply"

    def load(self, data):
        self.path.write_bytes(data)
        return read_ply(self.path)

    def test_rgb_export_layout_adds_opaque_alpha(self):
        data, header_bytes = ply_fixture()
        vertices, indices, metadata = self.load(data)
        np.testing.assert_array_equal(vertices["position"],
                                      [[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]])
        np.testing.assert_array_equal(vertices["color"],
                                      [[255, 0, 20, 255], [0, 128, 40, 255],
                                       [10, 30, 255, 255]])
        self.assertEqual(vertices["position"].dtype, np.dtype("float32"))
        self.assertEqual(vertices["color"].dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(indices, [0, 1, 2])
        self.assertEqual(indices.dtype, np.dtype("uint32"))
        self.assertEqual(metadata["vertex_count"], 3)
        self.assertEqual(metadata["triangle_count"], 1)
        self.assertEqual(metadata["header_bytes"], header_bytes)

    def test_rgba_preserves_alpha(self):
        vertices = [dict(vertex, alpha=alpha)
                    for vertex, alpha in zip(_VERTICES, (255, 128, 0))]
        data, _ = ply_fixture(properties=_PROPERTIES + (("uchar", "alpha"),),
                              vertices=vertices)
        loaded, _, _ = self.load(data)
        np.testing.assert_array_equal(loaded["color"][:, 3], [255, 128, 0])

    def test_extra_scalar_and_reordered_properties(self):
        properties = (("uchar", "blue"), ("float", "x"),
                      ("double", "confidence"), ("float", "y"),
                      ("uchar", "green"), ("float", "z"), ("uchar", "red"))
        data, _ = ply_fixture(properties=properties,
                              vertices=[dict(v, confidence=.75) for v in _VERTICES])
        loaded, indices, _ = self.load(data)
        np.testing.assert_array_equal(loaded["position"],
                                      [[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]])
        np.testing.assert_array_equal(loaded["color"][:, :3],
                                      [[255, 0, 20], [0, 128, 40], [10, 30, 255]])
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_rejects_unsupported_encodings(self):
        for encoding in ("ascii", "binary_big_endian"):
            with self.subTest(encoding=encoding):
                data, _ = ply_fixture(encoding=encoding)
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_rejects_missing_coordinate_or_color(self):
        for missing in ("z", "blue"):
            with self.subTest(missing=missing):
                properties = tuple(p for p in _PROPERTIES if p[1] != missing)
                data, _ = ply_fixture(properties=properties)
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_rejects_empty_geometry(self):
        for vertices, faces in (([], []), (_VERTICES, [])):
            with self.subTest(vertices=len(vertices), faces=len(faces)):
                data, _ = ply_fixture(vertices=vertices, faces=faces)
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_rejects_nonfinite_coordinates(self):
        for coordinate in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(coordinate=coordinate):
                vertices = [dict(v) for v in _VERTICES]
                vertices[1]["x"] = coordinate
                data, _ = ply_fixture(vertices=vertices)
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_rejects_nontriangular_faces(self):
        for face in ((0, 1), (0, 1, 2, 0)):
            with self.subTest(face=face):
                data, _ = ply_fixture(faces=(face,))
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_rejects_negative_or_out_of_range_indices(self):
        for face in ((-1, 1, 2), (0, 1, 3)):
            with self.subTest(face=face):
                data, _ = ply_fixture(faces=(face,))
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_rejects_truncated_data(self):
        data, header_bytes = ply_fixture()
        for label, truncated in (("header", b"ply\n"),
                                 ("vertices", data[:header_bytes + 5]),
                                 ("faces", data[:-1])):
            with self.subTest(section=label):
                with self.assertRaises(ValueError):
                    self.load(truncated)


if __name__ == "__main__":
    unittest.main()
