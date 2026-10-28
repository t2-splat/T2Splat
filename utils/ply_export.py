"""Stream static SH + per-face texture bakes to ordinary colored PLY meshes.

The caller resolves camera-dependent SH colors and any topology filtering first.
Texture bakes either fit original vertex colors or subdivide faces to retain detail.
This format stores fixed RGB samples, not SH coefficients or dynamic mip LOD.
"""
from __future__ import annotations

import json
import operator
import os
from pathlib import Path
import tempfile

import numpy as np

from utils.glb_export import ExportScene, face_mip, mip_sizes


_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
_FACE_DTYPE = np.dtype([("count", "u1"), ("vertex_indices", "<i4", (3,))])
_INDEX_LIMIT = 2**31


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def subdivision_grid(subdivisions):
    """Return barycentric grid points and consistently wound sub-triangles.

    With s subdivisions per edge, there are (s+1)(s+2)/2 points and s*s
    triangles. Columns move toward vertex 1; rows move toward vertex 2.
    """
    s = _integer(subdivisions, "subdivisions", 1)
    point_count = (s + 1) * (s + 2) // 2
    if point_count >= _INDEX_LIMIT or s * s >= _INDEX_LIMIT:
        raise ValueError("Subdivision grid exceeds signed 32-bit index/count limits")
    barycentric = np.empty((point_count, 3), dtype=np.float32)
    faces = np.empty((s * s, 3), dtype=np.int32)
    row_start = 0
    face_index = 0
    for row in range(s + 1):
        width = s + 1 - row
        columns = np.arange(width)
        barycentric[row_start:row_start + width, 0] = (s - row - columns) / s
        barycentric[row_start:row_start + width, 1] = columns / s
        barycentric[row_start:row_start + width, 2] = row / s
        if row < s:
            next_row = row_start + width
            for column in range(width - 1):
                faces[face_index] = (row_start + column,
                                     row_start + column + 1,
                                     next_row + column)
                face_index += 1
                if column < width - 2:
                    faces[face_index] = (row_start + column + 1,
                                         next_row + column + 1,
                                         next_row + column)
                    face_index += 1
        row_start += width
    return barycentric, faces


def _check_finite(array, name, chunk, float32=False):
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    for start in range(0, len(array), chunk):
        values = array[start:start + chunk]
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        if float32 and (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError(f"{name} contains coordinates outside float32 range")


def _sample_texture(texture, barycentric):
    """Bilinear sampling using the renderer's row=wC, column=wB convention."""
    size = texture.shape[1]
    x = np.clip(barycentric[:, 1].astype(np.float64) * (size - 1), 0, size - 1)
    y = np.clip(barycentric[:, 2].astype(np.float64) * (size - 1), 0, size - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, size - 1), np.minimum(y0 + 1, size - 1)
    dx = (x - x0)[None, :, None]
    dy = (y - y0)[None, :, None]
    return ((1 - dx) * (1 - dy) * texture[:, y0, x0]
            + dx * (1 - dy) * texture[:, y0, x1]
            + (1 - dx) * dy * texture[:, y1, x0]
            + dx * dy * texture[:, y1, x1])


def fit_vertex_texture(scene: ExportScene, colors, mip_level=0, chunk_faces=1024):
    """Approximate signed textures using the existing shared vertex colors.

    Each selected mip is sampled on a triangular grid with two intervals per
    texel interval. Bilinear samples include support from texels just beyond
    the hypotenuse and fit three corner residuals by least squares. Shared
    corners are averaged using source triangle area. The SH base colors are
    never fitted, averaged, or clamped here. Degenerate/unreferenced vertices
    receive zero residual; this approximation cannot preserve arbitrary seams
    or nonlinear appearance within an original face.
    """
    chunk_faces = _integer(chunk_faces, "chunk_faces", 1)
    mip_level = _integer(mip_level, "mip_level", 0)
    vertices, faces, colors = map(np.asarray, (scene.vertices, scene.faces, colors))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must have nonempty shape [V,3]")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in "iu"):
        raise ValueError("faces must have nonempty integer shape [F,3]")
    if colors.shape != vertices.shape:
        raise ValueError("Baked vertex colors must have shape [V,3]")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle indices exceed the vertex array")
    _check_finite(vertices, "vertices", chunk_faces, float32=True)
    _check_finite(colors, "baked vertex colors", chunk_faces)
    texture = None if scene.textures is None else np.asarray(scene.textures)
    resolution = 1
    if texture is not None:
        if (texture.ndim != 4 or texture.shape[0] != len(faces)
                or texture.shape[-1] != 3 or texture.shape[1] < 1
                or texture.shape[1] != texture.shape[2]):
            raise ValueError("face_textures must have shape [F,R,R,3]")
        resolution = texture.shape[1]
        _check_finite(texture, "face_textures", chunk_faces)
    sizes = mip_sizes(resolution)
    if mip_level >= len(sizes):
        raise ValueError(f"mip_level must be between 0 and {len(sizes) - 1}")
    size = sizes[mip_level]
    sample_subdivisions = 2 * (size - 1)
    sample_count = (sample_subdivisions + 1) * (sample_subdivisions + 2) // 2
    stats = {
        "mode": "vertex", "geometry": "original vertices and faces preserved",
        "method": "linear least-squares corner residual fit using bilinear samples on a triangular grid with two intervals per texel interval, including support across the hypotenuse; source-face-area averaging at shared vertices",
        "appearance": "static approximation; nonlinear per-face detail and discontinuous color seams may be lost",
        "source_texture_resolution": resolution, "baked_mip_level": mip_level,
        "baked_mip_resolution": size, "samples_per_face": sample_count,
        "sampling_intervals_per_texel_interval": 2,
        "sampling_subdivisions": sample_subdivisions,
        "sh_base": "preserved without averaging or pre-clamping its upper range",
        "residual_clipping": "none during fitting or averaging; final SH plus residual is clipped only on RGB serialization",
    }
    result = colors.astype(np.float64, copy=True)
    if texture is None:
        stats.update(affected_vertices=0, zero_area_faces=0,
                     vertices_without_positive_area=0, samples_per_face=0)
        return result, stats
    if size > 1:
        axis = np.arange(sample_subdivisions + 1)
        rows, columns = np.nonzero(np.add.outer(axis, axis) <= sample_subdivisions)
        barycentric = np.column_stack((sample_subdivisions - rows - columns, columns, rows)).astype(np.float64) / sample_subdivisions
        inverse = np.linalg.pinv(barycentric)
    else:
        # One texel is a constant residual on all three corners, rather than
        # an underdetermined one-point linear regression.
        barycentric = inverse = None
    residual_sum = np.zeros_like(result)
    area_sum = np.zeros(len(vertices), dtype=np.float64)
    zero_area_faces = 0
    for start in range(0, len(faces), chunk_faces):
        stop = min(start + chunk_faces, len(faces))
        indices = faces[start:stop]
        points = vertices[indices].astype(np.float64)
        area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                       points[:, 2] - points[:, 0]), axis=1) * .5
        zero_area_faces += int(np.count_nonzero(area == 0))
        native = face_mip(texture[start:stop].astype(np.float64), mip_level)
        if size == 1:
            corners = np.broadcast_to(native[:, 0, 0, None, :], (len(indices), 3, 3))
        else:
            samples = _sample_texture(native, barycentric)
            corners = np.einsum("kq,fqc->fkc", inverse, samples, optimize=True)
        flat = indices.reshape(-1)
        np.add.at(residual_sum, flat, (corners * area[:, None, None]).reshape(-1, 3))
        np.add.at(area_sum, flat, np.repeat(area, 3))
    observed = area_sum > 0
    result[observed] += residual_sum[observed] / area_sum[observed, None]
    stats.update(affected_vertices=int(np.count_nonzero(observed)),
                 zero_area_faces=zero_area_faces,
                 vertices_without_positive_area=int(np.count_nonzero(~observed)))
    return result, stats


def _write_vertices(stream, positions, colors, zup):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    clipped = int(np.count_nonzero((colors < 0) | (colors > 1)))
    rgb = np.floor(np.clip(colors, 0, 1) * 255 + .5).astype(np.uint8)
    records = np.empty(len(positions), dtype=_VERTEX_DTYPE)
    records["x"] = positions[:, 0]
    if zup:
        records["y"], records["z"] = positions[:, 2], -positions[:, 1]
    else:
        records["y"], records["z"] = positions[:, 1], positions[:, 2]
    for index, channel in enumerate(("red", "green", "blue")):
        records[channel] = rgb[:, index]
    stream.write(records.tobytes())
    return clipped


def _write_faces(stream, faces):
    faces = faces.reshape(-1, 3)
    records = np.empty(len(faces), dtype=_FACE_DTYPE)
    records["count"] = 3
    records["vertex_indices"] = faces
    stream.write(records.tobytes())


def write_ply(scene: ExportScene, colors, out, *, texture_mode="vertex", subdivisions=None, mip_level=0,
              chunk_faces=1024, zup=False, overwrite=False, bake_info=None):
    """Export static appearance with bounded per-face working memory.

    ``colors`` are the already evaluated, lower-clamped SH colors [V,3].
    They must retain values above one until signed texture residuals are added.
    The default fits texture residuals to original shared vertices without
    changing vertex/face indexing. Opt into texture_mode="subdivide" to sample
    the native R by R triangular grid (s=R-1) for textured meshes. Explicit s>1
    in subdivision mode also allows subdividing an untextured mesh.
    """
    if texture_mode not in ("subdivide", "vertex"):
        raise ValueError("texture_mode must be subdivide or vertex")
    out = Path(out).expanduser().resolve()
    report_path = out.with_name(out.name + ".export.json")
    if out.suffix.lower() != ".ply":
        raise ValueError("--out must end in .ply")
    if not overwrite and (out.exists() or report_path.exists()):
        raise FileExistsError(f"Output already exists; use --overwrite: {out}")
    if out.is_dir() or report_path.is_dir():
        raise ValueError("PLY output and export report must be files")
    chunk_faces = _integer(chunk_faces, "chunk_faces", 1)
    mip_level = _integer(mip_level, "mip_level", 0)
    vertices, faces, colors = map(np.asarray, (scene.vertices, scene.faces, colors))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must have nonempty shape [V,3]")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in "iu"):
        raise ValueError("faces must have nonempty integer shape [F,3]")
    if colors.shape != vertices.shape:
        raise ValueError("Baked vertex colors must have shape [V,3]")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle indices exceed the vertex array")
    texture = None if scene.textures is None else np.asarray(scene.textures)
    resolution = 1
    if texture is not None:
        if (texture.ndim != 4 or texture.shape[0] != len(faces)
                or texture.shape[-1] != 3 or texture.shape[1] < 1
                or texture.shape[1] != texture.shape[2]):
            raise ValueError("face_textures must have shape [F,R,R,3]")
        resolution = texture.shape[1]
    sizes = mip_sizes(resolution)
    if mip_level >= len(sizes):
        raise ValueError(f"mip_level must be between 0 and {len(sizes) - 1}")
    if subdivisions is not None:
        subdivisions = _integer(subdivisions, "subdivisions", 1)
    if texture_mode == "vertex" and subdivisions not in (None, 1):
        raise ValueError("subdivisions > 1 requires texture_mode='subdivide'")
    s = (1 if texture_mode == "vertex" else
         max(1, resolution - 1) if subdivisions is None else subdivisions)
    split_faces = texture_mode == "subdivide" and (texture is not None or s > 1)
    points_per_face = (s + 1) * (s + 2) // 2 if split_faces else None
    vertex_count = len(faces) * points_per_face if split_faces else len(vertices)
    face_count = len(faces) * s * s if split_faces else len(faces)
    if vertex_count >= _INDEX_LIMIT or face_count >= _INDEX_LIMIT:
        raise ValueError("Export exceeds signed 32-bit PLY vertex/face limits; reduce subdivisions")
    _check_finite(vertices, "vertices", chunk_faces, float32=True)
    _check_finite(colors, "baked vertex colors", chunk_faces)
    if texture is not None:
        _check_finite(texture, "face_textures", chunk_faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment T2-Splat static SH and per-face texture bake\n"
        "comment Fixed RGB colors; no dynamic SH or mip LOD\n"
        f"element vertex {vertex_count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    estimated_bytes = len(header) + vertex_count * _VERTEX_DTYPE.itemsize + face_count * _FACE_DTYPE.itemsize
    report = {
        "format": "t2splat-static-ply/1", "source_vertices": len(vertices),
        "source_faces": len(faces), "vertices": vertex_count, "faces": face_count,
        "exported_vertices": vertex_count, "exported_faces": face_count,
        "texture_mode": texture_mode, "subdivisions": s, "points_per_source_face": points_per_face,
        "triangles_per_source_face": s * s if split_faces else 1,
        "has_face_textures": texture is not None, "source_texture_resolution": resolution,
        "source_uses_mip": bool(scene.use_mip), "native_mip_sizes": sizes,
        "baked_mip_level": mip_level, "baked_mip_resolution": sizes[mip_level],
        "zup_to_yup": bool(zup), "split_vertices_at_face_seams": split_faces,
        "color_bake": bake_info or {}, "estimated_bytes": estimated_bytes,
        "color_formula": "interpolated max(SH(direction)+0.5,0) + signed face texture; clamp/quantize last",
        "texture_coordinates": "row=wC, column=wB; bilinear with align-corners coordinates",
        "geometry": ("source faces subdivided in place; independent vertices across original face seams"
                     if split_faces else "original vertex and face indexing preserved"),
        "appearance": "static RGB samples; view-dependent SH, dynamic mip LOD and soft opacity blending are not preserved",
        "mip_policy": "one selected native per-face mip level is baked into vertex RGB",
    }
    # Reject non-serializable/non-finite caller metadata before opening outputs.
    json.dumps(report, allow_nan=False)
    print(f"PLY bake: {len(vertices):,}/{len(faces):,} source vertices/faces -> "
          f"{vertex_count:,}/{face_count:,}; approximately {estimated_bytes / 1e9:.3f} GB", flush=True)
    barycentric = grid_faces = None
    if texture_mode == "vertex":
        colors, fit_stats = fit_vertex_texture(scene, colors, mip_level, chunk_faces)
        report["texture_bake"] = fit_stats
        report["color_formula"] = "max(SH(direction)+0.5,0) + area-averaged per-face least-squares signed residual fit; clamp/quantize last"
        report["appearance"] = "static vertex-color approximation; nonlinear face detail, seams, view-dependent SH, dynamic mip LOD and soft opacity blending are not preserved"
    if split_faces:
        barycentric, grid_faces = subdivision_grid(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped = 0
    with tempfile.TemporaryDirectory(prefix=".t2splat-ply-", dir=out.parent) as temporary:
        candidate = Path(temporary) / "mesh.ply"
        candidate_report = Path(temporary) / "report.json"
        with candidate.open("wb") as stream:
            stream.write(header)
            if split_faces:
                for start in range(0, len(faces), chunk_faces):
                    stop = min(start + chunk_faces, len(faces))
                    indices = faces[start:stop]
                    points = np.einsum("qk,fkc->fqc", barycentric,
                                       vertices[indices].astype(np.float64), optimize=True)
                    samples = np.einsum("qk,fkc->fqc", barycentric,
                                        colors[indices].astype(np.float64), optimize=True)
                    if texture is not None:
                        native = face_mip(texture[start:stop].astype(np.float64), mip_level)
                        samples += _sample_texture(native, barycentric)
                    clipped += _write_vertices(stream, points, samples, zup)
                for start in range(0, len(faces), chunk_faces):
                    stop = min(start + chunk_faces, len(faces))
                    offsets = np.arange(start, stop, dtype=np.int64) * points_per_face
                    indices = grid_faces[None] + offsets[:, None, None]
                    _write_faces(stream, indices)
            else:
                for start in range(0, len(vertices), chunk_faces):
                    clipped += _write_vertices(stream, vertices[start:start + chunk_faces],
                                               colors[start:start + chunk_faces], zup)
                for start in range(0, len(faces), chunk_faces):
                    _write_faces(stream, faces[start:start + chunk_faces])
        if candidate.stat().st_size != estimated_bytes:
            raise RuntimeError("PLY payload size does not match the declared vertex/face counts")
        report.update(bytes=estimated_bytes, clipped_color_values=clipped,
                      total_color_values=3 * vertex_count,
                      clipped_fraction=clipped / (3 * vertex_count))
        candidate_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if overwrite:
            os.replace(candidate, out)
            os.replace(candidate_report, report_path)
        else:
            # Hard-link publication is atomic and cannot replace a concurrently
            # created output. Both candidates are on the destination filesystem.
            os.link(candidate, out)
            try:
                os.link(candidate_report, report_path)
            except BaseException:
                if out.stat().st_ino == candidate.stat().st_ino:
                    out.unlink()
                raise
    return report
