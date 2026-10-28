# Project adaptations maintained by zhtang2004@gmail.com.
"""Experimental training-time selection of an orientable RDT surface.

This is a greedy subset of the RDT candidates, not a watertight reconstruction
or a global collision solver. Vertex coordinates and optimizer state are kept.
"""
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np


@lru_cache(maxsize=1)
def _native():
    # A pure pybind11 extension: avoid importing Torch's build helper after
    # Taichi/Open3D, which can conflict with setuptools' distutils override.
    import fcntl
    import hashlib
    import importlib.util
    import os
    import shlex
    import subprocess
    import sysconfig
    import pybind11
    source = Path(__file__).with_name("rdt_surface_selector.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    name = "_rdt_surface_selector_" + digest
    build = source.resolve().parents[1] / "build" / ("rdt_surface_" + sys.implementation.cache_tag)
    build.mkdir(parents=True, exist_ok=True)
    library = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    with (build / "compile.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix(library.suffix + ".tmp")
            command = shlex.split(os.environ.get("CXX", "c++")) + [
                "-O3", "-std=c++17", "-shared", "-fPIC", "-DTORCH_EXTENSION_NAME=" + name,
                "-I" + pybind11.get_include(), "-I" + sysconfig.get_paths()["include"],
                str(source), "-o", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("RDT selector compilation failed:\n" + result.stderr)
            temporary.replace(library)
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _faces(faces, vertex_count):
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (F, 3) array")
    if vertex_count < 0 or vertex_count >= 2**31 or len(faces) >= 2**31:
        raise ValueError("selector requires vertex and face counts below 2**31")
    if faces.size and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("face index out of range")
    return np.ascontiguousarray(faces, dtype=np.int32)


def select_manifold_faces(faces, vertex_count, scores=None):
    """Return oriented faces, original face IDs and diagnostic counts.

    Nonfinite scores and triangles with repeated vertex IDs are ineligible.
    Grow a component along its score-ordered frontier before seeding another.
    Equal scores use the original candidate index as a deterministic tie break.
    Boundaries and disconnected components are allowed; branching links are not.
    """
    started = time.monotonic()
    faces = _faces(faces, vertex_count)
    scores = np.ones(len(faces), np.float32) if scores is None else np.asarray(scores)
    if scores.shape != (len(faces),):
        raise ValueError("one score per face is required")
    eligible = np.isfinite(scores) & (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    ids = np.flatnonzero(eligible)
    # Pack undirected edges into uint64, retaining the cyclic face-edge order.
    a, b = faces.astype(np.uint64), np.roll(faces, -1, axis=1).astype(np.uint64)
    keys = (np.minimum(a, b) << np.uint64(32)) | np.maximum(a, b)
    del a, b
    _, inverse, counts = np.unique(keys.ravel(), return_inverse=True, return_counts=True)
    del keys
    if len(counts) >= 2**31:
        raise ValueError("selector requires edge count below 2**31")
    edge_faces = (np.argsort(inverse, kind="stable") // 3).astype(np.int32)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    face_edges = inverse.reshape(-1, 3).astype(np.int32)
    del inverse, counts
    order = ids[np.argsort(-scores[ids], kind="stable")].astype(np.int64)
    keep, flip, stats = _native().select_surface(faces, face_edges, order, offsets, edge_faces, int(vertex_count))
    kept_ids = np.flatnonzero(keep)
    selected = faces[kept_ids].copy()
    reverse = flip[kept_ids].astype(bool)
    selected[reverse] = selected[reverse][:, [0, 2, 1]]
    stats.update(input_faces=len(faces), eligible_faces=len(ids), ineligible_faces=len(faces)-len(ids),
                 flipped_faces=int(reverse.sum()), seconds=time.monotonic()-started)
    return selected, kept_ids, stats


def _geometry(vertices, faces):
    p = vertices[faces].astype(np.float64)
    e = p[:, [1, 2, 0]] - p
    squared = (e * e).sum(axis=2)
    normal = np.cross(e[:, 0], -e[:, 2])
    twice_area = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(twice_area[:, None], 1e-30)
    quality = 2 * np.sqrt(3.) * twice_area / np.maximum(squared.sum(axis=1), 1e-30)
    return p.mean(axis=1).astype(np.float32), normal, np.sqrt(squared.max(axis=1)), quality, twice_area


def score_surface_candidates(vertices, input_faces, candidates, *, max_distance_ratio=0.5,
                             min_normal_cos=0.15, min_quality=0.02, chunk_size=200000):
    """Rank by distance/alignment to the input surface and triangle quality.

    Distances are normalized by the closest input triangle's longest edge.
    Alignment is unsigned: the input triangle soup need not have coherent winding.
    The centroid check is a heuristic; it does not certify whole-face coverage.
    """
    import open3d as o3d
    started = time.monotonic()
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite (V, 3) coordinates")
    if not np.isfinite(max_distance_ratio) or max_distance_ratio <= 0 or not 0 <= min_normal_cos <= 1 or not 0 <= min_quality <= 1 or chunk_size <= 0:
        raise ValueError("invalid surface scoring thresholds")
    source = _faces(input_faces, len(vertices))
    candidates = _faces(candidates, len(vertices))
    if not len(source):
        raise ValueError("input surface is empty")
    # Degenerate source triangles are excluded from the closest-point index.
    normals = np.empty((len(source), 3), np.float32)
    scales = np.empty(len(source), np.float32)
    valid = np.empty(len(source), bool)
    for begin in range(0, len(source), chunk_size):
        end = begin + chunk_size
        _, n, scale, _, area = _geometry(vertices, source[begin:end])
        normals[begin:end], scales[begin:end], valid[begin:end] = n, scale, area > 0
    source, normals, scales = source[valid], normals[valid], scales[valid]
    if not len(source):
        raise ValueError("input surface has no nondegenerate triangles")
    ray = o3d.t.geometry.RaycastingScene(nthreads=8)
    ray.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices)),
                      o3d.core.Tensor(np.ascontiguousarray(source, dtype=np.uint32)))
    scores = np.full(len(candidates), -np.inf, np.float32)
    reasons = dict(distance_rejected=0, normal_rejected=0, quality_rejected=0)
    for begin in range(0, len(candidates), chunk_size):
        end = min(begin + chunk_size, len(candidates))
        centers, n, _, quality, area = _geometry(vertices, candidates[begin:end])
        closest = ray.compute_closest_points(o3d.core.Tensor(centers), nthreads=8)
        nearest = closest["primitive_ids"].numpy().astype(np.int64)
        distance = np.linalg.norm(centers - closest["points"].numpy(), axis=1)
        ratio = distance / np.maximum(scales[nearest], 1e-20)
        alignment = np.abs((n * normals[nearest]).sum(axis=1)).clip(0, 1)
        valid = (ratio <= max_distance_ratio) & (alignment >= min_normal_cos) & (quality >= min_quality) & (area > 0)
        score = alignment * np.exp(-4 * ratio) * (0.5 + 0.5 * quality)
        scores[begin:end] = np.where(valid, score, -np.inf)
        reasons["distance_rejected"] += int((ratio > max_distance_ratio).sum())
        reasons["normal_rejected"] += int((alignment < min_normal_cos).sum())
        reasons["quality_rejected"] += int(((quality < min_quality) | (area <= 0)).sum())
    reasons.update(seconds=time.monotonic()-started, max_distance_ratio=max_distance_ratio,
                   min_normal_cos=min_normal_cos, min_quality=min_quality)
    return scores, reasons


def select_rdt_surface(vertices, input_faces, candidates, **kwargs):
    scores, scoring = score_surface_candidates(vertices, input_faces, candidates, **kwargs)
    selected, ids, stats = select_manifold_faces(candidates, len(vertices), scores)
    if len(selected) == 0:
        raise RuntimeError("surface RDT selected no faces; keep the input mesh and inspect thresholds")
    stats["scoring"] = scoring
    print(f"[RDT surface] {len(candidates):,} -> {len(selected):,} faces; {stats}", flush=True)
    return selected, stats
