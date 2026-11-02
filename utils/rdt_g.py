"""CPU face selection with configurable topology constraints."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np


@lru_cache(maxsize=1)
def native():
    """Build the standalone pybind11 selector in the ignored project build dir."""
    import fcntl
    import hashlib
    import importlib.util
    import os
    import shlex
    import subprocess
    import sysconfig
    import pybind11

    source = Path(__file__).with_name("rdt_g_selector.cpp")
    name = "_rdt_g_selector_" + hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build = source.resolve().parents[1] / "build" / ("rdt_g_" + sys.implementation.cache_tag)
    build.mkdir(parents=True, exist_ok=True)
    library = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    with (build / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_name(library.name + ".tmp")
            command = shlex.split(os.environ.get("CXX", "c++")) + [
                "-O3", "-std=c++17", "-shared", "-fPIC", "-DTORCH_EXTENSION_NAME=" + name,
                "-I" + pybind11.get_include(), "-I" + sysconfig.get_paths()["include"],
                str(source), "-o", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("G RDT selector compilation failed:\n" + result.stderr)
            temporary.replace(library)
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_inputs(faces, vertex_count, scores):
    faces = np.asarray(faces)
    scores = np.asarray(scores)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Faces must be integer [F,3]')
    if not 0 <= vertex_count < 2**31 or len(faces) >= 2**31:
        raise ValueError('Face/vertex counts exceed native index range')
    if faces.size and (faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError('Vertex index out of range')
    if scores.shape != (len(faces),):
        raise ValueError('One score per candidate required')
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    eligible = (np.isfinite(scores) & (faces[:, 0] != faces[:, 1])
                & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0]))
    ids = np.flatnonzero(eligible)
    a = faces.astype(np.uint64)
    b = np.roll(faces, -1, axis=1).astype(np.uint64)
    keys = (np.minimum(a, b) << np.uint64(32)) | np.maximum(a, b)
    del a, b
    _, inverse, counts = np.unique(keys.ravel(), return_inverse=True, return_counts=True)
    if len(counts) >= 2**31:
        raise ValueError('Edge count exceeds native index range')
    edge_faces = (np.argsort(inverse, kind='stable') // 3).astype(np.int32)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    face_edges = inverse.reshape(-1, 3).astype(np.int32)
    order = ids[np.argsort(-scores[ids], kind='stable')].astype(np.int64)
    return faces, face_edges, order, offsets, edge_faces, int(vertex_count)


def select_faces(faces, vertex_count, scores, *, max_edge_faces=2, enforce_vertex_fan=True, enforce_orientation=True):
    """Return original-winding faces, ascending candidate IDs, stats and would-flip mask.

    enforce_orientation=True preserves the original first-owner parity rule.
    False skips orientation rejection and all DSU work. With edge cap>2 the
    enabled rule is an owner-star heuristic, not a manifold guarantee.
    Duplicate canonical faces are forbidden independently of the edge cap.
    """
    if isinstance(max_edge_faces, bool) or not isinstance(max_edge_faces, int) or not 1 <= max_edge_faces <= 254:
        raise ValueError('max_edge_faces must be an integer in [1,254]')
    if type(enforce_vertex_fan) is not bool or type(enforce_orientation) is not bool:
        raise ValueError('fan and orientation switches must be boolean')
    arrays = prepare_inputs(faces, vertex_count, scores)
    keep, flip, stats = native().select_surface(*arrays, max_edge_faces, enforce_vertex_fan, enforce_orientation)
    ids = np.flatnonzero(keep).astype(np.int64)
    selected = np.asarray(faces, dtype=np.int64)[ids].copy()
    would_flip = flip[ids].astype(bool)
    stats.update(input_faces=len(faces), eligible_faces=len(arrays[2]),
                 ineligible_faces=len(faces) - len(arrays[2]),
                 flipped_faces=0, computed_flipped_faces=int(would_flip.sum()),
                 max_edge_faces=max_edge_faces, enforce_vertex_fan=enforce_vertex_fan,
                 apply_orientation=False, enforce_orientation=enforce_orientation,
                 orientation_guard='owner_parity' if enforce_orientation else 'disabled')
    return selected, ids, stats, would_flip
