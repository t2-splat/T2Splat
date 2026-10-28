# Project adaptations maintained by zhtang2004@gmail.com.
"""Local topology guards and event diagnostics for training-time mesh edits.

These checks preserve valid local surface topology; they do not repair an
already non-manifold RDT mesh or detect global triangle intersections.
"""
import json
from pathlib import Path
import time

import numpy as np


def _surface_link(faces, vertex):
    """Return (neighbors, link edges, is_boundary), or None for a bad fan."""
    if len(faces) == 0 or np.any((faces == vertex).sum(axis=1) != 1):
        return None
    opposite = faces[faces != vertex].reshape(-1, 2)
    graph = {}
    link_edges = set()
    for a, b in opposite:
        a, b = int(a), int(b)
        edge = (min(a, b), max(a, b))
        if a == b or edge in link_edges:
            return None  # Degenerate or duplicate face, including reverse winding.
        link_edges.add(edge)
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    degree = [len(n) for n in graph.values()]
    if any(d not in (1, 2) for d in degree) or degree.count(1) not in (0, 2):
        return None
    visited, pending = set(), [next(iter(graph))]
    while pending:
        v = pending.pop()
        if v not in visited:
            visited.add(v)
            pending.extend(graph[v] - visited)
    if len(visited) != len(graph):
        return None  # Bow-tie vertex: more than one disconnected fan.
    return set(graph), link_edges, degree.count(1) == 2


class CollapseSafety:
    """Read-only vertex stars, reused within a batch of disjoint local edits."""

    def __init__(self, xyz, faces):
        self.xyz = xyz
        self.faces = faces
        flat = faces.ravel()
        self.face_order = np.argsort(flat, kind="stable") // 3
        self.offsets = np.r_[0, np.cumsum(np.bincount(flat, minlength=len(xyz)))]
        self.links = {}

    def incident(self, vertex):
        return self.face_order[self.offsets[vertex]:self.offsets[vertex + 1]]

    def link(self, vertex):
        if vertex not in self.links:
            self.links[vertex] = _surface_link(self.faces[self.incident(vertex)], vertex)
        return self.links[vertex]

    def check(self, u, v, position, blocked):
        """Return affected vertices if valid, else None. No arrays are mutated."""
        lu, lv = self.link(u), self.link(v)
        if lu is None or lv is None or u == v:
            return None
        neighborhood = lu[0] | lv[0] | {u, v}
        if neighborhood & blocked:
            return None
        face_ids = np.union1d(self.incident(u), self.incident(v))
        before = self.faces[face_ids]
        removed = np.any(before == u, axis=1) & np.any(before == v, axis=1)
        edge_faces = before[removed]
        if len(edge_faces) not in (1, 2):
            return None
        opposite = set(map(int, edge_faces[(edge_faces != u) & (edge_faces != v)]))
        # Full simplicial link condition includes link EDGES. Only comparing
        # neighbor vertices would accept a tetrahedron -> two duplicate faces.
        if lu[0] & lv[0] != opposite or lu[1] & lv[1]:
            return None
        # An interior chord joining two boundary vertices can pinch a disk.
        if lu[2] and lv[2] and len(edge_faces) != 1:
            return None
        surviving = before[~removed]
        after = np.where(surviving == v, u, surviving)
        if _surface_link(after, u) is None:
            return None  # Also prevents deleting the entire local component.
        old_xyz = self.xyz[surviving].astype(np.float64)
        new_xyz = old_xyz.copy()
        new_xyz[(surviving == u) | (surviving == v)] = position
        if not np.isfinite(new_xyz).all():
            return None
        edge_length = np.linalg.norm(self.xyz[u].astype(np.float64) - self.xyz[v])
        if (edge_length <= 0 or
                np.max(np.linalg.norm(position - self.xyz[[u, v]], axis=1)) > 2 * edge_length):
            return None  # Reject remote QEM minima, even on a short edge.
        old_normal = np.cross(old_xyz[:, 1] - old_xyz[:, 0], old_xyz[:, 2] - old_xyz[:, 0])
        new_normal = np.cross(new_xyz[:, 1] - new_xyz[:, 0], new_xyz[:, 2] - new_xyz[:, 0])
        old_area = np.linalg.norm(old_normal, axis=1)
        new_area = np.linalg.norm(new_normal, axis=1)
        if (np.any(old_area <= 0) or np.any(new_area <= 1e-6 * old_area)
                or np.any(np.einsum("ij,ij->i", old_normal, new_normal) <= 0)):
            return None  # No local face inversion or geometric degeneration.
        return neighborhood


def mesh_topology_metrics(vertices, faces):
    """Exact combinatorial counts, allowing boundaries; no self-intersection test.

    Invalid-index and repeated-index faces are reported and excluded from the
    edge/fan graphs. Reverse-wound duplicates count as duplicate faces.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    xyz = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n_vertices, n_faces = len(xyz), len(faces)
    good_index = np.all((faces >= 0) & (faces < n_vertices), axis=1)
    distinct = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                & (faces[:, 2] != faces[:, 0]))
    f = faces[good_index & distinct]
    result = dict(vertices=n_vertices, faces=n_faces,
                  invalid_index_faces=int((~good_index).sum()),
                  repeated_index_faces=int((~distinct).sum()),
                  nonfinite_vertices=int((~np.isfinite(xyz).all(axis=1)).sum()),
                  analyzed_faces=len(f))
    result.update(duplicate_faces=0, edges=0, boundary_edges=0,
                  nonmanifold_edges=0, nonmanifold_vertices=0,
                  disconnected_fan_vertices=0, winding_conflict_edges=0,
                  face_components=0, largest_component_faces=0,
                  unused_vertices=n_vertices, zero_area_faces=0,
                  nonfinite_faces=0, triangle_quality_p01=None,
                  triangle_quality_median=None)
    if not len(f):
        return result
    result["unused_vertices"] = n_vertices - len(np.unique(f))
    result["duplicate_faces"] = len(f) - len(np.unique(np.sort(f, axis=1), axis=0))
    tri = xyz[f].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    finite = np.isfinite(tri).all(axis=(1, 2))
    result["nonfinite_faces"] = int((~finite).sum())
    result["zero_area_faces"] = int(((area2 == 0) & finite).sum())
    # 1 for an equilateral triangle, 0 for a collapsed one (scale invariant).
    lengths2 = np.square(tri - np.roll(tri, 1, axis=1)).sum(axis=(1, 2))
    quality = np.zeros(len(f), dtype=np.float64)
    np.divide(2 * np.sqrt(3) * area2, lengths2, out=quality, where=lengths2 > 0)
    if finite.any():
        result["triangle_quality_p01"], result["triangle_quality_median"] = map(
            float, np.quantile(quality[finite], [0.01, 0.5]))
    del tri, cross, area2, lengths2, quality

    src, dst = f.ravel(), f[:, [1, 2, 0]].ravel()
    low, high = np.minimum(src, dst), np.maximum(src, dst)
    keys = low * max(n_vertices, 1) + high
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    result["edges"] = len(starts)
    result["boundary_edges"] = int((counts == 1).sum())
    bad_edges = order[starts[counts > 2]]
    result["nonmanifold_edges"] = len(bad_edges)
    two = starts[counts == 2]
    a, b = order[two], order[two + 1]
    result["winding_conflict_edges"] = int(((src[a] < dst[a]) == (src[b] < dst[b])).sum())

    shared = sorted_keys[1:] == sorted_keys[:-1]
    a, b = order[:-1][shared], order[1:][shared]
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a // 3, b // 3)),
                       shape=(len(f), len(f))).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    result["face_components"] = int(n_components)
    result["largest_component_faces"] = int(np.bincount(labels).max())
    del graph, labels

    # Face-corner components connect around the SAME vertex across shared
    # edges. Multiple components at one vertex expose bow-tie fans even
    # when every edge has at most two incident faces.
    def corner_ids(raw):
        base, c = (raw // 3) * 3, raw % 3
        nxt = (c + 1) % 3
        forward = src[raw] < dst[raw]
        return base + np.where(forward, c, nxt), base + np.where(forward, nxt, c)
    al, ah = corner_ids(a)
    bl, bh = corner_ids(b)
    row, col = np.r_[al, ah], np.r_[bl, bh]
    graph = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)),
                       shape=(3 * len(f), 3 * len(f))).tocsr()
    _, labels = connected_components(graph, directed=False)
    _, first_corner = np.unique(labels, return_index=True)
    fans = np.bincount(src[first_corner], minlength=n_vertices)
    bad_vertices = fans > 1
    result["disconnected_fan_vertices"] = int(bad_vertices.sum())
    bad_vertices[low[bad_edges]] = True
    bad_vertices[high[bad_edges]] = True
    result["nonmanifold_vertices"] = int(bad_vertices.sum())
    return result


class TopologyMonitor:
    """Append event metrics to the run directory without retaining meshes."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, iteration, stage, vertices, faces):
        start = time.perf_counter()
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        if hasattr(faces, "detach"):
            faces = faces.detach().cpu().numpy()
        metrics = mesh_topology_metrics(vertices, faces)
        metrics.update(iteration=int(iteration), stage=stage,
                       elapsed_seconds=time.perf_counter() - start)
        with self.path.open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(f"[Topology:{stage}] iter={iteration} F={metrics['faces']} "
              f"nonmanifold E/V={metrics['nonmanifold_edges']}/{metrics['nonmanifold_vertices']} "
              f"duplicates={metrics['duplicate_faces']} boundary={metrics['boundary_edges']}")
        return metrics
