#!/usr/bin/env python3
"""
Mesh topology statistics: Triangle Connectivity, Vertex-Face Ratio, Vertex Valence.

Usage:
    # From a trained model checkpoint
    python mesh_stats.py -m output/mipnerf360/garden
    python mesh_stats.py -m output/mipnerf360/garden --iteration 21000

    # From a PLY file
    python mesh_stats.py --ply mesh.ply

    # Save results to JSON
    python mesh_stats.py -m output/mipnerf360/garden --save
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_faces_vertices_from_checkpoint(model_path: str, iteration: int = -1):
    """Load triangle indices and vertices from a model checkpoint."""
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        sys.exit(f"point_cloud directory not found: {pc_dir}")

    if iteration == -1:
        best_json = os.path.join(model_path, "best_iteration.json")
        if os.path.exists(best_json):
            with open(best_json) as f:
                iteration = json.load(f).get("iteration", -1)
        if iteration == -1:
            iters = sorted(
                int(d.split("_")[-1])
                for d in os.listdir(pc_dir)
                if d.startswith("iteration_")
            )
            if not iters:
                sys.exit(f"No iteration_* dirs found in {pc_dir}")
            iteration = iters[-1]

    ckpt_path = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud_state_dict.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    faces = state["_triangle_indices"].detach().numpy().astype(np.int64)
    verts = state["triangles_points"].detach().numpy()
    return faces, verts, iteration


def load_faces_vertices_from_ply(ply_path: str):
    """Load triangle indices and vertices from a PLY file."""
    import trimesh
    mesh = trimesh.load(ply_path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return faces, verts


# ── Core Analysis ──────────────────────────────────────────────────────

def compute_triangle_connectivity(faces: np.ndarray):
    """
    For each triangle, count how many distinct adjacent triangles it has
    (sharing at least one edge). Non-manifold edges (3+ faces) contribute
    multiple neighbors, so the count can exceed 3.
    Returns: adjacency count per face [F], and the distribution dict
             with keys 0, 1, 2, 3, "4+".
    """
    F = faces.shape[0]

    e0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    e1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    e2 = np.stack([faces[:, 0], faces[:, 2]], axis=1)

    all_edges = np.concatenate([e0, e1, e2], axis=0)         # [3F, 2]
    all_edges = np.sort(all_edges, axis=1)
    face_ids = np.concatenate([np.arange(F)] * 3)

    edge_keys = all_edges[:, 0].astype(np.int64) * (faces.max() + 1) + all_edges[:, 1]

    sort_idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_idx]
    sorted_fids = face_ids[sort_idx]

    boundaries = np.where(np.diff(sorted_keys) != 0)[0] + 1
    groups_start = np.concatenate([[0], boundaries])
    groups_end = np.concatenate([boundaries, [len(sorted_keys)]])

    adj_count = np.zeros(F, dtype=np.int32)

    for i in range(len(groups_start)):
        s, e = groups_start[i], groups_end[i]
        n_faces = e - s
        if n_faces >= 2:
            fids = sorted_fids[s:e]
            for fid in fids:
                adj_count[fid] += (n_faces - 1)

    tri_conn_dist = {}
    for bucket in [0, 1, 2, 3]:
        tri_conn_dist[str(bucket)] = int((adj_count == bucket).sum())
    tri_conn_dist["4+"] = int((adj_count >= 4).sum())

    return adj_count, tri_conn_dist


def compute_vertex_valence(faces: np.ndarray, V: int):
    """
    For each vertex, count how many edges connect to it.
    Returns: valence per vertex [V], and distribution summary.
    """
    e0 = np.stack([faces[:, 0], faces[:, 1]], axis=1)
    e1 = np.stack([faces[:, 1], faces[:, 2]], axis=1)
    e2 = np.stack([faces[:, 0], faces[:, 2]], axis=1)

    all_edges = np.concatenate([e0, e1, e2], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    unique_edges = np.unique(all_edges, axis=0)

    valence = np.zeros(V, dtype=np.int32)
    np.add.at(valence, unique_edges[:, 0], 1)
    np.add.at(valence, unique_edges[:, 1], 1)

    return valence, unique_edges.shape[0]


def compute_vertex_face_count(faces: np.ndarray, V: int):
    """For each vertex, count how many faces reference it."""
    vf_count = np.zeros(V, dtype=np.int32)
    np.add.at(vf_count, faces[:, 0], 1)
    np.add.at(vf_count, faces[:, 1], 1)
    np.add.at(vf_count, faces[:, 2], 1)
    return vf_count


def percentile_summary(arr, name=""):
    """Return a dict of statistics for a numpy array."""
    return {
        "min": int(arr.min()),
        "p5": int(np.percentile(arr, 5)),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.median(arr)),
        "p75": int(np.percentile(arr, 75)),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def analyze_mesh(faces: np.ndarray, verts: np.ndarray):
    V = verts.shape[0]
    F = faces.shape[0]
    E_euler = V + F - 2  # Euler formula for genus-0

    print("=" * 60)
    print("  MESH TOPOLOGY STATISTICS")
    print("=" * 60)

    # ── 1. Basic counts ──
    print(f"\n[Basic]")
    print(f"  Vertices (V)      : {V:>12,}")
    print(f"  Faces (F)         : {F:>12,}")
    print(f"  V/F Ratio         : {V/F:>12.4f}")
    print(f"  F/V Ratio         : {F/V:>12.4f}")
    print(f"  Euler E (V+F-2)   : {E_euler:>12,}")

    # ── 2. Triangle Connectivity ──
    print(f"\n[Triangle Connectivity]")
    print(f"  (how many edge-neighbors each triangle has)")
    adj_count, tri_conn_dist = compute_triangle_connectivity(faces)
    for k in ["0", "1", "2", "3", "4+"]:
        cnt = tri_conn_dist[k]
        pct = 100.0 * cnt / F
        bar = "#" * int(pct / 2)
        print(f"  {k:>2s} neighbors : {cnt:>10,}  ({pct:5.1f}%)  {bar}")
    adj_mean = float(adj_count.mean())
    tri_conn_dist["mean"] = round(adj_mean, 2)
    print(f"  mean         : {adj_mean:>10.2f}")

    # ── 3. Vertex Valence (edge-degree) ──
    print(f"\n[Vertex Valence (edge degree)]")
    valence, n_unique_edges = compute_vertex_valence(faces, V)
    print(f"  Unique edges (E)  : {n_unique_edges:>12,}")
    vs = percentile_summary(valence)
    print(f"  min={vs['min']}  p5={vs['p5']}  p25={vs['p25']}  "
          f"median={vs['median']}  p75={vs['p75']}  p95={vs['p95']}  max={vs['max']}")
    print(f"  mean={vs['mean']:.2f}  std={vs['std']:.2f}")

    val_dist = Counter(valence.tolist())
    print(f"\n  Top-10 valence values:")
    for k, cnt in val_dist.most_common(10):
        pct = 100.0 * cnt / V
        print(f"    valence {k:>3d} : {cnt:>10,}  ({pct:5.1f}%)")

    isolated = int((valence == 0).sum())
    if isolated > 0:
        print(f"  WARNING: {isolated:,} isolated vertices (valence=0)")

    # ── 4. Vertex-Face count ──
    print(f"\n[Vertex-Face Count (faces per vertex)]")
    vf_count = compute_vertex_face_count(faces, V)
    vfs = percentile_summary(vf_count)
    print(f"  min={vfs['min']}  p5={vfs['p5']}  p25={vfs['p25']}  "
          f"median={vfs['median']}  p75={vfs['p75']}  p95={vfs['p95']}  max={vfs['max']}")
    print(f"  mean={vfs['mean']:.2f}  std={vfs['std']:.2f}")

    print("=" * 60)

    # Collect results for JSON
    results = {
        "vertices": V,
        "faces": F,
        "edges": n_unique_edges,
        "vf_ratio": V / F,
        "triangle_connectivity": tri_conn_dist,
        "vertex_valence": vs,
        "vertex_face_count": vfs,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Mesh topology statistics")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Path to trained model directory")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Checkpoint iteration (-1 = best or latest)")
    parser.add_argument("--ply", type=str, default=None,
                        help="Path to a PLY mesh file")
    parser.add_argument("--save", action="store_true",
                        help="Save results to mesh_stats.json")
    args = parser.parse_args()

    if args.ply:
        print(f"Loading PLY: {args.ply}")
        faces, verts = load_faces_vertices_from_ply(args.ply)
        iteration = None
        out_dir = os.path.dirname(args.ply) or "."
    elif args.model_path:
        faces, verts, iteration = load_faces_vertices_from_checkpoint(
            args.model_path, args.iteration
        )
        out_dir = args.model_path
    else:
        parser.error("Provide either -m MODEL_PATH or --ply PLY_PATH")

    if iteration is not None:
        print(f"Iteration: {iteration}")

    results = analyze_mesh(faces, verts)

    if args.save:
        out_path = os.path.join(out_dir, "mesh_stats.json")
        if iteration is not None:
            results["iteration"] = iteration
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
