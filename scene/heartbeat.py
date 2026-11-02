# Repository maintenance and English documentation: zhtang2004@gmail.com.
"""Dynamic topology editing after restricted Delaunay reconstruction.

A. Accumulate depth-compensated vertex-gradient statistics after backward.
B. Collapse low-cost edges and split high-gradient vertices using shared
   midpoints, with optional local topology and geometry guards.
C. Transfer area-weighted Adam moments and apply paired RMS-scaled split
   perturbations before replacing optimizer parameter objects."""

import torch
import numpy as np
from torch import nn
from utils.mesh_topology import CollapseSafety



# =====================================================================
# Phase A: gradient statistics
# =====================================================================

def init_heartbeat_stats(model):
    """Initialize per-vertex accumulated gradient magnitudes and observation counts.
    
    _hb_grad_accum and _hb_obs_count both have shape [V]."""
    V = model.vertices.shape[0]
    device = model.vertices.device
    model._hb_grad_accum = torch.zeros(V, device=device, dtype=torch.float32)
    model._hb_obs_count = torch.zeros(V, device=device, dtype=torch.long)


def accumulate_pm_stats(model, viewpoint_camera):
    """Accumulate perspective-compensated gradients after backward, before step.
    
    Multiply the 3D vertex-gradient norm by camera-space depth. Depth uses the
    transposed world_view_transform layout: m[2]*x + m[6]*y + m[10]*z + m[14],
    matching the renderer's transform_point_4x3. Reset buffers if vertex count
    changed during another prune or densification operation."""
    if model.vertices.grad is None:
        return

    with torch.no_grad():
        grad = model.vertices.grad                      # [V, 3]
        grad_norm = grad.norm(dim=-1)                    # [V]

        # World-to-camera depth
        W2C = viewpoint_camera.world_view_transform      # [4,4]
        m = W2C.flatten()
        verts = model.vertices                            # [V, 3]
        depth = (m[2]  * verts[:, 0] +
                 m[6]  * verts[:, 1] +
                 m[10] * verts[:, 2] + m[14])
        depth = depth.clamp(min=0.1)                      # Avoid unstable values near the camera plane.

        compensated = grad_norm * depth                   # Perspective compensation

        # Reinitialize if another prune/densify operation changed the vertex count.
        V_cur = model.vertices.shape[0]
        if (not hasattr(model, '_hb_grad_accum') or
                model._hb_grad_accum.shape[0] != V_cur):
            init_heartbeat_stats(model)

        model._hb_grad_accum.add_(compensated)
        model._hb_obs_count.add_(1)


def get_vertex_importance(model):
    """Return mean depth-compensated gradient magnitudes [V] for splitting."""
    if not hasattr(model, '_hb_grad_accum'):
        return torch.zeros(model.vertices.shape[0], device=model.vertices.device)
    count = model._hb_obs_count.clamp(min=1).float()
    return model._hb_grad_accum / count


# =====================================================================
# Phase B: batched NumPy topology edits
# =====================================================================

# B.0: utilities

def _edge_key(u, v):
    """Encode undirected edges as int64 keys: (min(u,v) << 32) | max(u,v).
    
    Sorted keys support vectorized searchsorted lookups."""
    lo = np.minimum(u, v).astype(np.int64)
    hi = np.maximum(u, v).astype(np.int64)
    return (lo << 32) | hi


# B.1: vectorized edge extraction

def extract_edges(indices_np):
    """Extract sorted unique undirected edges [E,2] from triangle indices [F,3].
    
    Each row satisfies u < v for valid triangles. Vectorized uniqueness costs
    O(3F log(3F))."""
    t0, t1, t2 = indices_np[:, 0], indices_np[:, 1], indices_np[:, 2]

    all_u = np.concatenate([t0, t1, t0])
    all_v = np.concatenate([t1, t2, t2])

    lo = np.minimum(all_u, all_v)
    hi = np.maximum(all_u, all_v)
    paired = np.stack([lo, hi], axis=1)                     # [3F, 2]
    edges = np.unique(paired, axis=0)                        # [E, 2]

    return edges


# B.2: joint geometry and appearance costs

def compute_edge_cost(xyz, fdc, opacity, edges,
                      w_spatial=1.0, w_color=0.5, w_opacity=0.3):
    """Return joint geometry/color/opacity edge costs [E] for the legacy path.
    
    cost = w_spatial * ||delta_xyz|| + w_color * ||delta_DC_SH||
           + w_opacity * abs(delta_opacity).
    Inputs: xyz [V,3], fdc [V,C_dc,3], activated opacity [V], edges [E,2]."""
    u, v = edges[:, 0], edges[:, 1]
    E = edges.shape[0]

    d_spatial = np.linalg.norm(xyz[u] - xyz[v], axis=1)                   # [E]
    d_color = np.linalg.norm(
        fdc[u].reshape(E, -1) - fdc[v].reshape(E, -1), axis=1)           # [E]
    d_opacity = np.abs(opacity[u] - opacity[v])                            # [E]

    cost = w_spatial * d_spatial + w_color * d_color + w_opacity * d_opacity

    return cost


# QEM geometry costs (Garland-Heckbert, 1997)

def _compute_face_planes(xyz, indices_np):
    """Return unit plane equations [a,b,c,d] as float64 [F,4].
    
    Degenerate faces get zero normals and contribute zero to the quadrics."""
    v0 = xyz[indices_np[:, 0]].astype(np.float64)
    v1 = xyz[indices_np[:, 1]].astype(np.float64)
    v2 = xyz[indices_np[:, 2]].astype(np.float64)

    normal = np.cross(v1 - v0, v2 - v0)                       # [F, 3]
    norm_len = np.linalg.norm(normal, axis=1, keepdims=True)
    safe = norm_len > 1e-20
    # Zero degenerate normals so K = p p^T contributes nothing to Q.
    normal = np.where(safe, normal / np.maximum(norm_len, 1e-20), 0.0)
    d = -(normal * v0).sum(axis=1, keepdims=True)             # [F, 1]
    return np.concatenate([normal, d], axis=1)                # [F, 4]


def _accumulate_vertex_quadrics(xyz, indices_np, planes_face, V,
                                boundary_weight=10.0):
    """Accumulate Q_v = sum(p p^T) over incident faces, returning [V,4,4].
    
    For boundary edges, add weighted virtual planes through the edge with normals
    normalize(edge_direction x face_normal). These constraints discourage inward
    boundary shrinkage (Garland 1998, Section 3.2). Set boundary_weight=0 to disable."""
    F = indices_np.shape[0]
    K_face = planes_face[:, :, None] * planes_face[:, None, :]   # [F, 4, 4]

    Q = np.zeros((V, 4, 4), dtype=np.float64)
    np.add.at(Q, indices_np[:, 0], K_face)
    np.add.at(Q, indices_np[:, 1], K_face)
    np.add.at(Q, indices_np[:, 2], K_face)

    if boundary_weight <= 0.0 or F == 0:
        return Q

    # Find boundary edges incident to exactly one face.
    t0 = indices_np[:, 0].astype(np.int64)
    t1 = indices_np[:, 1].astype(np.int64)
    t2 = indices_np[:, 2].astype(np.int64)
    all_u = np.concatenate([t0, t1, t0])
    all_v = np.concatenate([t1, t2, t2])
    all_face = np.tile(np.arange(F, dtype=np.int64), 3)

    lo = np.minimum(all_u, all_v)
    hi = np.maximum(all_u, all_v)
    keys = (lo << 32) | hi

    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    lo_s = lo[order]
    hi_s = hi[order]
    face_s = all_face[order]

    if len(keys_s) == 0:
        return Q

    boundaries = np.where(np.diff(keys_s) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(keys_s)]])
    bnd = np.where(ends - starts == 1)[0]
    if len(bnd) == 0:
        return Q

    sel = starts[bnd]
    be_lo = lo_s[sel]
    be_hi = hi_s[sel]
    be_face = face_s[sel]

    edge_dir = xyz[be_hi].astype(np.float64) - xyz[be_lo].astype(np.float64)
    edge_len = np.linalg.norm(edge_dir, axis=1, keepdims=True)
    edge_dir = edge_dir / np.maximum(edge_len, 1e-20)

    face_normal = planes_face[be_face, :3]
    virt_normal = np.cross(edge_dir, face_normal)
    virt_len = np.linalg.norm(virt_normal, axis=1, keepdims=True)
    virt_normal = np.where(
        virt_len > 1e-20,
        virt_normal / np.maximum(virt_len, 1e-20),
        0.0,
    )
    d_virt = -(virt_normal * xyz[be_lo].astype(np.float64)).sum(axis=1,
                                                                keepdims=True)
    virt_plane = np.concatenate([virt_normal, d_virt], axis=1)        # [Eb, 4]
    K_virt = boundary_weight * (virt_plane[:, :, None] *
                                virt_plane[:, None, :])

    np.add.at(Q, be_lo, K_virt)
    np.add.at(Q, be_hi, K_virt)
    return Q


def compute_qem_cost(xyz, fdc, opacity, edges, Q_vertex,
                     w_qem=1.0, w_color=0.0, w_opacity=0.0,
                     normalize_qem=True):
    """Return QEM collapse costs [E] and optimal positions [E,3].
    
    Solve the combined endpoint quadric with homogeneous coordinate fixed to 1.
    For singular systems, choose the cheapest endpoint or midpoint. Optionally
    normalize geometric costs by their median and add color/opacity differences.
    Position dtype matches xyz; costs and accumulated quadrics use float64."""
    u = edges[:, 0]
    v = edges[:, 1]
    E = edges.shape[0]

    if E == 0:
        return (np.zeros(0, dtype=np.float64),
                np.zeros((0, 3), dtype=xyz.dtype))

    Q_uv = Q_vertex[u] + Q_vertex[v]                          # [E, 4, 4]

    # Construct [Q[:3,:3] | Q[:3,3]; 0 0 0 1].
    A = Q_uv.copy()
    A[:, 3, :] = 0.0
    A[:, 3, 3] = 1.0
    # Batched np.linalg.solve expects b with shape (..., m, k).
    b = np.zeros((E, 4, 1), dtype=Q_uv.dtype)
    b[:, 3, 0] = 1.0

    A33 = Q_uv[:, :3, :3]
    det = np.linalg.det(A33)
    nonsing = np.abs(det) > 1e-12

    best_pos = np.empty((E, 3), dtype=xyz.dtype)
    cost_geom = np.empty(E, dtype=np.float64)

    if nonsing.any():
        x_sol = np.linalg.solve(A[nonsing], b[nonsing])[..., 0]   # [Es, 4]
        c_ns = np.einsum('eij,ei,ej->e',
                         Q_uv[nonsing], x_sol, x_sol)
        best_pos[nonsing] = x_sol[:, :3].astype(xyz.dtype)
        cost_geom[nonsing] = c_ns

    if (~nonsing).any():
        idx = np.where(~nonsing)[0]
        Es = len(idx)
        ones = np.ones((Es, 1), dtype=np.float64)
        cu = np.concatenate([xyz[u[idx]].astype(np.float64), ones], axis=1)
        cv = np.concatenate([xyz[v[idx]].astype(np.float64), ones], axis=1)
        mid = 0.5 * (xyz[u[idx]].astype(np.float64) +
                     xyz[v[idx]].astype(np.float64))
        cm = np.concatenate([mid, ones], axis=1)
        cands = np.stack([cu, cv, cm], axis=1)                # [Es, 3, 4]
        # cost_k = c_k^T Q c_k  → einsum: e k i, e i j, e k j -> e k
        c_costs = np.einsum('eki,eij,ekj->ek',
                            cands, Q_uv[idx], cands)
        winner = np.argmin(c_costs, axis=1)
        chosen = cands[np.arange(Es), winner, :3]
        best_pos[idx] = chosen.astype(xyz.dtype)
        cost_geom[idx] = c_costs[np.arange(Es), winner]

    # Clamp small negative costs caused by floating-point roundoff.
    cost_geom = np.maximum(cost_geom, 0.0)

    if normalize_qem:
        med = float(np.median(cost_geom))
        if med > 1e-20:
            cost_geom = cost_geom / med

    cost = w_qem * cost_geom

    if w_color > 0.0:
        d_color = np.linalg.norm(
            fdc[u].reshape(E, -1) - fdc[v].reshape(E, -1), axis=1)
        cost = cost + w_color * d_color
    if w_opacity > 0.0:
        d_op = np.abs(opacity[u] - opacity[v])
        cost = cost + w_opacity * d_op

    return cost.astype(np.float64), best_pos


# B.3: local topology utilities

def _build_vertex_neighbors(indices_np, V):
    """Return a list of adjacent-vertex sets for neighborhood queries.
    
    Common-neighbor checks alone are insufficient for full simplicial link
    preservation; guarded collapses use CollapseSafety for the complete test."""
    t0, t1, t2 = indices_np[:, 0], indices_np[:, 1], indices_np[:, 2]
    rows = np.concatenate([t0, t1, t1, t2, t2, t0]).astype(np.int64)
    cols = np.concatenate([t1, t0, t2, t1, t0, t2]).astype(np.int64)

    # Sort by row before grouping.
    order = np.argsort(rows, kind="stable")
    rows_s = rows[order]
    cols_s = cols[order]

    neighbors = [set() for _ in range(V)]
    # Find the start/end offsets for each row.
    if len(rows_s) > 0:
        boundaries = np.where(np.diff(rows_s) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(rows_s)]])
        for s, e in zip(starts, ends):
            v = int(rows_s[s])
            neighbors[v] = set(int(x) for x in cols_s[s:e])
    return neighbors


def _build_edge_to_opp_vertices(indices_np):
    """Return sorted edge keys and aligned opposite-vertex IDs.
    
    For a key, searchsorted(left/right) selects all opposite vertices. Keys use
    the same (min(u,v) << 32) | max(u,v) encoding as _edge_key."""
    t0 = indices_np[:, 0].astype(np.int64)
    t1 = indices_np[:, 1].astype(np.int64)
    t2 = indices_np[:, 2].astype(np.int64)

    all_u = np.concatenate([t0, t1, t0])
    all_v = np.concatenate([t1, t2, t2])
    all_opp = np.concatenate([t2, t0, t1])

    lo = np.minimum(all_u, all_v)
    hi = np.maximum(all_u, all_v)
    keys = (lo << 32) | hi

    order = np.argsort(keys, kind="stable")
    return keys[order], all_opp[order]


def _edge_face_count(indices_np):
    """Return unique edge keys and incident-face counts for safety filtering."""
    t0 = indices_np[:, 0].astype(np.int64)
    t1 = indices_np[:, 1].astype(np.int64)
    t2 = indices_np[:, 2].astype(np.int64)
    all_u = np.concatenate([t0, t1, t0])
    all_v = np.concatenate([t1, t2, t2])
    lo = np.minimum(all_u, all_v)
    hi = np.maximum(all_u, all_v)
    keys = (lo << 32) | hi
    uniq_keys, counts = np.unique(keys, return_counts=True)
    return uniq_keys, counts


# B.3: batched edge collapse


def _find_independent_set_greedy(edges, cost, safe_mask, max_count,
                                 xyz=None, indices_np=None,
                                 enable_safety=False,
                                 max_edge_length=None, candidate_positions=None):
    """Select low-cost edges; safe batches have disjoint vertex one-rings.

    Safety checks full links, boundary topology and actual collapse placement.
    Invalid candidates are skipped, even if the requested budget is not met.
    """
    order = np.argsort(cost, kind="stable")
    used, selected = set(), []
    guard = None
    if enable_safety:
        if xyz is None or indices_np is None or candidate_positions is None:
            raise ValueError("Safe collapse needs geometry, faces and proposed positions")
        guard = CollapseSafety(xyz, indices_np)
    for idx in order:
        if not safe_mask[idx] or not np.isfinite(cost[idx]):
            continue
        u, v = map(int, edges[idx])
        if u in used or v in used:
            continue
        if guard is not None:
            if max_edge_length is not None and np.linalg.norm(xyz[u] - xyz[v]) > max_edge_length:
                continue
            affected = guard.check(u, v, candidate_positions[idx], used)
            if affected is None:
                continue
            used.update(affected)
        else:
            used.update((u, v))
        selected.append(idx)
        if len(selected) >= max_count:
            break
    return np.asarray(selected, dtype=np.int64)


def _vertex_surface_areas(xyz, indices):
    """Barycentric vertex areas on the mesh BEFORE any collapse."""
    xyz64 = xyz.astype(np.float64, copy=False)
    face_areas = 0.5 * np.linalg.norm(
        np.cross(xyz64[indices[:, 1]] - xyz64[indices[:, 0]],
                 xyz64[indices[:, 2]] - xyz64[indices[:, 0]]), axis=1)
    areas = np.zeros(len(xyz), dtype=np.float64)
    for corner in range(3):
        areas += np.bincount(indices[:, corner], weights=face_areas / 3.0,
                             minlength=len(xyz))
    return areas


def batch_edge_collapse(xyz, fdc, frest, vw_logit, indices,
                        edges, opacity_floor,
                        collapse_ratio=0.05,
                        w_spatial=1.0, w_color=0.5, w_opacity=0.3,
                        enable_safety=False,
                        max_edge_length_ratio=2.0,
                        use_qem=True,
                        qem_boundary_weight=10.0):
    """Collapse a greedy independent set of low-cost edges.
    
    Use QEM positions by default; the legacy cost uses opacity-weighted positions.
    Appearance attributes always merge with opacity weights. Redirect faces,
    remove repeated-index triangles and compact vertex arrays.
    
    Return xyz, fdc, frest, vw_logit, indices, keep_mask, collapse_edges [K,2]
    (old retained/removed IDs), and pre-collapse area fractions [K,2] for Adam.
    
    Safety checks full local links, boundaries, placement, normals and areas,
    with disjoint affected one-rings. Overly long edges are rejected using
    max_edge_length_ratio * median_edge_length. These guards preserve valid local
    inputs; they do not repair existing non-manifold geometry or self-intersections.
    qem_boundary_weight controls virtual boundary planes; zero disables them."""
    V = xyz.shape[0]
    empty_edges = np.empty((0, 2), dtype=np.int64)
    empty_weights = np.empty((0, 2), dtype=np.float64)
    if collapse_ratio <= 0 or len(edges) == 0:
        return (xyz, fdc, frest, vw_logit, indices, np.ones(V, dtype=bool),
                empty_edges, empty_weights)

    # 1. Recover activated opacity for costs and attribute weights.
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(vw_logit.ravel(), -20, 20)))
    opacity = opacity_floor + (1.0 - opacity_floor) * sigmoid   # [V]

    # 2. Compute edge costs and optional optimal positions.
    if use_qem:
        planes_face = _compute_face_planes(xyz, indices)
        Q_vertex = _accumulate_vertex_quadrics(
            xyz, indices, planes_face, V,
            boundary_weight=qem_boundary_weight,
        )
        cost, best_pos = compute_qem_cost(
            xyz, fdc, opacity, edges, Q_vertex,
            w_qem=w_spatial, w_color=w_color, w_opacity=w_opacity,
            normalize_qem=True,
        )
    else:
        cost = compute_edge_cost(xyz, fdc, opacity, edges,
                                 w_spatial, w_color, w_opacity)
        u, v = edges[:, 0], edges[:, 1]
        denom = opacity[u] + opacity[v] + 1e-8
        best_pos = ((opacity[u] / denom)[:, None] * xyz[u]
                    + (opacity[v] / denom)[:, None] * xyz[v])

    # 3. Select independent edges, with optional topology guards.
    max_collapse = max(1, int(V * collapse_ratio))
    static_safe = np.ones(edges.shape[0], dtype=bool)

    max_edge_length = None
    if enable_safety:
        edge_len = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
        med = float(np.median(edge_len)) if len(edge_len) > 0 else 0.0
        max_edge_length = med * max_edge_length_ratio

    selected = _find_independent_set_greedy(
        edges, cost, static_safe, max_collapse,
        xyz=xyz, indices_np=indices,
        enable_safety=enable_safety,
        max_edge_length=max_edge_length,
        candidate_positions=best_pos,
    )

    if len(selected) == 0:
        return (xyz, fdc, frest, vw_logit, indices, np.ones(V, dtype=bool),
                empty_edges, empty_weights)

    col_u = edges[selected, 0]                                  # Retained endpoint
    col_v = edges[selected, 1]                                  # Removed endpoint

    # Momentum weights use the pre-collapse geometry, independently of
    # the opacity weights used below for appearance attributes.
    collapse_edges = edges[selected].copy()
    vertex_areas = _vertex_surface_areas(xyz, indices)
    area_pairs = vertex_areas[collapse_edges]
    total_area = area_pairs.sum(axis=1, keepdims=True)
    collapse_weights = np.full_like(area_pairs, 0.5)
    np.divide(area_pairs, total_area, out=collapse_weights, where=total_area > 0)

    # 5. Merge appearance by opacity weights and use QEM positions.
    op_u = opacity[col_u]
    op_v = opacity[col_v]
    denom = op_u + op_v + 1e-8
    wu = op_u / denom                                            # [K]
    wv = op_v / denom

    # Assign the exact positions validated during candidate selection.
    xyz[col_u] = best_pos[selected]

    # features_dc / features_rest: [K, C, 3]
    wu_f = wu[:, None, None]
    wv_f = wv[:, None, None]
    fdc[col_u] = wu_f * fdc[col_u] + wv_f * fdc[col_v]
    frest[col_u] = wu_f * frest[col_u] + wv_f * frest[col_v]

    # Average logits directly to avoid a sigmoid/logit round trip.
    vw_logit[col_u] = wu[:, None] * vw_logit[col_u] + wv[:, None] * vw_logit[col_v]

    # 6. Redirect face indices.
    remap = np.arange(V, dtype=np.int64)
    remap[col_v] = col_u                                        # v → u

    new_idx = remap[indices]

    # Remove faces with repeated vertex indices.
    degen = ((new_idx[:, 0] == new_idx[:, 1]) |
             (new_idx[:, 0] == new_idx[:, 2]) |
             (new_idx[:, 1] == new_idx[:, 2]))
    new_idx = new_idx[~degen]

    # 7. Compact vertex arrays and remap indices.
    keep = np.ones(V, dtype=bool)
    keep[col_v] = False

    compact_id = np.full(V, -1, dtype=np.int64)
    compact_id[keep] = np.arange(int(keep.sum()))
    new_idx = compact_id[new_idx]

    # Every face must refer to a retained vertex.
    assert (new_idx >= 0).all(), "[ecol] face references a removed vertex"

    return (xyz[keep].copy(), fdc[keep].copy(), frest[keep].copy(),
            vw_logit[keep].copy(), new_idx.astype(np.int32), keep,
            collapse_edges, collapse_weights)


# B.4: vertex splitting

def _subdivide_faces(indices, split_edges, new_base):
    """Conforming midpoint templates for every subset of a triangle's edges."""
    if len(split_edges) == 0 or len(indices) == 0:
        return indices.copy()
    t0, t1, t2 = indices[:, 0], indices[:, 1], indices[:, 2]
    su, sv = split_edges[:, 0], split_edges[:, 1]
    split_keys = _edge_key(su, sv)                               # [K]

    sort_sk = np.argsort(split_keys)
    sorted_keys = split_keys[sort_sk]
    sorted_mid_ids = (new_base + sort_sk).astype(np.int64)

    def _lookup_mid(hashes):
        """Match face edges to split midpoints with O(F log K) searchsorted lookups."""
        pos = np.searchsorted(sorted_keys, hashes)
        pos = np.clip(pos, 0, len(sorted_keys) - 1)
        found = sorted_keys[pos] == hashes
        mid_id = np.full(len(hashes), -1, dtype=np.int64)
        mid_id[found] = sorted_mid_ids[pos[found]]
        return mid_id

    h01 = _edge_key(t0.astype(np.int64), t1.astype(np.int64))
    h12 = _edge_key(t1.astype(np.int64), t2.astype(np.int64))
    h02 = _edge_key(t0.astype(np.int64), t2.astype(np.int64))

    m01 = _lookup_mid(h01)                                       # -1 marks an edge that was not split.
    m12 = _lookup_mid(h12)
    m02 = _lookup_mid(h02)

    split_01 = m01 >= 0
    split_12 = m12 >= 0
    split_02 = m02 >= 0
    n_splits = split_01.astype(np.int32) + split_12.astype(np.int32) + split_02.astype(np.int32)

    # 6. Group triangles by the number of split edges.
    #   n=0: retain the original face.
    #   n=1: split into two faces (the most common case).
    #   n=2/3: split into three/four faces; shared midpoints avoid T-junctions.

    new_faces_list = []

    # Only untouched faces remain unchanged.
    keep_mask = n_splits == 0
    if keep_mask.any():
        new_faces_list.append(indices[keep_mask])

    # Split edge (0,1).
    #   (a, b, c) -> (a, m, c) + (m, b, c)
    #   Winding is preserved: (m-a) x (c-a) = t * (b-a) x (c-a).
    mask = split_01 & (n_splits == 1)
    if mask.any():
        a, b, c, m = t0[mask], t1[mask], t2[mask], m01[mask]
        new_faces_list.append(np.stack([a, m, c], axis=1))
        new_faces_list.append(np.stack([m, b, c], axis=1))

    # Split edge (1,2).
    #   (a, b, c) -> (a, b, m) + (a, m, c)
    mask = split_12 & (n_splits == 1)
    if mask.any():
        a, b, c, m = t0[mask], t1[mask], t2[mask], m12[mask]
        new_faces_list.append(np.stack([a, b, m], axis=1))
        new_faces_list.append(np.stack([a, m, c], axis=1))

    # Split edge (0,2).
    #   (a, b, c) -> (a, b, m) + (m, b, c)
    mask = split_02 & (n_splits == 1)
    if mask.any():
        a, b, c, m = t0[mask], t1[mask], t2[mask], m02[mask]
        new_faces_list.append(np.stack([a, b, m], axis=1))
        new_faces_list.append(np.stack([m, b, c], axis=1))

    # Two split edges meet at one corner. Rotate that corner to a,
    # then triangulate the remaining quad with a fixed, shared-edge-safe rule.
    for mask, a, b, c, mab, mac in (
        (split_01 & split_02 & (n_splits == 2), t0, t1, t2, m01, m02),
        (split_01 & split_12 & (n_splits == 2), t1, t2, t0, m12, m01),
        (split_12 & split_02 & (n_splits == 2), t2, t0, t1, m02, m12),
    ):
        if mask.any():
            a, b, c, mab, mac = (x[mask] for x in (a, b, c, mab, mac))
            new_faces_list.extend([np.stack([a, mab, mac], axis=1),
                                   np.stack([mab, b, c], axis=1),
                                   np.stack([mab, c, mac], axis=1)])
    mask = n_splits == 3
    if mask.any():
        a, b, c, ab, bc, ac = (x[mask] for x in (t0, t1, t2, m01, m12, m02))
        new_faces_list.extend([np.stack([a, ab, ac], axis=1),
                               np.stack([ab, b, bc], axis=1),
                               np.stack([ac, bc, c], axis=1),
                               np.stack([ab, bc, ac], axis=1)])

    return np.concatenate(new_faces_list, axis=0)


def batch_vertex_split(xyz, fdc, frest, vw_logit, indices,
                       importance_np, top_k_percent=5.0,
                       enable_safety=False,
                       max_edge_length_ratio=2.0):
    """Split the longest incident edges of the top-K-percent important vertices.
    
    Append one midpoint per selected edge and share it across all incident faces.
    Faces with one, two or three split edges become two, three or four faces,
    respectively, preserving winding. Longest-edge selection uses lexsort;
    midpoint lookup uses sorted int64 edge keys.
    
    Return xyz, fdc, frest, vw_logit, indices, n_new_verts, split_parents.
    Parent IDs refer to input vertices and follow appended-midpoint order.
    The requesting vertex is the parent; duplicate requests choose higher
    importance, then the smaller vertex ID."""
    V = xyz.shape[0]
    empty_parents = np.empty(0, dtype=np.int64)
    if V == 0 or len(indices) == 0 or top_k_percent <= 0:
        return xyz, fdc, frest, vw_logit, indices, 0, empty_parents

    # 1. Select the top K percent of vertices.
    k = max(1, int(V * top_k_percent / 100.0))
    k = min(k, V)
    top_v = np.argpartition(importance_np, -k)[-k:]

    # 2. Extract unique edges and their lengths.
    t0, t1, t2 = indices[:, 0], indices[:, 1], indices[:, 2]

    all_u = np.concatenate([t0, t1, t0]).astype(np.int64)
    all_v = np.concatenate([t1, t2, t2]).astype(np.int64)
    lo = np.minimum(all_u, all_v)
    hi = np.maximum(all_u, all_v)
    paired = np.stack([lo, hi], axis=1)
    unique_edges, _ = np.unique(paired, axis=0, return_inverse=True)

    eu, ev = unique_edges[:, 0], unique_edges[:, 1]
    edge_len = np.linalg.norm(xyz[eu] - xyz[ev], axis=1)       # [E_unique]
    E_unique = unique_edges.shape[0]

    # 2.5. Optionally filter unsafe edges.
    #   Skip non-manifold edges (more than two incident faces).
    #   Skip overly long edges whose midpoints may lie far from the surface.
    if enable_safety and E_unique > 0:
        edge_keys = _edge_key(eu, ev)
        uniq_keys, counts = _edge_face_count(indices)
        pos = np.searchsorted(uniq_keys, edge_keys)
        pos = np.clip(pos, 0, len(uniq_keys) - 1)
        cnt_per_edge = np.where(uniq_keys[pos] == edge_keys, counts[pos], 0)
        manifold_mask = (cnt_per_edge >= 1) & (cnt_per_edge <= 2)

        med = float(np.median(edge_len)) if len(edge_len) > 0 else 0.0
        length_mask = edge_len <= med * max_edge_length_ratio

        safe_edge_mask = manifold_mask & length_mask & np.isfinite(edge_len) & (edge_len > 0)
    else:
        safe_edge_mask = np.ones(E_unique, dtype=bool)

    if len(top_v) == 0:
        return xyz, fdc, frest, vw_logit, indices, 0, empty_parents

    # 3. Find the longest incident edge of each selected vertex.
    vert_ids = np.concatenate([eu, ev])                          # [2E]
    edge_ids = np.tile(np.arange(E_unique), 2)                   # [2E]
    lengths = np.tile(edge_len, 2)                               # [2E]
    # Expand the safety mask to both directed copies of every edge.
    edge_safe_2e = np.tile(safe_edge_mask, 2)                    # [2E]

    mask_top = np.isin(vert_ids, top_v) & edge_safe_2e
    if mask_top.sum() == 0:
        return xyz, fdc, frest, vw_logit, indices, 0, empty_parents

    vert_top = vert_ids[mask_top]
    edge_top = edge_ids[mask_top]
    len_top = lengths[mask_top]

    # Sort by ascending vertex and descending length; the first edge is longest.
    sort_order = np.lexsort((-len_top, vert_top))
    vert_sorted = vert_top[sort_order]
    edge_sorted = edge_top[sort_order]
    _, first_idx = np.unique(vert_sorted, return_index=True)

    # Keep the requesting parent when deduplicating split edges. Each
    # parent requests only one edge, so retained/new children form disjoint
    # momentum pairs even when different split edges share an endpoint.
    requested_edges = edge_sorted[first_idx]
    requested_parents = vert_sorted[first_idx]
    parent_order = np.lexsort((
        requested_parents, -importance_np[requested_parents], requested_edges))
    split_edge_ids, first_request = np.unique(
        requested_edges[parent_order], return_index=True)
    split_parents = requested_parents[parent_order[first_request]].copy()
    split_edges = unique_edges[split_edge_ids]                   # [K, 2]
    if enable_safety and len(split_edges):
        # Edge counts alone miss bow-tie endpoints and duplicate faces.
        guard = CollapseSafety(xyz, indices)
        valid = np.array([guard.link(int(u)) is not None and guard.link(int(v)) is not None
                          for u, v in split_edges], dtype=bool)
        split_edges, split_parents = split_edges[valid], split_parents[valid]
    K = split_edges.shape[0]

    if K == 0:
        return xyz, fdc, frest, vw_logit, indices, 0, empty_parents

    # 4. Create midpoint vertices.
    su, sv = split_edges[:, 0], split_edges[:, 1]
    mid_xyz = (xyz[su] + xyz[sv]) / 2.0
    mid_fdc = (fdc[su] + fdc[sv]) / 2.0
    mid_frest = (frest[su] + frest[sv]) / 2.0
    mid_vw = (vw_logit[su] + vw_logit[sv]) / 2.0

    new_base = V                                                  # First new vertex ID

    all_faces = _subdivide_faces(indices, split_edges, new_base)

    # 7. Append new vertices.
    new_xyz = np.concatenate([xyz, mid_xyz], axis=0)
    new_fdc = np.concatenate([fdc, mid_fdc], axis=0)
    new_frest = np.concatenate([frest, mid_frest], axis=0)
    new_vw = np.concatenate([vw_logit, mid_vw], axis=0)

    return (new_xyz, new_fdc, new_frest, new_vw,
            all_faces.astype(np.int32), K, split_parents)


# =====================================================================
# Phase C: optimizer state transfer
# =====================================================================

@torch.no_grad()
def surgery_and_stitch_params(model, optimizer, keep_mask, n_new_verts,
                              xyz_np, fdc_np, frest_np, vw_np, indices_np,
                              *, collapse_edges, collapse_weights,
                              split_parents, split_momentum_std=1e-3):
    """Transfer Adam history through collapse, compaction, then splitting.

    Eq. 17: merge exp_avg and exp_avg_sq using pre-collapse vertex areas.
    Eq. 18: the retained requesting parent and appended midpoint inherit
    that parent's moments; their first moments receive +delta and -delta.
    For numerical stability, delta = std * sqrt(parent exp_avg_sq) * N(0, 1).
    The default std=1e-3 is dimensionless. This RMS-scaled adaptation of
    Eq. 18 avoids unbounded Adam updates for unobserved/low-gradient rows;
    an absolute perturbation is unsafe with the training Adam eps=1e-15.
    Set split_momentum_std=0 for deterministic inheritance without noise.

    collapse_edges uses OLD IDs; split_parents uses POST-COLLAPSE IDs.
    Adam step and parameter-group settings are preserved. No state is
    invented for parameters whose Adam state has not been initialized.
    Call after optimizer.step()/zero_grad(), before the next forward pass.
    """
    if not np.isfinite(split_momentum_std) or split_momentum_std < 0:
        raise ValueError("split_momentum_std must be finite and nonnegative")
    n_kept = int(keep_mask.sum())
    if len(split_parents) != n_new_verts:
        raise ValueError("Each new vertex must have one split parent")
    if (np.any(split_parents < 0) or np.any(split_parents >= n_kept)
            or len(np.unique(split_parents)) != n_new_verts):
        raise ValueError("Split parents must be unique post-collapse vertex IDs")
    if len(collapse_edges) != int((~keep_mask).sum()):
        raise ValueError("Each removed vertex must have a collapse mapping")

    device = model.vertices.device
    new_tensors = {
        "vertices":     torch.from_numpy(xyz_np).to(device, torch.float32),
        "f_dc":         torch.from_numpy(fdc_np).to(device, torch.float32),
        "f_rest":       torch.from_numpy(frest_np).to(device, torch.float32),
        "vertex_weight": torch.from_numpy(vw_np).to(device, torch.float32),
    }
    keep_mask_t = torch.from_numpy(keep_mask).to(device)
    edges_t = torch.as_tensor(collapse_edges, dtype=torch.long, device=device)
    compact_ids = np.cumsum(keep_mask, dtype=np.int64) - 1
    collapse_rows = torch.as_tensor(
        compact_ids[collapse_edges[:, 0]], dtype=torch.long, device=device)
    weights_t = torch.as_tensor(collapse_weights, device=device)
    parents_t = torch.as_tensor(split_parents, dtype=torch.long, device=device)

    for group in optimizer.param_groups:
        name = group["name"]
        if name not in new_tensors:
            continue
        new_data = new_tensors[name]
        old_param = group["params"][0]
        stored = optimizer.state.get(old_param)
        if stored:
            # max_exp_avg_sq is included for Adam's optional AMSGrad mode.
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key not in stored:
                    continue
                old_moment = stored[key]
                remapped = old_moment[keep_mask_t]
                if len(collapse_edges):
                    weight_shape = (-1,) + (1,) * (old_moment.ndim - 1)
                    weights = weights_t.to(dtype=old_moment.dtype)
                    remapped[collapse_rows] = (
                        old_moment[edges_t[:, 0]] * weights[:, 0].reshape(weight_shape)
                        + old_moment[edges_t[:, 1]] * weights[:, 1].reshape(weight_shape))
                if n_new_verts:
                    # Both children inherit the already-merged state.
                    inherited = remapped[parents_t]
                    remapped = torch.cat((remapped, inherited), dim=0)
                if remapped.shape != new_data.shape:
                    raise ValueError(
                        f"[Surgery] {name}/{key} shape {remapped.shape} "
                        f"!= param shape {new_data.shape}")
                stored[key] = remapped

            if n_new_verts and split_momentum_std > 0:
                # Scale only AFTER remapping v, so a collapsed-then-split
                # parent uses its area-weighted second moment. Zero-history
                # entries get no noise; epsilon must not become a noise floor.
                parent_m = stored["exp_avg"][parents_t]
                parent_rms = stored["exp_avg_sq"][parents_t].clamp_min(0).sqrt()
                noise = torch.randn_like(parent_m) * split_momentum_std * parent_rms
                stored["exp_avg"][parents_t] = parent_m + noise
                stored["exp_avg"][n_kept:] = parent_m - noise

        # Avoid leaving empty state entries under an obsolete Parameter.
        optimizer.state.pop(old_param, None)
        new_param = nn.Parameter(new_data, requires_grad=old_param.requires_grad)
        group["params"][0] = new_param
        if stored:
            optimizer.state[new_param] = stored

    # Synchronize model attributes by group name, independent of group order.
    for group in optimizer.param_groups:
        param = group["params"][0]
        if group["name"] == "vertices":
            model.vertices = param
        elif group["name"] == "f_dc":
            model._features_dc = param
        elif group["name"] == "f_rest":
            model._features_rest = param
        elif group["name"] == "vertex_weight":
            model.vertex_weight = param

    # Update face indices and reset per-triangle tracking buffers.
    model._triangle_indices = (
        torch.from_numpy(indices_np.astype(np.int64))
        .to(device).to(torch.int32)
    )
    F_new = model._triangle_indices.shape[0]
    model.image_size = torch.zeros(F_new, dtype=torch.float, device=device)
    model.importance_score = torch.zeros(F_new, dtype=torch.float, device=device)
    model.pixel_count = torch.zeros(F_new, dtype=torch.int, device=device)


# =====================================================================
# Entry point: one complete heartbeat_step
# =====================================================================

def heartbeat_step(model, collapse_ratio=0.12, split_top_k=2.0,
                   w_spatial=1.0, w_color=0.5, w_opacity=0.3,
                   mode="both",
                   enable_safety=False,
                   max_edge_length_ratio=2.0,
                   use_qem=True,
                   qem_boundary_weight=10.0,
                   split_momentum_std=1e-3,
                   topology_callback=None):
    """Run a complete topology-edit and optimizer-state-transfer event.
    
    mode is "both" (collapse then split), "collapse", or "split". QEM supplies
    collapse positions and geometry costs; color and opacity costs are optional.
    Safety guards local links, boundaries, flips, degeneracy, displacement and
    shared-midpoint splitting. Edge length is limited relative to the median.
    The optional topology_callback(stage, xyz, faces) must not mutate its arrays.
    
    split_momentum_std is dimensionless relative to sqrt(parent exp_avg_sq).
    The default 1e-3 uses RMS-scaled paired perturbations to avoid unstable Adam
    updates at near-zero second moments; zero disables perturbations.
    Call after optimizer.step()/zero_grad(). Statistics are collected after
    backward and before step."""
    if not np.isfinite(split_momentum_std) or split_momentum_std < 0:
        raise ValueError("split_momentum_std must be finite and nonnegative")
    if mode not in ("both", "collapse", "split"):
        raise ValueError(f"Unknown heartbeat mode: {mode}")
    do_collapse = mode in ("both", "collapse")
    do_split    = mode in ("both", "split")

    with torch.no_grad():
        xyz_np = model.vertices.detach().cpu().numpy().copy()
        fdc_np = model._features_dc.detach().cpu().numpy().copy()
        frest_np = model._features_rest.detach().cpu().numpy().copy()
        vw_np = model.vertex_weight.detach().cpu().numpy().copy()
        indices_np = model._triangle_indices.detach().cpu().numpy().astype(np.int64)

        importance_np = get_vertex_importance(model).cpu().numpy()
        V_orig = xyz_np.shape[0]
        opacity_floor = float(model.opacity_floor)

        keep_mask = np.ones(V_orig, dtype=bool)
        V_after_collapse = V_orig
        n_new = 0
        collapse_edges = np.empty((0, 2), dtype=np.int64)
        collapse_weights = np.empty((0, 2), dtype=np.float64)
        split_parents = np.empty(0, dtype=np.int64)

        if topology_callback is not None:
            topology_callback("before_dte", xyz_np, indices_np)

        # Edge collapse
        if do_collapse:
            edges = extract_edges(indices_np)
            (xyz_np, fdc_np, frest_np, vw_np,
             indices_np, keep_mask, collapse_edges, collapse_weights) = batch_edge_collapse(
                xyz_np, fdc_np, frest_np, vw_np, indices_np,
                edges, opacity_floor,
                collapse_ratio=collapse_ratio,
                w_spatial=w_spatial, w_color=w_color, w_opacity=w_opacity,
                enable_safety=enable_safety,
                max_edge_length_ratio=max_edge_length_ratio,
                use_qem=use_qem,
                qem_boundary_weight=qem_boundary_weight,
            )
            V_after_collapse = xyz_np.shape[0]
            importance_np = importance_np[keep_mask]
            if topology_callback is not None:
                topology_callback("after_collapse", xyz_np, indices_np)

        # Vertex splitting
        if do_split:
            (xyz_np, fdc_np, frest_np, vw_np,
             indices_np, n_new, split_parents) = batch_vertex_split(
                xyz_np, fdc_np, frest_np, vw_np, indices_np,
                importance_np,
                top_k_percent=split_top_k,
                enable_safety=enable_safety,
                max_edge_length_ratio=max_edge_length_ratio,
            )

        if do_split and topology_callback is not None:
            topology_callback("after_split", xyz_np, indices_np)

        V_final = xyz_np.shape[0]
        F_final = indices_np.shape[0]

        # Optimizer state transfer
        surgery_and_stitch_params(
            model, model.optimizer, keep_mask, n_new,
            xyz_np, fdc_np, frest_np, vw_np, indices_np,
            collapse_edges=collapse_edges, collapse_weights=collapse_weights,
            split_parents=split_parents, split_momentum_std=split_momentum_std,
        )

        init_heartbeat_stats(model)

        tag = mode.upper()
        if mode == "both":
            print(f"[Heartbeat] V: {V_orig} → {V_after_collapse} (collapse) "
                  f"→ {V_final} (split) | F: {F_final}")
        elif mode == "collapse":
            print(f"[Heartbeat-COLLAPSE] V: {V_orig} → {V_final} "
                  f"(-{V_orig - V_final}) | F: {F_final}")
        else:
            print(f"[Heartbeat-SPLIT] V: {V_orig} → {V_final} "
                  f"(+{V_final - V_orig}) | F: {F_final}")


@torch.no_grad()
def visibility_collapse_step(model, visibility, threshold, max_face_ratio=0.02,
                             max_edge_length_ratio=2.0, qem_boundary_weight=10.0):
    """Simplify low-visibility faces after an Adam step, before the next render.

    Only edges with ALL incident faces below the visibility threshold are
    candidates. High-contribution faces cannot be removed, but their shared
    vertices can move; QEM, appearance costs and placement checks limit this.
    Every collapse uses the full link/boundary guard and disjoint one-rings.
    A conservative floor(face_budget / 2) edge budget bounds removed faces,
    including boundary collapses. Impossible candidates are retained, never
    passed to an unrestricted face-pruning fallback. Textures are unsupported.
    """
    if getattr(model, "face_textures", None) is not None:
        raise RuntimeError("Visibility collapse must finish before face texture initialization")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Visibility threshold must lie in [0, 1]")
    if not np.isfinite(max_face_ratio) or not 0 < max_face_ratio < 1:
        raise ValueError("Visibility max_face_ratio must lie in (0, 1)")
    if not np.isfinite(max_edge_length_ratio) or max_edge_length_ratio <= 0:
        raise ValueError("Visibility edge-length ratio must be positive and finite")
    if not np.isfinite(qem_boundary_weight) or qem_boundary_weight < 0:
        raise ValueError("QEM boundary weight must be nonnegative and finite")
    # Never discard unconsumed optimizer gradients when replacing Parameters.
    if any(p.grad is not None for g in model.optimizer.param_groups for p in g["params"]):
        raise RuntimeError("Call visibility collapse after optimizer.step() and zero_grad(set_to_none=True)")
    if isinstance(visibility, torch.Tensor):
        visibility = visibility.detach().cpu().numpy()
    visibility = np.asarray(visibility).reshape(-1)
    faces = model._triangle_indices.detach().cpu().numpy().astype(np.int64)
    if visibility.shape != (len(faces),) or not np.isfinite(visibility).all():
        raise ValueError("Visibility must contain one finite value per current face")
    low = visibility <= threshold
    face_budget = int(len(faces) * max_face_ratio)
    edge_budget = face_budget // 2
    stats = dict(threshold=float(threshold), max_face_ratio=float(max_face_ratio),
                 vertices_before=len(model.vertices), faces_before=len(faces),
                 low_visibility_faces=int(low.sum()), face_budget=face_budget,
                 candidate_edges=0, collapsed_edges=0, removed_faces=0,
                 vertices_after=len(model.vertices), faces_after=len(faces),
                 remaining_low_visibility_faces=int(low.sum()))
    if not edge_budget or not low.any():
        return stats
    low_edges = extract_edges(faces[low])
    high_edges = extract_edges(faces[~low])
    keys = np.setdiff1d(_edge_key(low_edges[:, 0], low_edges[:, 1]),
                        _edge_key(high_edges[:, 0], high_edges[:, 1]), assume_unique=True)
    edges = np.column_stack((keys >> 32, keys & 0xFFFFFFFF))
    stats["candidate_edges"] = len(edges)
    if not len(edges):
        return stats

    data = [getattr(model, attr).detach().cpu().numpy().copy()
            for attr in ("vertices", "_features_dc", "_features_rest", "vertex_weight")]
    data.append(faces)
    # batch_edge_collapse budgets by vertex fraction. nextafter avoids floating
    # rounding below the intended integer budget; the floor remains <= budget.
    ratio = np.nextafter(edge_budget / len(model.vertices), np.inf)
    result = batch_edge_collapse(
        *data, edges=edges, opacity_floor=float(model.opacity_floor),
        collapse_ratio=ratio, enable_safety=True, use_qem=True,
        max_edge_length_ratio=max_edge_length_ratio, qem_boundary_weight=qem_boundary_weight)
    *new_data, keep, collapsed, weights = result
    if not len(collapsed):
        return stats  # Keep live Parameter objects and their state on a no-op.

    # Verify the face budget and high-visibility-face protection BEFORE touching
    # the model or its optimizer. The batch function only mutated private copies.
    remap = np.arange(len(model.vertices), dtype=np.int64)
    remap[collapsed[:, 1]] = collapsed[:, 0]
    redirected = remap[faces]
    removed = ((redirected[:, 0] == redirected[:, 1]) |
               (redirected[:, 1] == redirected[:, 2]) |
               (redirected[:, 0] == redirected[:, 2]))
    if (len(faces) - len(new_data[4]) != int(removed.sum())
            or int(removed.sum()) > face_budget or np.any(removed & ~low)):
        raise RuntimeError("Visibility collapse violated its face budget or protected-face mask")
    surgery_and_stitch_params(
        model, model.optimizer, keep, 0, *new_data,
        collapse_edges=collapsed, collapse_weights=weights,
        split_parents=np.empty(0, dtype=np.int64), split_momentum_std=0.0)
    init_heartbeat_stats(model)
    stats.update(collapsed_edges=len(collapsed), removed_faces=int(removed.sum()),
                 vertices_after=len(model.vertices), faces_after=len(model._triangle_indices),
                 remaining_low_visibility_faces=int((low & ~removed).sum()))
    return stats
