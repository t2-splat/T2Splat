# Repository maintenance and English documentation: zhtang2004@gmail.com.
#!/usr/bin/env python3
# triangles_to_ply.py
import argparse, os, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import trimesh

def parse_vec3(s):
    try:
        x, y, z = [float(v) for v in s.split(",")]
        return torch.tensor([x, y, z], dtype=torch.float32)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid vec3 '{s}', expected 'x,y,z'") from e

def load_scene(scene_dir, device="cpu"):
    path = os.path.join(scene_dir, "point_cloud_state_dict.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find '{path}'")
    sd = torch.load(path, map_location=device)

    # base mesh
    verts   = sd["triangles_points"].to(device).to(torch.float32)   # [V,3]
    faces   = sd["_triangle_indices"].to(device).to(torch.int64)    # [T,3]
    f_dc    = sd["features_dc"].to(device).to(torch.float32)        # [V,1,3]
    f_rest  = sd.get("features_rest", None)
    if f_rest is not None:
        f_rest = f_rest.to(device).to(torch.float32)                # [V,?,3] (unused here)
    act_deg = int(sd.get("active_sh_degree", 3))
    vert_w = sd.get("vertex_weight", None)
    if vert_w is not None:
        # TriangleModel.opacity_activation with opacity_floor = 0
        vert_w = torch.sigmoid(vert_w.detach().to(device).to(torch.float32)).reshape(-1)

    return verts, faces, f_dc, f_rest, act_deg, vert_w


def clean_mesh(mesh, min_faces_ratio=0.01, keep_largest=False, verbose=True, return_sources=False):
    """Remove small disconnected components and unreferenced vertices.
    
    Use SciPy sparse vertex adjacency to label connected components, then filter
    faces by component size. This avoids constructing a separate Trimesh object
    for each of the many fragments that RDT/heartbeat can produce."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    V = verts.shape[0]
    F = faces.shape[0]
    V_before, F_before = V, F

    if F == 0 or V == 0:
        return (mesh, np.arange(V), np.arange(F)) if return_sources else mesh

    # 1. Find vertex-connected components.
    # Build sparse vertex adjacency from the three edges of each triangle.
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.int8)
    A = sp.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
    n_cc, labels = connected_components(A, directed=False)

    # 2. Filter components by their face counts.
    face_labels = labels[faces[:, 0]]                          # Component containing each face
    face_counts = np.bincount(face_labels, minlength=n_cc)     # Number of faces in each component
    max_fc = int(face_counts.max())

    if keep_largest:
        keep_comp = np.zeros(n_cc, dtype=bool)
        keep_comp[int(face_counts.argmax())] = True
        threshold_str = "largest only"
    else:
        threshold = max(1, int(min_faces_ratio * max_fc))
        keep_comp = face_counts >= threshold
        threshold_str = str(threshold)

    face_mask = keep_comp[face_labels]
    if not face_mask.any():
        return (mesh, np.arange(V), np.arange(F)) if return_sources else mesh

    kept_faces = faces[face_mask]
    source_faces = np.flatnonzero(face_mask)

    # 3. Remove unreferenced vertices and remap indices.
    used = np.unique(kept_faces)
    remap = np.full(V, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    new_faces = remap[kept_faces]
    new_verts = verts[used]

    # Vertex colors, if available
    new_colors = None
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == V:
            new_colors = vc[used]
    except Exception:
        pass

    cleaned = trimesh.Trimesh(vertices=new_verts,
                              faces=new_faces,
                              vertex_colors=new_colors,
                              process=False)

    # 4. Remove duplicate faces in O(F log F).
    try:
        unique = cleaned.unique_faces()
        cleaned.update_faces(unique)
        source_faces = source_faces[unique]
    except Exception:
        pass

    F_after = len(cleaned.faces)
    V_after = len(cleaned.vertices)
    kept_n = int(keep_comp.sum())
    if verbose:
        print(f"[clean] components: {n_cc} → kept {kept_n} "
              f"(threshold faces >= {threshold_str})")
        print(f"[clean] vertices: {V_before} → {V_after} "
              f"(-{V_before - V_after}), faces: {F_before} → {F_after} "
              f"(-{F_before - F_after})")

    return (cleaned, used, source_faces) if return_sources else cleaned


def _split_nonmanifold_vertex_fans(vertices, faces, colors, vertices_to_split, return_sources=False):
    """Duplicate bow-tie vertices so each disconnected face fan gets one ID."""
    vertices_to_split = np.asarray(vertices_to_split, dtype=np.int64)
    if len(vertices_to_split) == 0 or len(faces) == 0:
        result = (vertices, faces, colors, 0)
        return (*result, np.arange(len(vertices))) if return_sources else result

    faces = np.asarray(faces, dtype=np.int64).copy()
    flat_faces = faces.reshape(-1)
    matched_positions = np.flatnonzero(
        np.isin(flat_faces, vertices_to_split, assume_unique=False))
    if len(matched_positions) == 0:
        result = (vertices, faces, colors, 0)
        return (*result, np.arange(len(vertices))) if return_sources else result

    matched_vertices = flat_faces[matched_positions]
    matched_face_ids = matched_positions // 3
    order = np.argsort(matched_vertices, kind="stable")
    matched_vertices = matched_vertices[order]
    matched_face_ids = matched_face_ids[order]
    boundaries = np.where(np.diff(matched_vertices) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(matched_vertices)]])

    added_vertices = []
    added_colors = []
    added_sources = []
    original_vertex_count = len(vertices)
    split_fans = 0

    for start, end in zip(starts, ends):
        vertex = int(matched_vertices[start])
        incident = np.unique(matched_face_ids[start:end])
        if len(incident) <= 1:
            continue

        # Two incident faces belong to the same fan when they share one of
        # the two edges touching this vertex.
        neighbor_faces = {}
        for face_id in incident:
            other = faces[face_id][faces[face_id] != vertex]
            if len(other) != 2:
                continue
            for neighbor in other:
                neighbor_faces.setdefault(int(neighbor), []).append(int(face_id))

        adjacency = {int(face_id): set() for face_id in incident}
        for linked_faces in neighbor_faces.values():
            if len(linked_faces) == 2:
                a, b = linked_faces
                adjacency[a].add(b)
                adjacency[b].add(a)

        components = []
        unseen = set(adjacency)
        while unseen:
            seed = unseen.pop()
            component = {seed}
            stack = [seed]
            while stack:
                current = stack.pop()
                new_faces = adjacency[current] & unseen
                unseen.difference_update(new_faces)
                component.update(new_faces)
                stack.extend(new_faces)
            components.append(np.fromiter(component, dtype=np.int64))

        if len(components) <= 1:
            continue

        # Keep the largest fan on the original vertex and duplicate the
        # vertex for every additional disconnected fan.
        components.sort(key=len, reverse=True)
        for component in components[1:]:
            new_vertex_id = len(vertices) + len(added_vertices)
            component_faces = faces[component]
            component_faces[component_faces == vertex] = new_vertex_id
            faces[component] = component_faces
            added_vertices.append(vertices[vertex].copy())
            added_sources.append(vertex)
            if colors is not None:
                added_colors.append(colors[vertex].copy())
            split_fans += 1

    if added_vertices:
        vertices = np.concatenate(
            [vertices, np.asarray(added_vertices, dtype=vertices.dtype)], axis=0)
        if colors is not None:
            colors = np.concatenate(
                [colors, np.asarray(added_colors, dtype=colors.dtype)], axis=0)

    result = (vertices, faces, colors, split_fans)
    sources = np.concatenate((np.arange(original_vertex_count), np.asarray(added_sources, dtype=np.int64)))
    return (*result, sources) if return_sources else result


def _source_face_indices(original, retained):
    """Recover face provenance after operations that only delete triangles."""
    original = np.ascontiguousarray(original, dtype=np.int64)
    retained = np.ascontiguousarray(retained, dtype=np.int64)
    record = np.dtype((np.void, original.dtype.itemsize * 3))
    keys, first = np.unique(original.view(record).ravel(), return_index=True)
    query = retained.view(record).ravel()
    positions = np.searchsorted(keys, query)
    if np.any(positions >= len(keys)) or not np.array_equal(keys[positions], query):
        raise ValueError("Mesh repair changed triangle order; cannot associate face textures")
    return first[positions]


def make_manifold(mesh, verbose=True, return_sources=False):
    """Repair non-manifold edges and bow-tie vertices at export time.

    Open3D removes the smallest-area excess triangles from edges shared by
    more than two faces.  Remaining non-manifold vertices are repaired without
    deleting geometry by duplicating the vertex once per disconnected face fan.
    Legitimate open boundaries are preserved.
    """
    import open3d as o3d

    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    colors = None
    try:
        vertex_colors = np.asarray(mesh.visual.vertex_colors)
        if vertex_colors.ndim == 2 and len(vertex_colors) == len(vertices):
            colors = vertex_colors.copy()
    except Exception:
        pass

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    if colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            colors[:, :3].astype(np.float64) / 255.0)

    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    # Keep the original vertex indices until face provenance is recovered.

    edge_before = len(o3d_mesh.get_non_manifold_edges(
        allow_boundary_edges=True))
    vertex_before = len(o3d_mesh.get_non_manifold_vertices())

    if edge_before:
        o3d_mesh.remove_non_manifold_edges()

    vertices = np.asarray(o3d_mesh.vertices).copy()
    source_faces = _source_face_indices(faces, np.asarray(o3d_mesh.triangles))
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int64).copy()
    if o3d_mesh.has_vertex_colors():
        colors_rgb = np.clip(
            np.asarray(o3d_mesh.vertex_colors) * 255.0, 0, 255).round()
        colors = colors_rgb.astype(np.uint8)
    else:
        colors = None

    vertices_to_split = np.asarray(
        o3d_mesh.get_non_manifold_vertices(), dtype=np.int64)
    vertices, faces, colors, split_fans, source_vertices = _split_nonmanifold_vertex_fans(
        vertices, faces, colors, vertices_to_split, return_sources=True)
    used = np.unique(faces)
    faces = np.searchsorted(used, faces)
    vertices = vertices[used]
    colors = colors[used] if colors is not None else None
    source_vertices = source_vertices[used]

    repaired = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
        process=False,
    )
    repaired.remove_unreferenced_vertices()

    # Verify the exported topology with Open3D after vertex fan splitting.
    check = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.asarray(repaired.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(
            np.asarray(repaired.faces, dtype=np.int32)),
    )
    edge_after = len(check.get_non_manifold_edges(
        allow_boundary_edges=True))
    vertex_after = len(check.get_non_manifold_vertices())

    if verbose:
        print(f"[manifold] non-manifold edges: {edge_before} → {edge_after}")
        print(f"[manifold] non-manifold vertices: "
              f"{vertex_before} → {vertex_after} "
              f"(duplicated {split_fans} disconnected fans)")
        print(f"[manifold] vertices/faces: "
              f"{len(mesh.vertices)}/{len(mesh.faces)} → "
              f"{len(repaired.vertices)}/{len(repaired.faces)}")

    if edge_after or vertex_after:
        print("[manifold] warning: topology repair left "
              f"{edge_after} non-manifold edges and "
              f"{vertex_after} non-manifold vertices")

    return (repaired, source_vertices, source_faces) if return_sources else repaired


def filter_faces(verts, faces, opacity=None, min_opacity=0.0, max_edge=0.0, verbose=True, return_mask=False):
    """Drop faces that the renderer barely uses but that F1 still pays for.

    The TNT evaluator samples the surface proportionally to face area, so
    near-transparent or oversized primitives contribute samples far from any real
    surface even though they contribute almost nothing to the image.
    """
    F_before = faces.shape[0]
    keep = np.ones(F_before, dtype=bool)

    if min_opacity > 0.0 and opacity is not None:
        keep &= opacity[faces].mean(axis=1) >= min_opacity
        if verbose:
            print(f"[filter] opacity >= {min_opacity}: kept {keep.sum()}/{F_before}")

    if max_edge > 0.0:
        tri = verts[faces]
        e = np.linalg.norm(np.stack([tri[:, 1] - tri[:, 0],
                                     tri[:, 2] - tri[:, 1],
                                     tri[:, 0] - tri[:, 2]]), axis=-1).max(axis=0)
        keep &= e <= max_edge
        if verbose:
            print(f"[filter] longest edge <= {max_edge}: kept {keep.sum()}/{F_before}")

    if not keep.any():
        print("[filter] every face was rejected, keeping the mesh unfiltered")
        return np.ones(F_before, dtype=bool) if return_mask else faces
    if verbose:
        print(f"[filter] faces: {F_before} → {int(keep.sum())} "
              f"({keep.mean() * 100:.2f}% kept)")
    return keep if return_mask else faces[keep]


def export_ply(out_path, verts_np, faces_np, colors_rgb, zup_to_yup=False,
               clean=False, min_faces_ratio=0.01, keep_largest=False,
               manifold=False):
    if zup_to_yup:
        transformed_verts = verts_np.copy()
        transformed_verts[:, 1] = verts_np[:, 2]
        transformed_verts[:, 2] = -verts_np[:, 1]
        verts_np = transformed_verts

    mesh = trimesh.Trimesh(vertices=verts_np.astype(np.float32),
                           faces=faces_np.astype(np.int32),
                           vertex_colors=colors_rgb.astype(np.uint8),
                           process=False)

    if clean:
        mesh = clean_mesh(mesh,
                          min_faces_ratio=min_faces_ratio,
                          keep_largest=keep_largest)
    if manifold:
        mesh = make_manifold(mesh)

    mesh.export(out_path, file_type='ply')
    return mesh

def prepare_bake_scene(scene, *, opacity=None, min_opacity=0.0, max_edge=0.0,
                       clean=False, min_faces_ratio=0.01, keep_largest=False, manifold=False):
    """Apply geometry edits before splitting color seams, retaining SH/texture IDs."""
    if not (min_opacity > 0 or max_edge > 0 or clean or manifold):
        return scene
    source_vertices = np.arange(len(scene.vertices))
    source_faces = np.arange(len(scene.faces))
    if min_opacity > 0 or max_edge > 0:
        mask = filter_faces(scene.vertices, scene.faces, opacity, min_opacity, max_edge, return_mask=True)
        source_faces = source_faces[mask]
    mesh = trimesh.Trimesh(vertices=scene.vertices, faces=scene.faces[source_faces], process=False)
    if clean:
        mesh, vertex_ids, face_ids = clean_mesh(mesh, min_faces_ratio, keep_largest, return_sources=True)
        source_vertices = source_vertices[vertex_ids]
        source_faces = source_faces[face_ids]
    if manifold:
        mesh, vertex_ids, face_ids = make_manifold(mesh, return_sources=True)
        source_vertices = source_vertices[vertex_ids]
        source_faces = source_faces[face_ids]
    if not len(mesh.faces):
        raise ValueError("No faces remain after export preprocessing")
    # No geometry operation changes corner order: texture barycentrics stay valid.
    return replace(scene, vertices=np.asarray(mesh.vertices, dtype=np.float32),
                   faces=np.asarray(mesh.faces, dtype=np.int32),
                   dc=scene.dc[source_vertices], rest=scene.rest[source_vertices],
                   textures=None if scene.textures is None else scene.textures[source_faces])


def main(argv=None):
    p = argparse.ArgumentParser(description="Bake full SH and per-face MipTex to standard vertex-colored PLY.")
    p.add_argument("scene_dir", help="Model root, checkpoint directory, or point_cloud_state_dict.pt")
    p.add_argument("--out", default=None, help="Default: <checkpoint directory>/mesh.ply")
    p.add_argument("--iteration", type=int, default=-1, help="Saved iteration; -1 selects the latest")
    p.add_argument("--color-mode", choices=("auto", "visible-mean", "camera", "dc"), default="auto",
                   help="auto: visible-mean unless --camera is given; dc ignores higher-order SH")
    p.add_argument("--camera", type=parse_vec3, default=None, help="Bake full SH from camera center x,y,z")
    p.add_argument("--degree", type=int, default=None, help="Override the evaluated SH degree")
    p.add_argument("--cameras", help="cameras.json; default is the model root")
    p.add_argument("--holdout", type=int, default=8, help="Exclude every Nth camera; 0 includes all")
    p.add_argument("--train-list", help="Explicit training image names, one per line; overrides holdout")
    p.add_argument("--visibility-max-size", type=int, default=1600)
    p.add_argument("--visibility-supersampling", type=int, default=2)
    p.add_argument("--texture-mode", choices=("vertex", "subdivide"), default="vertex",
                   help="vertex fits MipTex to the original vertices; subdivide preserves more interior detail")
    p.add_argument("--texture-subdivisions", type=int, default=None,
                   help="Samples per triangle edge minus one; default R-1 for RxR textures")
    p.add_argument("--mip-level", type=int, default=0, help="Fixed texture mip to bake (default highest detail)")
    p.add_argument("--chunk-faces", type=int, default=1024, help="Faces processed together during streamed export")
    p.add_argument("--geometry-only", action="store_true", help="Preserve geometry export with legacy DC colors; skip texture baking")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing appearance PLY and export report")
    p.add_argument("--cpu", action="store_true", help="CPU for camera/DC export; visible-mean requires CUDA")
    p.add_argument("--zup", action="store_true", help="Rotate Z-up geometry to Y-up after color baking")
    p.add_argument("--clean", action="store_true", help="Drop small components before appearance baking")
    p.add_argument("--min_faces_ratio", type=float, default=0.01)
    p.add_argument("--keep_largest", action="store_true")
    p.add_argument("--manifold", action="store_true", help="Repair topology before appearance baking")
    p.add_argument("--min_opacity", type=float, default=0.0)
    p.add_argument("--max_edge", type=float, default=0.0)
    args = p.parse_args(argv)

    from utils.glb_export import (load_scene as load_export_scene, resolve_checkpoint,
                                  load_training_cameras, bake_vertex_colors)
    from utils.ply_export import write_ply
    checkpoint, root = resolve_checkpoint(args.scene_dir, args.iteration)
    out = Path(args.out).expanduser().resolve() if args.out else checkpoint.parent / "mesh.ply"
    if out.suffix.lower() != ".ply":
        p.error("--out must end in .ply")
    if not args.geometry_only and not args.overwrite and (out.exists() or out.with_name(out.name + ".export.json").exists()):
        raise FileExistsError(f"Output already exists; use --overwrite: {out}")
    if not 0 <= args.min_opacity <= 1 or not 0 <= args.min_faces_ratio <= 1 or args.max_edge < 0:
        p.error("opacity and component ratio must be in [0,1], max_edge must be nonnegative")
    if args.texture_subdivisions is not None and args.texture_mode != "subdivide":
        p.error("--texture-subdivisions requires --texture-mode subdivide")
    if args.chunk_faces < 1 or args.texture_subdivisions is not None and args.texture_subdivisions < 1:
        p.error("--chunk-faces and --texture-subdivisions must be positive")
    if args.mip_level < 0 or args.visibility_max_size < 1 or args.visibility_supersampling < 1:
        p.error("mip level must be nonnegative; visibility dimensions must be positive")
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {checkpoint}", flush=True)
    scene = load_export_scene(checkpoint)
    opacity = None
    if args.min_opacity > 0:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if "vertex_weight" in state:
            opacity = torch.sigmoid(state["vertex_weight"].detach().float()).numpy().reshape(-1)
        else:
            print("[filter] checkpoint has no vertex_weight, opacity filter skipped")
        del state
    original_counts = {"vertices": len(scene.vertices), "faces": len(scene.faces)}
    scene = prepare_bake_scene(scene, opacity=opacity, min_opacity=args.min_opacity,
                               max_edge=args.max_edge, clean=args.clean, min_faces_ratio=args.min_faces_ratio,
                               keep_largest=args.keep_largest, manifold=args.manifold)
    if args.geometry_only:
        colors, _ = bake_vertex_colors(scene, "dc", device="cpu")
        colors = np.floor(np.clip(colors, 0, 1) * 255 + .5).astype(np.uint8)
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh = export_ply(out, scene.vertices, scene.faces, colors, zup_to_yup=args.zup)
        print(f"Saved geometry PLY: {out} ({len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces)")
        return
    if args.degree is not None:
        if not 0 <= args.degree <= 4 or (args.degree + 1)**2 > scene.rest.shape[1] + 1:
            p.error("--degree exceeds the SH coefficients in the checkpoint")
        scene = replace(scene, degree=args.degree)
    mode = ("camera" if args.camera is not None else "visible-mean") if args.color_mode == "auto" else args.color_mode
    if args.camera is not None and mode != "camera":
        p.error("--camera requires --color-mode camera (or auto)")
    if mode == "camera" and args.camera is None:
        p.error("--color-mode camera requires --camera x,y,z")
    cameras = None
    if mode == "visible-mean":
        camera_path = Path(args.cameras) if args.cameras else root / "cameras.json"
        if not camera_path.is_file():
            raise FileNotFoundError(f"Missing {camera_path}; supply --cameras or --camera x,y,z")
        if device == "cpu":
            raise RuntimeError("Training-view visibility baking requires CUDA; use --camera x,y,z for CPU baking")
        cameras = load_training_cameras(camera_path, args.holdout, args.train_list)
    colors, bake = bake_vertex_colors(scene, mode, cameras, device, args.visibility_max_size,
                                      args.visibility_supersampling,
                                      camera_center=None if args.camera is None else args.camera.numpy())
    bake.update(sh_degree=scene.degree, checkpoint=str(checkpoint),
                holdout=args.holdout if mode == "visible-mean" and args.train_list is None else None,
                explicit_train_list=args.train_list is not None,
                original_counts=original_counts,
                preprocessing={"clean": args.clean, "manifold": args.manifold,
                               "min_opacity": args.min_opacity, "max_edge": args.max_edge})
    report = write_ply(scene, colors, out, texture_mode=args.texture_mode,
                       subdivisions=args.texture_subdivisions, mip_level=args.mip_level,
                       chunk_faces=args.chunk_faces, zup=args.zup, overwrite=args.overwrite, bake_info=bake)
    print(f"Saved appearance PLY: {out}")
    print(f"Export report: {out.with_name(out.name + '.export.json')}")
    return report

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)