"""Bounded-memory export of SH + signed per-face texture to ordinary glTF 2.0.

No training/CUDA extensions are imported for DC export. Only visibility baking
needs nvdiffrast. GLB stores complete RGB PNGs, not raw signed residual images.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from PIL import Image
import torch

from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import C0, eval_sh


@dataclass
class ExportScene:
    vertices: np.ndarray
    faces: np.ndarray
    dc: np.ndarray
    rest: np.ndarray
    degree: int
    textures: np.ndarray | None
    use_mip: bool


def resolve_checkpoint(model, iteration=-1):
    model = Path(model).expanduser().resolve()
    if model.is_file():
        checkpoint = model
    elif (model / "point_cloud_state_dict.pt").is_file():
        checkpoint = model / "point_cloud_state_dict.pt"
    else:
        if iteration == -1:
            candidates = [(int(p.parent.name.removeprefix("iteration_")), p)
                          for p in (model / "point_cloud").glob("iteration_*/point_cloud_state_dict.pt")
                          if p.parent.name.removeprefix("iteration_").isdigit()]
            if not candidates:
                raise FileNotFoundError(f"No saved checkpoint under {model}")
            checkpoint = max(candidates)[1]
        else:
            checkpoint = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    root = checkpoint.parent
    if root.name.startswith("iteration_") and root.parent.name == "point_cloud":
        root = root.parent.parent
    return checkpoint, root


def _finite(array, name, chunk=8192):
    for start in range(0, len(array), chunk):
        if not np.isfinite(array[start:start + chunk]).all():
            raise ValueError(f"{name} contains non-finite values")


def load_scene(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise ValueError("Expected point_cloud_state_dict.pt, not an optimizer/resume checkpoint")

    def array(key):
        value = state.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint is missing tensor {key}")
        return value.detach().cpu().numpy()

    vertices = np.asarray(array("triangles_points"), dtype=np.float32)
    faces = array("_triangle_indices")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("triangles_points must be nonempty [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.dtype.kind not in "iu":
        raise ValueError("_triangle_indices must be nonempty integer [F,3]")
    if faces.min() < 0 or faces.max() >= len(vertices) or len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("Triangle indices exceed the vertex array/int32 range")
    faces = np.asarray(faces, dtype=np.int32)
    dc = np.asarray(array("features_dc"), dtype=np.float32)
    if dc.shape != (len(vertices), 1, 3):
        raise ValueError("features_dc must have shape [V,1,3]")
    degree = int(state.get("active_sh_degree", 0))
    rest = (np.asarray(array("features_rest"), dtype=np.float32) if "features_rest" in state
            else np.empty((len(vertices), 0, 3), dtype=np.float32))
    if not 0 <= degree <= 4 or rest.ndim != 3 or rest.shape[::2] != (len(vertices), 3):
        raise ValueError("Invalid SH degree or features_rest shape")
    if rest.shape[1] + 1 < (degree + 1) ** 2:
        raise ValueError("Not enough SH coefficients for active_sh_degree")
    textures = None
    if state.get("face_textures") is not None:
        textures = np.asarray(array("face_textures"), dtype=np.float32)
        if (textures.ndim != 4 or textures.shape[0] != len(faces) or textures.shape[-1] != 3
                or textures.shape[1] != textures.shape[2] or textures.shape[1] < 1):
            raise ValueError("face_textures must have shape [F,R,R,3]")
        if int(state.get("face_tex_res", textures.shape[1])) != textures.shape[1]:
            raise ValueError("face_tex_res does not match face_textures")
        _finite(textures, "face_textures")
    for name, value in (("vertices", vertices), ("features_dc", dc), ("features_rest", rest)):
        _finite(value, name)
    return ExportScene(vertices, faces, dc, rest, degree, textures,
                       bool(state.get("use_mip_texture", textures is not None)))


def load_training_cameras(path, holdout=8, train_list=None):
    if holdout < 0 or holdout == 1:
        raise ValueError("--holdout must be 0 or at least 2")
    cameras = json.loads(Path(path).read_text())
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("cameras.json must contain a nonempty camera list")
    names = [camera["img_name"] for camera in cameras]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("Camera img_name values must be unique strings")
    cameras = sorted(cameras, key=lambda camera: camera["img_name"])
    if train_list is not None:
        selected = {line.strip() for line in Path(train_list).read_text().splitlines() if line.strip()}
        unknown = selected - set(names)
        if unknown:
            raise ValueError(f"Unknown training camera names: {sorted(unknown)[:5]}")
        cameras = [c for c in cameras if c["img_name"] in selected]
    elif holdout:
        cameras = [c for i, c in enumerate(cameras) if i % holdout != 0]
    if not cameras:
        raise ValueError("Training camera selection is empty; check --holdout/--train-list")
    return cameras


def camera_matrices(camera, max_size):
    """Saved cameras.json uses a +Z-forward camera-to-world rotation."""
    width, height = int(camera["width"]), int(camera["height"])
    fx, fy = float(camera["fx"]), float(camera["fy"])
    c2w = np.eye(4)
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    center = np.asarray(camera["position"], dtype=np.float64)
    if rotation.shape != (3, 3) or center.shape != (3,):
        raise ValueError("Invalid camera rotation/position dimensions")
    c2w[:3, :3], c2w[:3, 3] = rotation, center
    if (min(width, height, fx, fy) <= 0 or not np.isfinite([fx, fy]).all()
            or not np.isfinite(c2w).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)):
        raise ValueError("Invalid camera dimensions, focal length or rotation")
    view = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32).T.copy())
    projection = getProjectionMatrix(.01, 100, 2 * math.atan(width / (2 * fx)),
                                     2 * math.atan(height / (2 * fy))).T
    scale = min(1.0, max_size / max(width, height))
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    return view @ projection, center.astype(np.float32), size


@torch.no_grad()
def bake_vertex_colors(scene, mode, cameras=None, device="cuda", max_size=1600, supersampling=2,
                       camera_center=None):
    base = np.maximum(scene.dc[:, 0] * C0 + .5, 0)
    if mode == "dc":
        return base, {"mode": "dc", "training_views": 0}
    if mode == "camera":
        center = np.asarray(camera_center, dtype=np.float32)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("camera mode requires a finite camera center [x,y,z]")
        result = np.empty_like(base)
        for start in range(0, len(scene.vertices), 131072):
            end = min(start + 131072, len(scene.vertices))
            vertices = torch.from_numpy(scene.vertices[start:end].copy()).to(device)
            direction = vertices - torch.from_numpy(center).to(device)
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            coefficients = torch.from_numpy(np.concatenate(
                (scene.dc[start:end], scene.rest[start:end]), axis=1)).to(device)
            result[start:end] = (eval_sh(scene.degree, coefficients.transpose(1, 2), direction)
                                + .5).clamp_min(0).cpu().numpy()
        return result, {"mode": mode, "training_views": 0, "camera_center": center.tolist()}
    if mode != "visible-mean" or not cameras:
        raise ValueError("visible-mean requires training cameras")
    if max_size < 1 or supersampling < 1:
        raise ValueError("Visibility size and supersampling must be positive")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("visible-mean requires CUDA; use --color-mode dc for CPU export")
    try:
        import nvdiffrast.torch as dr
    except ImportError as error:
        raise ImportError("Install/build submodules/nvdiffrast and source scripts/env.sh for visible-mean") from error
    vertices = torch.from_numpy(scene.vertices).to(device)
    faces = torch.from_numpy(np.ascontiguousarray(scene.faces)).to(device)
    coefficients = torch.from_numpy(np.concatenate((scene.dc, scene.rest), axis=1)).to(device)
    homogeneous = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=1)
    color_sum = torch.zeros_like(vertices)
    count = torch.zeros(len(vertices), dtype=torch.int32, device=device)
    context = dr.RasterizeCudaContext(device=device)
    for i, camera in enumerate(cameras):
        projection, center, (height, width) = camera_matrices(camera, max_size)
        clip = homogeneous @ projection.to(device)
        clip[:, 2] = 2 * clip[:, 2] - clip[:, 3]
        raster, _ = dr.rasterize(context, clip[None].contiguous(), faces,
                                 resolution=[height * supersampling, width * supersampling])
        face_ids = torch.unique(raster[0, :, :, 3].long())
        visible = torch.unique(faces[face_ids[face_ids > 0] - 1].flatten()).long()
        del raster, clip
        camera_center = torch.from_numpy(center).to(device)
        for start in range(0, len(visible), 131072):
            ids = visible[start:start + 131072]
            direction = vertices[ids] - camera_center
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            color_sum[ids] += (eval_sh(scene.degree, coefficients[ids].transpose(1, 2), direction) + .5).clamp_min(0)
            count[ids] += 1
        print(f"SH bake {i + 1}/{len(cameras)}: {camera['img_name']}, {len(visible):,} visible vertices", flush=True)
    observed = count > 0
    result = torch.where(observed[:, None], color_sum / count.clamp_min(1)[:, None],
                         torch.from_numpy(base).to(device)).cpu().numpy()
    return result, {"mode": mode, "training_views": len(cameras),
                    "training_camera_names": [c["img_name"] for c in cameras],
                    "observed_vertices": int(observed.sum()), "unobserved_fallback": "SH DC",
                    "visibility_max_size": max_size, "visibility_supersampling": supersampling}


def mip_sizes(resolution):
    sizes = [resolution]
    while sizes[-1] > 1:
        sizes.append(sizes[-1] // 2)
    return sizes


def face_mip(textures, level):
    """Match native avg_pool2d(2), independently per face, including odd sizes."""
    result = textures
    for _ in range(level):
        size = result.shape[1] // 2
        result = result[:, :2 * size, :2 * size].reshape(-1, size, 2, size, 2, 3).mean(axis=(2, 4))
    return result


def bake_tiles(vertex_colors, face_textures, tile_size, mip_level=0):
    # Texture row is wC, column is wB. Reflect only the SH extension outside
    # the triangular domain; keep the trained texture's entire square for mips.
    v, u = np.mgrid[:tile_size, :tile_size].astype(np.float32) / (tile_size - 1)
    outside = u + v > 1
    b, c = np.where(outside, 1 - v, u), np.where(outside, 1 - u, v)
    bary = np.stack((1 - b - c, b, c), axis=-1)
    full = np.einsum("hwk,fkc->fhwc", bary, vertex_colors, optimize=True)
    if face_textures is not None:
        texture = face_mip(face_textures, mip_level)
        size = texture.shape[1]
        if size == tile_size:
            full += texture
        else:
            x, y = u * (size - 1), v * (size - 1)
            x0, y0 = x.astype(np.int64), y.astype(np.int64)
            x1, y1 = np.minimum(x0 + 1, size - 1), np.minimum(y0 + 1, size - 1)
            dx, dy = (x - x0)[None, :, :, None], (y - y0)[None, :, :, None]
            full += ((1 - dx) * (1 - dy) * texture[:, y0, x0] + dx * (1 - dy) * texture[:, y0, x1]
                     + (1 - dx) * dy * texture[:, y1, x0] + dx * dy * texture[:, y1, x1])
    clipped = int(np.count_nonzero((full < 0) | (full > 1)))
    # Clip only AFTER adding the signed texture contribution.
    return np.floor(np.clip(full, 0, 1) * 255 + .5).astype(np.uint8), clipped


def _place_tiles(atlas, tiles, start, grid, cell):
    offset = 0
    while offset < len(tiles):
        row, col = divmod(start + offset, grid)
        count = min(grid - col, len(tiles) - offset)
        block = tiles[offset:offset + count].transpose(1, 0, 2, 3).reshape(cell, count * cell, 3)
        atlas[row * cell:(row + 1) * cell, col * cell:(col + count) * cell] = block
        offset += count


def write_glb(scene, colors, out, *, atlas_size=4096, padding=2, mip_level=0,
              chunk_faces=8192, zup=False, overwrite=False, bake_info=None):
    out = Path(out).expanduser().resolve()
    report_path = out.with_name(out.name + ".export.json")
    if out.suffix.lower() != ".glb":
        raise ValueError("--out must end in .glb")
    if not overwrite and (out.exists() or report_path.exists()):
        raise FileExistsError(f"Output already exists; use --overwrite: {out}")
    if atlas_size < 2 or atlas_size & (atlas_size - 1) or atlas_size > 16384:
        raise ValueError("--atlas-size must be a power of two between 2 and 16384")
    if padding < 1 or chunk_faces < 1:
        raise ValueError("--padding and --chunk-faces must be positive")
    resolution = scene.textures.shape[1] if scene.textures is not None else 1
    sizes = mip_sizes(resolution)
    if mip_level < 0 or mip_level >= len(sizes):
        raise ValueError(f"--mip-level must be between 0 and {len(sizes) - 1}")
    tile_size = max(2, resolution)
    cell = tile_size + 2 * padding
    grid = atlas_size // cell
    if grid < 1:
        raise ValueError("Atlas is smaller than a padded face tile")
    if colors.shape != scene.vertices.shape:
        raise ValueError("Baked vertex colors must have shape [V,3]")
    _finite(colors, "baked vertex colors")
    out.parent.mkdir(parents=True, exist_ok=True)
    capacity = grid * grid
    report = {"format": "t2splat-static-glb/1", "vertices": len(scene.vertices), "faces": len(scene.faces),
              "exported_corner_vertices": 3 * len(scene.faces), "source_texture_resolution": resolution,
              "has_face_textures": scene.textures is not None, "source_uses_mip": scene.use_mip,
              "native_mip_sizes": sizes, "baked_mip_level": mip_level, "tile_size": tile_size,
              "atlas_size": atlas_size, "padding": padding, "zup_to_yup": zup,
              "color_bake": bake_info or {},
              "color_formula": "interpolated max(SH(direction)+0.5,0) + signed face texture; clamp/quantize last",
              "mip_policy": "PNG base level; viewer-generated atlas mips, not native per-face LOD",
              "material": "KHR_materials_unlit, double-sided, OPAQUE",
              "geometry": "all spatial triangles preserved; vertices split at per-face UV seams"}
    gltf = {"asset": {"version": "2.0", "generator": "T2-Splat create_glb.py"},
            "extensionsUsed": ["KHR_materials_unlit"], "extensionsRequired": ["KHR_materials_unlit"],
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": out.stem}],
            "meshes": [{"primitives": []}], "buffers": [], "bufferViews": [], "accessors": [],
            "images": [], "textures": [], "materials": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9987 if scene.use_mip else 9729,
                          "wrapS": 33071, "wrapT": 33071}]}
    clipped = 0
    geometry_hash = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix=".t2splat-glb-", dir=out.parent) as temporary:
        binary_path = Path(temporary) / "payload.bin"
        with binary_path.open("wb") as binary:
            def buffer_view(data, target=None):
                binary.write(b"\0" * (-binary.tell() % 4))
                view = {"buffer": 0, "byteOffset": binary.tell(), "byteLength": len(data)}
                if target is not None:
                    view["target"] = target
                binary.write(data)
                if binary.tell() >= 2**32:
                    raise ValueError("GLB exceeds its 4 GiB format limit")
                gltf["bufferViews"].append(view)
                return len(gltf["bufferViews"]) - 1

            def accessor(values, kind, bounds=False):
                values = np.ascontiguousarray(values, dtype="<f4")
                view = buffer_view(values.tobytes(), 34962)
                entry = {"bufferView": view, "componentType": 5126, "count": len(values), "type": kind}
                if bounds:
                    entry.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
                gltf["accessors"].append(entry)
                return len(gltf["accessors"]) - 1

            for page, first in enumerate(range(0, len(scene.faces), capacity)):
                last = min(first + capacity, len(scene.faces))
                atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
                for start in range(first, last, chunk_faces):
                    end = min(start + chunk_faces, last)
                    texture = None if scene.textures is None else scene.textures[start:end]
                    tiles, n_clipped = bake_tiles(colors[scene.faces[start:end]], texture, tile_size, mip_level)
                    clipped += n_clipped
                    padded = np.pad(tiles, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="edge")
                    _place_tiles(atlas, padded, start - first, grid, cell)
                positions = scene.vertices[scene.faces[first:last]].reshape(-1, 3).copy()
                if zup:
                    positions = positions[:, [0, 2, 1]]
                    positions[:, 2] *= -1
                geometry_hash.update(np.ascontiguousarray(positions, dtype="<f4").tobytes())
                indices = np.arange(last - first)
                origins = np.stack((indices % grid, indices // grid), axis=-1) * cell + padding + .5
                uv = (origins[:, None] + np.array([[0, 0], [tile_size - 1, 0], [0, tile_size - 1]])) / atlas_size
                position_accessor = accessor(positions, "VEC3", bounds=True)
                uv_accessor = accessor(uv.reshape(-1, 2), "VEC2")
                png = io.BytesIO()
                Image.fromarray(atlas).save(png, format="PNG", compress_level=3)
                image_view = buffer_view(png.getvalue())
                gltf["images"].append({"bufferView": image_view, "mimeType": "image/png"})
                gltf["textures"].append({"source": page, "sampler": 0})
                gltf["materials"].append({"name": f"baked_page_{page}", "doubleSided": True, "alphaMode": "OPAQUE",
                    "extensions": {"KHR_materials_unlit": {}}, "pbrMetallicRoughness": {
                        "baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": page},
                        "metallicFactor": 0, "roughnessFactor": 1}})
                gltf["meshes"][0]["primitives"].append({"attributes": {
                    "POSITION": position_accessor, "TEXCOORD_0": uv_accessor}, "material": page, "mode": 4})
                print(f"Atlas {page + 1}/{math.ceil(len(scene.faces) / capacity)}: {last - first:,} faces", flush=True)
            length = binary.tell()
        report.update(atlas_count=len(gltf["images"]), expanded_positions_sha256=geometry_hash.hexdigest(),
                      clipped_channel_fraction=clipped / (len(scene.faces) * tile_size**2 * 3))
        gltf["extras"] = {k: report[k] for k in ("format", "source_texture_resolution", "source_uses_mip",
                                               "baked_mip_level", "mip_policy", "geometry")}
        gltf["buffers"] = [{"byteLength": length}]
        payload = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode()
        payload += b" " * (-len(payload) % 4)
        binary_length = length + (-length % 4)
        total = 12 + 8 + len(payload) + 8 + binary_length
        if total >= 2**32:
            raise ValueError("GLB exceeds its 4 GiB format limit")
        candidate = Path(temporary) / "model.glb"
        with candidate.open("wb") as stream:
            stream.write(struct.pack("<III", 0x46546C67, 2, total))
            stream.write(struct.pack("<II", len(payload), 0x4E4F534A))
            stream.write(payload)
            stream.write(struct.pack("<II", binary_length, 0x004E4942))
            with binary_path.open("rb") as binary:
                shutil.copyfileobj(binary, stream, 8 * 1024 * 1024)
            stream.write(b"\0" * (-length % 4))
        report["bytes"] = total
        candidate_report = Path(temporary) / "report.json"
        candidate_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        candidate.replace(out)
        candidate_report.replace(report_path)
    return report


def export_checkpoint(model, out, iteration=-1, color_mode="visible-mean", cameras=None,
                      holdout=8, train_list=None, visibility_max_size=1600,
                      visibility_supersampling=2, device="cuda", **options):
    # Fail before the expensive visibility pass if the destination is invalid.
    destination = Path(out).expanduser().resolve()
    if destination.suffix.lower() != ".glb":
        raise ValueError("--out must end in .glb")
    if not options.get("overwrite", False) and (destination.exists() or destination.with_name(destination.name + ".export.json").exists()):
        raise FileExistsError(f"Output already exists; use --overwrite: {destination}")
    checkpoint, root = resolve_checkpoint(model, iteration)
    selected = None
    if color_mode == "visible-mean":
        camera_path = Path(cameras) if cameras is not None else root / "cameras.json"
        if not camera_path.is_file():
            raise FileNotFoundError(f"Missing {camera_path}; supply --cameras or choose --color-mode dc")
        selected = load_training_cameras(camera_path, holdout, train_list)
    print(f"Loading {checkpoint}", flush=True)
    scene = load_scene(checkpoint)
    colors, bake = bake_vertex_colors(scene, color_mode, selected, device,
                                      visibility_max_size, visibility_supersampling)
    bake.update(holdout=holdout if train_list is None else None, explicit_train_list=train_list is not None)
    return write_glb(scene, colors, out, bake_info=bake, **options)
