# Adapted from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/evaluate_dtu_mesh.py
#
# Changes from original GOF script (to fit this repo / NeuS-preprocessed DTU):
#   1. load_dtu_camera() reads poses from `scan{id}/cameras.npz` (world_mat_i)
#      instead of `Calibration/cal18/pos_XXX.txt` (we don't have those files).
#   2. cull_mesh() uses `camera.gt_alpha_mask` when present, otherwise loads
#      `data/DTU/scan{id}/mask/*.png`. Our Camera class has `gt_alpha_mask`
#      not `gt_mask`.
#   3. Final eval.py path points to this repo's `eval_dtu/eval.py`.
#   4. Mask dilation radius exposed as CLI `--mask_dilate` (default 6 as in GOF).

import os
import sys
import glob
import subprocess
import math
from os import path
from argparse import ArgumentParser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import random
import trimesh
from PIL import Image
from skimage.morphology import binary_dilation, disk

from arguments import ModelParams, PipelineParams, get_combined_args
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def best_fit_transform(A, B):
    """Kabsch/Procrustes: find R,t that map A -> B (same shape Nxm)."""
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    T = np.eye(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t


def load_dtu_camera_from_npz(scan_dir):
    """Return dict[int image_idx -> 3x4 camtoworld pose] in DTU world coords."""
    cam_file = os.path.join(scan_dir, "cameras.npz")
    cd = np.load(cam_file)
    n = sum(1 for k in cd.files if k.startswith("world_mat_") and not k.startswith("world_mat_inv"))
    poses = {}
    for i in range(n):
        world_mat = cd[f"world_mat_{i}"].astype(np.float32)
        P = world_mat[:3, :4]
        K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        K = K / K[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        poses[i] = pose[:3]
    return poses


def _image_idx_from_name(image_name):
    """DTU image stems like '0000' / 'IMG_0001.jpg' → integer 0 / 1."""
    digits = "".join(c for c in image_name if c.isdigit())
    return int(digits)


def _get_cam_mask(camera, scan_dir, idx_in_scan):
    """Return HxW uint8/0-255 mask as tensor [1,H,W] on CPU.

    Priority:
        1. camera.gt_alpha_mask (if image had alpha channel).
        2. External mask file in `{scan_dir}/mask/`.
    """
    m = getattr(camera, "gt_alpha_mask", None)
    if m is not None:
        m = m.detach().cpu().float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        m = m[:1]
        m = (m * 255.0).clamp(0, 255)
        return m

    mask_candidates = sorted(
        glob.glob(os.path.join(scan_dir, "mask", "*.png"))
        + glob.glob(os.path.join(scan_dir, "mask", "*.jpg"))
    )
    if not mask_candidates:
        raise RuntimeError(f"No mask found for scan dir {scan_dir}")
    mp = mask_candidates[idx_in_scan % len(mask_candidates)]
    img = Image.open(mp)
    if img.mode in ("RGBA", "LA"):
        img = img.split()[-1]
    else:
        img = img.convert("L")
    arr = np.array(img, dtype=np.float32)
    if arr.max() <= 1.0:
        arr *= 255.0
    # resize to camera resolution
    arr = cv2.resize(arr, (camera.image_width, camera.image_height),
                     interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(arr).unsqueeze(0)


def cull_mesh(cameras, mesh, scan_dir, mask_dilate=6):
    vertices = torch.from_numpy(mesh.vertices).cuda().float()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1).permute(1, 0)

    sampled_masks = []
    for idx, camera in enumerate(cameras):
        c2w = camera.world_view_transform.T.inverse()
        w2c = torch.inverse(c2w).cuda()

        fx = fov2focal(camera.FoVx, camera.image_width)
        fy = fov2focal(camera.FoVy, camera.image_height)
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = camera.image_width / 2.0
        intrinsic[1, 2] = camera.image_height / 2.0
        intrinsic = intrinsic.cuda()

        W, H = camera.image_width, camera.image_height
        mask_t = _get_cam_mask(camera, scan_dir, idx)  # [1,H,W] on cpu, 0-255

        with torch.no_grad():
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1.) & (pix_coords < 1.)).all(dim=-1).float()

            maski = mask_t[0].numpy().astype(np.float32) / 256.0
            maski = torch.from_numpy(binary_dilation(maski, disk(mask_dilate))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(
                maski, pix_coords[None, None], mode="nearest",
                padding_mode="zeros", align_corners=True,
            )[0, -1, 0]

            sampled_mask = sampled_mask + (1.0 - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    return mesh


def _load_train_cameras(dataset: ModelParams):
    """Load train Camera objects without touching any mesh/triangle model.

    Mirrors what scene/__init__.py does, minus the TriangleModel load:
    Colmap reader → CameraInfo list → Camera list via cameraList_from_camInfos.
    """
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path, dataset.images, dataset.eval
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](
            dataset.source_path, dataset.white_background, dataset.eval
        )
    else:
        raise RuntimeError(f"Could not recognize scene type at {dataset.source_path}")
    # shuffle=False equivalent: we just don't shuffle.
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset)


def evaluate_mesh(dataset: ModelParams, iteration: int, DTU_PATH: str,
                  ply_name: str, mask_dilate: int, workers: int = 8, seed: int = 0):
    train_cameras = _load_train_cameras(dataset)

    scan_dir = dataset.source_path
    dtu_cam_by_idx = load_dtu_camera_from_npz(scan_dir)

    # Pair each train camera with its DTU pose by image index, not list position.
    # When --eval is set, train_cameras skips every 8th image (llffhold=8) so
    # train_cameras[i] is NOT necessarily image index i.
    points_list, gt_list = [], []
    for cam in train_cameras:
        img_idx = _image_idx_from_name(cam.image_name)
        if img_idx not in dtu_cam_by_idx:
            print(f"[warn] skip camera {cam.image_name}: no world_mat_{img_idx} in cameras.npz")
            continue
        c2w_learned = cam.world_view_transform.T.inverse()[:3, 3].cpu().numpy()
        points_list.append(c2w_learned)
        gt_list.append(dtu_cam_by_idx[img_idx][:, 3])
    points = np.array(points_list)
    gt_points = np.array(gt_list)
    print(f"[align] using {len(points)} paired cameras")

    # scale + Kabsch alignment
    scale_points = np.linalg.norm(points - points.mean(0), axis=1).mean()
    scale_gt = np.linalg.norm(gt_points - gt_points.mean(0), axis=1).mean()
    scaled = points * scale_gt / scale_points
    _, r, t = best_fit_transform(scaled, gt_points)

    # load & cull mesh (in training/normalized coords)
    mesh_file = os.path.join(dataset.model_path, ply_name)
    print(f"[load] {mesh_file}")
    mesh = trimesh.load(mesh_file)

    print(f"[cull] disk(dilate={mask_dilate})")
    mesh = cull_mesh(train_cameras, mesh, scan_dir, mask_dilate=mask_dilate)

    culled_mesh_file = os.path.join(dataset.model_path, "recon_culled.ply")
    mesh.export(culled_mesh_file)

    # transform to DTU world
    mesh.vertices = mesh.vertices * scale_gt / scale_points
    mesh.vertices = mesh.vertices @ r.T + t

    aligned_mesh_file = os.path.join(dataset.model_path, "recon_aligned.ply")
    mesh.export(aligned_mesh_file)

    out_dir = os.path.join(dataset.model_path, "vis")
    os.makedirs(out_dir, exist_ok=True)
    scan_id = int(os.path.basename(scan_dir.rstrip("/"))[4:])

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
    cmd = [sys.executable, eval_script, "--data", aligned_mesh_file,
           "--scan", str(scan_id), "--mode", "mesh", "--dataset_dir", DTU_PATH,
           "--vis_out_dir", out_dir, "--workers", str(workers), "--seed", str(seed)]
    print(cmd, flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="DTU mesh eval (GOF-style)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--DTU", type=str, required=True,
                        help="Path to Official DTU Dataset (with ObsMask/ and Points/stl/)")
    parser.add_argument("--ply_name", type=str, default="mesh.ply",
                        help="Mesh file under model_path to evaluate")
    parser.add_argument("--mask_dilate", type=int, default=6,
                        help="Mask dilation radius in pixels (GOF default: 6)")
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for mesh sampling and nearest neighbors")
    parser.add_argument("--seed", type=int, default=0, help="Seed for evaluation point shuffling")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print(f"evaluating {args.model_path}")
    print(f"ply_name:    {args.ply_name}")
    print(f"mask_dilate: {args.mask_dilate}")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    evaluate_mesh(model.extract(args), args.iteration,
                  args.DTU, args.ply_name, args.mask_dilate, args.workers, args.seed)
