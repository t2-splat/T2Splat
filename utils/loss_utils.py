#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def inverse_depth_prior_loss(prediction, target, mask):
    """Foreground-weighted inverse-depth L1, excluding invalid prior values."""
    valid = torch.isfinite(target) & (target > 0) & torch.isfinite(mask) & (mask > 0)
    weights = torch.where(valid, mask, torch.zeros_like(mask))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = (prediction - safe_target).abs()
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def normal_prior_loss(prediction, target, foreground_mask=None):
    """Camera-space normal consistency on valid foreground pixels only."""
    valid = torch.isfinite(target).all(dim=0, keepdim=True)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    valid = valid & (safe_target.norm(dim=0, keepdim=True) > 1e-6)
    target_unit = F.normalize(safe_target, dim=0, eps=1e-6)
    weights = valid.to(prediction.dtype)
    if foreground_mask is not None:
        weights = weights * torch.nan_to_num(foreground_mask, nan=0.0).clamp(0, 1)
    error = 1.0 - (prediction * target_unit).sum(dim=0, keepdim=True)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def vertex_normal_alignment_loss(vertices, triangle_indices, image_2D,
                                  vertex_rendered, gt_normal,
                                  world_view_transform):
    """
    Per-vertex normal alignment with GT normal map.

    1. Compute area-weighted vertex normals from the mesh
    2. Rotate to view space (same coordinate frame as gt_normal)
    3. Project visible vertices to screen, sample GT normal
    4. Cosine distance: 1 - dot(v_normal, gt_normal)
    """
    idx = triangle_indices.long()
    v0, v1, v2 = vertices[idx[:, 0]], vertices[idx[:, 1]], vertices[idx[:, 2]]
    face_cross = torch.cross(v1 - v0, v2 - v0, dim=-1)     # [F, 3] (area-weighted)

    # Area-weighted vertex normals via scatter_add
    V = vertices.shape[0]
    v_normal = torch.zeros(V, 3, device=vertices.device)
    face_ids = idx.reshape(-1)                                # [3F]
    face_normals_rep = face_cross.repeat(3, 1)                # [3F, 3]

    # idx[:,0], idx[:,1], idx[:,2] → scatter into their vertex slots
    v_normal.scatter_add_(0, torch.cat([idx[:, 0], idx[:, 1], idx[:, 2]]).unsqueeze(1).expand(-1, 3),
                          torch.cat([face_cross, face_cross, face_cross], dim=0))
    v_normal = F.normalize(v_normal, dim=-1, eps=1e-8)        # [V, 3]

    # Rotate to view space
    R = world_view_transform[:3, :3]                          # [3, 3]
    v_normal_view = v_normal @ R                              # [V, 3]
    v_normal_view = F.normalize(v_normal_view, dim=-1, eps=1e-8)

    # Screen projection — reuse image_2D from render_pkg
    H, W = gt_normal.shape[1], gt_normal.shape[2]
    v_coords = image_2D.to(gt_normal)
    y = v_coords[:, 1]
    x = v_coords[:, 0]

    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    visible = inside & (vertex_rendered > 0)
    if visible.sum() < 10:
        return torch.tensor(0.0, device=vertices.device)

    x_idx = torch.clamp(torch.round(x[visible]).long(), 0, W - 1)
    y_idx = torch.clamp(torch.round(y[visible]).long(), 0, H - 1)

    gt_n = gt_normal[:, y_idx, x_idx].permute(1, 0)          # [N, 3]
    gt_n = F.normalize(gt_n, dim=-1, eps=1e-8)

    # Only use pixels where GT normal is valid (nonzero)
    gt_len = gt_n.norm(dim=-1)
    valid = gt_len > 0.5
    if valid.sum() < 10:
        return torch.tensor(0.0, device=vertices.device)

    vn = v_normal_view[visible][valid]
    gn = gt_n[valid]

    cosine = (vn * gn).sum(dim=-1)
    return (1.0 - cosine.abs()).mean()


def vertex_depth_loss_hr(vertex_depth_out, image_2D, vertex_rendered, depth_full_hr, max_diff_threshold=0.5, iteration=10):
    assert depth_full_hr.dim() == 3 and depth_full_hr.size(0) == 1, "depth_full_hr must be (1,H,W)"

    v_depths = vertex_depth_out
    v_coords = image_2D.to(depth_full_hr)  # [y, x] as float

    H, W = depth_full_hr.shape[1], depth_full_hr.shape[2]

    # SWAPPED: v_coords stores [y, x] not [x, y]
    y = v_coords[:, 1]  # height dimension - FIRST COLUMN
    x = v_coords[:, 0]  # width dimension - SECOND COLUMN

    # inside image
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    keep = inside
    
    if not torch.any(keep):
        print("NO values")
        return torch.tensor(0.0, device=vertex_depth_out.device, dtype=vertex_depth_out.dtype)

    x = x[keep]; y = y[keep]; v_depths = v_depths[keep]

    # round to nearest pixel, then clamp for safety
    x_idx = torch.clamp(torch.round(x).long(), 0, W - 1)  # width indices
    y_idx = torch.clamp(torch.round(y).long(), 0, H - 1)  # height indices

    # Sampling with proper indices
    sampled = depth_full_hr[0, y_idx, x_idx]

    valid = torch.isfinite(sampled) & (sampled > 0)
    if not torch.any(valid):
        return torch.tensor(0.0, device=vertex_depth_out.device, dtype=vertex_depth_out.dtype)

    diffs = torch.abs((v_depths[valid] - sampled[valid]))
    use = diffs < max_diff_threshold
    if not torch.any(use):
        return torch.tensor(0.0, device=vertex_depth_out.device, dtype=vertex_depth_out.dtype)
    
    return diffs[use].mean()


def u_shaped_opacity_loss(x, center=0.1, width=0.03):
    # Normalized distance to center (e.g., 0.1)
    penalty = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    return penalty.mean()

def binarization_loss(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)  # avoid log(0)
    return -x * torch.log(x) - (1 - x) * torch.log(1 - x)


def mesh_regularization_losses(vertices, triangle_indices,
                               smooth_method="uniform",
                               compute_smooth=True,
                               compute_edge=True,
                               compute_nc=True):
    """
    Compute mesh regularization losses with shared edge/face tables.

    Normal consistency uses only edges with exactly two incident faces.
    It assumes consistent winding; it does not enforce manifold topology.

    Returns dict with keys ``"smooth"``, ``"edge"``, ``"nc"``
    (each ``0.0`` when disabled).
    """
    idx = triangle_indices.long()
    nF = idx.shape[0]
    V = vertices.shape[0]
    device = vertices.device
    i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]

    result = {"smooth": 0.0, "edge": 0.0, "nc": 0.0}
    if nF == 0:
        zero = vertices.sum() * 0.0
        return {key: zero if enabled else 0.0 for key, enabled in
                (("smooth", compute_smooth), ("edge", compute_edge), ("nc", compute_nc))}

    # ── shared: face vertices & normals (needed by smooth-cot and nc) ──
    v0 = v1 = v2 = face_normals = None
    need_face_verts = (compute_smooth and smooth_method != "uniform") or compute_nc
    if need_face_verts:
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        face_normals = F.normalize(
            torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)     # [F, 3]

    # ── shared: unique edge table (needed by edge and nc) ──
    unique_edges = edge_keys = inverse = None
    if compute_edge or compute_nc:
        e_src = torch.cat([i0, i1, i0])                         # [3F]
        e_dst = torch.cat([i1, i2, i2])                         # [3F]
        e_lo = torch.min(e_src, e_dst)
        e_hi = torch.max(e_src, e_dst)
        V_max = V + 1
        edge_keys = e_lo.long() * V_max + e_hi.long()           # [3F]
        unique_keys, inverse, edge_counts = torch.unique(
            edge_keys, return_inverse=True, return_counts=True)

        if compute_edge:
            # Decode keys directly; reversed scatter indices select unrelated edges.
            ue_lo = torch.div(unique_keys, V_max, rounding_mode="floor")
            ue_hi = unique_keys.remainder(V_max)
            unique_edges = torch.stack([ue_lo, ue_hi], dim=1)   # [E, 2]

    # ── 1. Laplacian smoothing ──
    if compute_smooth:
        if smooth_method == "uniform":
            src = torch.cat([i0, i1, i2, i1, i2, i0])
            dst = torch.cat([i1, i2, i0, i0, i1, i2])
            nbr_sum = torch.zeros_like(vertices)
            nbr_cnt = torch.zeros(V, 1, device=device)
            nbr_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3),
                                 vertices[src])
            nbr_cnt.scatter_add_(0, dst.unsqueeze(1),
                                 torch.ones(dst.shape[0], 1, device=device))
            nbr_cnt = nbr_cnt.clamp(min=1)
            Lv = nbr_sum / nbr_cnt - vertices
        else:
            A = (v2 - v1).norm(dim=1)
            B = (v0 - v2).norm(dim=1)
            C = (v1 - v0).norm(dim=1)
            s = 0.5 * (A + B + C)
            area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
            A2, B2, C2 = A * A, B * B, C * C
            cota = (B2 + C2 - A2) / (4.0 * area)
            cotb = (A2 + C2 - B2) / (4.0 * area)
            cotc = (A2 + B2 - C2) / (4.0 * area)

            w_nbr = torch.zeros(V, 3, device=device)
            w_sum = torch.zeros(V, 1, device=device)
            for w, si, di in [(cota, i1, i2), (cotb, i2, i0), (cotc, i0, i1)]:
                w3 = w.unsqueeze(1).expand(-1, 3)
                w1 = w.unsqueeze(1)
                w_nbr.scatter_add_(0, di.unsqueeze(1).expand(-1, 3),
                                   w3 * vertices[si])
                w_nbr.scatter_add_(0, si.unsqueeze(1).expand(-1, 3),
                                   w3 * vertices[di])
                w_sum.scatter_add_(0, di.unsqueeze(1), w1)
                w_sum.scatter_add_(0, si.unsqueeze(1), w1)
            Lv = w_nbr - w_sum * vertices

            if smooth_method == "cotcurv":
                v_area = torch.zeros(V, 1, device=device)
                fa = area.unsqueeze(1)
                v_area.scatter_add_(0, i0.unsqueeze(1), fa)
                v_area.scatter_add_(0, i1.unsqueeze(1), fa)
                v_area.scatter_add_(0, i2.unsqueeze(1), fa)
                Lv = Lv / v_area.clamp(min=1e-8)

        result["smooth"] = Lv.norm(dim=1).mean()

    # ── 2. Edge loss ──
    if compute_edge:
        v_src = vertices[unique_edges[:, 0]]
        v_dst = vertices[unique_edges[:, 1]]
        lengths = (v_src - v_dst).norm(dim=1)
        result["edge"] = (lengths ** 2).mean()

    # ── 3. Normal consistency ──
    if compute_nc:
        E = unique_keys.shape[0]
        face_ids = torch.arange(nF, device=device).repeat(3)    # [3F]
        face_a = torch.full((E,), nF, dtype=torch.long, device=device)
        face_b = torch.full((E,), -1, dtype=torch.long, device=device)
        face_a.scatter_reduce_(0, inverse, face_ids, reduce="amin")
        face_b.scatter_reduce_(0, inverse, face_ids, reduce="amax")
        valid = (edge_counts == 2) & (face_a != face_b)
        fa, fb = face_a[valid], face_b[valid]
        dot = (face_normals[fa] * face_normals[fb]).sum(dim=-1)
        # Empty adjacency must stay finite and differentiable.
        result["nc"] = (1.0 - dot.clamp(-1.0, 1.0)).sum() / valid.sum().clamp(min=1)

    return result


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def lp_loss(pred, target, p=0.7, eps=1e-6):
    """
    Computes Lp loss with 0 < p < 1.
    Args:
        pred: (N, C, H, W) predicted image
        target: (N, C, H, W) groundtruth image
        p: norm degree < 1
        eps: small constant for numerical stability
    """
    diff = torch.abs(pred - target) + eps
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

