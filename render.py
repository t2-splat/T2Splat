# Repository maintenance and English documentation: zhtang2004@gmail.com.
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
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import torch.nn.functional as F
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import loadBestIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel

def depth_to_colormap(depth, cmap_name="turbo"):
    """Convert a [1, H, W] depth tensor to a [3, H, W] colormap image."""
    import matplotlib.cm as cm
    d = depth.squeeze(0).cpu().numpy()
    valid = d > 0
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        if hi - lo < 1e-6:
            hi = lo + 1.0
        normed = np.clip((d - lo) / (hi - lo), 0, 1)
    else:
        normed = np.zeros_like(d)
    normed[~valid] = 0
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(normed)[:, :, :3]
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def normal_to_blue_purple_image(n, world_view_transform=None):
    """
    Classic blue-ish normal map look: encode view-space normals as RGB.

    In camera/view space we force each direction into the +Z hemisphere (toward
    the camera in OpenGL-style view), then map (nx,ny,nz) -> (nx,ny,nz)*0.5+0.5.
    Flat surfaces facing the camera read ~ (0.5, 0.5, 1.0) — blue/purple tint.

    If ``world_view_transform`` is given, ``n`` is treated as world-space and
    rotated to view space first (for depth-derived normals).
    """
    if world_view_transform is not None:
        R = world_view_transform[:3, :3]
        n = torch.einsum("ij,jhw->ihw", R, n)
    n = F.normalize(n, dim=0, eps=1e-8)
    # Face +Z hemisphere (toward camera in GL view space) → flat areas → ~(0.5,0.5,1) blue
    flip = (n[2:3] < 0).to(n.dtype)
    n = n * (1.0 - 2.0 * flip)
    n = F.normalize(n, dim=0, eps=1e-8)
    return (n * 0.5 + 0.5).clamp(0, 1)


def render_set(model_path, name, iteration, views, triangles, pipeline,
               background, save_depth=False, save_normal=False):
    base = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(base, "renders")
    gts_path = os.path.join(base, "gt")
    depth_path = os.path.join(base, "depth")
    depth_npy_path = os.path.join(base, "depth_npy")
    # normal/ and surf_normal/: view-space normals facing the camera, with Y flipped.
    normal_path = os.path.join(base, "normal")
    surf_normal_path = os.path.join(base, "surf_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if save_depth:
        makedirs(depth_path, exist_ok=True)
        makedirs(depth_npy_path, exist_ok=True)
    if save_normal:
        makedirs(normal_path, exist_ok=True)
        makedirs(surf_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, triangles, pipeline, background)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        # For alpha-masked ground truth, cameras.py already zeroes the background.
        # Rendering fills it with the configured background (white when requested).
        # Composite the same background into saved ground truth for consistent metrics.
        if getattr(view, "gt_alpha_mask", None) is not None:
            alpha = view.gt_alpha_mask.to(gt.device)
            bg = background.to(gt.device).view(3, 1, 1)
            gt = (gt + bg * (1.0 - alpha)).clamp(0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if save_depth:
            surf_depth = render_pkg["surf_depth"]           # [1, H, W]
            np.save(os.path.join(depth_npy_path, '{0:05d}.npy'.format(idx)),
                    surf_depth.squeeze(0).cpu().numpy())
            depth_vis = depth_to_colormap(surf_depth)
            torchvision.utils.save_image(depth_vis,
                os.path.join(depth_path, '{0:05d}.png'.format(idx)))

        if save_normal:
            wvt = view.world_view_transform
            n_view = render_pkg["rend_normal_view"]
            torchvision.utils.save_image(
                normal_to_blue_purple_image(n_view),
                os.path.join(normal_path, '{0:05d}.png'.format(idx)))
            surf_w = render_pkg["surf_normal_vis"]
            torchvision.utils.save_image(
                normal_to_blue_purple_image(surf_w, world_view_transform=wvt),
                os.path.join(surf_normal_path, '{0:05d}.png'.format(idx)))

def render_sets(dataset : ModelParams, iteration, pipeline : PipelineParams,
                skip_train : bool, skip_test : bool,
                save_depth : bool = False, save_normal : bool = False):
    with torch.no_grad():
        if iteration == "best":
            best = loadBestIteration(dataset.model_path)
            if best is not None:
                iteration = best
                print(f"[render] Using best model at iteration {iteration}")
            else:
                iteration = -1
                print("[render] No best_iteration.json found, falling back to latest")

        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=iteration,
                  shuffle=False)

        # Restore use_mip_texture from the checkpoint to match training.

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter,
                        scene.getTrainCameras(), triangles, pipeline,
                        background, save_depth=save_depth,
                        save_normal=save_normal)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter,
                        scene.getTestCameras(), triangles, pipeline,
                        background, save_depth=save_depth,
                        save_normal=save_normal)

def _parse_iteration(value):
    if value == "best":
        return "best"
    return int(value)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=_parse_iteration)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--save_depth", action="store_true",
                        help="Save per-view depth maps (colormap PNG + raw .npy)")
    parser.add_argument("--save_normal", action="store_true",
                        help="Save per-view normal maps (rend_normal + surf_normal)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test,
                save_depth=args.save_depth, save_normal=args.save_normal)