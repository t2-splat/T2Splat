# Project adaptations maintained by zhtang2004@gmail.com.
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

import os
import json
import time
import torch
from random import randint
from utils.loss_utils import (l1_loss, ssim, vertex_depth_loss_hr,
                              vertex_normal_alignment_loss,
                              mesh_regularization_losses, inverse_depth_prior_loss,
                              normal_prior_loss)
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
    print("Using fused SSIM for faster training.")
except:
    FUSED_SSIM_AVAILABLE = False
try:
    from diff_triangle_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
    print("Sparse Adam optimizer available")
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.render_utils import (
    generate_path,
    generate_smooth_orbit_path,
    create_videos,
    save_img_u8,
    write_mesh_evolution_video,
)
from scene.heartbeat import heartbeat_step, accumulate_pm_stats, visibility_collapse_step
from utils.visibility_pruning import (VisibilityPruningSchedule, collect_face_visibility,
                                      finalize_visibility_pruning, require_surface_topology)
from utils.mesh_topology import TopologyMonitor
from utils.training_snapshots import save_training_snapshot
from scene.triangle_model import build_face_tex_mirror
from scene.colmap_loader import read_images_binary_name_order, read_images_text_name_order



def training(
        dataset,   
        opt, 
        pipe,
        testing_iterations,
        checkpoint, 
        debug_from,
        scene_name,
        use_sparse_adam=False,
        saving_iterations=(),
        save_topology_snapshots=False,
        ):
    
    saving_iterations = set(saving_iterations)
    visibility_schedule = VisibilityPruningSchedule.from_options(opt)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)

    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)

    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_training_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # define the scheduler for sigma and opacity
    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter

    lambda_weight = opt.lambda_weight
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    
    need_delaunay = False

    topology_interval = int(getattr(opt, "topology_monitor_interval", 0))
    topology_monitor = (TopologyMonitor(os.path.join(dataset.model_path, "topology_metrics.jsonl"))
                        if topology_interval > 0 or visibility_schedule.enabled else None)

    def record_topology(stage, iteration, vertices=None, faces=None):
        if topology_monitor is not None:
            return topology_monitor.record(
                iteration, stage,
                triangles.vertices if vertices is None else vertices,
                triangles._triangle_indices if faces is None else faces)

    mesh_viz_interval = int(getattr(opt, "mesh_viz_interval", 0) or 0)
    mesh_viz_dir = None
    mesh_viz_orbit = bool(getattr(opt, "mesh_viz_orbit", False))
    mesh_viz_use_train_views = bool(getattr(opt, "mesh_viz_use_train_views", False))
    mesh_viz_rdt_snapshots = mesh_viz_interval > 0 and bool(getattr(opt, "mesh_viz_rdt_snapshots", True))
    rdt_probe = bool(getattr(opt, "rdt_probe", False))
    if rdt_probe:
        from utils.rdt_probe import evaluate_views, save_rdt_event
    mesh_viz_include_test_views = bool(getattr(opt, "mesh_viz_include_test_views", True))
    mesh_viz_train_cam_indexing = str(
        getattr(opt, "mesh_viz_train_cam_indexing", "interval") or "interval"
    ).lower()
    mesh_viz_traj = None
    mesh_viz_train_cams_ordered = None
    mesh_viz_frame_idx = 0
    if mesh_viz_interval > 0:
        mesh_viz_dir = os.path.join(dataset.model_path, "mesh_evolution")
        os.makedirs(mesh_viz_dir, exist_ok=True)
        if mesh_viz_orbit:
            n_orbit = int(getattr(opt, "mesh_viz_orbit_frames", 480) or 480)
            n_orbit = max(1, n_orbit)
            orbit_cams = list(scene.getTrainCameras() or [])
            if mesh_viz_include_test_views:
                orbit_cams = orbit_cams + list(scene.getTestCameras() or [])
            if not orbit_cams:
                orbit_cams = list(scene.getTrainCameras() or [])
            orbit_style = str(getattr(opt, "mesh_viz_orbit_style", "smooth") or "smooth").lower()
            if orbit_style == "legacy":
                mesh_viz_traj = generate_path(orbit_cams, n_frames=n_orbit)
                print(f"[mesh_viz] every {mesh_viz_interval} iters → {mesh_viz_dir} "
                      f"(orbit=legacy ellipse+PCA, frames={n_orbit})")
            else:
                mesh_viz_traj = generate_smooth_orbit_path(
                    orbit_cams,
                    n_frames=n_orbit,
                    radius_scale=float(getattr(opt, "mesh_viz_orbit_radius_scale", 1.0) or 1.0),
                    height_frac=float(getattr(opt, "mesh_viz_orbit_height_frac", 0.08) or 0.0),
                    ease=str(getattr(opt, "mesh_viz_orbit_ease", "smoothstep") or "smoothstep"),
                    loops=float(getattr(opt, "mesh_viz_orbit_loops", 1.0) or 1.0),
                    focus_mode=str(
                        getattr(opt, "mesh_viz_orbit_focus", "cameras_mean") or "cameras_mean"
                    ),
                )
                print(f"[mesh_viz] every {mesh_viz_interval} iters → {mesh_viz_dir} "
                      f"(orbit=smooth, frames={n_orbit}, ease={getattr(opt, 'mesh_viz_orbit_ease', 'smoothstep')}, "
                      f"focus={getattr(opt, 'mesh_viz_orbit_focus', 'cameras_mean')})")
        elif mesh_viz_use_train_views:
            # Scene() shuffles train cameras for training; for visualization we want a stable order.
            import re
            from pathlib import Path

            def _natural_key(s: str):
                # split into text / number chunks: "img_12.png" -> ["img_", 12, ".png"]
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

            cams_train = scene.getTrainCameras()
            cams_test = scene.getTestCameras() if mesh_viz_include_test_views else []
            cams = (cams_train or []) + (cams_test or [])
            if cams:
                order_mode = str(getattr(opt, "mesh_viz_camera_order", "auto") or "auto").lower()

                def _stem_key(name: str) -> str:
                    x = str(name).strip().replace("\\", "/").split("/")[-1]
                    return os.path.splitext(x)[0].lower()

                def _full_name_key(name: str) -> str:
                    return str(name).strip().replace("\\", "/").split("/")[-1].lower()

                def _add_key(bucket: dict, k: str, cam):
                    if not k:
                        return
                    bucket.setdefault(k, []).append(cam)

                def _build_stem_bucket(cam_list):
                    stem_to_cams = {}
                    for c in cam_list:
                        if not hasattr(c, "image_name"):
                            continue
                        raw = str(getattr(c, "image_name", ""))
                        _add_key(stem_to_cams, _stem_key(raw), c)
                        if "." in raw:
                            _add_key(stem_to_cams, _full_name_key(raw), c)
                    return stem_to_cams

                def _pop_cam(bucket: dict, k: str):
                    lst = bucket.get(k, None)
                    if not lst:
                        return None
                    cam = lst.pop(0)
                    if not lst:
                        del bucket[k]
                    return cam

                def _order_from_names(name_list, stem_bucket, allowed_stems):
                    ordered_local = []
                    for raw_name in name_list:
                        sk = _stem_key(raw_name)
                        if allowed_stems is not None and sk not in allowed_stems:
                            continue
                        cam = _pop_cam(stem_bucket, sk)
                        if cam is None:
                            cam = _pop_cam(stem_bucket, _full_name_key(raw_name))
                        if cam is not None:
                            ordered_local.append(cam)
                    return ordered_local

                # Directory used for optional filtering / disk ordering
                override_dir = str(getattr(opt, "mesh_viz_images_dir", "") or "").strip()
                images_dir = override_dir if override_dir else str(getattr(dataset, "images", "images"))
                images_path = Path(images_dir)
                if not images_path.is_absolute():
                    images_path = Path(dataset.source_path) / images_dir

                sparse_dir = Path(dataset.source_path) / "sparse" / "0"
                images_bin = sparse_dir / "images.bin"
                images_txt = sparse_dir / "images.txt"

                if order_mode == "auto":
                    if images_bin.is_file():
                        order_mode = "colmap"
                    elif images_txt.is_file():
                        order_mode = "colmap"
                    else:
                        order_mode = "disk"

                stem_bucket = _build_stem_bucket(cams)
                ordered = None
                order_src = ""

                allowed_stems = None
                if images_path.is_dir():
                    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                    disk_files = [p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
                    allowed_stems = {_stem_key(p.name) for p in disk_files}

                if order_mode == "colmap":
                    if images_bin.is_file():
                        colmap_names = read_images_binary_name_order(str(images_bin))
                        order_src = f"colmap:{images_bin}"
                    elif images_txt.is_file():
                        colmap_names = read_images_text_name_order(str(images_txt))
                        order_src = f"colmap:{images_txt}"
                    else:
                        order_mode = "disk"
                        print("[mesh_viz][WARN] colmap order requested/missing images.bin/.txt — fall back to disk.")

                if order_mode == "colmap":
                    # If user points mesh_viz_images_dir at images_4, filter COLMAP names to that folder only.
                    filt = allowed_stems if (override_dir.strip() and images_path.is_dir()) else None
                    ordered = _order_from_names(colmap_names, stem_bucket, filt)
                if order_mode == "disk":
                    if not images_path.is_dir():
                        ordered = None
                    else:
                        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                        files = [p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
                        files.sort(key=lambda p: _natural_key(p.name))
                        order_src = f"disk:{images_path}"
                        ordered = _order_from_names([p.name for p in files], stem_bucket, None)

                if ordered is not None:
                    used = {id(c) for c in ordered}
                    rest = [c for c in cams if id(c) not in used]
                    rest.sort(key=lambda c: _natural_key(str(getattr(c, "image_name", getattr(c, "uid", 0)))))
                    ordered.extend(rest)

                    matched = len(ordered) - len(rest)
                    print(f"[mesh_viz] camera order ({order_src})  matched={matched}  "
                          f"total_cams={len(cams)} (train={len(cams_train or [])}, test={len(cams_test or [])})")
                    if matched == 0 and len(cams) > 0:
                        print("[mesh_viz][WARN] 0 cameras matched ordering — check COLMAP names vs Camera.image_name, "
                              "or set --mesh_viz_camera_order disk and verify --mesh_viz_images_dir.")

                if ordered is not None and len(ordered) > 0:
                    mesh_viz_train_cams_ordered = ordered
                else:
                    # Fallback: stable natural sort by image_name / uid
                    if hasattr(cams[0], "image_name"):
                        mesh_viz_train_cams_ordered = sorted(
                            cams, key=lambda c: _natural_key(str(getattr(c, "image_name", "")))
                        )
                    else:
                        mesh_viz_train_cams_ordered = sorted(cams, key=lambda c: int(getattr(c, "uid", 0)))
            print(f"[mesh_viz] every {mesh_viz_interval} iters → {mesh_viz_dir} "
                  f"(train cam indexing={mesh_viz_train_cam_indexing})")
        else:
            print(f"[mesh_viz] every {mesh_viz_interval} iters → {mesh_viz_dir} "
                  f"(cam index {getattr(opt, 'mesh_viz_cam_index', 0)})")

    run_restricted_delaunay = (
        opt.run_restricted_delaunay
        if getattr(opt, "run_restricted_delaunay", -1) and opt.run_restricted_delaunay > 0
        else opt.densify_until_iter + 1000
    )

    def _mesh_viz_pick_camera(iteration=None, advance: bool = True):
        nonlocal mesh_viz_frame_idx
        cams_train = scene.getTrainCameras()
        cams_test = scene.getTestCameras() if mesh_viz_include_test_views else []
        cams = (cams_train or []) + (cams_test or [])
        if not cams:
            return None
        if mesh_viz_orbit and mesh_viz_traj is not None and len(mesh_viz_traj) > 0:
            cam = mesh_viz_traj[mesh_viz_frame_idx % len(mesh_viz_traj)]
            if advance:
                mesh_viz_frame_idx += 1
            return cam
        if mesh_viz_use_train_views:
            ordered = mesh_viz_train_cams_ordered if (mesh_viz_train_cams_ordered is not None and len(mesh_viz_train_cams_ordered) > 0) else cams
            if mesh_viz_train_cam_indexing == "interval" and iteration is not None and mesh_viz_interval > 0:
                it = int(iteration)
                # Map: 500 -> images_4 order #1, 1000 -> #2, ...  (k = it / interval)
                if it % mesh_viz_interval == 0:
                    k = it // mesh_viz_interval
                elif it == opt.iterations:
                    k = (it + mesh_viz_interval - 1) // mesh_viz_interval
                else:
                    k = it // mesh_viz_interval
                k = max(k, 1)
                slot = (k - 1) % len(ordered)
                return ordered[slot]
            cam = ordered[mesh_viz_frame_idx % len(ordered)]
            if advance:
                mesh_viz_frame_idx += 1
            return cam
        cam_idx = int(getattr(opt, "mesh_viz_cam_index", 0) or 0)
        return cams[cam_idx % len(cams)]

    def _mesh_viz_save(tag, iteration, face_idx_override=None, advance_camera: bool = True, seq=None):
        if mesh_viz_interval <= 0 or mesh_viz_dir is None:
            return
        cam = _mesh_viz_pick_camera(iteration=iteration, advance=advance_camera)
        if cam is None:
            return
        with torch.no_grad():
            original_faces = None
            if face_idx_override is not None:
                original_faces = triangles._triangle_indices
                triangles._triangle_indices = face_idx_override
            rp = render(cam, triangles, pipe, background)
            img_np = torch.clamp(rp["render"], 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
            if seq is None:
                fname = f"{tag}_{iteration:06d}.png"
            else:
                fname = f"{tag}_{iteration:06d}_{int(seq):02d}.png"
            save_img_u8(img_np, os.path.join(mesh_viz_dir, fname))
            if original_faces is not None:
                triangles._triangle_indices = original_faces

    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)

    # Inline face texture schedule: parse "20000:4,25000:6" → [(20000,4), (25000,6)]
    face_tex_schedule = []
    if opt.face_tex_res_schedule:
        for entry in opt.face_tex_res_schedule.split(","):
            it_str, res_str = entry.strip().split(":")
            face_tex_schedule.append((int(it_str), int(res_str)))
        face_tex_schedule.sort(key=lambda x: x[0])
        print(f"[FaceTexture] Inline basecolor schedule: {face_tex_schedule}")
    visibility_schedule.validate_training(
        opt, run_restricted_delaunay,
        texture_start=face_tex_schedule[0][0] if face_tex_schedule else None)
    if visibility_schedule.enabled:
        print(f"[Visibility] gradual safe collapse {visibility_schedule.start}..{visibility_schedule.end} "
              f"every {visibility_schedule.interval}, threshold "
              f"{visibility_schedule.threshold_start}..{visibility_schedule.threshold_end}, "
              f"face budget {visibility_schedule.max_face_ratio:.1%}; no final face pruning")
    elif visibility_schedule.mode == "none":
        print("[Visibility] disabled after RDT; no gradual collapse, texture pre-prune "
              "or final face pruning; compact unused vertices only")
    face_tex_phase = 0
    face_tex_mirror = None  # (inv_r, inv_c, mir_r, mir_c)

    for iteration in range(first_iter, opt.iterations + 1):

        if need_delaunay:
            record_topology("before_rdt", iteration)
            if save_topology_snapshots:
                save_training_snapshot(scene, iteration, "before_rdt")
            if rdt_probe:
                scene.save(iteration - 1)
                evaluate_views(scene, triangles, pipe, background, render, "before", iteration)
            if mesh_viz_rdt_snapshots:
                # RDT stages should share the same viewpoint for fair visual comparison.
                _mesh_viz_save("rdt_00_before", iteration, advance_camera=False)
            with torch.no_grad():
                rdt_debug = triangles.run_restricted_delaunay(
                    return_debug=mesh_viz_rdt_snapshots or rdt_probe,
                    backend=opt.rdt_backend, max_distance_ratio=opt.rdt_max_distance_ratio,
                    min_normal_cos=opt.rdt_min_normal_cos, min_quality=opt.rdt_min_quality)
            if rdt_probe:
                record_topology("legacy_rdt_candidate", iteration, faces=rdt_debug["selected_faces"])
                evaluate_views(scene, triangles, pipe, background, render, "legacy", iteration,
                               faces=rdt_debug["selected_faces"])
                evaluate_views(scene, triangles, pipe, background, render, "surface", iteration)
                save_rdt_event(scene, triangles, rdt_debug, iteration)
            record_topology("after_rdt", iteration)
            if save_topology_snapshots:
                save_training_snapshot(scene, iteration, "after_rdt")
            if mesh_viz_rdt_snapshots and rdt_debug is not None:
                def _faces_to_cuda_i32(x):
                    if x is None:
                        return None
                    arr = np.asarray(x, dtype=np.int64)
                    if arr.size == 0:
                        return None
                    return torch.as_tensor(arr, device="cuda").to(torch.int32).contiguous()

                # Render the actual RDT pipeline stages directly.
                cand = _faces_to_cuda_i32(rdt_debug.get("candidate_faces"))
                if cand is not None:
                    _mesh_viz_save("rdt_10_candidate", iteration, face_idx_override=cand, advance_camera=False)

                sel = _faces_to_cuda_i32(rdt_debug.get("selected_faces"))
                if sel is not None:
                    _mesh_viz_save("rdt_20_selected", iteration, face_idx_override=sel, advance_camera=False)

                rej = _faces_to_cuda_i32(rdt_debug.get("rejected_faces"))
                if rej is not None:
                    _mesh_viz_save("rdt_15_rejected", iteration, face_idx_override=rej, advance_camera=False)

                _mesh_viz_save("rdt_30_after", iteration, advance_camera=False)

                # Optional extra: progressive "surface tetrahedra disappear" sequence.
                # Start from all candidate faces, then gradually remove rejected faces.
                rdt_cand_steps = int(getattr(opt, "mesh_viz_rdt_candidate_steps", 0) or 0)
                if rdt_cand_steps > 0 and cand is not None:
                    if rej is not None and rej.shape[0] > 0:
                        # Build a fast lookup from rejected faces to candidate indices.
                        cand_np = cand.detach().cpu().numpy().astype(np.int64)
                        rej_np = rej.detach().cpu().numpy().astype(np.int64)
                        cand_keys = {
                            (int(f[0]), int(f[1]), int(f[2])): idx
                            for idx, f in enumerate(cand_np)
                        }
                        rej_idx = []
                        for f in rej_np:
                            k = (int(f[0]), int(f[1]), int(f[2]))
                            if k in cand_keys:
                                rej_idx.append(cand_keys[k])
                        rej_idx = np.asarray(rej_idx, dtype=np.int64)
                        keep_mask = np.ones(cand_np.shape[0], dtype=bool)
                        n_rej = rej_idx.shape[0]

                        for si in range(1, rdt_cand_steps + 1):
                            # Slow-start disappearance so early frames keep more tetrahedra.
                            # si=1 -> almost no removal, last step -> remove all rejected.
                            if rdt_cand_steps <= 1:
                                prog = 1.0
                            else:
                                prog = float(si - 1) / float(rdt_cand_steps - 1)
                            n_drop = int(round((prog ** 1.8) * n_rej))
                            n_drop = min(n_rej, max(0, n_drop))
                            keep_mask[:] = True
                            if n_drop > 0:
                                keep_mask[rej_idx[:n_drop]] = False
                            sub_np = cand_np[keep_mask]
                            if sub_np.shape[0] == 0:
                                sub_np = cand_np[:1]
                            sub = torch.as_tensor(sub_np, device="cuda").to(torch.int32).contiguous()
                            _mesh_viz_save(
                                "rdt_candprog", iteration, face_idx_override=sub,
                                advance_camera=False, seq=si,
                            )
                    else:
                        # No rejected set available: fallback to previous prefix growth.
                        for si in range(1, rdt_cand_steps + 1):
                            n_keep = max(1, (si * cand.shape[0]) // rdt_cand_steps)
                            sub = cand[:n_keep].contiguous()
                            _mesh_viz_save(
                                "rdt_candprog", iteration, face_idx_override=sub,
                                advance_camera=False, seq=si,
                            )
                elif rdt_cand_steps > 0 and sel is not None:
                    # Rare fallback: if candidate is unavailable, interpolate on selected set.
                    for si in range(1, rdt_cand_steps + 1):
                        n_keep = max(1, (si * sel.shape[0]) // rdt_cand_steps)
                        sub = sel[:n_keep].contiguous()
                        _mesh_viz_save(
                            "rdt_candprog", iteration, face_idx_override=sub,
                            advance_camera=False, seq=si,
                        )
            del rdt_debug
            need_delaunay = False

        # Supersampling
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)

        # Sigma schedule
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = (iteration - sigma_start) / (total_iters - sigma_start)
            progress = min(progress, 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        # Inline face texture: bake at first phase, upsample at later phases
        if face_tex_phase < len(face_tex_schedule):
            trigger_iter, trigger_res = face_tex_schedule[face_tex_phase]
            if iteration == trigger_iter:
                from scene.triangle_model import build_face_tex_mirror
                remaining = opt.iterations - trigger_iter
                if face_tex_phase == 0:
                    if visibility_schedule.mode == "legacy":
                        with torch.no_grad():
                            n_before = triangles._triangle_indices.shape[0]
                            triangles.importance_score = torch.zeros(
                                n_before, dtype=torch.float, device="cuda")
                            for vc in scene.getTrainCameras():
                                rp = render(vc, triangles, pipe, background)
                                imp = rp["max_blending"].detach()
                                better = imp > triangles.importance_score
                                triangles.importance_score[better] = imp[better]
                            keep = triangles.importance_score > 0.5
                            if not keep.all():
                                triangles.prune_triangles(keep)
                                device = triangles.vertices.device
                                used = torch.zeros(triangles.vertices.shape[0],
                                                   dtype=torch.bool, device=device)
                                if triangles._triangle_indices.numel() > 0:
                                    used[triangles._triangle_indices.flatten()] = True
                                triangles._prune_vertices(used)
                                n_after = triangles._triangle_indices.shape[0]
                                print(f"[Prune@{iteration}] {n_before} → {n_after} faces "
                                      f"(removed {n_before - n_after})")
                            else:
                                print(f"[Prune@{iteration}] No faces to prune ({n_before} kept)")

                    triangles.bake_face_textures_basecolor(
                        face_tex_res=trigger_res, lr=opt.lr_face_texture,
                        lr_final=opt.lr_face_texture_final, max_steps=remaining)
                    print(f"[FaceTexture@{iteration}] Baked basecolor R={trigger_res}: "
                          f"{triangles.face_textures.shape}")
                else:
                    triangles.upsample_face_textures(
                        new_res=trigger_res, lr=opt.lr_face_texture,
                        lr_final=opt.lr_face_texture_final, max_steps=remaining)
                    print(f"[FaceTexture@{iteration}] Upsampled to R={trigger_res}: "
                          f"{triangles.face_textures.shape}")
                triangles.update_face_tex_lr_origin(trigger_iter)
                inv_r, inv_c, mir_r, mir_c, _ = build_face_tex_mirror(
                    trigger_res, device=triangles.face_textures.device)
                face_tex_mirror = (inv_r, inv_c, mir_r, mir_c)
                from train_face_texture import mirror_invalid_texels
                mirror_invalid_texels(triangles.face_textures, inv_r, inv_c, mir_r, mir_c)

                if face_tex_phase == 0:
                    lr_sh_tex = 0.004
                    for group in triangles.optimizer.param_groups:
                        if group["name"] == "vertices":
                            group["lr"] = 0.0
                        elif group["name"] == "vertex_weight":
                            group["lr"] = 0.0
                        elif group["name"] == "f_rest":
                            group["lr"] = lr_sh_tex / 20.0
                    print(f"[FaceTexture@{iteration}] Froze vertices, vertex_weight; "
                          f"f_rest lr={lr_sh_tex/20.0:.6f}")

                face_tex_phase += 1

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device="cuda") # reset to 0 to ensure that everything is deleted with an importance score of 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # GT may use alpha (e.g. DTU RGBA); composite with `bg` so aligned pixels match the render.
        if getattr(viewpoint_cam, "gt_alpha_mask", None) is not None:
            alpha = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image + bg.view(3, 1, 1) * (1.0 - alpha)
        if getattr(viewpoint_cam, "normal_map", None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            seg_hr = gt_normal.unsqueeze(0)  # -> [1, 3, H, W]
            seg_ds_area = F.interpolate(seg_hr, size=(gt_image.shape[1], gt_image.shape[2]), mode="area")  # [1, 3, H0, W0]
            gt_normal = seg_ds_area.squeeze(0)  # -> [3, H0, W0]
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        if not (hasattr(triangles, 'face_textures') and face_tex_phase > 0):
            image_size = render_pkg["scaling"].detach()
            mask = image_size > triangles.image_size
            triangles.image_size[mask] = image_size[mask]

            importance_score = render_pkg["max_blending"].detach()
            mask = importance_score > triangles.importance_score
            triangles.importance_score[mask] = importance_score[mask]

            pixel_count = render_pkg["triangle_was_rendered"].detach()
            mask = pixel_count > triangles.pixel_count
            triangles.pixel_count[mask] = pixel_count[mask]

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
    

        # FINAL LOSS
        loss = loss_image.clone()
        L_mesh_smooth = 0.0
        L_mesh_edge = 0.0
        L_mesh_nc = 0.0
        Lweight = 0
        Lvertex_depth = 0
        Ll1depth = 0
        Lnormal = 0
        normal_loss_super = 0

        in_face_tex_phase = hasattr(triangles, 'face_textures') and face_tex_phase > 0

        if not in_face_tex_phase:
            # Opacity loss
            Lweight_pure = 0.0
            lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
            if lambda_weight > 0:
                mask_out = triangles.vertices.shape[0]
                vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
                Lweight_pure = vertex_weights.mean()
                Lweight = lambda_weight * Lweight_pure
                loss += Lweight

            # Vertex depth regularization
            Lvertex_depth_pure = 0.0
            lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
            lambda_vnormal = opt.lambda_vertex_normal if iteration > opt.start_vertex_normal else 0
            if lambda_vertex > 0 or lambda_vnormal > 0:
                image_2D = render_pkg["image_2D"]
                vertex_rendered = render_pkg["vertex_rendered"]
            if lambda_vertex > 0:
                depth_down = render_pkg["surf_depth"]
                vertex_depth_out = render_pkg["vertex_depth_out"]
                Lvertex_depth_pure = vertex_depth_loss_hr(
                    vertex_depth_out,
                    image_2D,
                    vertex_rendered,
                    depth_down,
                    max_diff_threshold=opt.max_diff_threshold,
                )
                Lvertex_depth = lambda_vertex * Lvertex_depth_pure
                loss += Lvertex_depth

            # Depth loss
            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "invdepthmap", None) is not None:
                invDepth = 1.0 / render_pkg["expected_depth"].clamp(min=0.01)
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                foreground_mask = getattr(viewpoint_cam, "gt_alpha_mask", None)
                if foreground_mask is not None:
                    depth_mask = depth_mask * foreground_mask.to(depth_mask)
                Ll1depth_pure = inverse_depth_prior_loss(invDepth, mono_invdepth, depth_mask)
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth

            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']

            # Normal regularization (2DGS)
            Lnormal_pure = 0.0
            lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
            if lambda_normal > 0:
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                Lnormal_pure = normal_error.mean()
                Lnormal = lambda_normal * Lnormal_pure
                loss += Lnormal

            # supervised normal loss
            # The priors in normals/ are camera-space (Metric3D), so compare against
            # the view-space render, not the world-space rend_normal.
            if gt_normal is not None:
                lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
                foreground_mask = getattr(viewpoint_cam, "gt_alpha_mask", None)
                if foreground_mask is not None:
                    foreground_mask = foreground_mask.to(gt_normal)
                normal_loss_super = lambda_normals_super * normal_prior_loss(
                    render_pkg['rend_normal_view'], gt_normal, foreground_mask)
                loss += normal_loss_super

            # Vertex normal alignment with rendered surface normal (self-supervised)
            if lambda_vnormal > 0:
                L_vnormal = vertex_normal_alignment_loss(
                    triangles.vertices, triangles._triangle_indices,
                    image_2D, vertex_rendered, surf_normal,
                    viewpoint_cam.world_view_transform,
                )
                loss += lambda_vnormal * L_vnormal

            # PyTorch3D-style mesh regularization (shared edge computation)
            do_smooth = iteration > opt.mesh_smooth_start_iter and opt.lambda_mesh_smooth > 0
            do_edge = iteration > opt.mesh_edge_start_iter and opt.lambda_mesh_edge > 0
            do_nc = iteration > opt.mesh_nc_start_iter and opt.lambda_mesh_nc > 0
            if do_smooth or do_edge or do_nc:
                mesh_reg = mesh_regularization_losses(
                    triangles.vertices, triangles._triangle_indices,
                    smooth_method=opt.mesh_smooth_method,
                    compute_smooth=do_smooth,
                    compute_edge=do_edge,
                    compute_nc=do_nc,
                )
                if do_smooth:
                    L_mesh_smooth = mesh_reg["smooth"]
                    loss += opt.lambda_mesh_smooth * L_mesh_smooth
                if do_edge:
                    L_mesh_edge = mesh_reg["edge"]
                    loss += opt.lambda_mesh_edge * L_mesh_edge
                if do_nc:
                    L_mesh_nc = mesh_reg["nc"]
                    loss += opt.lambda_mesh_nc * L_mesh_nc

        if torch.isfinite(loss):
            loss.backward()
            if (opt.enable_heartbeat
                    and opt.heartbeat_start_iter < iteration <= (
                        opt.heartbeat_end_iter if opt.heartbeat_end_iter > 0
                        else opt.iterations)):
                accumulate_pm_stats(triangles, viewpoint_cam)
        else:
            img_w = loss_image.item()
            depth_w = Ll1depth.item() if isinstance(Ll1depth, torch.Tensor) else float(Ll1depth)
            normal_w = Lnormal.item() if isinstance(Lnormal, torch.Tensor) else float(Lnormal)
            weight_w = Lweight.item() if isinstance(Lweight, torch.Tensor) else float(Lweight)
            ms_w = (opt.lambda_mesh_smooth * L_mesh_smooth).item() if isinstance(L_mesh_smooth, torch.Tensor) else 0.0
            me_w = (opt.lambda_mesh_edge * L_mesh_edge).item() if isinstance(L_mesh_edge, torch.Tensor) else 0.0
            mnc_w = (opt.lambda_mesh_nc * L_mesh_nc).item() if isinstance(L_mesh_nc, torch.Tensor) else 0.0
            print(f"[WARN@{iteration}] loss={loss.item()}, skipping backward"
                  f"  img={img_w:.6f}  depth={depth_w:.6f}  normal={normal_w:.6f}"
                  f"  opacity={weight_w:.6f}"
                  f"  mesh_smooth={ms_w:.6f}  mesh_edge={me_w:.6f}  mesh_nc={mnc_w:.6f}")
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 1000 == 0:
                img_val = loss_image.item()
                depth_w = Ll1depth.item() if isinstance(Ll1depth, torch.Tensor) else float(Ll1depth)
                normal_w = Lnormal.item() if isinstance(Lnormal, torch.Tensor) else float(Lnormal)
                normal_super_w = normal_loss_super.item() if isinstance(normal_loss_super, torch.Tensor) else float(normal_loss_super)
                weight_w = Lweight.item() if isinstance(Lweight, torch.Tensor) else float(Lweight)
                ms_w = (opt.lambda_mesh_smooth * L_mesh_smooth).item() if isinstance(L_mesh_smooth, torch.Tensor) else 0.0
                me_w = (opt.lambda_mesh_edge * L_mesh_edge).item() if isinstance(L_mesh_edge, torch.Tensor) else 0.0
                mnc_w = (opt.lambda_mesh_nc * L_mesh_nc).item() if isinstance(L_mesh_nc, torch.Tensor) else 0.0
                print(f"[Loss@{iteration}] total={loss.item():.6f}  img={img_val:.6f}"
                      f"  depth={depth_w:.6f}  normal={normal_w:.6f}"
                      f"  normal_super={normal_super_w:.6f}"
                      f"  opacity={weight_w:.6f}"
                      f"  mesh_smooth={ms_w:.6f}  mesh_edge={me_w:.6f}  mesh_nc={mnc_w:.6f}")

            # Log and save
            
            training_report(tb_writer, scene_name, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),
                            final_only_monitoring=opt.final_only_monitoring)

            # Handle pruning operations
            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                
                # Building masks to delete triangles
                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                ) 
                min_weights = triangle_vertex_weights.min(dim=1).values

                mask_opacity     = (min_weights <= prune_triangles).squeeze()              # delete if too low
                mask_importance  = (triangles.importance_score <= prune_triangles).squeeze()  # delete if too low
                mask_size        = (triangles.image_size > prune_size).squeeze()                 # delete if too big

                delete_mask = mask_opacity | mask_size

                if number_of_training_views < 500: # only delete if the number of views are below 500. Otherwise, we might delete too much
                    delete_mask = delete_mask | mask_importance

                keep_mask   = ~delete_mask 

                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
             
                # We prune vertices that are no longer used
                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], 
                                            dtype=torch.bool, 
                                            device=device)
                if triangles._triangle_indices.numel() > 0:
                    flat_indices = triangles._triangle_indices.flatten()
                    used_vertex_mask[flat_indices] = True
                
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask

                triangles._prune_vertices(vertex_mask)


                triangle_vertex_weights = triangles.opacity_activation(
                    triangles.vertex_weight[triangles._triangle_indices]
                )  # [T,3]

                needs_densification = (iteration < opt.densify_until_iter and 
                                     iteration % opt.densification_interval == 0 and 
                                     iteration > opt.densify_from_iter)
                
                if needs_densification:
                    triangles.add_new_gs(iteration, cap_max=opt.max_points, splitt_large_triangles=splitt_large_triangles)
   

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:

                if iteration > opt.start_opacity_floor:
                    start_iter = opt.start_opacity_floor
                    end_iter = total_iters_opacity  # the iteration where you want to reach final_opacity
                    a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                    current_opacity = init_opacity + (final_opacity - init_opacity) * a
                    current_opacity = min(current_opacity, final_opacity)
                    triangles.update_min_weight(current_opacity)

                    prune_triangles += 0.01 
                    mask_out = triangles.vertices.shape[0]
                    triangle_vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]

            if iteration < opt.iterations:
                for p in [triangles.vertices, triangles._features_dc,
                          triangles._features_rest, triangles.vertex_weight]:
                    if p.grad is not None:
                        bad = ~torch.isfinite(p.grad)
                        if bad.any():
                            p.grad[bad] = 0.0
                if face_tex_mirror is not None and hasattr(triangles, 'face_textures'):
                    from train_face_texture import redirect_mirror_gradients, mirror_invalid_texels
                    inv_r, inv_c, mir_r, mir_c = face_tex_mirror
                    redirect_mirror_gradients(triangles.face_textures, inv_r, inv_c, mir_r, mir_c)
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                if face_tex_mirror is not None and hasattr(triangles, 'face_textures'):
                    mirror_invalid_texels(triangles.face_textures, inv_r, inv_c, mir_r, mir_c)

            # Consume this iteration's gradients before replacing Parameters;
            # Heartbeat then transfers the freshly updated Adam history.
            hb_end = opt.heartbeat_end_iter if opt.heartbeat_end_iter > 0 else opt.iterations
            if (opt.enable_heartbeat
                    and opt.heartbeat_start_iter < iteration <= hb_end
                    and iteration % opt.heartbeat_interval == 0):

                split_end = opt.heartbeat_split_end
                collapse_start = opt.heartbeat_collapse_start

                if split_end > 0 and collapse_start > 0:
                    if split_end < collapse_start:
                        # split first, then collapse (original)
                        if iteration <= split_end:
                            hb_mode = "split"
                        elif iteration >= collapse_start:
                            hb_mode = "collapse"
                        else:
                            hb_mode = None
                    else:
                        # collapse first, then split (reversed)
                        boundary = (collapse_start + split_end) // 2
                        if iteration <= boundary:
                            hb_mode = "collapse"
                        elif iteration <= split_end:
                            hb_mode = "split"
                        else:
                            hb_mode = None
                else:
                    hb_mode = "both"

                if hb_mode is not None:
                    hb_w_spatial = opt.heartbeat_w_spatial
                    if hb_mode == "collapse":
                        hb_w_spatial = getattr(opt, 'heartbeat_collapse_w_spatial',
                                               opt.heartbeat_w_spatial)

                    heartbeat_step(
                        triangles,
                        collapse_ratio=opt.heartbeat_collapse_ratio,
                        split_top_k=opt.heartbeat_split_top_k,
                        w_spatial=hb_w_spatial,
                        w_color=opt.heartbeat_w_color,
                        w_opacity=opt.heartbeat_w_opacity,
                        mode=hb_mode,
                        enable_safety=getattr(opt, 'heartbeat_safety', False),
                        max_edge_length_ratio=getattr(
                            opt, 'heartbeat_max_edge_len_ratio', 2.0),
                        use_qem=getattr(opt, 'heartbeat_use_qem', True),
                        qem_boundary_weight=getattr(
                            opt, 'heartbeat_qem_boundary_weight', 10.0),
                        split_momentum_std=getattr(
                            opt, 'heartbeat_split_momentum_std', 1e-3),
                        topology_callback=(
                            lambda stage, xyz, faces: record_topology(stage, iteration, xyz, faces)
                        ) if topology_monitor is not None else None,
                    )

            visibility_threshold = visibility_schedule.threshold_at(iteration)
            if visibility_threshold is not None:
                started = time.monotonic()
                before_metrics = record_topology("before_visibility_collapse", iteration)
                require_surface_topology(before_metrics)
                visibility = collect_face_visibility(
                    triangles, scene.getTrainCameras(), render, pipe, background)
                stats = visibility_collapse_step(
                    triangles, visibility, visibility_threshold,
                    max_face_ratio=visibility_schedule.max_face_ratio,
                    max_edge_length_ratio=opt.heartbeat_max_edge_len_ratio,
                    qem_boundary_weight=opt.heartbeat_qem_boundary_weight)
                del visibility
                after_metrics = record_topology("after_visibility_collapse", iteration)
                require_surface_topology(after_metrics)
                euler_before = (before_metrics["vertices"] - before_metrics["unused_vertices"]
                                - before_metrics["edges"] + before_metrics["faces"])
                euler_after = (after_metrics["vertices"] - after_metrics["unused_vertices"]
                               - after_metrics["edges"] + after_metrics["faces"])
                if (before_metrics["face_components"] != after_metrics["face_components"]
                        or euler_before != euler_after):
                    raise RuntimeError("Visibility collapse changed surface components or Euler characteristic")
                stats.update(euler_before=euler_before, euler_after=euler_after,
                             components_before=before_metrics["face_components"],
                             components_after=after_metrics["face_components"])
                stats.update(iteration=iteration, seconds=time.monotonic() - started)
                with open(os.path.join(dataset.model_path, "visibility_pruning.jsonl"), "a") as log:
                    log.write(json.dumps(stats, allow_nan=False) + "\n")
                print(f"[Visibility@{iteration}] threshold={visibility_threshold:.3f} "
                      f"F: {stats['faces_before']} -> {stats['faces_after']} "
                      f"({stats['collapsed_edges']} safe collapses; budget {stats['face_budget']} faces)")

            if (topology_monitor is not None and topology_interval > 0
                    and not opt.final_only_monitoring
                    and iteration > run_restricted_delaunay
                    and iteration % topology_interval == 0):
                record_topology("periodic", iteration)

            if mesh_viz_interval > 0 and mesh_viz_dir is not None:
                if (iteration % mesh_viz_interval == 0) or (iteration == opt.iterations):
                    adv = (
                        mesh_viz_orbit
                        or (
                            mesh_viz_use_train_views
                            and mesh_viz_train_cam_indexing == "sequential"
                        )
                    )
                    _mesh_viz_save("frame", iteration, advance_camera=adv)

            # Save after parameter updates and topology edits. The final model is
            # saved below, after the existing finalization step.
            if iteration in saving_iterations and iteration < opt.iterations:
                if save_topology_snapshots:
                    save_training_snapshot(scene, iteration, "iteration_end")
                else:
                    scene.save(iteration)

            if opt.stop_after_iteration > 0 and iteration >= opt.stop_after_iteration:
                record_topology("probe_stop", iteration)
                if rdt_probe:
                    evaluate_views(scene, triangles, pipe, background, render, "after_training", iteration)
                scene.save(iteration)
                progress_bar.close()
                print(f"[Training probe] stopped at {iteration}; full schedule={opt.iterations}; no final pruning")
                return

    if visibility_schedule.mode == "legacy":
        record_topology("before_final_prune", opt.iterations)
    final_stats = finalize_visibility_pruning(
        triangles, scene.getTrainCameras(), render, pipe, background, visibility_schedule.mode,
        final_visibility_threshold=opt.final_visibility_threshold)
    if visibility_schedule.mode != "legacy":
        record_topology("training_complete", opt.iterations)
        print(f"[Visibility] training complete; retained all {final_stats['faces_after']} faces; "
              "no final visibility pruning")
    else:
        record_topology("after_final_prune", opt.iterations)

    if mesh_viz_interval > 0 and mesh_viz_dir is not None:
        cams_train = scene.getTrainCameras()
        cams_test = scene.getTestCameras() if mesh_viz_include_test_views else []
        cams = (cams_train or []) + (cams_test or [])
        if cams:
            if mesh_viz_orbit and mesh_viz_traj is not None and len(mesh_viz_traj) > 0:
                cam = mesh_viz_traj[(mesh_viz_frame_idx - 1) % len(mesh_viz_traj)]
            else:
                cam = _mesh_viz_pick_camera(iteration=iteration, advance=False)
            rp = render(cam, triangles, pipe, background)
            img_np = torch.clamp(rp["render"], 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
            save_img_u8(img_np, os.path.join(mesh_viz_dir, "frame_final.png"))

    final_checkpoint = scene.save(iteration)
    if save_topology_snapshots:
        save_training_snapshot(scene, iteration, "training_complete", final_checkpoint)
    print("Training is done")

    if mesh_viz_interval > 0 and mesh_viz_dir is not None and not getattr(
            opt, "mesh_viz_skip_encode", False):
        out_mp4 = os.path.join(dataset.model_path, "mesh_evolution.mp4")
        write_mesh_evolution_video(
            mesh_viz_dir,
            out_mp4,
            fps=float(getattr(opt, "mesh_viz_fps", 24.0)),
        )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and os.environ.get("T2SPLAT_DISABLE_TENSORBOARD") != "1":
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, scene_name, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    final_only_monitoring=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set; scalar logs above are unchanged.
    if (iteration in testing_iterations if final_only_monitoring else iteration % 1000 == 0):
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # Same as training: composite GT with eval background for comparable metrics.
                    if getattr(viewpoint, "gt_alpha_mask", None) is not None:
                        eval_bg = renderArgs[1]
                        alpha = viewpoint.gt_alpha_mask.to("cuda")
                        gt_image = torch.clamp(gt_image + eval_bg.view(3, 1, 1) * (1.0 - alpha), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                fps = 1000.0 / total_time
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} FPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test, fps))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_topology_snapshots", action="store_true",
                        help="Save raw mesh/model snapshots immediately before and after RDT and at save_iterations")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--wandb_name', default="Test", type=str)
    parser.add_argument('--scene_name', default="Garden", type=str)
    parser.add_argument("--use_sparse_adam", action="store_true", default=True)
    parser.add_argument("--indoor", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)

    if args.indoor:
        ops = update_indoor(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps,
             ops,
             pps,
             args.test_iterations,
             args.start_checkpoint,
             args.debug_from,
             args.scene_name,
             use_sparse_adam=args.use_sparse_adam,
             saving_iterations=args.save_iterations,
             save_topology_snapshots=args.save_topology_snapshots,
             )
    
    # All done
    print("\nTraining complete.")