# Project adaptations maintained by zhtang2004@gmail.com.
"""
Stage 2: Per-Face Base Color Texture  (physics-based decoupling)

Architecture:   pixel_color = Bilinear(face_tex_basecolor) + Gouraud(vertex_SH_rest)

  - face_textures [F, R, R, 3]  — stores diffuse base color
    (initialized from vertex _features_dc via barycentric interpolation)
  - _features_dc  — zeroed & frozen after bake
  - _features_rest — high-order SH only, models specular / view-dependent effects

Legacy residual mode (--legacy_residual) preserves the old architecture:
    pixel_color = Gouraud(vertex_SH) + bilinear(face_textures_residual)

Two training modes (choose with --progressive):

  Fixed resolution (default):
      python train_face_texture.py -m OUT --face_tex_res 4 --stage2_iterations 4000

  Progressive multi-resolution:
      python train_face_texture.py -m OUT --progressive \\
          --face_tex_res 4,8 --phase_iters 1500,2500

Face-texture LR: constant until --lr_decay_start, then exponential decay to
lr_face_texture * 0.01 by end of each phase.
"""

import os
import sys
import math
import shutil
import torch
import cv2
import numpy as np
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image

from scene import Scene
from scene.triangle_model import TriangleModel, build_face_tex_mirror
from triangle_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from utils.system_utils import saveBestIteration
from utils.face_texture_regularization import accumulate_face_tex_regularization

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips


# ── helpers ──────────────────────────────────────────────────────────────

def face_tex_tv_loss(face_tex: torch.Tensor) -> torch.Tensor:
    """Squared total-variation on [F, R, R, C] face textures."""
    tv_x = (face_tex[:, :, 1:, :] - face_tex[:, :, :-1, :]).pow(2).mean()
    tv_y = (face_tex[:, 1:, :, :] - face_tex[:, :-1, :, :]).pow(2).mean()
    return tv_x + tv_y


def redirect_mirror_gradients(
    face_textures: torch.nn.Parameter,
    inv_r, inv_c, mir_r, mir_c,
):
    """Move gradients from invalid texels to their valid mirrors."""
    g = face_textures.grad
    if g is None or inv_r.numel() == 0:
        return
    g.data[:, mir_r, mir_c, :] += g.data[:, inv_r, inv_c, :]
    g.data[:, inv_r, inv_c, :] = 0.0


def mirror_invalid_texels(
    face_textures: torch.nn.Parameter,
    inv_r, inv_c, mir_r, mir_c,
):
    """Copy valid texel values to their out-of-triangle mirrors."""
    if inv_r.numel() == 0:
        return
    face_textures.data[:, inv_r, inv_c, :] = \
        face_textures.data[:, mir_r, mir_c, :]


def evaluate_test(triangles, test_cameras, pipeline, background, lpips_fn):
    """Run evaluation on test cameras, return (psnr, ssim, lpips)."""
    psnr_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0
    for view in test_cameras:
        img = torch.clamp(
            render(view, triangles, pipeline, background)["render"], 0, 1,
        )
        gt = torch.clamp(view.original_image.cuda(), 0, 1)
        psnr_sum += psnr(img, gt).mean().double()
        ssim_sum += ssim(img, gt).mean().double()
        lpips_sum += lpips_fn(img, gt).mean().double()
    n = len(test_cameras)
    return psnr_sum / n, ssim_sum / n, lpips_sum / n


def encode_png_sequence_opencv(frames_dir: str, output_mp4: str, fps: float):
    if not os.path.isdir(frames_dir):
        return
    names = sorted(
        n for n in os.listdir(frames_dir)
        if n.lower().endswith(".png")
    )
    if not names:
        return

    first = cv2.imread(os.path.join(frames_dir, names[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        output_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        return
    wrote = 0
    for nm in names:
        img = cv2.imread(os.path.join(frames_dir, nm))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            continue
        writer.write(img)
        wrote += 1
    writer.release()
    if wrote > 0:
        print(f"[track_test00001] encoded {wrote} frames -> {output_mp4} @ {fps} fps")


# ── main training loop ──────────────────────────────────────────────────

def train_face_texture(
    dataset,
    pipeline,
    load_iteration: int,
    load_from_model_path: str = None,
    face_tex_res_list: list = None,
    phase_iters_list: list = None,
    progressive: bool = False,
    lr_face_texture: float = 0.005,
    lr_geometry: float = 0.0,
    lr_sh: float = 0.0,
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
    lambda_tv: float = 0.005,
    lambda_l2: float = 0.005,
    lr_decay_start: int = 500,
    lr_decay_ratio: float = 0.1,
    no_lr_decay: bool = False,
    test_interval: int = 500,
    save_interval: int = 2000,
    track_test00001_interval: int = 0,
    track_test_render_index: int = 1,
    track_test00001_fps: float = 24.0,
    quiet: bool = False,
    legacy_residual: bool = False,
    use_mip_texture: bool = False,
    texture_adam_offload: bool = False,
    texture_adam_chunk_faces: int = 65536,
):
    if face_tex_res_list is None:
        face_tex_res_list = [4]
    if phase_iters_list is None:
        phase_iters_list = [4000]
    assert len(face_tex_res_list) == len(phase_iters_list)

    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = 4

    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=load_iteration,
        shuffle=True,
        load_from_model_path=load_from_model_path,
    )
    base_iter = scene.loaded_iter
    print(f"Loaded Stage 1 model at iteration {base_iter}")
    print(f"  Vertices : {triangles.vertices.shape[0]}")
    print(f"  Triangles: {triangles._triangle_indices.shape[0]}")
    mode = "progressive multi-resolution" if progressive else "fixed resolution"
    print(f"  Stage 2 mode: {mode}")
    print(f"  Phases     : {list(zip(face_tex_res_list, phase_iters_list))}")

    # Init first resolution
    first_res = face_tex_res_list[0]
    if legacy_residual:
        triangles.bake_face_textures(face_tex_res=first_res, lr=lr_face_texture)
        print(f"  [Legacy] Face textures (RGB residual, zero-init): "
              f"{triangles.face_textures.shape}")
    else:
        triangles.bake_face_textures_basecolor(face_tex_res=first_res, lr=lr_face_texture)
        print(f"  [BaseColor] Face textures (diffuse, DC-init): "
              f"{triangles.face_textures.shape}")
        print(f"  _features_dc zeroed & frozen  →  vertices only carry SH rest (specular)")

    if texture_adam_offload:
        from utils.texture_offload_adam import TextureOffloadAdam
        triangles.optimizer = TextureOffloadAdam.from_adam(
            triangles.optimizer, chunk_faces=texture_adam_chunk_faces)
        print(f"  Texture Adam: FP32 moments on CPU, GPU updates in chunks of {texture_adam_chunk_faces} faces")

    triangles.use_mip_texture = use_mip_texture
    if use_mip_texture:
        F_count = triangles.face_textures.shape[0]
        R_base = triangles.face_textures.shape[1]
        r = R_base
        total_texels = 0
        level_info = []
        while r >= 1:
            level_info.append(f"{r}x{r}={r*r}")
            total_texels += r * r
            if r == 1:
                break
            r //= 2
        mem_mb = F_count * total_texels * 3 * 4 / (1024**2)
        print(f"  Mip-texture (trilinear filtering): ENABLED")
        print(f"    Pyramid levels ({len(level_info)}): {' → '.join(level_info)}")
        print(f"    Texels/face: {total_texels}  |  Total: {F_count}×{total_texels}×3 = {F_count*total_texels*3:,} floats")
        print(f"    Packed buffer: {mem_mb:.1f} MB")

    # Set LR for vertex SH & positions
    for group in triangles.optimizer.param_groups:
        if group["name"] == "f_dc":
            if legacy_residual:
                group["lr"] = lr_sh
            else:
                group["lr"] = 0.0      # always frozen in basecolor mode
        elif group["name"] == "f_rest":
            group["lr"] = lr_sh / 20.0 if lr_sh > 0 else 0.0
        elif group["name"] == "vertices":
            group["lr"] = lr_geometry
        elif group["name"] == "vertex_weight":
            group["lr"] = 0.0

    if not legacy_residual:
        if lr_sh > 0:
            print(f"  Specular (f_rest) co-optimization: lr={lr_sh/20.0:.6f}")
        else:
            print(f"  Specular (f_rest) frozen (lr_sh=0)")
    elif lr_sh > 0:
        print(f"  SH co-optimization enabled: f_dc lr={lr_sh}, f_rest lr={lr_sh/20.0}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    lpips_fn = lpips.LPIPS(net="vgg").to(device="cuda")

    tb_writer = None
    if TENSORBOARD_FOUND:
        log_dir = os.path.join(dataset.model_path, "tb_face_texture")
        tb_writer = SummaryWriter(log_dir)

    viewpoint_stack = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras()
    test00001_cam = None
    if track_test00001_interval > 0 and test_cameras:
        # Use render index semantics (00000, 00001, ...), matching render.py output order.
        ridx = int(track_test_render_index)
        ridx = max(0, min(ridx, len(test_cameras) - 1))
        test00001_cam = test_cameras[ridx]
        print(
            f"[track_test00001] every {track_test00001_interval} iters, "
            f"render_index={ridx:05d}, image_name={getattr(test00001_cam, 'image_name', 'N/A')}"
        )
    track_dir = os.path.join(dataset.model_path, "texture_evolution")
    if track_test00001_interval > 0:
        os.makedirs(track_dir, exist_ok=True)

    best_psnr = 0.0
    best_save_tag = -1
    global_iter = 0

    for phase_idx, (cur_res, phase_len) in enumerate(
        zip(face_tex_res_list, phase_iters_list)
    ):
        # ── Upsample if not first phase ──
        if phase_idx > 0:
            triangles.upsample_face_textures(cur_res, lr=lr_face_texture)
            print(f"  Upsampled face textures: {triangles.face_textures.shape}")
            if use_mip_texture:
                F_count = triangles.face_textures.shape[0]
                R_base = triangles.face_textures.shape[1]
                r = R_base
                total_texels = 0
                level_info = []
                while r >= 1:
                    level_info.append(f"{r}x{r}={r*r}")
                    total_texels += r * r
                    if r == 1:
                        break
                    r //= 2
                mem_mb = F_count * total_texels * 3 * 4 / (1024**2)
                print(f"    Mip pyramid ({len(level_info)} levels): {' → '.join(level_info)}")
                print(f"    Texels/face: {total_texels}  |  Packed buffer: {mem_mb:.1f} MB")

        # Build mirror map for current resolution
        inv_r, inv_c, mir_r, mir_c, valid_mask = build_face_tex_mirror(
            cur_res, device=triangles.face_textures.device
        )
        n_valid = int(valid_mask.sum())
        n_total = cur_res * cur_res
        mirror_invalid_texels(triangles.face_textures, inv_r, inv_c, mir_r, mir_c)

        lr_final = lr_face_texture * lr_decay_ratio
        ema_loss = 0.0
        if progressive and len(face_tex_res_list) > 1:
            tag = f"Phase {phase_idx} R={cur_res}"
        elif progressive:
            tag = f"Face-Tex R={cur_res}"
        else:
            tag = "Face-Tex Stage 2"
        progress = tqdm(range(1, phase_len + 1), desc=tag)

        for local_iter in progress:
            global_iter += 1

            # LR schedule: constant or exponential decay
            if no_lr_decay:
                cur_lr = lr_face_texture
            elif local_iter >= lr_decay_start:
                denom = max(phase_len - lr_decay_start, 1)
                t = (local_iter - lr_decay_start) / denom
                t = min(t, 1.0)
                cur_lr = lr_face_texture * math.exp(
                    math.log(lr_final / lr_face_texture) * t
                )
            else:
                cur_lr = lr_face_texture
            for group in triangles.optimizer.param_groups:
                if group.get("name") == "face_textures":
                    group["lr"] = cur_lr

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            render_pkg = render(viewpoint_cam, triangles, pipeline, background)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                Lssim = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                Lssim = 1.0 - ssim(image, gt_image)

            loss = lambda_l1 * Ll1 + lambda_ssim * Lssim

            # Release the renderer's backward buffers before differentiating
            # TV/L2. Chunking preserves the full-texture objective while
            # avoiding several simultaneous multi-GB regularization tensors.
            loss.backward()
            reg_loss = accumulate_face_tex_regularization(
                triangles.face_textures, lambda_tv, lambda_l2
            )
            loss = loss.detach() + reg_loss

            with torch.no_grad():
                ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                if local_iter % 10 == 0:
                    progress.set_postfix({"loss": f"{ema_loss:.5f}",
                                          "lr": f"{cur_lr:.1e}"})

                if tb_writer and global_iter % 100 == 0:
                    tb_writer.add_scalar("face_tex/loss", loss.item(), global_iter)
                    tb_writer.add_scalar("face_tex/lr", cur_lr, global_iter)

                redirect_mirror_gradients(
                    triangles.face_textures, inv_r, inv_c, mir_r, mir_c
                )
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)
                mirror_invalid_texels(
                    triangles.face_textures, inv_r, inv_c, mir_r, mir_c
                )

                # Evaluation
                if local_iter % test_interval == 0 and test_cameras:
                    avg_psnr, avg_ssim, avg_lpips = evaluate_test(
                        triangles, test_cameras, pipeline, background, lpips_fn,
                    )
                    improved = avg_psnr > best_psnr
                    if improved:
                        best_psnr = avg_psnr
                    print(f"\n[{tag} iter {local_iter}/{phase_len}] "
                          f"PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}  "
                          f"LPIPS={avg_lpips:.4f}"
                          f"{'  *best*' if improved else ''}")
                    if tb_writer:
                        tb_writer.add_scalar("face_tex/test_psnr", avg_psnr, global_iter)
                        tb_writer.add_scalar("face_tex/test_ssim", avg_ssim, global_iter)
                    if improved:
                        best_save_tag = base_iter + global_iter
                    torch.cuda.empty_cache()

                if (
                    track_test00001_interval > 0
                    and test00001_cam is not None
                    and (global_iter % track_test00001_interval == 0)
                ):
                    tracked = render(test00001_cam, triangles, pipeline, background)["render"]
                    tracked_np = (
                        torch.clamp(tracked, 0.0, 1.0)
                        .detach().cpu().permute(1, 2, 0).numpy()
                    )
                    tracked_u8 = (tracked_np * 255.0).round().astype(np.uint8)
                    Image.fromarray(tracked_u8).save(
                        os.path.join(track_dir, f"tex_{global_iter:06d}.png")
                    )

        print(f"\n  Phase {phase_idx} (R={cur_res}) complete")

    if tb_writer:
        tb_writer.close()
    final_tag = base_iter + global_iter
    scene.save(final_tag)
    saveBestIteration(dataset.model_path, final_tag, best_psnr)
    if track_test00001_interval > 0:
        encode_png_sequence_opencv(
            track_dir,
            os.path.join(dataset.model_path, "texture_evolution.mp4"),
            fps=track_test00001_fps,
        )
    print(f"Stage 2 complete. Best test PSNR = {best_psnr:.2f} (saved at iteration {final_tag})")


if __name__ == "__main__":
    parser = ArgumentParser(description="Stage 2: Per-Face Texture Training")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                        help="Stage 1 iteration to load (-1 = latest)")
    parser.add_argument("--progressive", action="store_true",
                        help="Enable multi-resolution phases (use comma lists in "
                             "--face_tex_res and --phase_iters). "
                             "Omit for single fixed R and --stage2_iterations.")
    parser.add_argument("--face_tex_res", default="4", type=str,
                        help="Fixed mode: one integer, e.g. 4. "
                             "Progressive: comma list, e.g. 4,8 (requires --progressive)")
    parser.add_argument("--stage2_iterations", default=4000, type=int,
                        help="Total iterations when NOT using --progressive")
    parser.add_argument("--phase_iters", default="1500,2500", type=str,
                        help="Comma-separated iterations per phase (only with --progressive)")
    parser.add_argument("--lr_face_texture", default=0.005, type=float,
                        help="LR for face textures. "
                             "BaseColor mode: recommend 0.002~0.005 (fine-tuning from DC init). "
                             "Legacy mode: 0.005 (learning from zero).")
    parser.add_argument("--lr_geometry", default=0.0, type=float)
    parser.add_argument("--lr_sh", default=0.0, type=float,
                        help="LR for vertex SH in Stage 2. "
                             "BaseColor mode: controls f_rest (specular) only, f_dc is always frozen. "
                             "Legacy mode: controls both f_dc and f_rest. "
                             "0 = frozen (default). Try 0.0002 to co-adapt specular with textures.")
    parser.add_argument("--lambda_l1", default=0.8, type=float)
    parser.add_argument("--lambda_ssim", default=0.2, type=float)
    parser.add_argument("--lambda_tv", default=0.005, type=float,
                        help="Total-variation weight for face textures")
    parser.add_argument("--lambda_l2", default=0.005, type=float,
                        help="L2 regularization on face texture values. "
                             "BaseColor mode: penalizes deviation from gray; "
                             "consider reducing to 0.001 or 0. "
                             "Legacy mode: penalizes residual magnitude.")
    parser.add_argument("--lr_decay_start", default=500, type=int,
                        help="Per phase: keep lr_face_texture until this local iter, "
                             "then exponential decay by lr_decay_ratio by phase end")
    parser.add_argument("--lr_decay_ratio", default=0.1, type=float,
                        help="Final LR = lr_face_texture * ratio. "
                             "0.1 = 10x decay (default), 0.01 = 100x (legacy), 1.0 = no decay")
    parser.add_argument("--no_lr_decay", action="store_true",
                        help="Keep face texture LR constant (no decay). "
                             "Recommended for basecolor mode where gradients are "
                             "naturally small due to good DC initialization.")
    parser.add_argument("--test_interval", default=500, type=int)
    parser.add_argument("--save_interval", default=2000, type=int)
    parser.add_argument("--track_test00001_interval", default=0, type=int,
                        help="If >0, every N iters render fixed test 00001 view to texture_evolution/*.png.")
    parser.add_argument("--track_test_render_index", default=1, type=int,
                        help="Fixed test camera render index (00000->0, 00001->1, ...).")
    parser.add_argument("--track_test00001_fps", default=24.0, type=float,
                        help="FPS for auto-encoded texture_evolution.mp4.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--legacy_residual", action="store_true",
                        help="Use legacy zero-init residual architecture instead of "
                             "physics-based base-color decoupling. "
                             "Default: base-color mode (face_tex=diffuse, vertex=specular).")
    parser.add_argument("--use_mip_texture", action="store_true",
                        help="Enable mip-mapped trilinear filtering for face textures. "
                             "Reduces aliasing from view-distance changes and improves "
                             "PSNR/SSIM. Requires face_tex_res >= 2.")
    parser.add_argument("--texture_adam_offload", action="store_true",
                        help="Keep texture Adam moments in CPU memory; update on GPU in chunks.")
    parser.add_argument("--texture_adam_chunk_faces", default=65536, type=int)
    parser.add_argument("--init_from", default="", type=str,
                        help="Directory with completed Stage-1 (cfg_args + point_cloud/iteration_*). "
                             "Load checkpoint from here; weights and best_iteration save under -m. "
                             "Use this to re-train only Stage-2 (texture) into a new output path.")

    args = get_combined_args(parser)
    _init = (args.init_from or "").strip()
    _out = (args.model_path or "").strip()
    load_from = None
    if _init:
        if os.path.abspath(_init) == os.path.abspath(_out):
            print("[init_from] same as -m, loading from model_path only")
        else:
            load_from = _init
            os.makedirs(_out, exist_ok=True)
            src_cfg = os.path.join(_init, "cfg_args")
            dst_cfg = os.path.join(_out, "cfg_args")
            if os.path.isfile(src_cfg) and (not os.path.isfile(dst_cfg)):
                shutil.copy2(src_cfg, dst_cfg)
                print(f"[init_from] copied cfg_args -> {dst_cfg}")
    print(f"Face-texture training: {args.model_path}"
          + (f"  (load checkpoint from: {load_from})" if load_from else ""))
    safe_state(args.quiet)

    fts = args.face_tex_res.strip()
    if args.progressive:
        res_list = [int(x.strip()) for x in fts.split(",") if x.strip()]
        iter_list = [int(x.strip()) for x in args.phase_iters.split(",") if x.strip()]
        if len(res_list) != len(iter_list):
            raise ValueError(
                f"--progressive: len(face_tex_res)={len(res_list)} != "
                f"len(phase_iters)={len(iter_list)}"
            )
        if len(res_list) < 1:
            raise ValueError("--progressive requires at least one resolution in --face_tex_res")
    else:
        if "," in fts:
            raise ValueError(
                "Comma in --face_tex_res requires --progressive "
                "(or use a single integer, e.g. --face_tex_res 4)"
            )
        r = int(fts)
        res_list = [r]
        iter_list = [args.stage2_iterations]

    train_face_texture(
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        load_iteration=args.iteration,
        load_from_model_path=load_from,
        face_tex_res_list=res_list,
        phase_iters_list=iter_list,
        progressive=args.progressive,
        lr_face_texture=args.lr_face_texture,
        lr_geometry=args.lr_geometry,
        lr_sh=args.lr_sh,
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_tv=args.lambda_tv,
        lambda_l2=args.lambda_l2,
        lr_decay_start=args.lr_decay_start,
        lr_decay_ratio=args.lr_decay_ratio,
        no_lr_decay=args.no_lr_decay,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        track_test00001_interval=args.track_test00001_interval,
        track_test_render_index=args.track_test_render_index,
        track_test00001_fps=args.track_test00001_fps,
        quiet=args.quiet,
        legacy_residual=args.legacy_residual,
        use_mip_texture=args.use_mip_texture,
        texture_adam_offload=args.texture_adam_offload,
        texture_adam_chunk_faces=args.texture_adam_chunk_faces,
    )
