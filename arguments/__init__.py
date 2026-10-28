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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.disable_priors = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2

        self.densification_interval = 500

        self.densify_from_iter = 500
        self.densify_until_iter = 10000
        # Iteration at which the restricted Delaunay remesh is triggered.
        # Set to -1 to use the default schedule (densify_until_iter + 1000).
        self.run_restricted_delaunay = -1
        # Experimental candidate selection. Legacy remains the default.
        self.rdt_backend = "legacy"  # legacy | surface
        self.rdt_max_distance_ratio = 0.5
        self.rdt_min_normal_cos = 0.15
        self.rdt_min_quality = 0.02
        self.rdt_probe = False  # save same-state RDT comparisons locally
        # Diagnostic stop: preserves the full training schedule and skips final pruning.
        self.stop_after_iteration = 0

        # API default retains the legacy baseline. The MipNeRF shell enables
        # progressive collapse together with surface RDT.
        self.visibility_pruning = "legacy"  # legacy | progressive_collapse | none (post-RDT)
        self.final_visibility_threshold = 0.5  # final legacy prune keeps scores strictly above this
        self.final_only_monitoring = False  # restrict full evaluation/topology statistics to final/events
        self.visibility_prune_start_iter = 21000
        self.visibility_prune_end_iter = 27000
        self.visibility_prune_interval = 1000
        self.visibility_prune_threshold_start = 0.05
        self.visibility_prune_threshold_end = 0.5
        self.visibility_prune_max_face_ratio = 0.02

        self.random_background = False
        
        self.feature_lr = 0.0016 # 0.0025
        self.max_points = 4000000

        # Opacity & weight
        self.set_weight = 0.28
        self.weight_lr =  0.03
        self.lambda_weight = 1.9e-06

        # Normal loss
        self.iteration_mesh = 5000
        self.lambda_normals = 0.00005
        self.lambda_normals_super = 0.01

        # Heartbeat (topology surgery after RDT; see scene/heartbeat.py)
        self.enable_heartbeat = False
        self.heartbeat_start_iter = 12000
        self.heartbeat_interval = 500
        self.heartbeat_collapse_ratio = 0.05
        self.heartbeat_split_top_k = 5.0
        # Eq. 18 stability adaptation: std relative to sqrt(exp_avg_sq); 0 disables.
        self.heartbeat_split_momentum_std = 1e-3
        self.heartbeat_w_spatial = 1.0
        self.heartbeat_w_color = 0.5
        self.heartbeat_w_opacity = 0.3
        self.heartbeat_end_iter = 0
        # Phased scheduling: split-only phase then collapse-only phase.
        # 0 = disabled (use "both" mode throughout).
        self.heartbeat_split_end = 0
        self.heartbeat_collapse_start = 0
        self.heartbeat_collapse_w_spatial = 5.0
        # Heartbeat safety: full local links, boundaries, placement and disjoint one-rings.
        # API opt-in via --heartbeat_safety; enabled by the MipNeRF shell script.
        self.heartbeat_safety = False
        self.heartbeat_max_edge_len_ratio = 2.0
        # 0: off; positive: RDT/DTE events + periodic metrics after RDT.
        self.topology_monitor_interval = 0
        # Heartbeat QEM (Garland-Heckbert) edge-collapse cost. Default ON.
        # Disable with --heartbeat_use_qem False to fall back to (Δxyz + Δcolor + Δopacity).
        # qem_boundary_weight: weight of "virtual perpendicular planes" added on
        # boundary edges (Garland 1998). 0 disables boundary protection.
        self.heartbeat_use_qem = True
        self.heartbeat_qem_boundary_weight = 10.0

        self.add_percentage = 1.23

        self.set_sigma = 1.0

        # Add new triangles or vertices
        self.intervall_add_triangles = 500

        # Prune triangles and vertices
        self.prune_triangles_threshold = 0.235

        # PARAMETER SECOND STAGE
        self.lr_triangles_points_init = 0.0015

        self.start_opacity_floor = 5000

        self.start_pruning = 4000
        self.sigma_until = 30000
        self.final_opacity_iter = 24000

        self.sigma_start = 0

        self.splitt_large_triangles = 100
        self.start_upsampling = 20000
        self.upscaling_factor = 2

        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 0.0

        self.prune_size = 1400

        self.lambda_vertex = 0.00025
        self.max_diff_threshold = 0.5
        self.start_vertex_opt = 12000

        self.lambda_vertex_normal = 0
        self.start_vertex_normal = 12000

        # PyTorch3D-style mesh regularization
        self.lambda_mesh_smooth = 0.0
        self.mesh_smooth_start_iter = 5000
        self.mesh_smooth_method = "uniform"  # "uniform", "cot", "cotcurv"
        self.lambda_mesh_edge = 0.0
        self.mesh_edge_start_iter = 5000
        self.lambda_mesh_nc = 0.0
        self.mesh_nc_start_iter = 5000

        self.lamba_depth = 0.05

        self.depth_lambda_init = 0.01
        self.depth_lambda_final = 0.001

        # Inline face texture (basecolor) — baked during Stage 1 training
        self.face_tex_start_iter = 0        # 0 = disabled; e.g. 20000 to start at iter 20k
        self.face_tex_res_schedule = ""     # comma-separated "iter:res", e.g. "20000:4,25000:6"
        self.lr_face_texture = 0.003
        self.lr_face_texture_final = 0.0003  # final LR after exponential decay

        # Mesh evolution video: fixed-view renders every N iters → mesh_evolution/*.png + mesh_evolution.mp4
        self.mesh_viz_interval = 0  # 0 = disabled; e.g. 500
        self.mesh_viz_cam_index = 0
        # If True, use the training camera list sequentially (real training image poses).
        self.mesh_viz_use_train_views = False
        # Override the image folder whose filename order defines the camera order.
        # Example: --mesh_viz_images_dir images_4
        self.mesh_viz_images_dir = ""
        # mesh_viz_camera_order:
        #   auto  — if sparse/0/images.bin exists, use COLMAP file order; else disk order under mesh_viz_images_dir / dataset.images
        #   colmap — force COLMAP images.bin/.txt order
        #   disk — force directory listing order (natural sort)
        self.mesh_viz_camera_order = "auto"
        # If True, also include test cameras in visualization ordering/matching.
        self.mesh_viz_include_test_views = True
        # How to pick the training-camera viewpoint for mesh visualization:
        #   interval   — k=floor(iter/interval); cam = ordered[k-1] (500→#1, 1000→#2 in ordered list)
        #   sequential — advance one camera on every saved frame (legacy)
        self.mesh_viz_train_cam_indexing = "interval"
        self.mesh_viz_orbit = False
        self.mesh_viz_orbit_frames = 480
        # Orbit camera for mesh_viz: smooth = horizontal orbit around focus_point (default); legacy = old PCA+ellipse path.
        self.mesh_viz_orbit_style = "smooth"
        self.mesh_viz_orbit_radius_scale = 1.0
        self.mesh_viz_orbit_height_frac = 0.08
        self.mesh_viz_orbit_ease = "smoothstep"
        self.mesh_viz_orbit_loops = 1.0
        # Orbit look-at: cameras_mean = centroid of camera positions (robust for 360°);
        # rays = least-squares ray intersection (NeRF focus_point_fn); hybrid = average of both.
        self.mesh_viz_orbit_focus = "cameras_mean"
        self.mesh_viz_fps = 24.0
        # If True, only write PNG frames; skip MP4 at end (encode later with utils.render_utils.write_mesh_evolution_video).
        self.mesh_viz_skip_encode = False
        # If True, save extra frames right before/after RDT (restricted Delaunay remesh).
        self.mesh_viz_rdt_snapshots = True
        # RDT: number of progressive snapshots while growing the candidate face set (0 = off).
        # Each step renders the first (i/N) of candidate faces for a smoother "process" video.
        self.mesh_viz_rdt_candidate_steps = 16

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    # Optional: read cfg_args from another directory (e.g. Stage-1 output while saving to -m elsewhere)
    init_from = getattr(args_cmdline, "init_from", "") or ""
    cfg_dir = init_from.strip() if init_from else args_cmdline.model_path
    cfgfilepath = os.path.join(cfg_dir, "cfg_args")

    try:
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, OSError, FileNotFoundError) as e:
        print("Config file not found or unreadable at {}: {}".format(cfgfilepath, e))
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def update_indoor(params):
    params.add_percentage = 1.27
    params.densify_from_iter = 1000
    params.densify_until_iter = 10000
    params.feature_lr = 0.004
    params.size_probs_zero = 0.0
    params.splitt_large_triangles = 500
    params.start_pruning = 3000
    params.weight_lr = 0.05
    params.lambda_weight = 0.0
    params.lambda_normals = 0.00001
    params.lambda_normals_super = 0.01
    params.prune_size = 1300
    params.lambda_vertex = 0.00025
    params.lambda_vertex_normal = 0.00025
    params.depth_lambda_init = 0.0
    params.depth_lambda_final = 0.0
    params.iteration_mesh = 12000
    return params

