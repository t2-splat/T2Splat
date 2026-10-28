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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_triangles(
    vertices,
    triangles_indices,
    vertex_weights,
    sigma,
    sh,
    colors_precomp,
    scaling,
    raster_settings,
    face_textures=None,
    mip_textures=None,
    mip_offsets=None,
):
    if face_textures is None:
        face_textures = torch.empty(0, device=vertices.device, dtype=torch.float32)
    if mip_textures is None:
        mip_textures = torch.empty(0, device=vertices.device, dtype=torch.float32)
    if mip_offsets is None:
        mip_offsets = torch.empty(0, device=vertices.device, dtype=torch.int32)
    return _RasterizeTriangles.apply(
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        sh,
        colors_precomp,
        scaling,
        face_textures,
        mip_textures,
        mip_offsets,
        raster_settings,
    )

class _RasterizeTriangles(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        sh,
        colors_precomp,
        scaling,
        face_textures,
        mip_textures,
        mip_offsets,
        raster_settings,
    ):

        args = (
            raster_settings.bg, 
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            colors_precomp,
            scaling,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            face_textures,
            mip_textures,
            mip_offsets,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, depth, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending = _C.rasterize_triangles(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending = _C.rasterize_triangles(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.sigma = sigma
        ctx.save_for_backward(vertices, triangles_indices, vertex_weights, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, face_textures, mip_textures, mip_offsets)
        return color, radii, scaling, depth, max_blending, was_rendered

    @staticmethod
    def backward(ctx, grad_out_color, _, __, grad_depth, _____, _______):

        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        sigma = ctx.sigma
        vertices, triangles_indices, vertex_weights, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, face_textures, mip_textures, mip_offsets = ctx.saved_tensors

        args = (raster_settings.bg,
                vertices,
                triangles_indices,
                vertex_weights,
                sigma,
                radii, 
                colors_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                face_textures,
                mip_textures,
                mip_offsets)

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_vertices, grad_vertice_weights, grad_colors_precomp, grad_sh, grad_face_textures, grad_mip_textures = _C.rasterize_triangles_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_vertices, grad_vertice_weights, grad_colors_precomp, grad_sh, grad_face_textures, grad_mip_textures = _C.rasterize_triangles_backward(*args)


        grads = (
            grad_vertices,
            None,               # triangles_indices
            grad_vertice_weights,
            None,               # sigma
            grad_sh,
            grad_colors_precomp,
            None,               # scaling
            grad_face_textures if grad_face_textures.numel() > 0 else None,
            grad_mip_textures if mip_textures.numel() > 0 else None,
            None,               # mip_offsets (int, no grad)
            None,               # raster_settings
        )

        return grads

class TriangleRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class TriangleRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, vertices, triangles_indices, vertex_weights, sigma, scaling,
                shs=None, colors_precomp=None, face_textures=None,
                mip_textures=None, mip_offsets=None):
        
        raster_settings = self.raster_settings

        if face_textures is not None and face_textures.numel() > 0:
            if shs is None:
                shs = torch.Tensor([])
            if colors_precomp is None:
                colors_precomp = torch.Tensor([])
        else:
            if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
                raise Exception('Please provide excatly one of either SHs or precomputed colors!')
            if shs is None:
                shs = torch.Tensor([])
            if colors_precomp is None:
                colors_precomp = torch.Tensor([])

        return rasterize_triangles(
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            shs,
            colors_precomp,
            scaling,
            raster_settings,
            face_textures,
            mip_textures,
            mip_offsets,
        )


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)
