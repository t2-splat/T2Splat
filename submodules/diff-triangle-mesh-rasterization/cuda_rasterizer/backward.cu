// English comments maintained by zhtang2004@gmail.com.
#include <cstdint>
/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

 #include "backward.h"
 #include "auxiliary.h"
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 namespace cg = cooperative_groups;

 __device__ void circumcenter_backward(float3 A, float3 B, float3 C, float3 dU, float3* dA, float3* dB, float3* dC) {
    float3 AB = make_float3(B.x-A.x, B.y-A.y, B.z-A.z);
    float3 AC = make_float3(C.x-A.x, C.y-A.y, C.z-A.z);
    
    float3 N = make_float3(
        AB.y*AC.z - AB.z*AC.y,
        AB.z*AC.x - AB.x*AC.z,
        AB.x*AC.y - AB.y*AC.x
    );
    
    float AB2 = dot_float3(AB, AB);
    float AC2 = dot_float3(AC, AC);
    float denom = 2.0f * dot_float3(N, N);
    float inv_denom = 1.0f / denom;
    
    float3 term_sum = make_float3(
        (N.y*AB.z - N.z*AB.y)*AC2 + (AC.y*N.z - AC.z*N.y)*AB2,
        (N.z*AB.x - N.x*AB.z)*AC2 + (AC.z*N.x - AC.x*N.z)*AB2,
        (N.x*AB.y - N.y*AB.x)*AC2 + (AC.x*N.y - AC.y*N.x)*AB2
    );
    
    float3 dterm_sum = scale_float3(dU, inv_denom);
    float ddenom = -dot_float3(term_sum, dU) / (denom*denom);
    
    float dAB2 = 0.0f, dAC2 = 0.0f;
    float3 dN = make_float3(0,0,0);
    float3 dAB = make_float3(0,0,0);
    float3 dAC = make_float3(0,0,0);
    
    for (int i = 0; i < 2; i++) {
        float3 cross_vec = (i == 0) ? 
            make_float3(N.y*AB.z - N.z*AB.y, N.z*AB.x - N.x*AB.z, N.x*AB.y - N.y*AB.x) :
            make_float3(AC.y*N.z - AC.z*N.y, AC.z*N.x - AC.x*N.z, AC.x*N.y - AC.y*N.x);
            
        float scalar = (i == 0) ? AC2 : AB2;
        float3 dcross_vec = scale_float3(dterm_sum, scalar);
        float dscalar = dot_float3(dterm_sum, cross_vec);
        
        if (i == 0) dAC2 += dscalar;
        else dAB2 += dscalar;
        
        if (i == 0) {
            dN = add_float3(dN, cross_float3(AB, dcross_vec));
            dAB = add_float3(dAB, cross_float3(dcross_vec, N));
        } else {
            dAC = add_float3(dAC, cross_float3(N, dcross_vec));
            dN = add_float3(dN, cross_float3(dcross_vec, AC));
        }
    }
    
    dN = add_float3(dN, scale_float3(N, 2.0f * ddenom * 2.0f));
    
    dAB = add_float3(dAB, cross_float3(AC, dN));
    dAC = add_float3(dAC, cross_float3(dN, AB));
    
    dAB = add_float3(dAB, scale_float3(AB, 2.0f * dAB2));
    dAC = add_float3(dAC, scale_float3(AC, 2.0f * dAC2));
    
    *dA = add_float3(scale_float3(dAB, -1.0f), scale_float3(dAC, -1.0f));
    *dB = dAB;
    *dC = dAC;
	*dA = add_float3(*dA, dU);
}

// ═══════════════════════════════════════════════════════════════════════
// Per-vertex SH backward (legacy path)
// ═══════════════════════════════════════════════════════════════════════
 __device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dvertices3D, glm::vec3* dL_dshs)
 {
	 glm::vec3 pos = means;
	 glm::vec3 dir_orig = pos - campos;
	 glm::vec3 dir = dir_orig / glm::length(dir_orig);

	 glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	 glm::vec3 dL_dRGB = dL_dcolor[idx];
	 dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	 dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	 dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	 glm::vec3 dRGBdx(0, 0, 0);
	 glm::vec3 dRGBdy(0, 0, 0);
	 glm::vec3 dRGBdz(0, 0, 0);
	 float x = dir.x;
	 float y = dir.y;
	 float z = dir.z;

	 glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	 float dRGBdsh0 = SH_C0;
	 dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	 if (deg > 0)
	 {
		 float dRGBdsh1 = -SH_C1 * y;
		 float dRGBdsh2 = SH_C1 * z;
		 float dRGBdsh3 = -SH_C1 * x;
		 dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		 dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		 dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		 dRGBdx = -SH_C1 * sh[3];
		 dRGBdy = -SH_C1 * sh[1];
		 dRGBdz = SH_C1 * sh[2];

		 if (deg > 1)
		 {
			 float xx = x * x, yy = y * y, zz = z * z;
			 float xy = x * y, yz = y * z, xz = x * z;

			 float dRGBdsh4 = SH_C2[0] * xy;
			 float dRGBdsh5 = SH_C2[1] * yz;
			 float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			 float dRGBdsh7 = SH_C2[3] * xz;
			 float dRGBdsh8 = SH_C2[4] * (xx - yy);
			 dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			 dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			 dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			 dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			 dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			 dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			 dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			 dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			 if (deg > 2)
			 {
				 float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				 float dRGBdsh10 = SH_C3[1] * xy * z;
				 float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				 float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				 float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				 float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				 float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				 dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				 dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				 dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				 dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				 dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				 dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				 dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				 dRGBdx += (
					 SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					 SH_C3[1] * sh[10] * yz +
					 SH_C3[2] * sh[11] * -2.f * xy +
					 SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					 SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					 SH_C3[5] * sh[14] * 2.f * xz +
					 SH_C3[6] * sh[15] * 3.f * (xx - yy));

				 dRGBdy += (
					 SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					 SH_C3[1] * sh[10] * xz +
					 SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					 SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					 SH_C3[4] * sh[13] * -2.f * xy +
					 SH_C3[5] * sh[14] * -2.f * yz +
					 SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				 dRGBdz += (
					 SH_C3[1] * sh[10] * xy +
					 SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					 SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					 SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					 SH_C3[5] * sh[14] * (xx - yy));
			 }
		 }
	 }
 }

// ═══════════════════════════════════════════════════════════════════════
// Per-pixel SH backward: from dL/dRGB to dL/dsh (48-dim)
// Re-evaluates SH forward to determine clamping, then backpropagates
// ═══════════════════════════════════════════════════════════════════════
__device__ void computeLocalSHBackward(
	int deg,
	const float* __restrict__ pixel_sh,
	const float3& dir,
	const glm::vec3& dL_dcolor_in,
	float* __restrict__ dL_dsh_out)
{
	const glm::vec3* sh = (const glm::vec3*)pixel_sh;

	glm::vec3 result = SH_C0 * sh[0];
	float x = dir.x, y = dir.y, z = dir.z;

	if (deg > 0) {
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
		if (deg > 1) {
			float xx = x*x, yy = y*y, zz = z*z;
			float xy = x*y, yz = y*z, xz = x*z;
			result = result +
				SH_C2[0]*xy*sh[4] + SH_C2[1]*yz*sh[5] +
				SH_C2[2]*(2.0f*zz-xx-yy)*sh[6] +
				SH_C2[3]*xz*sh[7] + SH_C2[4]*(xx-yy)*sh[8];
			if (deg > 2) {
				result = result +
					SH_C3[0]*y*(3.0f*xx-yy)*sh[9] +
					SH_C3[1]*xy*z*sh[10] +
					SH_C3[2]*y*(4.0f*zz-xx-yy)*sh[11] +
					SH_C3[3]*z*(2.0f*zz-3.0f*xx-3.0f*yy)*sh[12] +
					SH_C3[4]*x*(4.0f*zz-xx-yy)*sh[13] +
					SH_C3[5]*z*(xx-yy)*sh[14] +
					SH_C3[6]*x*(xx-3.0f*yy)*sh[15];
			}
		}
	}
	result += 0.5f;

	glm::vec3 dL_dRGB = dL_dcolor_in;
	if (result.x < 0) dL_dRGB.x = 0;
	if (result.y < 0) dL_dRGB.y = 0;
	if (result.z < 0) dL_dRGB.z = 0;

	glm::vec3* dL_dsh = (glm::vec3*)dL_dsh_out;

	dL_dsh[0] = SH_C0 * dL_dRGB;
	if (deg > 0) {
		dL_dsh[1] = -SH_C1 * y * dL_dRGB;
		dL_dsh[2] = SH_C1 * z * dL_dRGB;
		dL_dsh[3] = -SH_C1 * x * dL_dRGB;
		if (deg > 1) {
			float xx = x*x, yy = y*y, zz = z*z;
			float xy = x*y, yz = y*z, xz = x*z;
			dL_dsh[4] = SH_C2[0]*xy * dL_dRGB;
			dL_dsh[5] = SH_C2[1]*yz * dL_dRGB;
			dL_dsh[6] = SH_C2[2]*(2.0f*zz-xx-yy) * dL_dRGB;
			dL_dsh[7] = SH_C2[3]*xz * dL_dRGB;
			dL_dsh[8] = SH_C2[4]*(xx-yy) * dL_dRGB;
			if (deg > 2) {
				dL_dsh[9]  = SH_C3[0]*y*(3.0f*xx-yy) * dL_dRGB;
				dL_dsh[10] = SH_C3[1]*xy*z * dL_dRGB;
				dL_dsh[11] = SH_C3[2]*y*(4.0f*zz-xx-yy) * dL_dRGB;
				dL_dsh[12] = SH_C3[3]*z*(2.0f*zz-3.0f*xx-3.0f*yy) * dL_dRGB;
				dL_dsh[13] = SH_C3[4]*x*(4.0f*zz-xx-yy) * dL_dRGB;
				dL_dsh[14] = SH_C3[5]*z*(xx-yy) * dL_dRGB;
				dL_dsh[15] = SH_C3[6]*x*(xx-3.0f*yy) * dL_dRGB;
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════
// Distribute dL/dsh to the 4 bilinear neighbors via atomicAdd
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void scatterFaceTextureGrad(
	float* __restrict__ dL_dface_tex,
	int R, int C,
	float local_x, float local_y,
	const float* __restrict__ dL_dsh,
	int face_id)
{
	int ix0 = __float2int_rd(local_x);
	int iy0 = __float2int_rd(local_y);
	ix0 = max(0, min(ix0, R - 1));
	iy0 = max(0, min(iy0, R - 1));
	int ix1 = min(ix0 + 1, R - 1);
	int iy1 = min(iy0 + 1, R - 1);

	float fx = local_x - (float)ix0;
	float fy = local_y - (float)iy0;
	fx = fmaxf(0.0f, fminf(fx, 1.0f));
	fy = fmaxf(0.0f, fminf(fy, 1.0f));

	float w00 = (1.0f - fx) * (1.0f - fy);
	float w10 = fx * (1.0f - fy);
	float w01 = (1.0f - fx) * fy;
	float w11 = fx * fy;

	const int stride_face = R * R * C;
	const int stride_row  = R * C;
	float* base = dL_dface_tex + static_cast<uint64_t>(face_id) * stride_face;

	float* p00 = base + iy0 * stride_row + ix0 * C;
	float* p10 = base + iy0 * stride_row + ix1 * C;
	float* p01 = base + iy1 * stride_row + ix0 * C;
	float* p11 = base + iy1 * stride_row + ix1 * C;

	for (int c = 0; c < C; c++) {
		float g = dL_dsh[c];
		atomicAdd(p00 + c, w00 * g);
		atomicAdd(p10 + c, w10 * g);
		atomicAdd(p01 + c, w01 * g);
		atomicAdd(p11 + c, w11 * g);
	}
}

// ═══════════════════════════════════════════════════════════════════════
// Bilinear sampling helper (same as forward, duplicated here to avoid
// cross-TU device function linkage issues)
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void sampleFaceTextureBwd(
	const float* __restrict__ face_tex,
	int R, int C,
	float local_x, float local_y,
	float* __restrict__ out_sh)
{
	int ix0 = __float2int_rd(local_x);
	int iy0 = __float2int_rd(local_y);
	ix0 = max(0, min(ix0, R - 1));
	iy0 = max(0, min(iy0, R - 1));
	int ix1 = min(ix0 + 1, R - 1);
	int iy1 = min(iy0 + 1, R - 1);

	float fx = local_x - (float)ix0;
	float fy = local_y - (float)iy0;
	fx = fmaxf(0.0f, fminf(fx, 1.0f));
	fy = fmaxf(0.0f, fminf(fy, 1.0f));

	float w00 = (1.0f - fx) * (1.0f - fy);
	float w10 = fx * (1.0f - fy);
	float w01 = (1.0f - fx) * fy;
	float w11 = fx * fy;

	const int stride = R * C;
	const float* r00 = face_tex + iy0 * stride + ix0 * C;
	const float* r10 = face_tex + iy0 * stride + ix1 * C;
	const float* r01 = face_tex + iy1 * stride + ix0 * C;
	const float* r11 = face_tex + iy1 * stride + ix1 * C;

	for (int c = 0; c < C; c++) {
		out_sh[c] = w00 * r00[c] + w10 * r10[c]
		          + w01 * r01[c] + w11 * r11[c];
	}
}

// ═══════════════════════════════════════════════════════════════════════
// Trilinear sampling for backward recomputation (same logic as forward)
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void sampleMipTrilinearBwd(
	const float* __restrict__ mip_face_base,
	const int* __restrict__ mip_offsets,
	int num_levels, int base_R, int C,
	float wB, float wC, float lod,
	float* __restrict__ out)
{
	lod = fmaxf(0.0f, fminf(lod, (float)(num_levels - 1)));
	int lo = __float2int_rd(lod);
	lo = max(0, min(lo, num_levels - 1));
	int hi = min(lo + 1, num_levels - 1);
	float frac = lod - (float)lo;

	int R_lo = base_R >> lo;
	int R_hi = base_R >> hi;
	R_lo = max(R_lo, 1);
	R_hi = max(R_hi, 1);

	float lx_lo = wB * (float)(R_lo - 1);
	float ly_lo = wC * (float)(R_lo - 1);
	float lx_hi = wB * (float)(R_hi - 1);
	float ly_hi = wC * (float)(R_hi - 1);

	float val_lo[3], val_hi[3];
	sampleFaceTextureBwd(mip_face_base + mip_offsets[lo] * C, R_lo, C, lx_lo, ly_lo, val_lo);
	sampleFaceTextureBwd(mip_face_base + mip_offsets[hi] * C, R_hi, C, lx_hi, ly_hi, val_hi);

	for (int c = 0; c < C; c++)
		out[c] = (1.0f - frac) * val_lo[c] + frac * val_hi[c];
}

// ═══════════════════════════════════════════════════════════════════════
// Scatter gradient to a single mip level (bilinear adjoint within level)
// level_base: start of this face's data for this mip level in dL buffer
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void scatterFaceTextureLevelGrad(
	float* __restrict__ dL_level_base,
	int R, int C,
	float local_x, float local_y,
	const float* __restrict__ dL_dsh,
	int face_id, int texels_per_face)
{
	int ix0 = __float2int_rd(local_x);
	int iy0 = __float2int_rd(local_y);
	ix0 = max(0, min(ix0, R - 1));
	iy0 = max(0, min(iy0, R - 1));
	int ix1 = min(ix0 + 1, R - 1);
	int iy1 = min(iy0 + 1, R - 1);

	float fx = local_x - (float)ix0;
	float fy = local_y - (float)iy0;
	fx = fmaxf(0.0f, fminf(fx, 1.0f));
	fy = fmaxf(0.0f, fminf(fy, 1.0f));

	float w00 = (1.0f - fx) * (1.0f - fy);
	float w10 = fx * (1.0f - fy);
	float w01 = (1.0f - fx) * fy;
	float w11 = fx * fy;

	const int stride_row = R * C;
	float* base = dL_level_base + static_cast<uint64_t>(face_id) * texels_per_face * C;

	float* p00 = base + iy0 * stride_row + ix0 * C;
	float* p10 = base + iy0 * stride_row + ix1 * C;
	float* p01 = base + iy1 * stride_row + ix0 * C;
	float* p11 = base + iy1 * stride_row + ix1 * C;

	for (int c = 0; c < C; c++) {
		float g = dL_dsh[c];
		atomicAdd(p00 + c, w00 * g);
		atomicAdd(p10 + c, w10 * g);
		atomicAdd(p01 + c, w01 * g);
		atomicAdd(p11 + c, w11 * g);
	}
}

// ═══════════════════════════════════════════════════════════════════════
// Trilinear gradient scatter: distribute dL/dresidual to two mip levels
// ═══════════════════════════════════════════════════════════════════════
__device__ __forceinline__ void scatterMipFaceTextureGrad(
	float* __restrict__ dL_dmip_tex,
	const int* __restrict__ mip_offsets,
	int num_levels, int base_R, int C,
	float wB, float wC, float lod,
	const float* __restrict__ dL_dout,
	int face_id, int texels_per_face)
{
	lod = fmaxf(0.0f, fminf(lod, (float)(num_levels - 1)));
	int lo = __float2int_rd(lod);
	lo = max(0, min(lo, num_levels - 1));
	int hi = min(lo + 1, num_levels - 1);
	float frac = lod - (float)lo;

	int R_lo = base_R >> lo;
	int R_hi = base_R >> hi;
	R_lo = max(R_lo, 1);
	R_hi = max(R_hi, 1);

	float lx_lo = wB * (float)(R_lo - 1);
	float ly_lo = wC * (float)(R_lo - 1);
	float lx_hi = wB * (float)(R_hi - 1);
	float ly_hi = wC * (float)(R_hi - 1);

	float dL_scaled[3];
	for (int c = 0; c < C; c++) dL_scaled[c] = dL_dout[c] * (1.0f - frac);
	scatterFaceTextureLevelGrad(
		dL_dmip_tex + mip_offsets[lo] * C,
		R_lo, C, lx_lo, ly_lo, dL_scaled, face_id, texels_per_face);

	if (hi != lo) {
		for (int c = 0; c < C; c++) dL_scaled[c] = dL_dout[c] * frac;
		scatterFaceTextureLevelGrad(
			dL_dmip_tex + mip_offsets[hi] * C,
			R_hi, C, lx_hi, ly_hi, dL_scaled, face_id, texels_per_face);
	}
}


 __global__ void computeVertexColorsCUDA(
    int V, int D, int M,
	int W, int H,
	const float* viewmatrix,
	const float* proj,
	const float* vertices,
	const float* shs,
	const bool* clamped,
	const glm::vec3* campos,
	const float* dL_dcolor,
	const float* dL_dpoints2D,
	glm::vec3* dL_dvertices3D,
	float* dL_dsh,
	const float* dL_dvertice_depth)
{	
    auto idx = cg::this_grid().thread_rank();
    if (idx >= V)
        return;

	glm::vec3 vertex {
		vertices[3 * idx + 0],
		vertices[3 * idx + 1],
		vertices[3 * idx + 2]
	};

	computeColorFromSH(idx, D, M, vertex, *campos,
		shs,
		clamped,
		(glm::vec3*)dL_dcolor,   
		(glm::vec3*)dL_dvertices3D, 
		(glm::vec3*)dL_dsh
	);

	float3 dL_ddepht = {0.0f, 0.0f, dL_dvertice_depth[idx]};
	float3 transposed_dL_ddepth = transformPoint4x3Transpose(dL_ddepht, viewmatrix);

	dL_dvertices3D[idx].x = transposed_dL_ddepth.x;
	dL_dvertices3D[idx].y = transposed_dL_ddepth.y;
	dL_dvertices3D[idx].z = transposed_dL_ddepth.z;

}

 template<int C>
 __global__ void preprocessCUDA(
	 int P, int D, int M,
	 const float* vertices,
	 const int* triangles_indices,
	 const float* vertex_weights,
	 int W, int H,
	 const int* radii,
	 const float* shs,
	 const bool* clamped,
	 const float* proj,
	 const float* viewmatrix,
	 float2* points_xy_image,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 const glm::vec3* campos,
	 glm::vec3* dL_dvertices3D,
	 float* dL_dvertice_weights,
	 const float2* dL_dnormals,
	 const float* dL_doffsets,
	 float3* dL_dmean2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolor,
	 float* dL_dsh)
 {
	 auto idx = cg::this_grid().thread_rank();

	 if (idx >= P || !(radii[idx] > 0))
		 return;

	 const int cumsum_for_triangle = 3 * idx;;
	 const int offset = 3 * cumsum_for_triangle;
	 float3 center_triangle = {0.0f, 0.0f, 0.0f};
	 float sum_x[MAX_NB_POINTS] = {0.0f};
	 float sum_y[MAX_NB_POINTS] = {0.0f};
	 float sum_z[MAX_NB_POINTS] = {0.0f};

	 float min_weight = INFINITY;
	 int id_lowest_weight = -1;
	 for (int i = 0; i < 3; i++) {
		int vertex_index = triangles_indices[cumsum_for_triangle + i];
		center_triangle.x += vertices[3 * vertex_index];
		center_triangle.y += vertices[3 * vertex_index + 1];
		center_triangle.z += vertices[3 * vertex_index + 2];
		float weight = vertex_weights[vertex_index];
		if (weight < min_weight) {
			id_lowest_weight = vertex_index;
			min_weight = weight;
		}
	 }

	 float3 total_sum = {center_triangle.x, center_triangle.y, center_triangle.z};

	 center_triangle.x /= 3;
	 center_triangle.y /= 3;
	 center_triangle.z /= 3;

	 float3 p_view_triangle;
	 if (!in_frustum_triangle(idx, center_triangle, viewmatrix, proj, false, p_view_triangle)){
		 return;
	 }

	 float loss_points_x[MAX_NB_POINTS] = {0.0f};
	 float loss_points_y[MAX_NB_POINTS] = {0.0f};

	 for (int i = 0; i < 3; i++) {
		float dL_dnormal_x = dL_dnormals[cumsum_for_triangle + i].x;
		float dL_dnormal_y = dL_dnormals[cumsum_for_triangle + i].y;
		float dL_doffset = dL_doffsets[cumsum_for_triangle + i];

		float2 p1_conv = p_image[cumsum_for_triangle + i];
		float2 p2_conv = p_image[cumsum_for_triangle + (i + 1) % 3];

		float nx = p2_conv.y - p1_conv.y;
		float ny = -(p2_conv.x - p1_conv.x);
		float norm = __fsqrt_rn(nx * nx + ny * ny);
		float inv_norm = 1.0f / norm;
	
		float2 normal = {nx * inv_norm, ny * inv_norm};
		float offset_ = -(normal.x * p1_conv.x + normal.y * p1_conv.y);

		if (normal.x * points_xy_image[idx].x + normal.y * points_xy_image[idx].y + offset_ > 0) {
			dL_dnormal_x = -dL_dnormal_x;
			dL_dnormal_y = -dL_dnormal_y;
			dL_doffset = -dL_doffset;
		}

		dL_dnormal_x += (-p1_conv.x) * dL_doffset;
		dL_dnormal_y += (-p1_conv.y) * dL_doffset;
	
		float norm_sq = norm * norm;
		float inv_norm_cubed = 1.0f / (norm * norm_sq);
		float dL_dnx = (dL_dnormal_x * ny * ny - dL_dnormal_y * nx * ny) * inv_norm_cubed;
		float dL_dny = (-dL_dnormal_x * nx * ny + dL_dnormal_y * nx * nx) * inv_norm_cubed;
	
		float2 dL_dp1_conv = {dL_dny, -dL_dnx};
		float2 dL_dp2_conv = {-dL_dny, dL_dnx};
	
		dL_dp1_conv.x += (-normal.x) * dL_doffset;
		dL_dp1_conv.y += (-normal.y) * dL_doffset;

		loss_points_x[indices[cumsum_for_triangle + i]] += dL_dp1_conv.x;
    	loss_points_y[indices[cumsum_for_triangle + i]] += dL_dp1_conv.y;

		loss_points_x[indices[cumsum_for_triangle + (i + 1) % 3]] += dL_dp2_conv.x;
    	loss_points_y[indices[cumsum_for_triangle + (i + 1) % 3]] += dL_dp2_conv.y;
	}

	 float3 dL_ddepht = {0.0f, 0.0f, dL_dmean2D[idx].x};
	 float3 transposed_dL_ddepth = transformPoint4x3Transpose(dL_ddepht, viewmatrix);

	 for (int i = 0; i < 3; i++) {
		int vertex_index = triangles_indices[cumsum_for_triangle + i];

		float mul1 = (proj[0] * vertices[3 * vertex_index] + proj[4] * vertices[3 * vertex_index + 1] + proj[8] * vertices[3 * vertex_index + 2] + proj[12]) * p_w[cumsum_for_triangle + i] * p_w[cumsum_for_triangle + i];
		float mul2 = (proj[1] * vertices[3 * vertex_index] + proj[5] * vertices[3 * vertex_index + 1] + proj[9] * vertices[3 * vertex_index + 2] + proj[13]) * p_w[cumsum_for_triangle + i] * p_w[cumsum_for_triangle + i];
				
		dL_dvertices3D[vertex_index].x += (proj[0] * p_w[cumsum_for_triangle + i] - proj[3] * mul1) * loss_points_x[i]  + (proj[1] * p_w[cumsum_for_triangle + i] - proj[3] * mul2) * loss_points_y[i] + transposed_dL_ddepth.x / 3;
		dL_dvertices3D[vertex_index].y += (proj[4] * p_w[cumsum_for_triangle + i] - proj[7] * mul1) * loss_points_x[i] + (proj[5] * p_w[cumsum_for_triangle + i] - proj[7] * mul2) * loss_points_y[i] + transposed_dL_ddepth.y / 3;
		dL_dvertices3D[vertex_index].z += (proj[8] * p_w[cumsum_for_triangle + i] - proj[11] * mul1) * loss_points_x[i] + (proj[9] * p_w[cumsum_for_triangle + i] - proj[11] * mul2) * loss_points_y[i] + transposed_dL_ddepth.z / 3;
	 }

	float3 normal_cvx = {0.0f, 0.0f, 0.0f};
	int vertex_index = triangles_indices[cumsum_for_triangle];
	float3 p0 = make_float3(vertices[3 * vertex_index + 0], vertices[3 * vertex_index + 1], vertices[3 * vertex_index + 2]);
	vertex_index = triangles_indices[cumsum_for_triangle + 1];
	float3 p1 = make_float3(vertices[3 * vertex_index + 0], vertices[3 * vertex_index + 1], vertices[3 * vertex_index + 2]);
	vertex_index = triangles_indices[cumsum_for_triangle + 2];
	float3 p2 = make_float3(vertices[3 * vertex_index + 0], vertices[3 * vertex_index + 1], vertices[3 * vertex_index + 2]);
	float3 v1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
	float3 v2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
	float3 v3 = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);

	float3 unnorm_ross_prod = make_float3(
		 v1.y * v2.z - v1.z * v2.y,
		 v1.z * v2.x - v1.x * v2.z,
		 v1.x * v2.y - v1.y * v2.x
	);
	float3 cross_prod = transformVec4x3(unnorm_ross_prod, viewmatrix);

	float length = sqrtf(cross_prod.x*cross_prod.x + cross_prod.y*cross_prod.y + cross_prod.z*cross_prod.z);
	if (length > 1e-8f) {
		 cross_prod.x /= length;
		 cross_prod.y /= length;
		 cross_prod.z /= length;
	}
	normal_cvx = cross_prod;

	float3 p_view_triangle_ = transformPoint4x3(center_triangle, viewmatrix);
	float length_viewpoint = sqrtf(p_view_triangle_.x*p_view_triangle_.x + p_view_triangle_.y*p_view_triangle_.y + p_view_triangle_.z*p_view_triangle_.z);
	length_viewpoint = max(length_viewpoint, 1e-4f);
	float3 normalized_camera_center;
	normalized_camera_center.x = p_view_triangle_.x / length_viewpoint;
	normalized_camera_center.y = p_view_triangle_.y / length_viewpoint;
	normalized_camera_center.z = p_view_triangle_.z / length_viewpoint;

	float3 dir = make_float3(
		 normalized_camera_center.x * normal_cvx.x,
		 normalized_camera_center.y * normal_cvx.y,
		 normalized_camera_center.z * normal_cvx.z
	);
	 
	float cos = -sumf3(dir);
	 
	const float threshold = 0.001f;
	if (fabsf(cos) < threshold) {
		return;
	}
		
	float multiplier = cos > 0 ? 1 : -1;
	normal_cvx = {cross_prod.x * multiplier, cross_prod.y * multiplier, cross_prod.z * multiplier};

	float3 dL_dtn = {dL_dnormal3D[idx * 3 + 0]*multiplier, dL_dnormal3D[idx * 3 + 1]*multiplier, dL_dnormal3D[idx * 3 + 2]*multiplier};

	float matrix_w0[9], matrix_w1[9], matrix_w2[9];
	matrix_w0[0] = 0.0f;   matrix_w0[1] = -v3.z; matrix_w0[2] = v3.y;
	matrix_w0[3] = v3.z;   matrix_w0[4] = 0.0f;  matrix_w0[5] = -v3.x;
	matrix_w0[6] = -v3.y;  matrix_w0[7] = v3.x;  matrix_w0[8] = 0.0f;

	matrix_w1[0] = 0.0f;   matrix_w1[1] = v2.z;  matrix_w1[2] = -v2.y;
	matrix_w1[3] = -v2.z;  matrix_w1[4] = 0.0f;  matrix_w1[5] = v2.x;
	matrix_w1[6] = v2.y;   matrix_w1[7] = -v2.x; matrix_w1[8] = 0.0f;

	matrix_w2[0] = 0.0f;   matrix_w2[1] = -v1.z; matrix_w2[2] = v1.y;
	matrix_w2[3] = v1.z;   matrix_w2[4] = 0.0f;  matrix_w2[5] = -v1.x;
	matrix_w2[6] = -v1.y;  matrix_w2[7] = v1.x;  matrix_w2[8] = 0.0f;

	float normal_transpose[9];
	normal_transpose[0] = normal_cvx.x*normal_cvx.x; 
	normal_transpose[1] = normal_transpose[3] = normal_cvx.x*normal_cvx.y;
	normal_transpose[2] = normal_transpose[6] = normal_cvx.x*normal_cvx.z;
	normal_transpose[4] = normal_cvx.y*normal_cvx.y;
	normal_transpose[5] = normal_transpose[7] = normal_cvx.y*normal_cvx.z;
	normal_transpose[8] = normal_cvx.z*normal_cvx.z;

	float length_inv = 1 / length;

	float projection_matrix[9];
	projection_matrix[0] = viewmatrix[0]; projection_matrix[1] = viewmatrix[4]; projection_matrix[2] = viewmatrix[8];
	projection_matrix[3] = viewmatrix[1]; projection_matrix[4] = viewmatrix[5]; projection_matrix[5] = viewmatrix[9];
	projection_matrix[6] = viewmatrix[2]; projection_matrix[7] = viewmatrix[6]; projection_matrix[8] = viewmatrix[10];

	float matrix_w0_transformed[9], matrix_w1_transformed[9], matrix_w2_transformed[9];
	transformMat3x3(projection_matrix, matrix_w0, matrix_w0_transformed);
	transformMat3x3(projection_matrix, matrix_w1, matrix_w1_transformed);
	transformMat3x3(projection_matrix, matrix_w2, matrix_w2_transformed);

	float norm_times_matrix0[9], norm_times_matrix1[9], norm_times_matrix2[9];
	transformMat3x3(normal_transpose, matrix_w0_transformed, norm_times_matrix0);
	transformMat3x3(normal_transpose, matrix_w1_transformed, norm_times_matrix1);
	transformMat3x3(normal_transpose, matrix_w2_transformed, norm_times_matrix2);

	float matrix_substraction0[9], matrix_substraction1[9], matrix_substraction2[9];
	substractionMat3x3(matrix_w0_transformed, norm_times_matrix0, matrix_substraction0);
	substractionMat3x3(matrix_w1_transformed, norm_times_matrix1, matrix_substraction1);
	substractionMat3x3(matrix_w2_transformed, norm_times_matrix2, matrix_substraction2);

	vertex_index = triangles_indices[cumsum_for_triangle];
	vertex_index = triangles_indices[cumsum_for_triangle + 1];
	vertex_index = triangles_indices[cumsum_for_triangle + 2];

	atomicAdd(&dL_dvertice_weights[id_lowest_weight], dL_dopacity[idx]);
 }

// ═══════════════════════════════════════════════════════════════════════
// Backward render kernel — supports BOTH vertex-color and face-texture paths
// ═══════════════════════════════════════════════════════════════════════
 template <uint32_t C>
 __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
 renderCUDA(
	 const uint2* __restrict__ ranges,
	 const uint32_t* __restrict__ point_list,
	 int W, int H,
	 const float* __restrict__ bg_color,
	 const float sigma,
	 const int* __restrict__ triangles_indices,
	 const float2* __restrict__ normals,
	 const float* __restrict__ offsets,
	 const float4* __restrict__ conic_opacity,
	 const float* __restrict__ depths,
	 const float2* __restrict__ means2D,
	 const float2* __restrict__ phi_center,
	 const float2* __restrict__ p_image,
	 const float* __restrict__ vertex_depth,
	 const float* __restrict__ colors,
	 const float* __restrict__ final_Ts,
	 const uint32_t* __restrict__ n_contrib,
	 const float* __restrict__ dL_dpixels,
	 const float* __restrict__ dL_depths,
	 float2* __restrict__ dL_dnormals,
	 float* __restrict__ dL_doffsets,
	 float3* __restrict__ dL_dmean2D,
	 float* __restrict__ dL_dopacity,
	 float* __restrict__ dL_dnormal3D,
	 float* __restrict__ dL_dcolors,
	 float* __restrict__ dL_dpoints2D,
	 float* __restrict__ dL_dvertice_depth,
	 // ── Per-Face Texture ──
	 const float* __restrict__ face_textures,
	 float* __restrict__ dL_dface_textures,
	 const int face_tex_res,
	 const int face_tex_channels,
	 const float* __restrict__ vertices,
	 const glm::vec3* __restrict__ campos,
	 const int sh_degree,
	 // ── Mip-Texture ──
	 const float* __restrict__ mip_textures,
	 float* __restrict__ dL_dmip_textures,
	 const int* __restrict__ mip_offsets,
	 const int num_mip_levels,
	 const int mip_texels_per_face,
	 const float* __restrict__ tri_lod)
 {
	 auto block = cg::this_thread_block();
	 const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	 const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	 const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	 const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	 const uint32_t pix_id = W * pix.y + pix.x;
	 const float2 pixf = { (float)pix.x, (float)pix.y };

	 const bool inside = pix.x < W && pix.y < H;
	 const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	 const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	 bool done = !inside;
	 int toDo = range.y - range.x;

	 const bool use_face_tex = (face_textures != nullptr && face_tex_res > 0);
	 const bool use_mip_tex = (mip_textures != nullptr && num_mip_levels > 0);

	 __shared__ int collected_id[BLOCK_SIZE];
	 __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	 __shared__ float collected_colors[C * BLOCK_SIZE];
	 __shared__ float2 collected_p_images[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float2 collected_normals[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_offsets[BLOCK_SIZE * MAX_NB_POINTS];
	 __shared__ float collected_depths[BLOCK_SIZE];
	 __shared__ float2 collected_xy[BLOCK_SIZE];
	 __shared__ float2 collected_phi_center[BLOCK_SIZE];
	 __shared__ float collected_lod[BLOCK_SIZE];

	 const float T_final = inside ? final_Ts[pix_id] : 0;
	 float T = T_final;

	 uint32_t contributor = toDo;
	 const int last_contributor = inside ? n_contrib[pix_id] : 0;

	 float accum_rec[C] = { 0 };
	 float dL_dpixel[C];
	 if (inside)
		 for (int i = 0; i < C; i++)
			 dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
	}

	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};

	 float last_alpha = 0;
	 float last_color[C] = { 0 };

	 for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	 {
		 block.sync();
		 const int progress = i * BLOCK_SIZE + block.thread_rank();
		 if (range.x + progress < range.y)
		 {
			 const int coll_id = point_list[range.y - progress - 1];
			 collected_id[block.thread_rank()] = coll_id;
			 collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			 if (!use_face_tex && !use_mip_tex) {
				 for (int ii = 0; ii < C; ii++)
					 collected_colors[ii * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ii];
			 }
			 collected_depths[block.thread_rank()] = depths[coll_id];
			 collected_xy[block.thread_rank()] = means2D[coll_id];
			 for (int k = 0; k < 3; k++) {
				collected_normals[MAX_NB_POINTS * block.thread_rank() + k] = normals[3 * coll_id + k];
				collected_offsets[MAX_NB_POINTS * block.thread_rank() + k] = offsets[3 * coll_id + k];
				collected_p_images[MAX_NB_POINTS * block.thread_rank() + k] = p_image[3 * coll_id + k];
			}
			collected_phi_center[block.thread_rank()] = phi_center[coll_id];
			collected_lod[block.thread_rank()] = (use_mip_tex && tri_lod != nullptr) ? tri_lod[coll_id] : 0.0f;
		 }
		 block.sync();

		 for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		 {
			 contributor--;
			 if (contributor >= last_contributor)
				 continue;

			 int j_id = collected_id[j];
			 float4 con_o = collected_conic_opacity[j];
			 float normal[3] = {con_o.x, con_o.y, con_o.z};
			 float2 phi_center_min = collected_phi_center[j];
			 float distances[MAX_NB_POINTS];
			 float max_val = -INFINITY;
			 int base = j * MAX_NB_POINTS;
			 bool outside = false;

			 for (int k = 0; k < 3; k++) {
				 float dist = (collected_normals[base + k].x * pixf.x
						  + collected_normals[base + k].y * pixf.y
						  + collected_offsets[base + k]);
				 if (dist > 0) {
					outside = true;
					break;
				 }
				 distances[k] = dist;
				 max_val = fmaxf(max_val, dist);
			 }

			 if (outside)
				 continue;

			 float phi_x = max_val;
			 float phi_final = phi_x * phi_center_min.x;
			 float Cx = fmaxf(0.0f,  __powf(phi_final, sigma));

			 const float alpha = min(0.99f, con_o.w * Cx);
			 if (alpha < 1.0f / 255.0f)
				 continue;

			 T = T / (1.f - alpha);
			 const float dchannel_dcolor = alpha * T;

			 // ── Barycentric coordinates ──
			 float2 uv0 = collected_p_images[j * 3 + 0];
			 float2 uv1 = collected_p_images[j * 3 + 1];
			 float2 uv2 = collected_p_images[j * 3 + 2];

			 float2 v0 = { uv1.x - uv0.x, uv1.y - uv0.y };
			 float2 v1 = { uv2.x - uv0.x, uv2.y - uv0.y };
			 float2 v2 = { pixf.x  - uv0.x, pixf.y  - uv0.y };

			 float denom = v0.x * v1.y - v1.x * v0.y;
			 float invDen = 1.0f / denom;

			 float b0 = ( v2.x * v1.y - v1.x * v2.y) * invDen;
			 float b1 = (-v2.x * v0.y + v0.x * v2.y) * invDen;
			 float b2 = 1.0f - b0 - b1;

			 int aux = 3 * j_id;
			 int vertex_idx0 = triangles_indices[aux];
			 int vertex_idx1 = triangles_indices[aux + 1];
			 int vertex_idx2 = triangles_indices[aux + 2];

			 float depth_vertex_0 = vertex_depth[vertex_idx0];
			 float depth_vertex_1 = vertex_depth[vertex_idx1];
			 float depth_vertex_2 = vertex_depth[vertex_idx2];

			 float wA = b2;
			 float wB = b0;
			 float wC = b1;

			 // ════════════════════════════════════════════════════════════
			 //  COLOR BACKWARD: Gouraud + optional per-face RGB residual
			 // ════════════════════════════════════════════════════════════

			 float interp_color[C];
			 float dL_dalpha = 0.0f;
			 const int global_id = collected_id[j];

			 // Recompute Gouraud color (+ residual if enabled)
			 for (int ch = 0; ch < C; ++ch) {
				 float c0 = colors[vertex_idx0 * C + ch];
				 float c1 = colors[vertex_idx1 * C + ch];
				 float c2 = colors[vertex_idx2 * C + ch];
				 interp_color[ch] = wA * c0 + wB * c1 + wC * c2;
			 }

			 float local_x_bwd = 0, local_y_bwd = 0;
			 float lod_bwd = 0;
			 if (use_mip_tex) {
				 float residual[3];
				 const float* mip_face_base = mip_textures
					 + static_cast<uint64_t>(j_id) * mip_texels_per_face * face_tex_channels;
				 lod_bwd = collected_lod[j];
				 sampleMipTrilinearBwd(mip_face_base, mip_offsets,
					 num_mip_levels, face_tex_res, face_tex_channels,
					 wB, wC, lod_bwd, residual);
				 for (int ch = 0; ch < C; ++ch)
					 interp_color[ch] += residual[ch];
			 } else if (use_face_tex) {
				 float residual[3];
				 local_x_bwd = wB * (float)(face_tex_res - 1);
				 local_y_bwd = wC * (float)(face_tex_res - 1);
				 const float* face_tex = face_textures
					 + static_cast<uint64_t>(j_id) * face_tex_res * face_tex_res * face_tex_channels;
				 sampleFaceTextureBwd(face_tex, face_tex_res, face_tex_channels,
									  local_x_bwd, local_y_bwd, residual);
				 for (int ch = 0; ch < C; ++ch)
					 interp_color[ch] += residual[ch];
			 }

			 // Alpha-blending backward + vertex color gradients
			 for (int ch = 0; ch < C; ++ch) {
				 accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				 last_color[ch] = interp_color[ch];
				 const float dL_dchannel = dL_dpixel[ch];
				 dL_dalpha += (interp_color[ch] - accum_rec[ch]) * dL_dchannel;

				 float grad0 = dchannel_dcolor * dL_dchannel * b2;
				 float grad1 = dchannel_dcolor * dL_dchannel * b0;
				 float grad2 = dchannel_dcolor * dL_dchannel * b1;

				 atomicAdd(&dL_dcolors[vertex_idx0*C + ch], grad0);
				 atomicAdd(&dL_dcolors[vertex_idx1*C + ch], grad1);
				 atomicAdd(&dL_dcolors[vertex_idx2*C + ch], grad2);
			 }

			 // Scatter residual gradient to mip texture or face texture
			 if (use_mip_tex) {
				 float dL_dresidual[3];
				 for (int ch = 0; ch < C; ++ch)
					 dL_dresidual[ch] = dchannel_dcolor * dL_dpixel[ch];
				 scatterMipFaceTextureGrad(
					 dL_dmip_textures, mip_offsets,
					 num_mip_levels, face_tex_res, face_tex_channels,
					 wB, wC, lod_bwd, dL_dresidual,
					 j_id, mip_texels_per_face);
			 } else if (use_face_tex) {
				 float dL_dresidual[3];
				 for (int ch = 0; ch < C; ++ch)
					 dL_dresidual[ch] = dchannel_dcolor * dL_dpixel[ch];
				 scatterFaceTextureGrad(
					 dL_dface_textures, face_tex_res, face_tex_channels,
					 local_x_bwd, local_y_bwd, dL_dresidual, j_id);
			 }

			 // ── Depth backward (shared by both paths) ──
			 float depth_interp = wA * depth_vertex_0 + wB * depth_vertex_1 + wC * depth_vertex_2;
			 float weight_here = alpha * T;
			 atomicAdd(&dL_dvertice_depth[vertex_idx0], dL_ddepth * weight_here * wA);
			 atomicAdd(&dL_dvertice_depth[vertex_idx1], dL_ddepth * weight_here * wB);
			 atomicAdd(&dL_dvertice_depth[vertex_idx2], dL_ddepth * weight_here * wC);

			 if (contributor == median_contributor-1) {
				  atomicAdd(&dL_dvertice_depth[vertex_idx0], dL_dmedian_depth * wA);
				  atomicAdd(&dL_dvertice_depth[vertex_idx1], dL_dmedian_depth * wB);
				  atomicAdd(&dL_dvertice_depth[vertex_idx2], dL_dmedian_depth * wC);
			 }

			 // ── Alpha, depth, normal backward (shared) ──
			 accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			 last_depth = depth_interp;
			 dL_dalpha += (depth_interp - accum_depth_rec) * dL_ddepth;

			 accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			 dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			 for (int ch = 0; ch < 3; ch++) {
				 accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				 last_normal[ch] = normal[ch];
				 dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				 atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			 }

			 dL_dalpha *= T;
			 last_alpha = alpha;

			 float bg_dot_dpixel = 0;
			 for (int ii = 0; ii < C; ii++)
				 bg_dot_dpixel += bg_color[ii] * dL_dpixel[ii];
			 dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			 const float dL_dC = con_o.w * dL_dalpha;
			 float dL_dphi_x = dL_dC * (sigma / phi_x) * Cx;

			 #pragma unroll
			 for (int k = 0; k < 3; k++) {
				if (fabsf(distances[k] - max_val) < 1e-6f) {
					float dL_dnormal_x = dL_dphi_x * pixf.x;
					float dL_dnormal_y = dL_dphi_x * pixf.y;
					atomicAdd(&(dL_dnormals[aux + k].x), dL_dnormal_x);
					atomicAdd(&(dL_dnormals[aux + k].y), dL_dnormal_y);
					atomicAdd(&(dL_doffsets[aux + k]), dL_dphi_x);
				}
			 }

			 atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * Cx);
		 }
	 }
 }

 void BACKWARD::preprocess(
	 int P, int D, int M,
	 const float* vertices,
	 const int* triangles_indices,
	 const float* vertex_weights,
	 int W, int H,
	 const int* radii,
	 const float* shs,
	 const bool* clamped,
	 const float* viewmatrix,
	 const float* projmatrix,
	 float2* points_xy_image,
	 float* p_w,
	 float2* p_image,
	 int* indices,
	 const float focal_x, float focal_y,
	 const float tan_fovx, float tan_fovy,
	 const glm::vec3* campos,
	 glm::vec3* dL_dvertices3D, 
	 float* dL_dvertice_weights,
	 const float2* dL_dnormals,
	 const float* dL_doffsets,
	 float3* dL_dmean2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolor,
	 float* dL_dsh
	 )
 {
	 preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		 P, D, M,
		 vertices,
		 triangles_indices,
		 vertex_weights,
		 W, H,
		 radii,
		 shs,
		 clamped,
		 projmatrix,
		 viewmatrix,
		 points_xy_image,
		 p_w,
		 p_image,
		 indices,
		 campos,
		 (glm::vec3*)dL_dvertices3D,
		 (float*) dL_dvertice_weights,
		 (float2*) dL_dnormals,
		 dL_doffsets,
		 (float3*)dL_dmean2D,
		 dL_dopacity,
		 dL_dnormal3D,
		 dL_dcolor,
		 dL_dsh
		 );
 }

 void BACKWARD::computeVertexColorGradients(
    int V, int D, int M,
	int W, int H,
	const float* viewmatrix,
	const float* projmatrix,
	float* p_w,
	const float* vertices,
	const float* shs,
	const bool* clamped,
	const glm::vec3* campos,
	const float* dL_dcolor,
	const float* dL_dpoints2D,
	glm::vec3* dL_dvertices3D,
	float* dL_dsh,
	const float* dL_dvertice_depth)
 {
    computeVertexColorsCUDA<<<(V + 255) / 256, 256>>>(
        V, D, M, W, H, viewmatrix, projmatrix, vertices, shs, clamped, campos, dL_dcolor, dL_dpoints2D, (glm::vec3*)dL_dvertices3D, dL_dsh, dL_dvertice_depth
    );
 }

 void BACKWARD::render(
	 const dim3 grid, const dim3 block,
	 const uint2* ranges,
	 const uint32_t* point_list,
	 int W, int H,
	 const float* bg_color,
	 const float sigma,
	 const int* triangles_indices,
	 const float2* normals,
	 const float* offsets,
	 const float4* conic_opacity,
	 const float* depths,
	 const float2* means2D,
	 const float2* phi_center,
	 const float2* p_image,
	 const float* vertex_depth,
	 const float* colors,
	 const float* final_Ts,
	 const uint32_t* n_contrib,
	 const float* dL_dpixels,
	 const float* dL_depths,
	 float2* dL_dnormals,
	 float* dL_doffsets,
	 float3* dL_dmean2D,
	 float* dL_dopacity,
	 float* dL_dnormal3D,
	 float* dL_dcolors,
	 float* dL_dpoints2D,
	 float* dL_dvertice_depth,
	 const float* face_textures,
	 float* dL_dface_textures,
	 int face_tex_res,
	 int face_tex_channels,
	 const float* vertices,
	 const glm::vec3* campos,
	 int sh_degree,
	 const float* mip_textures,
	 float* dL_dmip_textures,
	 const int* mip_offsets,
	 int num_mip_levels,
	 int mip_texels_per_face,
	 const float* tri_lod)
 {
	 renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		 ranges, point_list, W, H,
		 bg_color, sigma,
		 triangles_indices, normals, offsets,
		 conic_opacity, depths, means2D, phi_center, p_image,
		 vertex_depth, colors,
		 final_Ts, n_contrib,
		 dL_dpixels, dL_depths,
		 dL_dnormals, dL_doffsets, dL_dmean2D,
		 dL_dopacity, dL_dnormal3D, dL_dcolors,
		 dL_dpoints2D, dL_dvertice_depth,
		 face_textures, dL_dface_textures,
		 face_tex_res, face_tex_channels,
		 vertices, campos, sh_degree,
		 mip_textures, dL_dmip_textures,
		 mip_offsets, num_mip_levels,
		 mip_texels_per_face, tri_lod
		 );
 }
