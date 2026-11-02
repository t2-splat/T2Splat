 #ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
 #define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>

 namespace BACKWARD
 {
	 void render(
		 const dim3 grid, dim3 block,
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
		 const float* mip_textures = nullptr,
		 float* dL_dmip_textures = nullptr,
		 const int* mip_offsets = nullptr,
		 int num_mip_levels = 0,
		 int mip_texels_per_face = 0,
		 const float* tri_lod = nullptr);

	 void computeVertexColorGradients(
        int V, int D, int M,
		int W, int H,
		const float* view,
		const float* proj,
		float* p_w,
        const float* vertices,
        const float* shs,
        const bool* clamped,
        const glm::vec3* campos,
        const float* dL_dcolor,
		const float* dL_dpoints2D,
        glm::vec3* dL_dvertices3D,
        float* dL_dsh,
		const float* dL_dvertice_depth);

	 void preprocess(
		 int P, int D, int M,
		 const float* vertices,
		 const int* triangles_indices,
		 const float* vertex_weights,
		 int W, int H,
		 const int* radii,
		 const float* shs,
		 const bool* clamped,
		 const float* view,
		 const float* proj,
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
		 );
 }

 #endif
