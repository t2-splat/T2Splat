 #ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
 #define CUDA_RASTERIZER_FORWARD_H_INCLUDED

 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>

 namespace FORWARD
 {
	 void preprocess(int P, int D, int M,
		 const float* vertices,
		 const int* triangles_indices,
		 const float* vertex_weights,
		 const float sigma,
		 float* scaling,
		 const float* shs,
		 bool* clamped,
		 const float* colors_precomp,
		 const float* viewmatrix,
		 const float* projmatrix,
		 const glm::vec3* cam_pos,
		 const int W, int H,
		 const float focal_x, float focal_y,
		 const float tan_fovx, float tan_fovy,
		 int* radii,
		 float2* normals,
		 float* offsets,
		 float* p_w,
		 float2* p_image,
		 int* indices,
		 float2* points_xy_image,
		 float* depths,
		 float4* conic_opacity,
		 float2* phi_center,
		 uint2* rect_min,
		 uint2* rect_max,
		 const dim3 grid,
		 uint32_t* tiles_touched,
		 bool prefiltered,
		 float* tri_lod = nullptr,
		 int face_tex_res = 0);

	void computeVertexColors(
		int V, int D, int M,
		const float* vertices,
		const float* shs,
		bool* clamped,
		float* rgb,
		float* vertex_depth,
		const float* viewmatrix,
		const glm::vec3* cam_pos);

	void computeVertexDepths(
		int V,
		const float* vertices,
		const float* viewmatrix,
		float* vertex_depth);

	 void render(
		 const dim3 grid, dim3 block,
		 const uint2* ranges,
		 const uint32_t* point_list,
		 int W, int H,
		 const float2* normals,
		 const float* offsets,
		 const float2* points_xy_image,
		 const float* vertex_depth,
		 const int* triangles_indices,
		 const float sigma,
		 const float* features,
		 const float4* conic_opacity,
		 const float* depths,
		 const float2* phi_center,
		 const float2* p_image,
		 float* final_T,
		 uint32_t* n_contrib,
		 const float* bg_color,
		 float* out_color,
		 float* out_others,
		 float* max_blending,
		 int* was_rendered,
		 const float* face_textures,
		 int face_tex_res,
		 int face_tex_channels,
		 const float* vertices,
		 const glm::vec3* campos,
		 int sh_degree,
		 const float* mip_textures = nullptr,
		 const int* mip_offsets = nullptr,
		 int num_mip_levels = 0,
		 int mip_texels_per_face = 0,
		 const float* tri_lod = nullptr);
 }


 #endif
