#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, const int V, int D, int M,
			const float* background,
			const int width, int height,
			const float* vertices,
			const int* triangles_indices,
			const float* vertex_weights,
			const float sigma,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			float* scaling,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			float* max_blending,
			int* radii = nullptr,
			int* was_rendered = nullptr,
			bool debug = false,
			const float* face_textures = nullptr,
			int face_tex_res = 0,
			int face_tex_channels = 0,
			const float* mip_textures = nullptr,
			const int* mip_offsets = nullptr,
			int num_mip_levels = 0,
			int mip_texels_per_face = 0);

		static void backward(
			const int P, const int V, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* vertices,
			const int* triangles_indices,
			const float* vertex_weights,
			const float sigma,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			float* dL_dmeans2D,
			float* dL_dnormal3D,
			float* dL_dvertices3D,
			float* dL_dvertex_weights,
			float* dL_dnormals,
			float* dL_doffsets,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsh,
			float* dL_dpoints2D,
			float* dL_dvertice_depth,
			bool debug,
			const float* face_textures = nullptr,
			float* dL_dface_textures = nullptr,
			int face_tex_res = 0,
			int face_tex_channels = 0,
			const float* mip_textures = nullptr,
			float* dL_dmip_textures = nullptr,
			const int* mip_offsets = nullptr,
			int num_mip_levels = 0,
			int mip_texels_per_face = 0);
	};
};

#endif
