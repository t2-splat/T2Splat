 #pragma once
 #include <torch/extension.h>
 #include <cstdio>
 #include <tuple>
 #include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizetrianglesCUDA(
	const torch::Tensor& background,
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const torch::Tensor& vertex_weights,
	const float sigma,
    const torch::Tensor& colors,
	torch::Tensor& scaling,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& face_textures,
	const torch::Tensor& mip_textures,
	const torch::Tensor& mip_offsets);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizetrianglesBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const torch::Tensor& vertex_weights,
	const float sigma,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const torch::Tensor& face_textures,
	const torch::Tensor& mip_textures,
	const torch::Tensor& mip_offsets);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max);


void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
);
