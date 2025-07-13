/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

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
    const int P, int D, int M,
    const float* background,
    const int width, int height,
    const int8_t* means3D_quantized, float means3D_scale, int means3D_zero_point,
    const int8_t* shs_quantized, float shs_scale, int shs_zero_point,
    const int8_t* colors_precomp_quantized, float colors_precomp_scale, int colors_precomp_zero_point,
    const int8_t* opacities_quantized, float opacities_scale, int opacities_zero_point,
    const int8_t* scales_quantized, float scales_scale, int scales_zero_point,
    const float scale_modifier,
    const int8_t* rotations_quantized, float rotations_scale, int rotations_zero_point,
    const int8_t* cov3D_precomp_quantized, float cov3D_precomp_scale, int cov3D_precomp_zero_point,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    int8_t* out_color_quantized, float out_color_scale, int out_color_zero_point,
    int* radii = nullptr,
    bool debug = false);


		static void visible_filter(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			bool debug);
		
		
		
		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif