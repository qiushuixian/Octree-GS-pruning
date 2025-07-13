#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
'''
from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def quantize_to_int8(tensor):
    # 计算量化参数（scale 和 zero_point）
    scale = torch.max(torch.abs(tensor)) / 127.0  # 假设数据范围在 [-127, 127]
    zero_point = 0
    quantized_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, torch.qint8)
    return quantized_tensor, scale, zero_point
'''
# 在调用 rasterize_gaussians 之前量化输入张量
means3D_quantized = quantize_to_int8(means3D)
colors_precomp_quantized = quantize_to_int8(colors_precomp)
opacities_quantized = quantize_to_int8(opacities)
scales_quantized = quantize_to_int8(scales)
rotations_quantized = quantize_to_int8(rotations)
cov3Ds_precomp_quantized = quantize_to_intized(cov3Ds_precomp)
'''
def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster,
):
    '''
    means3D_quantized, means3D_scale, means3D_zero_point = quantize_to_int8(means3D)
    colors_precomp_quantized, colors_precomp_scale, colors_precomp_zero_point = quantize_to_int8(colors_precomp)
    opacities_quantized, opacities_scale, opacities_zero_point = quantize_to_int8(opacities)
    scales_quantized, scales_scale, scales_zero_point = quantize_to_int8(scales)
    rotations_quantized, rotations_scale, rotations_zero_point = quantize_to_int8(rotations)
    cov3Ds_precomp_quantized, cov3Ds_precomp_scale, cov3Ds_precomp_zero_point = quantize_to_int8(cov3Ds_precomp)
    '''
    # 调用 C++/CUDA 扩展
    return _RasterizeGaussians.apply(
        means3D_quantized,
        means2D,
        sh,
        colors_precomp_quantized,
        opacities_quantized,
        scales_quantized,
        rotations_quantized,
        cov3Ds_precomp_quantized,
        raster_settings,
        means3D_scale,
        means3D_zero_point,
        colors_precomp_scale,
        colors_precomp_zero_point,
        opacities_scale,
        opacities_zero_point,
        scales_scale,
        scales_zero_point,
        rotations_scale,
        rotations_zero_point,
        cov3Ds_precomp_scale,
        cov3Ds_precomp_zero_point,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        means3D_scale,
        means3D_zero_point,
        colors_precomp_scale,
        colors_precomp_zero_point,
        opacities_scale,
        opacities_zero_point,
        scales_scale,
        scales_zero_point,
        rotations_scale,
        rotations_zero_point,
        cov3Ds_precomp_scale,
        cov3Ds_precomp_zero_point,
    ):
        # 保存量化参数
        ctx.means3D_scale = means3D_scale
        ctx.means3D_zero_point = means3D_zero_point
        ctx.colors_precomp_scale = colors_precomp_scale
        ctx.colors_precomp_zero_point = colors_precomp_zero_point
        ctx.opacities_scale = opacities_scale
        ctx.opacities_zero_point = opacities_zero_point
        ctx.scales_scale = scales_scale
        ctx.scales_zero_point = scales_zero_point
        ctx.rotations_scale = rotations_scale
        ctx.rotations_zero_point = rotations_zero_point
        ctx.cov3Ds_precomp_scale = cov3Ds_precomp_scale
        ctx.cov3Ds_precomp_zero_point = cov3Ds_precomp_zero_point

        # 调用 C++/CUDA 扩展
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
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
            means3D_scale,
            means3D_zero_point,
            colors_precomp_scale,
            colors_precomp_zero_point,
            opacities_scale,
            opacities_zero_point,
            scales_scale,
            scales_zero_point,
            rotations_scale,
            rotations_zero_point,
            cov3Ds_precomp_scale,
            cov3Ds_precomp_zero_point,
        )
        
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, depths,geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii,depths,geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # 保存必要的张量用于反向传播
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        
        return color,radii,depth

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # 恢复必要的值
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 重新构造参数
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            ctx.means3D_scale,
            ctx.means3D_zero_point,
            ctx.colors_precomp_scale,
            ctx.colors_precomp_zero_point,
            ctx.opacities_scale,
            ctx.opacities_zero_point,
            ctx.scales_scale,
            ctx.scales_zero_point,
            ctx.rotations_scale,
            ctx.rotations_zero_point,
            ctx.cov3Ds_precomp_scale,
            ctx.cov3Ds_precomp_zero_point,
        )

        

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
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

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_aussians_filter(means3D,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.debug)
        return  radii
    
    


