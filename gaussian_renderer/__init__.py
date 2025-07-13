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
import torch
from einops import repeat
import config
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import confignew
def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)
            
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    #concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    #masked = concatenated_all[mask]
    #scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    # post-process cov
    scaling_repeat,repeat_anchor=concatenated_repeated.split([6,3],dim=-1)
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]

    
    '''
    
    if config.iteration_para>12500:

        clamp_factor=max(1,200-(config.iteration_para-12500)//50)
        N = offsets.shape[0]
        num_blocks = N // 10  # 块的数量
        remainder = N % 10  # 处理剩余的元素

        # 为每个 offset 创建对应的 level
        a = level[:num_blocks].repeat_interleave(10)  # 重复每个 level 对应的 10 次
        if remainder > 0:
            a = torch.cat((a, level[num_blocks].view(1).repeat(remainder)))
        b=torch.ones_like(torch.tensor(a, device=a.device))
        # 将 offsets 限制在 [-0.044 * (0.5 * a), 0.044 * (0.5 * a)]
        offsets = torch.clamp(offsets, min=-0.5*(clamp_factor)*(a+1) .view(-1, 1), max=0.5 * clamp_factor*(a+1).view(-1, 1))
        #scaling = torch.clamp(offsets, min=-2.0*(clamp_factor)*(1 ** a).view(-1, 1), max=2.0 * clamp_factor*(1 ** a).view(-1, 1))
        #offsets = torch.clamp(offsets.detach(), -max_offset, max_offset)
    
    if config.start_render==1:

        N = offsets.shape[0]
        num_blocks = N // 10  # 块的数量
        remainder = N % 10  # 处理剩余的元素

        # 为每个 offset 创建对应的 level
        a = level[:num_blocks].repeat_interleave(10)  # 重复每个 level 对应的 10 次
        if remainder > 0:
            a = torch.cat((a, level[num_blocks].view(1).repeat(remainder)))
        b=torch.ones_like(torch.tensor(a, device=a.device))
        #=torch.ones_like(torch.tensor(offsets[:,0], device=a.device))
        # 将 offsets 限制在 [-0.044 * (0.5 * a), 0.044 * (0.5 * a)]
        offsets = torch.clamp(offsets, min=-0.5*  (a+1).view(-1, 1), max=0.5*(a+1).view(-1, 1))
    '''
    '''
    if config.iteration_para > 12500:
        clamp_factor = max(1, 200 - (config.iteration_para - 12500) // 50)
        N = offsets.shape[0]
        num_blocks = N // 10
        remainder = N % 10

        # 为每个 offset 创建对应的 level
        a = level[:num_blocks].repeat_interleave(10)
        if remainder > 0:
            a = torch.cat((a, level[num_blocks].view(1).repeat(remainder)))
    
        # 计算动态范围边界
        scale = 1.5 * clamp_factor * (a + 3).view(-1, 1)
        scale = scale.expand(-1, offsets.shape[1])  # [N, 3]，扩展到与 offsets 相同的通道数
    
        # 超出范围的值使用对数压缩 
        # 1. 处理超过上限的值
        over_upper = offsets > scale
        if torch.any(over_upper):
            over_ratio = offsets[over_upper] / scale[over_upper]
            compressed = scale[over_upper] * (1.0 + torch.log(over_ratio) / 5.0)
            offsets[over_upper] = torch.min(compressed, scale[over_upper] * 1.2)  # 限制最大压缩值
    
        # 2. 处理低于下限的值
        under_lower = offsets < -scale
        if torch.any(under_lower):
            under_ratio = (-scale[under_lower]) / offsets[under_lower]  # 计算超出比例
            compressed = -scale[under_lower] * (1.0 + torch.log(under_ratio) / 5.0)
            offsets[under_lower] = torch.max(compressed, -scale[under_lower] * 1.2)  # 限制最大压缩值
    if config.start_render==1:
        clamp_factor =1
        N = offsets.shape[0]
        num_blocks = N // 10
        remainder = N % 10

        # 为每个 offset 创建对应的 level
        a = level[:num_blocks].repeat_interleave(10)
        if remainder > 0:
            a = torch.cat((a, level[num_blocks].view(1).repeat(remainder)))
    
        # 计算动态范围边界
        scale = 1.5 * clamp_factor * (a + 3).view(-1, 1)
        scale = scale.expand(-1, offsets.shape[1])  # [N, 3]，扩展到与 offsets 相同的通道数
    
        # 超出范围的值使用对数压缩
        # 1. 处理超过上限的值
        over_upper = offsets > scale
        if torch.any(over_upper):
            over_ratio = offsets[over_upper] / scale[over_upper]
            compressed = scale[over_upper] * (1.0 + torch.log(over_ratio) / 5.0)
            offsets[over_upper] = torch.min(compressed, scale[over_upper] * 1.2)  # 限制最大压缩值
    
        # 2. 处理低于下限的值
        under_lower = offsets < -scale
        if torch.any(under_lower):
            under_ratio = (-scale[under_lower]) / offsets[under_lower]  # 计算超出比例
            compressed = -scale[under_lower] * (1.0 + torch.log(under_ratio) / 5.0)
            offsets[under_lower] = torch.max(compressed, -scale[under_lower] * 1.2)  # 限制最大压缩值
   
    '''
    #scaling = torch.clamp(offsets, min=-2.0*(clamp_factor) .view(-1, 1), max=2.0 * clamp_factor.view(-1, 1))
    #print(f'offset:{offsets}')
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    # 示例用法()
    #offsets=torch.zeros_like(offsets)
    xyz = repeat_anchor + offsets 
    #if config.iteration_para%1000==0:
        #pc.plot_levels()
        #print(anchor.shape)
        #print(confignew.viewmatrix_new)
    
    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code)
    
   
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
        # ==== 新增：高斯点筛选 ====
    # scaling: [N, 3], opacity: [N, 1] or [N]
   
    scaling_prod = scaling.prod(dim=1)*10000  # [N]
    
    z_vals =xyz[:, 2].abs()                   # [N]
  
    # 防止除0，建议加一个极小值
    eps = 0.0
    #score = (scaling_prod * opacity.flatten()*opacity.flatten()*opacity.flatten()*opacity.flatten()*opacity.flatten()*opacity.flatten()/((z_vals))).abs() -z_vals/sigma
    #print(f"z_vals device: {z_vals.device}")
    #print(f"sigma device: {sigma.device}")
    
    score = ((scaling_prod * (opacity.flatten().abs())**12.95 )/z_vals.abs()).abs()
    
    print(f"[Score Stats] min={score.min().item():.4g} max={score.max().item():.4g} mean={score.mean().item():.4g} std={score.std().item():.4g}")
    print(f"[Score Quantiles] 0%={score.quantile(0).item():.4g}, 15%={score.quantile(0.2).item():.4g}, 50%={score.quantile(0.5).item():.4g}, 75%={score.quantile(0.75).item():.4g}, 100%={score.quantile(1).item():.4g}")
    mask_filter = score >= 0.8e-20
    #score.quantile(0.2).item()
    #0.000000005
    # 应用 mask 到所有属性
    
    xyz =xyz[mask_filter]
    screenspace_points = screenspace_points[mask_filter]
    color = color[mask_filter]
    scaling = scaling[mask_filter]
    opacity = opacity[mask_filter]
    rot=rot[mask_filter]
    
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image,radii= rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None,
        )
    #print(depths)
    #print(f'depth_print')
    #print(f'{rendered_image.shape}')
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor[pc._anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii_pure >0
    return visible_mask
