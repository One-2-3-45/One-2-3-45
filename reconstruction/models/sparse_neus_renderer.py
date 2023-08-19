"""
The codes are heavily borrowed from NeuS
"""

import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from models.render_utils import sample_pdf

from models.projector import Projector
from tsparse.torchsparse_utils import sparse_to_dense_channel

from models.fast_renderer import FastRenderer

from models.patch_projector import PatchProjector


class SparseNeuSRenderer(nn.Module):
    """
    conditional neus render;
    optimize on normalized world space;
    warped by nn.Module to support DataParallel traning
    """

    def __init__(self,
                 rendering_network_outside,
                 sdf_network,
                 variance_network,
                 rendering_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 perturb,
                 alpha_type='div',
                 conf=None
                 ):
        super(SparseNeuSRenderer, self).__init__()

        self.conf = conf
        self.base_exp_dir = conf['general.base_exp_dir']

        # network setups
        self.rendering_network_outside = rendering_network_outside
        self.sdf_network = sdf_network
        self.variance_network = variance_network
        self.rendering_network = rendering_network

        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.perturb = perturb
        self.alpha_type = alpha_type

        self.rendering_projector = Projector()  # used to obtain features for generalized rendering

        self.h_patch_size = self.conf.get_int('model.h_patch_size', default=3)
        self.patch_projector = PatchProjector(self.h_patch_size)

        self.ray_tracer = FastRenderer()  # ray_tracer to extract depth maps from sdf_volume

        # - fitted rendering or general rendering
        try:
            self.if_fitted_rendering = self.sdf_network.if_fitted_rendering
        except:
            self.if_fitted_rendering = False

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_variance,
                  conditional_valid_mask_volume=None):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

        if conditional_valid_mask_volume is not None:
            pts_mask = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), conditional_valid_mask_volume)
            pts_mask = pts_mask.reshape(batch_size, n_samples)
            pts_mask = pts_mask[:, :-1] * pts_mask[:, 1:]  # [batch_size, n_samples-1]
        else:
            pts_mask = torch.ones([batch_size, n_samples]).to(pts.device)

        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        dot_val = None
        if self.alpha_type == 'uniform':
            dot_val = torch.ones([batch_size, n_samples - 1]) * -1.0
        else:
            dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
            prev_dot_val = torch.cat([torch.zeros([batch_size, 1]).to(device), dot_val[:, :-1]], dim=-1)
            dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)
            dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)
            dot_val = dot_val.clip(-10.0, 0.0) * pts_mask
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
        next_esti_sdf = mid_sdf + dot_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_variance)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_variance)
        alpha_sdf = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        alpha = alpha_sdf

        # - apply pts_mask
        alpha = pts_mask * alpha

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, lod,
                   sdf_network, gru_fusion,
                   # * related to conditional feature
                   conditional_volume=None,
                   conditional_valid_mask_volume=None
                   ):
        device = rays_o.device
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]

        if conditional_valid_mask_volume is not None:
            pts_mask = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), conditional_valid_mask_volume)
            pts_mask = pts_mask.reshape(batch_size, n_importance)
            pts_mask_bool = (pts_mask > 0).view(-1)
        else:
            pts_mask = torch.ones([batch_size, n_importance]).to(pts.device)

        new_sdf = torch.ones([batch_size * n_importance, 1]).to(pts.dtype).to(device) * 100

        if torch.sum(pts_mask) > 1:
            new_outputs = sdf_network.sdf(pts.reshape(-1, 3)[pts_mask_bool], conditional_volume, lod=lod)
            new_sdf[pts_mask_bool] = new_outputs['sdf_pts_scale%d' % lod]  # .reshape(batch_size, n_importance)

        new_sdf = new_sdf.view(batch_size, n_importance)

        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        sdf = torch.cat([sdf, new_sdf], dim=-1)

        z_vals, index = torch.sort(z_vals, dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    @torch.no_grad()
    def get_pts_mask_for_conditional_volume(self, pts, mask_volume):
        """

        :param pts: [N, 3]
        :param mask_volume: [1, 1, X, Y, Z]
        :return:
        """
        num_pts = pts.shape[0]
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])

        pts_mask = F.grid_sample(mask_volume, pts, mode='nearest')  # [1, c, 1, 1, num_pts]
        pts_mask = pts_mask.view(-1, num_pts).permute(1, 0).contiguous()  # [num_pts, 1]

        return pts_mask

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    lod,
                    sdf_network,
                    rendering_network,
                    background_alpha=None,  # - no use here
                    background_sampled_color=None,  # - no use here
                    background_rgb=None,  # - no use here
                    alpha_inter_ratio=0.0,
                    # * related to conditional feature
                    conditional_volume=None,
                    conditional_valid_mask_volume=None,
                    # * 2d feature maps
                    feature_maps=None,
                    color_maps=None,
                    w2cs=None,
                    intrinsics=None,
                    img_wh=None,
                    query_c2w=None,  # - used for testing
                    if_general_rendering=True,
                    if_render_with_grad=True,
                    # * used for blending mlp rendering network
                    img_index=None,
                    rays_uv=None,
                    # * used for clear bg and fg
                    bg_num=0
                    ):
        device = rays_o.device
        N_rays = rays_o.shape[0]
        _, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(device)], -1)

        mid_z_vals = z_vals + dists * 0.5
        mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # * if conditional_volume is restored from sparse volume, need mask for pts
        if conditional_valid_mask_volume is not None:
            pts_mask = self.get_pts_mask_for_conditional_volume(pts, conditional_valid_mask_volume)
            pts_mask = pts_mask.reshape(N_rays, n_samples).float().detach()
            pts_mask_bool = (pts_mask > 0).view(-1)

            if torch.sum(pts_mask_bool.float()) < 1:  # ! when render out image, may meet this problem
                pts_mask_bool[:100] = True

        else:
            pts_mask = torch.ones([N_rays, n_samples]).to(pts.device)
        # import ipdb; ipdb.set_trace()
        # pts_valid = pts[pts_mask_bool]
        sdf_nn_output = sdf_network.sdf(pts[pts_mask_bool], conditional_volume, lod=lod)

        sdf = torch.ones([N_rays * n_samples, 1]).to(pts.dtype).to(device) * 100
        sdf[pts_mask_bool] = sdf_nn_output['sdf_pts_scale%d' % lod]  # [N_rays*n_samples, 1]
        feature_vector_valid = sdf_nn_output['sdf_features_pts_scale%d' % lod]
        feature_vector = torch.zeros([N_rays * n_samples, feature_vector_valid.shape[1]]).to(pts.dtype).to(device)
        feature_vector[pts_mask_bool] = feature_vector_valid

        # * estimate alpha from sdf
        gradients = torch.zeros([N_rays * n_samples, 3]).to(pts.dtype).to(device)
        # import ipdb; ipdb.set_trace()
        gradients[pts_mask_bool] = sdf_network.gradient(
            pts[pts_mask_bool], conditional_volume, lod=lod).squeeze()

        sampled_color_mlp = None
        rendering_valid_mask_mlp = None
        sampled_color_patch = None
        rendering_patch_mask = None

        if self.if_fitted_rendering:  # used for fine-tuning
            position_latent = sdf_nn_output['sampled_latent_scale%d' % lod]
            sampled_color_mlp = torch.zeros([N_rays * n_samples, 3]).to(pts.dtype).to(device)
            sampled_color_mlp_mask = torch.zeros([N_rays * n_samples, 1]).to(pts.dtype).to(device)

            # - extract pixel
            pts_pixel_color, pts_pixel_mask = self.patch_projector.pixel_warp(
                pts[pts_mask_bool][:, None, :], color_maps, intrinsics,
                w2cs, img_wh=None)  # [N_rays * n_samples,1, N_views,  3] , [N_rays*n_samples, 1, N_views]
            pts_pixel_color = pts_pixel_color[:, 0, :, :]  # [N_rays * n_samples, N_views,  3]
            pts_pixel_mask = pts_pixel_mask[:, 0, :]  # [N_rays*n_samples, N_views]

            # - extract patch
            if_patch_blending = False if rays_uv is None else True
            pts_patch_color, pts_patch_mask = None, None
            if if_patch_blending:
                pts_patch_color, pts_patch_mask = self.patch_projector.patch_warp(
                    pts.reshape([N_rays, n_samples, 3]),
                    rays_uv, gradients.reshape([N_rays, n_samples, 3]),
                    color_maps,
                    intrinsics[0], intrinsics,
                    query_c2w[0], torch.inverse(w2cs), img_wh=None
                )  # (N_rays, n_samples, N_src, Npx, 3), (N_rays, n_samples, N_src, Npx)
                N_src, Npx = pts_patch_mask.shape[2:]
                pts_patch_color = pts_patch_color.view(N_rays * n_samples, N_src, Npx, 3)[pts_mask_bool]
                pts_patch_mask = pts_patch_mask.view(N_rays * n_samples, N_src, Npx)[pts_mask_bool]

                sampled_color_patch = torch.zeros([N_rays * n_samples, Npx, 3]).to(device)
                sampled_color_patch_mask = torch.zeros([N_rays * n_samples, 1]).to(device)

            sampled_color_mlp_, sampled_color_mlp_mask_, \
            sampled_color_patch_, sampled_color_patch_mask_ = sdf_network.color_blend(
                pts[pts_mask_bool],
                position_latent,
                gradients[pts_mask_bool],
                dirs[pts_mask_bool],
                feature_vector[pts_mask_bool],
                img_index=img_index,
                pts_pixel_color=pts_pixel_color,
                pts_pixel_mask=pts_pixel_mask,
                pts_patch_color=pts_patch_color,
                pts_patch_mask=pts_patch_mask

            )  # [n, 3], [n, 1]
            sampled_color_mlp[pts_mask_bool] = sampled_color_mlp_
            sampled_color_mlp_mask[pts_mask_bool] = sampled_color_mlp_mask_.float()
            sampled_color_mlp = sampled_color_mlp.view(N_rays, n_samples, 3)
            sampled_color_mlp_mask = sampled_color_mlp_mask.view(N_rays, n_samples)
            rendering_valid_mask_mlp = torch.mean(pts_mask * sampled_color_mlp_mask, dim=-1, keepdim=True) > 0.5

            # patch blending
            if if_patch_blending:
                sampled_color_patch[pts_mask_bool] = sampled_color_patch_
                sampled_color_patch_mask[pts_mask_bool] = sampled_color_patch_mask_.float()
                sampled_color_patch = sampled_color_patch.view(N_rays, n_samples, Npx, 3)
                sampled_color_patch_mask = sampled_color_patch_mask.view(N_rays, n_samples)
                rendering_patch_mask = torch.mean(pts_mask * sampled_color_patch_mask, dim=-1,
                                                  keepdim=True) > 0.5  # [N_rays, 1]
            else:
                sampled_color_patch, rendering_patch_mask = None, None

        if if_general_rendering:  # used for general training
            # [512, 128, 16]; [4, 512, 128, 59]; [4, 512, 128, 4]
            ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, _, _ = self.rendering_projector.compute(
                pts.view(N_rays, n_samples, 3),
                # * 3d geometry feature volumes
                geometryVolume=conditional_volume[0],
                geometryVolumeMask=conditional_valid_mask_volume[0],
                # * 2d rendering feature maps
                rendering_feature_maps=feature_maps, # [n_views, 56, 256, 256]
                color_maps=color_maps,
                w2cs=w2cs,
                intrinsics=intrinsics,
                img_wh=img_wh,
                query_img_idx=0,  # the index of the N_views dim for rendering
                query_c2w=query_c2w,
            )

            # (N_rays, n_samples, 3)
            if if_render_with_grad:
                # import ipdb; ipdb.set_trace()
                # [nrays, 3] [nrays, 1]
                sampled_color, rendering_valid_mask = rendering_network(
                    ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask)
                # import ipdb; ipdb.set_trace()
            else:
                with torch.no_grad():
                    sampled_color, rendering_valid_mask = rendering_network(
                        ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask)
        else:
            sampled_color, rendering_valid_mask = None, None

        inv_variance = self.variance_network(feature_vector)[:, :1].clip(1e-6, 1e6)

        true_dot_val = (dirs * gradients).sum(-1, keepdim=True)  # * calculate

        iter_cos = -(F.relu(-true_dot_val * 0.5 + 0.5) * (1.0 - alpha_inter_ratio) + F.relu(
            -true_dot_val) * alpha_inter_ratio)  # always non-positive

        iter_cos = iter_cos * pts_mask.view(-1, 1)

        true_estimate_sdf_half_next = sdf + iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5
        true_estimate_sdf_half_prev = sdf - iter_cos.clip(-10.0, 10.0) * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(true_estimate_sdf_half_prev * inv_variance)
        next_cdf = torch.sigmoid(true_estimate_sdf_half_next * inv_variance)

        p = prev_cdf - next_cdf
        c = prev_cdf

        if self.alpha_type == 'div':
            alpha_sdf = ((p + 1e-5) / (c + 1e-5)).reshape(N_rays, n_samples).clip(0.0, 1.0)
        elif self.alpha_type == 'uniform':
            uniform_estimate_sdf_half_next = sdf - dists.reshape(-1, 1) * 0.5
            uniform_estimate_sdf_half_prev = sdf + dists.reshape(-1, 1) * 0.5
            uniform_prev_cdf = torch.sigmoid(uniform_estimate_sdf_half_prev * inv_variance)
            uniform_next_cdf = torch.sigmoid(uniform_estimate_sdf_half_next * inv_variance)
            uniform_alpha = F.relu(
                (uniform_prev_cdf - uniform_next_cdf + 1e-5) / (uniform_prev_cdf + 1e-5)).reshape(
                N_rays, n_samples).clip(0.0, 1.0)
            alpha_sdf = uniform_alpha
        else:
            assert False

        alpha = alpha_sdf

        # - apply pts_mask
        alpha = alpha * pts_mask

        # pts_radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(N_rays, n_samples)
        # inside_sphere = (pts_radius < 1.0).float().detach()
        # relax_inside_sphere = (pts_radius < 1.2).float().detach()
        inside_sphere = pts_mask
        relax_inside_sphere = pts_mask

        weights = alpha * torch.cumprod(torch.cat([torch.ones([N_rays, 1]).to(device), 1. - alpha + 1e-7], -1), -1)[:,
                          :-1]  # n_rays, n_samples
        weights_sum = weights.sum(dim=-1, keepdim=True)
        alpha_sum = alpha.sum(dim=-1, keepdim=True)

        if bg_num > 0:
            weights_sum_fg = weights[:, :-bg_num].sum(dim=-1, keepdim=True)
        else:
            weights_sum_fg = weights_sum

        if sampled_color is not None:
            color = (sampled_color * weights[:, :, None]).sum(dim=1)
        else:
            color = None
        # import ipdb; ipdb.set_trace()

        if background_rgb is not None and color is not None:
            color = color + background_rgb * (1.0 - weights_sum)
            # print("color device:" + str(color.device))
        # if color is not None:
        #     # import ipdb; ipdb.set_trace()
        #     color = color + (1.0 - weights_sum)


        ###################*  mlp color rendering  #####################
        color_mlp = None
        # import ipdb; ipdb.set_trace()
        if sampled_color_mlp is not None:
            color_mlp = (sampled_color_mlp * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None and color_mlp is not None:
            color_mlp = color_mlp + background_rgb * (1.0 - weights_sum)

        ############################ *  patch blending  ################
        blended_color_patch = None
        if sampled_color_patch is not None:
            blended_color_patch = (sampled_color_patch * weights[:, :, None, None]).sum(dim=1)  # [N_rays, Npx, 3]

        ######################################################

        gradient_error = (torch.linalg.norm(gradients.reshape(N_rays, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        # ! the gradient normal should be masked out, the pts out of the bounding box should also be penalized
        gradient_error = (pts_mask * gradient_error).sum() / (
                (pts_mask).sum() + 1e-5)

        depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        # print("[TEST]: weights_sum in render_core", weights_sum.mean())
        # print("[TEST]: weights_sum in render_core NAN number", weights_sum.isnan().sum())
        # if weights_sum.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()
        return {
            'color': color,
            'color_mask': rendering_valid_mask,  # (N_rays, 1)
            'color_mlp': color_mlp,
            'color_mlp_mask': rendering_valid_mask_mlp,
            'sdf': sdf,  # (N_rays, n_samples)
            'depth': depth,  # (N_rays, 1)
            'dists': dists,
            'gradients': gradients.reshape(N_rays, n_samples, 3),
            'variance': 1.0 / inv_variance,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weights_sum': weights_sum,
            'alpha_sum': alpha_sum,
            'alpha_mean': alpha.mean(),
            'cdf': c.reshape(N_rays, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'blended_color_patch': blended_color_patch,
            'blended_color_patch_mask': rendering_patch_mask,
            'weights_sum_fg': weights_sum_fg
        }

    def render(self, rays_o, rays_d, near, far, sdf_network, rendering_network,
               perturb_overwrite=-1,
               background_rgb=None,
               alpha_inter_ratio=0.0,
               # * related to conditional feature
               lod=None,
               conditional_volume=None,
               conditional_valid_mask_volume=None,
               # * 2d feature maps
               feature_maps=None,
               color_maps=None,
               w2cs=None,
               intrinsics=None,
               img_wh=None,
               query_c2w=None,  # -used for testing
               if_general_rendering=True,
               if_render_with_grad=True,
               # * used for blending mlp rendering network
               img_index=None,
               rays_uv=None,
               # * importance sample for second lod network
               pre_sample=False,  # no use here
               # * for clear foreground
               bg_ratio=0.0
               ):
        device = rays_o.device
        N_rays = len(rays_o)
        # sample_dist = 2.0 / self.n_samples
        sample_dist = ((far - near) / self.n_samples).mean().item()
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        bg_num = int(self.n_samples * bg_ratio)

        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(N_rays, 1)

        if bg_num > 0:
            z_vals_bg = z_vals[:, self.n_samples - bg_num:]
            z_vals = z_vals[:, :self.n_samples - bg_num]

        n_samples = self.n_samples - bg_num
        perturb = self.perturb

        # - significantly speed up training, for the second lod network
        if pre_sample:
            z_vals = self.sample_z_vals_from_maskVolume(rays_o, rays_d, near, far,
                                                        conditional_valid_mask_volume)

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        background_alpha = None
        background_sampled_color = None
        z_val_before = z_vals.clone()
        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

                sdf_outputs = sdf_network.sdf(
                    pts.reshape(-1, 3), conditional_volume, lod=lod)
                # pdb.set_trace()
                sdf = sdf_outputs['sdf_pts_scale%d' % lod].reshape(N_rays, self.n_samples - bg_num)

                n_steps = 4
                for i in range(n_steps):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_importance // n_steps,
                                                64 * 2 ** i,
                                                conditional_valid_mask_volume=conditional_valid_mask_volume,
                                                )

                    # if new_z_vals.isnan().sum() > 0:
                    #     import ipdb; ipdb.set_trace()

                    z_vals, sdf = self.cat_z_vals(
                        rays_o, rays_d, z_vals, new_z_vals, sdf, lod,
                        sdf_network, gru_fusion=False,
                        conditional_volume=conditional_volume,
                        conditional_valid_mask_volume=conditional_valid_mask_volume,
                    )

                del sdf

            n_samples = self.n_samples + self.n_importance

        # Background
        ret_outside = None

        # Render
        if bg_num > 0:
            z_vals = torch.cat([z_vals, z_vals_bg], dim=1)
        # if z_vals.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    lod,
                                    sdf_network,
                                    rendering_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    alpha_inter_ratio=alpha_inter_ratio,
                                    # * related to conditional feature
                                    conditional_volume=conditional_volume,
                                    conditional_valid_mask_volume=conditional_valid_mask_volume,
                                    # * 2d feature maps
                                    feature_maps=feature_maps,
                                    color_maps=color_maps,
                                    w2cs=w2cs,
                                    intrinsics=intrinsics,
                                    img_wh=img_wh,
                                    query_c2w=query_c2w,
                                    if_general_rendering=if_general_rendering,
                                    if_render_with_grad=if_render_with_grad,
                                    # * used for blending mlp rendering network
                                    img_index=img_index,
                                    rays_uv=rays_uv
                                    )

        color_fine = ret_fine['color']

        if self.n_outside > 0:
            color_fine_mask = torch.logical_or(ret_fine['color_mask'], ret_outside['color_mask'])
        else:
            color_fine_mask = ret_fine['color_mask']

        weights = ret_fine['weights']
        weights_sum = ret_fine['weights_sum']

        gradients = ret_fine['gradients']
        mid_z_vals = ret_fine['mid_z_vals']

        # depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        depth = ret_fine['depth']
        depth_varaince = ((mid_z_vals - depth) ** 2 * weights[:, :n_samples]).sum(dim=-1, keepdim=True)
        variance = ret_fine['variance'].reshape(N_rays, n_samples).mean(dim=-1, keepdim=True)

        # - randomly sample points from the volume, and maximize the sdf
        pts_random = torch.rand([1024, 3]).float().to(device) * 2 - 1  # normalized to (-1, 1)
        sdf_random = sdf_network.sdf(pts_random, conditional_volume, lod=lod)['sdf_pts_scale%d' % lod]

        result = {
            'depth': depth,
            'color_fine': color_fine,
            'color_fine_mask': color_fine_mask,
            'color_outside': ret_outside['color'] if ret_outside is not None else None,
            'color_outside_mask': ret_outside['color_mask'] if ret_outside is not None else None,
            'color_mlp': ret_fine['color_mlp'],
            'color_mlp_mask': ret_fine['color_mlp_mask'],
            'variance': variance.mean(),
            'cdf_fine': ret_fine['cdf'],
            'depth_variance': depth_varaince,
            'weights_sum': weights_sum,
            'weights_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'alpha_sum': ret_fine['alpha_sum'].mean(),
            'alpha_mean': ret_fine['alpha_mean'],
            'gradients': gradients,
            'weights': weights,
            'gradient_error_fine': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'sdf': ret_fine['sdf'],
            'sdf_random': sdf_random,
            'blended_color_patch': ret_fine['blended_color_patch'],
            'blended_color_patch_mask': ret_fine['blended_color_patch_mask'],
            'weights_sum_fg': ret_fine['weights_sum_fg']
        }

        return result

    @torch.no_grad()
    def sample_z_vals_from_sdfVolume(self, rays_o, rays_d, near, far, sdf_volume, mask_volume):
        # ? based on sdf to do importance sampling, seems that too biased on pre-estimation
        device = rays_o.device
        N_rays = len(rays_o)
        n_samples = self.n_samples * 2

        z_vals = torch.linspace(0.0, 1.0, n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(N_rays, 1)

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

        sdf = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), sdf_volume).reshape([N_rays, n_samples])

        new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_samples,
                                    200,
                                    conditional_valid_mask_volume=mask_volume,
                                    )
        return new_z_vals

    @torch.no_grad()
    def sample_z_vals_from_maskVolume(self, rays_o, rays_d, near, far, mask_volume):  # don't use
        device = rays_o.device
        N_rays = len(rays_o)
        n_samples = self.n_samples * 2

        z_vals = torch.linspace(0.0, 1.0, n_samples).to(device)
        z_vals = near + (far - near) * z_vals[None, :]

        if z_vals.shape[0] == 1:
            z_vals = z_vals.repeat(N_rays, 1)

        mid_z_vals = (z_vals[:, 1:] + z_vals[:, :-1]) * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]

        pts_mask = self.get_pts_mask_for_conditional_volume(pts.view(-1, 3), mask_volume).reshape(
            [N_rays, n_samples - 1])

        # empty voxel set to 0.1, non-empty voxel set to 1
        weights = torch.where(pts_mask > 0, torch.ones_like(pts_mask).to(device),
                              0.1 * torch.ones_like(pts_mask).to(device))

        # sample more pts in non-empty voxels
        z_samples = sample_pdf(z_vals, weights, self.n_samples, det=True).detach()
        return z_samples

    @torch.no_grad()
    def filter_pts_by_depthmaps(self, coords, pred_depth_maps, proj_matrices,
                                partial_vol_origin, voxel_size,
                                near, far, depth_interval, d_plane_nums):
        """
        Use the pred_depthmaps to remove redundant pts (pruned by sdf, sdf always have two sides, the back side is useless)
        :param coords: [n, 3]  int coords
        :param pred_depth_maps: [N_views, 1, h, w]
        :param proj_matrices: [N_views, 4, 4]
        :param partial_vol_origin: [3]
        :param voxel_size: 1
        :param near: 1
        :param far: 1
        :param depth_interval: 1
        :param d_plane_nums: 1
        :return:
        """
        device = pred_depth_maps.device
        n_views, _, sizeH, sizeW = pred_depth_maps.shape

        if len(partial_vol_origin.shape) == 1:
            partial_vol_origin = partial_vol_origin[None, :]
        pts = coords * voxel_size + partial_vol_origin

        rs_grid = pts.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()  # [n_views, 3, n_pts]
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).to(device)], dim=1)  # [n_views, 4, n_pts]

        # Project grid
        im_p = proj_matrices @ rs_grid  # - transform world pts to image UV space   # [n_views, 4, n_pts]
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (sizeW - 1) - 1, 2 * im_y / (sizeH - 1) - 1], dim=-1)

        im_grid = im_grid.view(n_views, 1, -1, 2)
        sampled_depths = torch.nn.functional.grid_sample(pred_depth_maps, im_grid, mode='bilinear',
                                                         padding_mode='zeros',
                                                         align_corners=True)[:, 0, 0, :]  # [n_views, n_pts]
        sampled_depths_valid = (sampled_depths > 0.5 * near).float()
        valid_d_min = (sampled_depths - d_plane_nums * depth_interval).clamp(near.item(),
                                                                             far.item()) * sampled_depths_valid
        valid_d_max = (sampled_depths + d_plane_nums * depth_interval).clamp(near.item(),
                                                                             far.item()) * sampled_depths_valid

        mask = im_grid.abs() <= 1
        mask = mask[:, 0]  # [n_views, n_pts, 2]
        mask = (mask.sum(dim=-1) == 2) & (im_z > valid_d_min) & (im_z < valid_d_max)

        mask = mask.view(n_views, -1)
        mask = mask.permute(1, 0).contiguous()  # [num_pts, nviews]

        mask_final = torch.sum(mask.float(), dim=1, keepdim=False) > 0

        return mask_final

    @torch.no_grad()
    def get_valid_sparse_coords_by_sdf_depthfilter(self, sdf_volume, coords_volume, mask_volume, feature_volume,
                                                   pred_depth_maps, proj_matrices,
                                                   partial_vol_origin, voxel_size,
                                                   near, far, depth_interval, d_plane_nums,
                                                   threshold=0.02, maximum_pts=110000):
        """
        assume batch size == 1, from the first lod to get sparse voxels
        :param sdf_volume: [1, X, Y, Z]
        :param coords_volume: [3, X, Y, Z]
        :param mask_volume: [1, X, Y, Z]
        :param feature_volume: [C, X, Y, Z]
        :param threshold:
        :return:
        """
        device = coords_volume.device
        _, dX, dY, dZ = coords_volume.shape

        def prune(sdf_pts, coords_pts, mask_volume, threshold):
            occupancy_mask = (torch.abs(sdf_pts) < threshold).squeeze(1)  # [num_pts]
            valid_coords = coords_pts[occupancy_mask]

            # - filter backside surface by depth maps
            mask_filtered = self.filter_pts_by_depthmaps(valid_coords, pred_depth_maps, proj_matrices,
                                                         partial_vol_origin, voxel_size,
                                                         near, far, depth_interval, d_plane_nums)
            valid_coords = valid_coords[mask_filtered]

            # - dilate
            occupancy_mask = sparse_to_dense_channel(valid_coords, 1, [dX, dY, dZ], 1, 0, device)  # [dX, dY, dZ, 1]

            # - dilate
            occupancy_mask = occupancy_mask.float()
            occupancy_mask = occupancy_mask.view(1, 1, dX, dY, dZ)
            occupancy_mask = F.avg_pool3d(occupancy_mask, kernel_size=7, stride=1, padding=3)
            occupancy_mask = occupancy_mask.view(-1, 1) > 0

            final_mask = torch.logical_and(mask_volume, occupancy_mask)[:, 0]  # [num_pts]

            return final_mask, torch.sum(final_mask.float())

        C, dX, dY, dZ = feature_volume.shape
        sdf_volume = sdf_volume.permute(1, 2, 3, 0).contiguous().view(-1, 1)
        coords_volume = coords_volume.permute(1, 2, 3, 0).contiguous().view(-1, 3)
        mask_volume = mask_volume.permute(1, 2, 3, 0).contiguous().view(-1, 1)
        feature_volume = feature_volume.permute(1, 2, 3, 0).contiguous().view(-1, C)

        # - for check
        # sdf_volume = torch.rand_like(sdf_volume).float().to(sdf_volume.device) * 0.02

        final_mask, valid_num = prune(sdf_volume, coords_volume, mask_volume, threshold)

        while (valid_num > maximum_pts) and (threshold > 0.003):
            threshold = threshold - 0.002
            final_mask, valid_num = prune(sdf_volume, coords_volume, mask_volume, threshold)

        valid_coords = coords_volume[final_mask]  # [N, 3]
        valid_feature = feature_volume[final_mask]  # [N, C]

        valid_coords = torch.cat([torch.ones([valid_coords.shape[0], 1]).to(valid_coords.device) * 0,
                                  valid_coords], dim=1)  # [N, 4], append batch idx

        # ! if the valid_num is still larger than maximum_pts, sample part of pts
        if valid_num > maximum_pts:
            valid_num = valid_num.long()
            occupancy = torch.ones([valid_num]).to(device) > 0
            choice = np.random.choice(valid_num.cpu().numpy(), valid_num.cpu().numpy() - maximum_pts,
                                      replace=False)
            ind = torch.nonzero(occupancy).to(device)
            occupancy[ind[choice]] = False
            valid_coords = valid_coords[occupancy]
            valid_feature = valid_feature[occupancy]

            print(threshold, "randomly sample to save memory")

        return valid_coords, valid_feature

    @torch.no_grad()
    def get_valid_sparse_coords_by_sdf(self, sdf_volume, coords_volume, mask_volume, feature_volume, threshold=0.02,
                                       maximum_pts=110000):
        """
        assume batch size == 1, from the first lod to get sparse voxels
        :param sdf_volume: [num_pts, 1]
        :param coords_volume: [3, X, Y, Z]
        :param mask_volume: [1, X, Y, Z]
        :param feature_volume: [C, X, Y, Z]
        :param threshold:
        :return:
        """

        def prune(sdf_volume, mask_volume, threshold):
            occupancy_mask = torch.abs(sdf_volume) < threshold  # [num_pts, 1]

            # - dilate
            occupancy_mask = occupancy_mask.float()
            occupancy_mask = occupancy_mask.view(1, 1, dX, dY, dZ)
            occupancy_mask = F.avg_pool3d(occupancy_mask, kernel_size=7, stride=1, padding=3)
            occupancy_mask = occupancy_mask.view(-1, 1) > 0

            final_mask = torch.logical_and(mask_volume, occupancy_mask)[:, 0]  # [num_pts]

            return final_mask, torch.sum(final_mask.float())

        C, dX, dY, dZ = feature_volume.shape
        coords_volume = coords_volume.permute(1, 2, 3, 0).contiguous().view(-1, 3)
        mask_volume = mask_volume.permute(1, 2, 3, 0).contiguous().view(-1, 1)
        feature_volume = feature_volume.permute(1, 2, 3, 0).contiguous().view(-1, C)

        final_mask, valid_num = prune(sdf_volume, mask_volume, threshold)

        while (valid_num > maximum_pts) and (threshold > 0.003):
            threshold = threshold - 0.002
            final_mask, valid_num = prune(sdf_volume, mask_volume, threshold)

        valid_coords = coords_volume[final_mask]  # [N, 3]
        valid_feature = feature_volume[final_mask]  # [N, C]

        valid_coords = torch.cat([torch.ones([valid_coords.shape[0], 1]).to(valid_coords.device) * 0,
                                  valid_coords], dim=1)  # [N, 4], append batch idx

        # ! if the valid_num is still larger than maximum_pts, sample part of pts
        if valid_num > maximum_pts:
            device = sdf_volume.device
            valid_num = valid_num.long()
            occupancy = torch.ones([valid_num]).to(device) > 0
            choice = np.random.choice(valid_num.cpu().numpy(), valid_num.cpu().numpy() - maximum_pts,
                                      replace=False)
            ind = torch.nonzero(occupancy).to(device)
            occupancy[ind[choice]] = False
            valid_coords = valid_coords[occupancy]
            valid_feature = valid_feature[occupancy]

            print(threshold, "randomly sample to save memory")

        return valid_coords, valid_feature

    @torch.no_grad()
    def extract_fields(self, bound_min, bound_max, resolution, query_func, device,
                       # * related to conditional feature
                       **kwargs
                       ):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).to(device).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).to(device).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).to(device).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)

                        # ! attention, the query function is different for extract geometry and fields
                        output = query_func(pts, **kwargs)
                        sdf = output['sdf_pts_scale%d' % kwargs['lod']].reshape(len(xs), len(ys),
                                                                                len(zs)).detach().cpu().numpy()

                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = -1 * sdf
        return u

    @torch.no_grad()
    def extract_geometry(self, sdf_network, bound_min, bound_max, resolution, threshold, device, occupancy_mask=None,
                         # * 3d feature volume
                         **kwargs
                         ):
        # logging.info('threshold: {}'.format(threshold))

        u = self.extract_fields(bound_min, bound_max, resolution,
                                lambda pts, **kwargs: sdf_network.sdf(pts, **kwargs),
                                # - sdf need to be multiplied by -1
                                device,
                                # * 3d feature volume
                                **kwargs
                                )
        if occupancy_mask is not None:
            dX, dY, dZ = occupancy_mask.shape
            empty_mask = 1 - occupancy_mask
            empty_mask = empty_mask.view(1, 1, dX, dY, dZ)
            # - dilation
            # empty_mask = F.avg_pool3d(empty_mask, kernel_size=7, stride=1, padding=3)
            empty_mask = F.interpolate(empty_mask, [resolution, resolution, resolution], mode='nearest')
            empty_mask = empty_mask.view(resolution, resolution, resolution).cpu().numpy() > 0
            u[empty_mask] = -100
            del empty_mask

        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles, u

    @torch.no_grad()
    def extract_depth_maps(self, sdf_network, con_volume, intrinsics, c2ws, H, W, near, far):
        """
        extract depth maps from the density volume
        :param con_volume: [1, 1+C, dX, dY, dZ]  can by con_volume or sdf_volume
        :param c2ws: [B, 4, 4]
        :param H:
        :param W:
        :param near:
        :param far:
        :return:
        """
        device = con_volume.device
        batch_size = intrinsics.shape[0]

        with torch.no_grad():
            ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H),
                                    torch.linspace(0, W - 1, W), indexing="ij")  # pytorch's meshgrid has indexing='ij'
            p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

            intrinsics_inv = torch.inverse(intrinsics)

            p = p.view(-1, 3).float().to(device)  # N_rays, 3
            p = torch.matmul(intrinsics_inv[:, None, :3, :3], p[:, :, None]).squeeze()  # Batch, N_rays, 3
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # Batch, N_rays, 3
            rays_v = torch.matmul(c2ws[:, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # Batch, N_rays, 3
            rays_o = c2ws[:, None, :3, 3].expand(rays_v.shape)  # Batch, N_rays, 3
            rays_d = rays_v

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        ################## - sphere tracer to extract depth maps               ######################
        depth_masks_sphere, depth_maps_sphere = self.ray_tracer.extract_depth_maps(
            rays_o, rays_d,
            near[None, :].repeat(rays_o.shape[0], 1),
            far[None, :].repeat(rays_o.shape[0], 1),
            sdf_network, con_volume
        )

        depth_maps = depth_maps_sphere.view(batch_size, 1, H, W)
        depth_masks = depth_masks_sphere.view(batch_size, 1, H, W)

        depth_maps = torch.where(depth_masks, depth_maps,
                                 torch.zeros_like(depth_masks.float()).to(device))  # fill invalid pixels by 0

        return depth_maps, depth_masks
