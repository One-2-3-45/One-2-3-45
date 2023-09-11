"""
decouple the trainer with the renderer
"""
import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import trimesh

from utils.misc_utils import visualize_depth_numpy

from utils.training_utils import numpy2tensor

from loss.depth_loss import DepthLoss, DepthSmoothLoss

from models.sparse_neus_renderer import SparseNeuSRenderer


class GenericTrainer(nn.Module):
    def __init__(self,
                 rendering_network_outside,
                 pyramid_feature_network_lod0,
                 pyramid_feature_network_lod1,
                 sdf_network_lod0,
                 sdf_network_lod1,
                 variance_network_lod0,
                 variance_network_lod1,
                 rendering_network_lod0,
                 rendering_network_lod1,
                 n_samples_lod0,
                 n_importance_lod0,
                 n_samples_lod1,
                 n_importance_lod1,
                 n_outside,
                 perturb,
                 alpha_type='div',
                 conf=None,
                 timestamp="",
                 mode='train',
                 base_exp_dir=None,
                 ):
        super(GenericTrainer, self).__init__()

        self.conf = conf
        self.timestamp = timestamp

        
        self.base_exp_dir = base_exp_dir 
        

        self.anneal_start = self.conf.get_float('train.anneal_start', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.anneal_start_lod1 = self.conf.get_float('train.anneal_start_lod1', default=0.0)
        self.anneal_end_lod1 = self.conf.get_float('train.anneal_end_lod1', default=0.0)

        # network setups
        self.rendering_network_outside = rendering_network_outside
        self.pyramid_feature_network_geometry_lod0 = pyramid_feature_network_lod0  # 2D pyramid feature network for geometry
        self.pyramid_feature_network_geometry_lod1 = pyramid_feature_network_lod1  # use differnet networks for the two lods

        # when num_lods==2, may consume too much memeory
        self.sdf_network_lod0 = sdf_network_lod0
        self.sdf_network_lod1 = sdf_network_lod1

        # - warpped by ModuleList to support DataParallel
        self.variance_network_lod0 = variance_network_lod0
        self.variance_network_lod1 = variance_network_lod1

        self.rendering_network_lod0 = rendering_network_lod0
        self.rendering_network_lod1 = rendering_network_lod1

        self.n_samples_lod0 = n_samples_lod0
        self.n_importance_lod0 = n_importance_lod0
        self.n_samples_lod1 = n_samples_lod1
        self.n_importance_lod1 = n_importance_lod1
        self.n_outside = n_outside
        self.num_lods = conf.get_int('model.num_lods')  # the number of octree lods
        self.perturb = perturb
        self.alpha_type = alpha_type

        # - the two renderers
        self.sdf_renderer_lod0 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod0,
            self.variance_network_lod0,
            self.rendering_network_lod0,
            self.n_samples_lod0,
            self.n_importance_lod0,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.sdf_renderer_lod1 = SparseNeuSRenderer(
            self.rendering_network_outside,
            self.sdf_network_lod1,
            self.variance_network_lod1,
            self.rendering_network_lod1,
            self.n_samples_lod1,
            self.n_importance_lod1,
            self.n_outside,
            self.perturb,
            alpha_type='div',
            conf=self.conf)

        self.if_fix_lod0_networks = self.conf.get_bool('train.if_fix_lod0_networks')

        # sdf network weights
        self.sdf_igr_weight = self.conf.get_float('train.sdf_igr_weight')
        self.sdf_sparse_weight = self.conf.get_float('train.sdf_sparse_weight', default=0)
        self.sdf_decay_param = self.conf.get_float('train.sdf_decay_param', default=100)
        self.fg_bg_weight = self.conf.get_float('train.fg_bg_weight', default=0.00)
        self.bg_ratio = self.conf.get_float('train.bg_ratio', default=0.0)

        self.depth_loss_weight = self.conf.get_float('train.depth_loss_weight', default=1.00)

        print("depth_loss_weight: ", self.depth_loss_weight)
        self.depth_criterion = DepthLoss()

        # - DataParallel mode, cannot modify attributes in forward()
        # self.iter_step = 0
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')

        # - True for finetuning; False for general training
        self.if_fitted_rendering = self.conf.get_bool('train.if_fitted_rendering', default=False)

        self.prune_depth_filter = self.conf.get_bool('model.prune_depth_filter', default=False)

    def get_trainable_params(self):
        # set trainable params

        self.params_to_train = []

        if not self.if_fix_lod0_networks:
            #  load pretrained featurenet
            self.params_to_train += list(self.pyramid_feature_network_geometry_lod0.parameters())
            self.params_to_train += list(self.sdf_network_lod0.parameters())
            self.params_to_train += list(self.variance_network_lod0.parameters())

            if self.rendering_network_lod0 is not None:
                self.params_to_train += list(self.rendering_network_lod0.parameters())

        if self.sdf_network_lod1 is not None:
            #  load pretrained featurenet
            self.params_to_train += list(self.pyramid_feature_network_geometry_lod1.parameters())

            self.params_to_train += list(self.sdf_network_lod1.parameters())
            self.params_to_train += list(self.variance_network_lod1.parameters())
            if self.rendering_network_lod1 is not None:
                self.params_to_train += list(self.rendering_network_lod1.parameters())

        return self.params_to_train

    def train_step(self, sample,
                   perturb_overwrite=-1,
                   background_rgb=None,
                   alpha_inter_ratio_lod0=0.0,
                   alpha_inter_ratio_lod1=0.0,
                   iter_step=0,
                   ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['near_fars'][0, 0, :1], sample['near_fars'][0, 0, 1:]

        # the full-size ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        c2ws = sample['c2ws'][0]
        proj_matrices = sample['affine_mats']
        scale_mat = sample['scale_mat']
        trans_mat = sample['trans_mat']

        # ***********************     Lod==0     ***********************
        if not self.if_fix_lod0_networks:
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs)

            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, 1:, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices[:,1:],
                # proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )

        else:
            with torch.no_grad():
                geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)
                conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                    feature_maps=geometry_feature_maps[None, 1:, :, :, :],
                    partial_vol_origin=partial_vol_origin,
                    proj_mats=proj_matrices[:,1:],
                    # proj_mats=proj_matrices,
                    sizeH=sizeH,
                    sizeW=sizeW,
                    lod=0,
                )

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']

        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']

        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        # * extract depth maps for all the images
        depth_maps_lod0, depth_masks_lod0 = None, None
        if self.num_lods > 1:
            sdf_volume_lod0 = self.sdf_network_lod0.get_sdf_volume(
                con_volume_lod0, con_valid_mask_volume_lod0,
                coords_lod0, partial_vol_origin)  # [1, 1, dX, dY, dZ]

        if self.prune_depth_filter:
            depth_maps_lod0_l4x, depth_masks_lod0_l4x = self.sdf_renderer_lod0.extract_depth_maps(
                self.sdf_network_lod0, sdf_volume_lod0, intrinsics_l_4x, c2ws,
                sizeH // 4, sizeW // 4, near * 1.5, far)
            depth_maps_lod0 = F.interpolate(depth_maps_lod0_l4x, size=(sizeH, sizeW), mode='bilinear',
                                            align_corners=True)

        # *************** losses
        loss_lod0, losses_lod0, depth_statis_lod0 = None, None, None

        if not self.if_fix_lod0_networks:

            render_out = self.sdf_renderer_lod0.render(
                rays_o, rays_d, near, far,
                self.sdf_network_lod0,
                self.rendering_network_lod0,
                background_rgb=background_rgb,
                alpha_inter_ratio=alpha_inter_ratio_lod0,
                # * related to conditional feature
                lod=0,
                conditional_volume=con_volume_lod0,
                conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                # * 2d feature maps
                feature_maps=geometry_feature_maps,
                color_maps=imgs,
                w2cs=w2cs,
                intrinsics=intrinsics,
                img_wh=[sizeW, sizeH],
                if_general_rendering=True,
                if_render_with_grad=True,
            )

            loss_lod0, losses_lod0, depth_statis_lod0 = self.cal_losses_sdf(render_out, sample_rays,
                                                                                     iter_step, lod=0)

        # ***********************     Lod==1     ***********************

        loss_lod1, losses_lod1, depth_statis_lod1 = None, None, None

        if self.num_lods > 1:
            geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)
            # geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)
            if self.prune_depth_filter:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf_depthfilter(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0],
                    depth_maps_lod0, proj_matrices[0],
                    partial_vol_origin, self.sdf_network_lod0.voxel_size,
                    near, far, self.sdf_network_lod0.voxel_size, 12)
            else:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0])

            pre_coords[:, 1:] = pre_coords[:, 1:] * 2

            # ? It seems that training gru_fusion, this part should be trainable too
            conditional_features_lod1 = self.sdf_network_lod1.get_conditional_volume(
                feature_maps=geometry_feature_maps_lod1[None, 1:, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices[:,1:],
                # proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                pre_coords=pre_coords,
                pre_feats=pre_feats,
            )

            con_volume_lod1 = conditional_features_lod1['dense_volume_scale1']
            con_valid_mask_volume_lod1 = conditional_features_lod1['valid_mask_volume_scale1']

            # if not self.if_gru_fusion_lod1:
            render_out_lod1 = self.sdf_renderer_lod1.render(
                rays_o, rays_d, near, far,
                self.sdf_network_lod1,
                self.rendering_network_lod1,
                background_rgb=background_rgb,
                alpha_inter_ratio=alpha_inter_ratio_lod1,
                # * related to conditional feature
                lod=1,
                conditional_volume=con_volume_lod1,
                conditional_valid_mask_volume=con_valid_mask_volume_lod1,
                # * 2d feature maps
                feature_maps=geometry_feature_maps_lod1,
                color_maps=imgs,
                w2cs=w2cs,
                intrinsics=intrinsics,
                img_wh=[sizeW, sizeH],
                bg_ratio=self.bg_ratio,
            )
            loss_lod1, losses_lod1, depth_statis_lod1 = self.cal_losses_sdf(render_out_lod1, sample_rays,
                                                                                     iter_step, lod=1)


        # # - extract mesh
        if iter_step % self.val_mesh_freq == 0:
            torch.cuda.empty_cache()
            self.validate_mesh(self.sdf_network_lod0,
                               self.sdf_renderer_lod0.extract_geometry,
                               conditional_volume=con_volume_lod0, lod=0,
                               threshold=0,
                               # occupancy_mask=con_valid_mask_volume_lod0[0, 0],
                               mode='train_bg', meta=meta,
                               iter_step=iter_step, scale_mat=scale_mat,
                               trans_mat=trans_mat)
            torch.cuda.empty_cache()

            if self.num_lods > 1:
                self.validate_mesh(self.sdf_network_lod1,
                                   self.sdf_renderer_lod1.extract_geometry,
                                   conditional_volume=con_volume_lod1, lod=1,
                                   # occupancy_mask=con_valid_mask_volume_lod1[0, 0].detach(),
                                   mode='train_bg', meta=meta,
                                   iter_step=iter_step, scale_mat=scale_mat,
                                   trans_mat=trans_mat)

        losses = {
            # - lod 0
            'loss_lod0': loss_lod0,
            'losses_lod0': losses_lod0,
            'depth_statis_lod0': depth_statis_lod0,

            # - lod 1
            'loss_lod1': loss_lod1,
            'losses_lod1': losses_lod1,
            'depth_statis_lod1': depth_statis_lod1,

        }

        return losses

    def val_step(self, sample,
                 perturb_overwrite=-1,
                 background_rgb=None,
                 alpha_inter_ratio_lod0=0.0,
                 alpha_inter_ratio_lod1=0.0,
                 iter_step=0,
                 chunk_size=512,
                 save_vis=False,
                 ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        H, W = sizeH, sizeW

        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:]

        # the ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        rays_ndc_uv = sample_rays['rays_ndc_uv'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        c2ws = sample['c2ws'][0]
        proj_matrices = sample['affine_mats']

        # render_img_idx = sample['render_img_idx'][0]
        # true_img = sample['images'][0][render_img_idx]

        # - the image to render
        scale_mat = sample['scale_mat']  # [1,4,4]  used to convert mesh into true scale
        trans_mat = sample['trans_mat']
        query_c2w = sample['query_c2w']  # [1,4,4]
        query_w2c = sample['query_w2c']  # [1,4,4]
        true_img = sample['query_image'][0]
        true_img = np.uint8(true_img.permute(1, 2, 0).cpu().numpy() * 255)

        depth_min, depth_max = near.cpu().numpy(), far.cpu().numpy()

        scale_factor = sample['scale_factor'][0].cpu().numpy()
        true_depth = sample['query_depth'] if 'query_depth' in sample.keys() else None
        if true_depth is not None:
            true_depth = true_depth[0].cpu().numpy()
            true_depth_colored = visualize_depth_numpy(true_depth, [depth_min, depth_max])[0]
        else:
            true_depth_colored = None

        rays_o = rays_o.reshape(-1, 3).split(chunk_size)
        rays_d = rays_d.reshape(-1, 3).split(chunk_size)

        # - obtain conditional features
        with torch.no_grad():
            # - obtain conditional features
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)

            # - lod 0
            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, :, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        if self.num_lods > 1:
            sdf_volume_lod0 = self.sdf_network_lod0.get_sdf_volume(
                con_volume_lod0, con_valid_mask_volume_lod0,
                coords_lod0, partial_vol_origin)  # [1, 1, dX, dY, dZ]

        depth_maps_lod0, depth_masks_lod0 = None, None
        if self.prune_depth_filter:
            depth_maps_lod0_l4x, depth_masks_lod0_l4x = self.sdf_renderer_lod0.extract_depth_maps(
                self.sdf_network_lod0, sdf_volume_lod0,
                intrinsics_l_4x, c2ws,
                sizeH // 4, sizeW // 4, near * 1.5, far)  # - near*1.5 is a experienced number
            depth_maps_lod0 = F.interpolate(depth_maps_lod0_l4x, size=(sizeH, sizeW), mode='bilinear',
                                            align_corners=True)
            depth_masks_lod0 = F.interpolate(depth_masks_lod0_l4x.float(), size=(sizeH, sizeW), mode='nearest')

            #### visualize the depth_maps_lod0 for checking
            colored_depth_maps_lod0 = []
            for i in range(depth_maps_lod0.shape[0]):
                colored_depth_maps_lod0.append(
                    visualize_depth_numpy(depth_maps_lod0[i, 0].cpu().numpy(), [depth_min, depth_max])[0])

            colored_depth_maps_lod0 = np.concatenate(colored_depth_maps_lod0, axis=0).astype(np.uint8)
            os.makedirs(os.path.join(self.base_exp_dir, 'depth_maps_lod0'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'depth_maps_lod0',
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       colored_depth_maps_lod0[:, :, ::-1])

        if self.num_lods > 1:
            geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)

            if self.prune_depth_filter:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf_depthfilter(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0],
                    depth_maps_lod0, proj_matrices[0],
                    partial_vol_origin, self.sdf_network_lod0.voxel_size,
                    near, far, self.sdf_network_lod0.voxel_size, 12)
            else:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0])

            pre_coords[:, 1:] = pre_coords[:, 1:] * 2

            with torch.no_grad():
                conditional_features_lod1 = self.sdf_network_lod1.get_conditional_volume(
                    feature_maps=geometry_feature_maps_lod1[None, :, :, :, :],
                    partial_vol_origin=partial_vol_origin,
                    proj_mats=proj_matrices,
                    sizeH=sizeH,
                    sizeW=sizeW,
                    pre_coords=pre_coords,
                    pre_feats=pre_feats,
                )

            con_volume_lod1 = conditional_features_lod1['dense_volume_scale1']
            con_valid_mask_volume_lod1 = conditional_features_lod1['valid_mask_volume_scale1']

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        out_rgb_fine_lod1 = []
        out_normal_fine_lod1 = []
        out_depth_fine_lod1 = []

        # out_depth_fine_explicit = []
        if save_vis:
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):

                # ****** lod 0 ****
                render_out = self.sdf_renderer_lod0.render(
                    rays_o_batch, rays_d_batch, near, far,
                    self.sdf_network_lod0,
                    self.rendering_network_lod0,
                    background_rgb=background_rgb,
                    alpha_inter_ratio=alpha_inter_ratio_lod0,
                    # * related to conditional feature
                    lod=0,
                    conditional_volume=con_volume_lod0,
                    conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                    # * 2d feature maps
                    feature_maps=geometry_feature_maps,
                    color_maps=imgs,
                    w2cs=w2cs,
                    intrinsics=intrinsics,
                    img_wh=[sizeW, sizeH],
                    query_c2w=query_c2w,
                    if_render_with_grad=False,
                )

                feasible = lambda key: ((key in render_out) and (render_out[key] is not None))

                if feasible('depth'):
                    out_depth_fine.append(render_out['depth'].detach().cpu().numpy())

                # if render_out['color_coarse'] is not None:
                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                if feasible('gradients') and feasible('weights'):
                    if render_out['inside_sphere'] is not None:
                        out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                          :self.n_samples_lod0 + self.n_importance_lod0,
                                                                          None] * render_out['inside_sphere'][
                                                    ..., None]).sum(dim=1).detach().cpu().numpy())
                    else:
                        out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                          :self.n_samples_lod0 + self.n_importance_lod0,
                                                                          None]).sum(dim=1).detach().cpu().numpy())
                del render_out

            # ****************** lod 1 **************************
            if self.num_lods > 1:
                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    render_out_lod1 = self.sdf_renderer_lod1.render(
                        rays_o_batch, rays_d_batch, near, far,
                        self.sdf_network_lod1,
                        self.rendering_network_lod1,
                        background_rgb=background_rgb,
                        alpha_inter_ratio=alpha_inter_ratio_lod1,
                        # * related to conditional feature
                        lod=1,
                        conditional_volume=con_volume_lod1,
                        conditional_valid_mask_volume=con_valid_mask_volume_lod1,
                        # * 2d feature maps
                        feature_maps=geometry_feature_maps_lod1,
                        color_maps=imgs,
                        w2cs=w2cs,
                        intrinsics=intrinsics,
                        img_wh=[sizeW, sizeH],
                        query_c2w=query_c2w,
                        if_render_with_grad=False,
                    )

                    feasible = lambda key: ((key in render_out_lod1) and (render_out_lod1[key] is not None))

                    if feasible('depth'):
                        out_depth_fine_lod1.append(render_out_lod1['depth'].detach().cpu().numpy())

                    # if render_out['color_coarse'] is not None:
                    if feasible('color_fine'):
                        out_rgb_fine_lod1.append(render_out_lod1['color_fine'].detach().cpu().numpy())
                    if feasible('gradients') and feasible('weights'):
                        if render_out_lod1['inside_sphere'] is not None:
                            out_normal_fine_lod1.append((render_out_lod1['gradients'] * render_out_lod1['weights'][:,
                                                                                        :self.n_samples_lod1 + self.n_importance_lod1,
                                                                                        None] *
                                                         render_out_lod1['inside_sphere'][
                                                             ..., None]).sum(dim=1).detach().cpu().numpy())
                        else:
                            out_normal_fine_lod1.append((render_out_lod1['gradients'] * render_out_lod1['weights'][:,
                                                                                        :self.n_samples_lod1 + self.n_importance_lod1,
                                                                                        None]).sum(
                                dim=1).detach().cpu().numpy())
                    del render_out_lod1

            # - save visualization of lod 0

            self.save_visualization(true_img, true_depth_colored, out_depth_fine, out_normal_fine,
                                    query_w2c[0], out_rgb_fine, H, W,
                                    depth_min, depth_max, iter_step, meta, "val_lod0", true_depth=true_depth, scale_factor=scale_factor)

            if self.num_lods > 1:
                self.save_visualization(true_img, true_depth_colored, out_depth_fine_lod1, out_normal_fine_lod1,
                                        query_w2c[0], out_rgb_fine_lod1, H, W,
                                        depth_min, depth_max, iter_step, meta, "val_lod1", true_depth=true_depth, scale_factor=scale_factor)

        # - extract mesh
        if (iter_step % self.val_mesh_freq == 0):
            torch.cuda.empty_cache()
            self.validate_mesh(self.sdf_network_lod0,
                               self.sdf_renderer_lod0.extract_geometry,
                               conditional_volume=con_volume_lod0, lod=0,
                               threshold=0,
                               # occupancy_mask=con_valid_mask_volume_lod0[0, 0],
                               mode='val_bg', meta=meta,
                               iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat)
            torch.cuda.empty_cache()

            if self.num_lods > 1:
                self.validate_mesh(self.sdf_network_lod1,
                                   self.sdf_renderer_lod1.extract_geometry,
                                   conditional_volume=con_volume_lod1, lod=1,
                                   # occupancy_mask=con_valid_mask_volume_lod1[0, 0].detach(),
                                   mode='val_bg', meta=meta,
                                   iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat)

            torch.cuda.empty_cache()

    @torch.no_grad()
    def get_metrics_step(self, sample,
                   perturb_overwrite=-1,
                   background_rgb=None,
                   alpha_inter_ratio_lod0=0.0,
                   alpha_inter_ratio_lod1=0.0,
                   iter_step=0,
                   ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['near_fars'][0, 0, :1], sample['near_fars'][0, 0, 1:]

        # the full-size ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        c2ws = sample['c2ws'][0]
        proj_matrices = sample['affine_mats']
        scale_mat = sample['scale_mat']
        trans_mat = sample['trans_mat']

        # ***********************     Lod==0     ***********************
        if not self.if_fix_lod0_networks:
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs)

            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, 1:, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices[:,1:],
                # proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )

        else:
            with torch.no_grad():
                geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)
                # geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)
                conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                    feature_maps=geometry_feature_maps[None, 1:, :, :, :],
                    partial_vol_origin=partial_vol_origin,
                    proj_mats=proj_matrices[:,1:],
                    # proj_mats=proj_matrices,
                    sizeH=sizeH,
                    sizeW=sizeW,
                    lod=0,
                )
        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']

        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        # * extract depth maps for all the images
        depth_maps_lod0, depth_masks_lod0 = None, None
        if self.num_lods > 1:
            sdf_volume_lod0 = self.sdf_network_lod0.get_sdf_volume(
                con_volume_lod0, con_valid_mask_volume_lod0,
                coords_lod0, partial_vol_origin)  # [1, 1, dX, dY, dZ]

        if self.prune_depth_filter:
            depth_maps_lod0_l4x, depth_masks_lod0_l4x = self.sdf_renderer_lod0.extract_depth_maps(
                self.sdf_network_lod0, sdf_volume_lod0, intrinsics_l_4x, c2ws,
                sizeH // 4, sizeW // 4, near * 1.5, far)
            depth_maps_lod0 = F.interpolate(depth_maps_lod0_l4x, size=(sizeH, sizeW), mode='bilinear',
                                            align_corners=True)
            depth_masks_lod0 = F.interpolate(depth_masks_lod0_l4x.float(), size=(sizeH, sizeW), mode='nearest')

        # *************** losses
        loss_lod0, losses_lod0, depth_statis_lod0 = None, None, None

        if not self.if_fix_lod0_networks:

            render_out = self.sdf_renderer_lod0.render(
                rays_o, rays_d, near, far,
                self.sdf_network_lod0,
                self.rendering_network_lod0,
                background_rgb=background_rgb,
                alpha_inter_ratio=alpha_inter_ratio_lod0,
                # * related to conditional feature
                lod=0,
                conditional_volume=con_volume_lod0,
                conditional_valid_mask_volume=con_valid_mask_volume_lod0,
                # * 2d feature maps
                feature_maps=geometry_feature_maps,
                color_maps=imgs,
                w2cs=w2cs,
                intrinsics=intrinsics,
                img_wh=[sizeW, sizeH],
                if_general_rendering=True,
                if_render_with_grad=True,
            )

            loss_lod0, losses_lod0, depth_statis_lod0 = self.cal_losses_sdf(render_out, sample_rays,
                                                                                     iter_step, lod=0)

        # ***********************     Lod==1     ***********************

        loss_lod1, losses_lod1, depth_statis_lod1 = None, None, None

        if self.num_lods > 1:
            geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)
            # geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)
            if self.prune_depth_filter:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf_depthfilter(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0],
                    depth_maps_lod0, proj_matrices[0],
                    partial_vol_origin, self.sdf_network_lod0.voxel_size,
                    near, far, self.sdf_network_lod0.voxel_size, 12)
            else:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0])

            pre_coords[:, 1:] = pre_coords[:, 1:] * 2

            # ? It seems that training gru_fusion, this part should be trainable too
            conditional_features_lod1 = self.sdf_network_lod1.get_conditional_volume(
                feature_maps=geometry_feature_maps_lod1[None, 1:, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices[:,1:],
                # proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                pre_coords=pre_coords,
                pre_feats=pre_feats,
            )

            con_volume_lod1 = conditional_features_lod1['dense_volume_scale1']
            con_valid_mask_volume_lod1 = conditional_features_lod1['valid_mask_volume_scale1']

            # if not self.if_gru_fusion_lod1:
            render_out_lod1 = self.sdf_renderer_lod1.render(
                rays_o, rays_d, near, far,
                self.sdf_network_lod1,
                self.rendering_network_lod1,
                background_rgb=background_rgb,
                alpha_inter_ratio=alpha_inter_ratio_lod1,
                # * related to conditional feature
                lod=1,
                conditional_volume=con_volume_lod1,
                conditional_valid_mask_volume=con_valid_mask_volume_lod1,
                # * 2d feature maps
                feature_maps=geometry_feature_maps_lod1,
                color_maps=imgs,
                w2cs=w2cs,
                intrinsics=intrinsics,
                img_wh=[sizeW, sizeH],
                bg_ratio=self.bg_ratio,
            )
            loss_lod1, losses_lod1, depth_statis_lod1 = self.cal_losses_sdf(render_out_lod1, sample_rays,
                                                                                     iter_step, lod=1)


        # # - extract mesh
        if iter_step % self.val_mesh_freq == 0:
            torch.cuda.empty_cache()
            self.validate_mesh(self.sdf_network_lod0,
                               self.sdf_renderer_lod0.extract_geometry,
                               conditional_volume=con_volume_lod0, lod=0,
                               threshold=0,
                               # occupancy_mask=con_valid_mask_volume_lod0[0, 0],
                               mode='train_bg', meta=meta,
                               iter_step=iter_step, scale_mat=scale_mat,
                               trans_mat=trans_mat)
            torch.cuda.empty_cache()

            if self.num_lods > 1:
                self.validate_mesh(self.sdf_network_lod1,
                                   self.sdf_renderer_lod1.extract_geometry,
                                   conditional_volume=con_volume_lod1, lod=1,
                                   # occupancy_mask=con_valid_mask_volume_lod1[0, 0].detach(),
                                   mode='train_bg', meta=meta,
                                   iter_step=iter_step, scale_mat=scale_mat,
                                   trans_mat=trans_mat)

        losses = {
            # - lod 0
            'loss_lod0': loss_lod0,
            'losses_lod0': losses_lod0,
            'depth_statis_lod0': depth_statis_lod0,

            # - lod 1
            'loss_lod1': loss_lod1,
            'losses_lod1': losses_lod1,
            'depth_statis_lod1': depth_statis_lod1,

        }

        return losses


    def export_mesh_step(self, sample,
                        iter_step=0,
                        chunk_size=512,
                        resolution=360,
                        save_vis=False,
                        ):
        # * only support batch_size==1
        # ! attention: the list of string cannot be splited in DataParallel
        batch_idx = sample['batch_idx'][0]
        meta = sample['meta'][batch_idx]  # the scan lighting ref_view info

        sizeW = sample['img_wh'][0][0]
        sizeH = sample['img_wh'][0][1]
        H, W = sizeH, sizeW

        partial_vol_origin = sample['partial_vol_origin']  # [B, 3]
        near, far = sample['query_near_far'][0, :1], sample['query_near_far'][0, 1:]

        # the ray variables
        sample_rays = sample['rays']
        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]

        imgs = sample['images'][0]
        intrinsics = sample['intrinsics'][0]
        intrinsics_l_4x = intrinsics.clone()
        intrinsics_l_4x[:, :2] *= 0.25
        w2cs = sample['w2cs'][0]
        # target_candidate_w2cs = sample['target_candidate_w2cs'][0]
        proj_matrices = sample['affine_mats']


        # - the image to render
        scale_mat = sample['scale_mat']  # [1,4,4]  used to convert mesh into true scale
        trans_mat = sample['trans_mat']
        query_c2w = sample['query_c2w']  # [1,4,4]
        true_img = sample['query_image'][0]
        true_img = np.uint8(true_img.permute(1, 2, 0).cpu().numpy() * 255)

        # depth_min, depth_max = near.cpu().numpy(), far.cpu().numpy()

        # scale_factor = sample['scale_factor'][0].cpu().numpy()
        # true_depth = sample['query_depth'] if 'query_depth' in sample.keys() else None
        # # if true_depth is not None:
        # #     true_depth = true_depth[0].cpu().numpy()
        # #     true_depth_colored = visualize_depth_numpy(true_depth, [depth_min, depth_max])[0]
        # # else:
        # #     true_depth_colored = None

        rays_o = rays_o.reshape(-1, 3).split(chunk_size)
        rays_d = rays_d.reshape(-1, 3).split(chunk_size)

        # - obtain conditional features
        with torch.no_grad():
            # - obtain conditional features
            geometry_feature_maps = self.obtain_pyramid_feature_maps(imgs, lod=0)
            # - lod 0
            conditional_features_lod0 = self.sdf_network_lod0.get_conditional_volume(
                feature_maps=geometry_feature_maps[None, :, :, :, :],
                partial_vol_origin=partial_vol_origin,
                proj_mats=proj_matrices,
                sizeH=sizeH,
                sizeW=sizeW,
                lod=0,
            )

        con_volume_lod0 = conditional_features_lod0['dense_volume_scale0']
        con_valid_mask_volume_lod0 = conditional_features_lod0['valid_mask_volume_scale0']
        coords_lod0 = conditional_features_lod0['coords_scale0']  # [1,3,wX,wY,wZ]

        if self.num_lods > 1:
            sdf_volume_lod0 = self.sdf_network_lod0.get_sdf_volume(
                con_volume_lod0, con_valid_mask_volume_lod0,
                coords_lod0, partial_vol_origin)  # [1, 1, dX, dY, dZ]

        depth_maps_lod0, depth_masks_lod0 = None, None


        if self.num_lods > 1:
            geometry_feature_maps_lod1 = self.obtain_pyramid_feature_maps(imgs, lod=1)

            if self.prune_depth_filter:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf_depthfilter(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0],
                    depth_maps_lod0, proj_matrices[0],
                    partial_vol_origin, self.sdf_network_lod0.voxel_size,
                    near, far, self.sdf_network_lod0.voxel_size, 12)
            else:
                pre_coords, pre_feats = self.sdf_renderer_lod0.get_valid_sparse_coords_by_sdf(
                    sdf_volume_lod0[0], coords_lod0[0], con_valid_mask_volume_lod0[0], con_volume_lod0[0])

            pre_coords[:, 1:] = pre_coords[:, 1:] * 2

            with torch.no_grad():
                conditional_features_lod1 = self.sdf_network_lod1.get_conditional_volume(
                    feature_maps=geometry_feature_maps_lod1[None, :, :, :, :],
                    partial_vol_origin=partial_vol_origin,
                    proj_mats=proj_matrices,
                    sizeH=sizeH,
                    sizeW=sizeW,
                    pre_coords=pre_coords,
                    pre_feats=pre_feats,
                )

            con_volume_lod1 = conditional_features_lod1['dense_volume_scale1']
            con_valid_mask_volume_lod1 = conditional_features_lod1['valid_mask_volume_scale1']


        # - extract mesh
        if (iter_step % self.val_mesh_freq == 0):
            torch.cuda.empty_cache()
            self.validate_colored_mesh(
                                        density_or_sdf_network=self.sdf_network_lod0,
                                        func_extract_geometry=self.sdf_renderer_lod0.extract_geometry,
                                        resolution=resolution,
                                        conditional_volume=con_volume_lod0,
                                        conditional_valid_mask_volume = con_valid_mask_volume_lod0,
                                        feature_maps=geometry_feature_maps,
                                        color_maps=imgs,
                                        w2cs=w2cs,
                                        target_candidate_w2cs=None,
                                        intrinsics=intrinsics,
                                        rendering_network=self.rendering_network_lod0,
                                        rendering_projector=self.sdf_renderer_lod0.rendering_projector,
                                        lod=0,
                                        threshold=0,
                                        query_c2w=query_c2w,
                                        mode='val_bg', meta=meta,
                                        iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat
                                    )
            torch.cuda.empty_cache()

            if self.num_lods > 1:
                self.validate_colored_mesh(
                            density_or_sdf_network=self.sdf_network_lod1,
                            func_extract_geometry=self.sdf_renderer_lod1.extract_geometry,
                            resolution=resolution,
                            conditional_volume=con_volume_lod1,
                            conditional_valid_mask_volume = con_valid_mask_volume_lod1,
                            feature_maps=geometry_feature_maps,
                            color_maps=imgs,
                            w2cs=w2cs,
                            target_candidate_w2cs=None,
                            intrinsics=intrinsics,
                            rendering_network=self.rendering_network_lod1,
                            rendering_projector=self.sdf_renderer_lod1.rendering_projector,
                            lod=1,
                            threshold=0,
                            query_c2w=query_c2w,
                            mode='val_bg', meta=meta,
                            iter_step=iter_step, scale_mat=scale_mat, trans_mat=trans_mat
                        )
            torch.cuda.empty_cache()




    def save_visualization(self, true_img, true_colored_depth, out_depth, out_normal, w2cs, out_color, H, W,
                           depth_min, depth_max, iter_step, meta, comment, out_color_mlp=[], true_depth=None, scale_factor=1.0):
        if len(out_color) > 0:
            img_fine = (np.concatenate(out_color, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        if len(out_color_mlp) > 0:
            img_mlp = (np.concatenate(out_color_mlp, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        if len(out_normal) > 0:
            normal_img = np.concatenate(out_normal, axis=0)
            rot = w2cs[:3, :3].detach().cpu().numpy()
            # - convert normal from world space to camera space
            normal_img = (np.matmul(rot[None, :, :],
                                    normal_img[:, :, None]).reshape([H, W, 3]) * 128 + 128).clip(0, 255)
        if len(out_depth) > 0:
            pred_depth = np.concatenate(out_depth, axis=0).reshape([H, W])
            pred_depth_colored = visualize_depth_numpy(pred_depth, [depth_min, depth_max])[0]

        if len(out_depth) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'depths_' + comment), exist_ok=True)
            if true_colored_depth is not None:

                if true_depth is not None:
                    depth_error_map = np.abs(true_depth - pred_depth) * 2.0 / scale_factor
                    # [256, 256, 1] -> [256, 256, 3]
                    depth_error_map = np.tile(depth_error_map[:, :, None], [1, 1, 3])

                    depth_visualized = np.concatenate(
                            [(depth_error_map * 255).astype(np.uint8), true_colored_depth, pred_depth_colored, true_img], axis=1)[:, :, ::-1]
                    # print("depth_visualized.shape: ", depth_visualized.shape)
                    # write  depth error result text on img, the input is a numpy array of [256, 1024, 3]
                    # cv.putText(depth_visualized.copy(), "depth_error_mean: {:.4f}".format(depth_error_map.mean()), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    depth_visualized = np.concatenate(
                            [true_colored_depth, pred_depth_colored, true_img])[:, :, ::-1]
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'depths_' + comment,
                                 '{:0>8d}_{}.png'.format(iter_step, meta)), depth_visualized
                    )
            else:
                cv.imwrite(
                    os.path.join(self.base_exp_dir, 'depths_' + comment,
                                 '{:0>8d}_{}.png'.format(iter_step, meta)),
                    np.concatenate(
                        [pred_depth_colored, true_img])[:, :, ::-1])
        if len(out_color) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'synthesized_color_' + comment), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'synthesized_color_' + comment,
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       np.concatenate(
                           [img_fine, true_img])[:, :, ::-1])  # bgr2rgb
            # compute psnr (image pixel lie in [0, 255])
            # mse_loss = np.mean((img_fine - true_img) ** 2)
            # psnr = 10 * np.log10(255 ** 2 / mse_loss)
            
        if len(out_color_mlp) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'synthesized_color_mlp_' + comment), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'synthesized_color_mlp_' + comment,
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       np.concatenate(
                           [img_mlp, true_img])[:, :, ::-1])  # bgr2rgb

        if len(out_normal) > 0:
            os.makedirs(os.path.join(self.base_exp_dir, 'normals_' + comment), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'normals_' + comment,
                                    '{:0>8d}_{}.png'.format(iter_step, meta)),
                       normal_img[:, :, ::-1])

    def forward(self, sample,
                perturb_overwrite=-1,
                background_rgb=None,
                alpha_inter_ratio_lod0=0.0,
                alpha_inter_ratio_lod1=0.0,
                iter_step=0,
                mode='train',
                save_vis=False,
                resolution=360,
                ):

        if mode == 'train':
            return self.train_step(sample,
                                   perturb_overwrite=perturb_overwrite,
                                   background_rgb=background_rgb,
                                   alpha_inter_ratio_lod0=alpha_inter_ratio_lod0,
                                   alpha_inter_ratio_lod1=alpha_inter_ratio_lod1,
                                   iter_step=iter_step
                                   )
        elif mode == 'val':
            import time
            begin = time.time()
            result =  self.val_step(sample,
                                 perturb_overwrite=perturb_overwrite,
                                 background_rgb=background_rgb,
                                 alpha_inter_ratio_lod0=alpha_inter_ratio_lod0,
                                 alpha_inter_ratio_lod1=alpha_inter_ratio_lod1,
                                 iter_step=iter_step,
                                 save_vis=save_vis,
                                 )
            end = time.time()
            print("val_step time: ", end - begin)
            return result
        elif mode == 'export_mesh':
            import time
            begin = time.time()
            result =  self.export_mesh_step(sample,
                                    iter_step=iter_step,
                                    save_vis=save_vis,
                                    resolution=resolution,
                                    )
            end = time.time()
            print("export mesh time: ", end - begin)
            return result
        elif mode == 'get_metrics':
            return self.get_metrics_step(sample,
                        perturb_overwrite=perturb_overwrite,
                        background_rgb=background_rgb,
                        alpha_inter_ratio_lod0=alpha_inter_ratio_lod0,
                        alpha_inter_ratio_lod1=alpha_inter_ratio_lod1,
                        iter_step=iter_step
                        )
    def obtain_pyramid_feature_maps(self, imgs, lod=0):
        """
        get feature maps of all conditional images
        :param imgs:
        :return:
        """

        if lod == 0:
            extractor = self.pyramid_feature_network_geometry_lod0
        elif lod >= 1:
            extractor = self.pyramid_feature_network_geometry_lod1

        pyramid_feature_maps = extractor(imgs)

        # * the pyramid features are very important, if only use the coarst features, hard to optimize
        fused_feature_maps = torch.cat([
            F.interpolate(pyramid_feature_maps[0], scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(pyramid_feature_maps[1], scale_factor=2, mode='bilinear', align_corners=True),
            pyramid_feature_maps[2]
        ], dim=1)

        return fused_feature_maps

    def cal_losses_sdf(self, render_out, sample_rays, iter_step=-1, lod=0):

        # loss weight schedule; the regularization terms should be added in later training stage
        def get_weight(iter_step, weight):
            if lod == 1:
                anneal_start = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = anneal_end * 2
            else:
                anneal_start = self.anneal_start if lod == 0 else self.anneal_start_lod1
                anneal_end = self.anneal_end if lod == 0 else self.anneal_end_lod1
                anneal_end = anneal_end * 2

            if iter_step < 0:
                return weight

            if anneal_end == 0.0:
                return weight
            elif iter_step < anneal_start:
                return 0.0
            else:
                return np.min(
                    [1.0,
                     (iter_step - anneal_start) / (anneal_end - anneal_start)]) * weight

        rays_o = sample_rays['rays_o'][0]
        rays_d = sample_rays['rays_v'][0]
        true_rgb = sample_rays['rays_color'][0]

        if 'rays_depth' in sample_rays.keys():
            true_depth = sample_rays['rays_depth'][0]
        else:
            true_depth = None
        mask = sample_rays['rays_mask'][0]

        color_fine = render_out['color_fine']
        color_fine_mask = render_out['color_fine_mask']
        depth_pred = render_out['depth']

        variance = render_out['variance']
        cdf_fine = render_out['cdf_fine']
        weight_sum = render_out['weights_sum']

        gradient_error_fine = render_out['gradient_error_fine']

        sdf = render_out['sdf']

        # * color generated by mlp
        color_mlp = render_out['color_mlp']
        color_mlp_mask = render_out['color_mlp_mask']

        if color_fine is not None:
            # Color loss
            color_mask = color_fine_mask if color_fine_mask is not None else mask
            color_mask = color_mask[..., 0]
            color_error = (color_fine[color_mask] - true_rgb[color_mask])
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error).to(color_error.device),
                                        reduction='mean')
            psnr = 20.0 * torch.log10(
                1.0 / (((color_fine[color_mask] - true_rgb[color_mask]) ** 2).mean() / (3.0)).sqrt())
        else:
            color_fine_loss = 0.
            psnr = 0.

        if color_mlp is not None:
            # Color loss
            color_mlp_mask = color_mlp_mask[..., 0]
            color_error_mlp = (color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask])
            color_mlp_loss = F.l1_loss(color_error_mlp,
                                       torch.zeros_like(color_error_mlp).to(color_error_mlp.device),
                                       reduction='mean')

            psnr_mlp = 20.0 * torch.log10(
                1.0 / (((color_mlp[color_mlp_mask] - true_rgb[color_mlp_mask]) ** 2).mean() / (3.0)).sqrt())
        else:
            color_mlp_loss = 0.
            psnr_mlp = 0.

        # depth loss is only used for inference, not included in total loss
        if true_depth is not None:
            # depth_loss = self.depth_criterion(depth_pred, true_depth, mask)
            depth_loss = self.depth_criterion(depth_pred, true_depth)

            depth_statis = None
        else:
            depth_loss = 0.
            depth_statis = None

        sparse_loss_1 = torch.exp(
            -1 * torch.abs(render_out['sdf_random']) * self.sdf_decay_param).mean()  # - should equal
        sparse_loss_2 = torch.exp(-1 * torch.abs(sdf) * self.sdf_decay_param).mean()
        sparse_loss = (sparse_loss_1 + sparse_loss_2) / 2

        sdf_mean = torch.abs(sdf).mean()
        sparseness_1 = (torch.abs(sdf) < 0.01).to(torch.float32).mean()
        sparseness_2 = (torch.abs(sdf) < 0.02).to(torch.float32).mean()

        # Eikonal loss
        gradient_error_loss = gradient_error_fine
        # ! the first 50k, don't use bg constraint
        fg_bg_weight = 0.0 if iter_step < 50000 else get_weight(iter_step, self.fg_bg_weight)

        # Mask loss, optional
        # The images of DTU dataset contain large black regions (0 rgb values),
        # can use this data prior to make fg more clean
        background_loss = 0.0
        fg_bg_loss = 0.0
        if self.fg_bg_weight > 0 and torch.mean((mask < 0.5).to(torch.float32)) > 0.02:
            weights_sum_fg = render_out['weights_sum_fg']
            fg_bg_error = (weights_sum_fg - mask)
            fg_bg_loss = F.l1_loss(fg_bg_error,
                                   torch.zeros_like(fg_bg_error).to(fg_bg_error.device),
                                   reduction='mean')


        loss = self.depth_loss_weight * depth_loss + color_fine_loss + color_mlp_loss + \
               sparse_loss * get_weight(iter_step, self.sdf_sparse_weight) + \
               fg_bg_loss * fg_bg_weight + \
               gradient_error_loss * self.sdf_igr_weight  # ! gradient_error_loss need a mask

        losses = {
            "loss": loss,
            "depth_loss": depth_loss,
            "color_fine_loss": color_fine_loss,
            "color_mlp_loss": color_mlp_loss,
            "gradient_error_loss": gradient_error_loss,
            "background_loss": background_loss,
            "sparse_loss": sparse_loss,
            "sparseness_1": sparseness_1,
            "sparseness_2": sparseness_2,
            "sdf_mean": sdf_mean,
            "psnr": psnr,
            "psnr_mlp": psnr_mlp,
            "weights_sum": render_out['weights_sum'],
            "weights_sum_fg": render_out['weights_sum_fg'],
            "alpha_sum": render_out['alpha_sum'],
            "variance": render_out['variance'],
            "sparse_weight": get_weight(iter_step, self.sdf_sparse_weight),
            "fg_bg_weight": fg_bg_weight,
            "fg_bg_loss": fg_bg_loss, 
        }
        losses = numpy2tensor(losses, device=rays_o.device)
        return loss, losses, depth_statis

    @torch.no_grad()
    def validate_mesh(self, density_or_sdf_network, func_extract_geometry, world_space=True, resolution=360,
                      threshold=0.0, mode='val',
                      # * 3d feature volume
                      conditional_volume=None, lod=None, occupancy_mask=None,
                      bound_min=[-1, -1, -1], bound_max=[1, 1, 1], meta='', iter_step=0, scale_mat=None,
                      trans_mat=None
                      ):

        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)

        vertices, triangles, fields = func_extract_geometry(
            density_or_sdf_network,
            bound_min, bound_max, resolution=resolution,
            threshold=threshold, device=conditional_volume.device,
            # * 3d feature volume
            conditional_volume=conditional_volume, lod=lod,
            occupancy_mask=occupancy_mask
        )


        if scale_mat is not None:
            scale_mat_np = scale_mat.cpu().numpy()
            vertices = vertices * scale_mat_np[0][0, 0] + scale_mat_np[0][:3, 3][None]

        if trans_mat is not None: # w2c_ref_inv
            trans_mat_np = trans_mat.cpu().numpy()
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            vertices = np.matmul(trans_mat_np, vertices_homo[:, :, None])[:, :3, 0]

        mesh = trimesh.Trimesh(vertices, triangles)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes_' + mode), exist_ok=True)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes_' + mode,
                                 'mesh_{:0>8d}_{}_lod{:0>1d}.ply'.format(iter_step, meta, lod)))



    def validate_colored_mesh(self, density_or_sdf_network, func_extract_geometry, world_space=True, resolution=360,
                                threshold=0.0, mode='val',
                                # * 3d feature volume
                                conditional_volume=None,
                                conditional_valid_mask_volume=None,
                                feature_maps=None,
                                color_maps = None,
                                w2cs=None,
                                target_candidate_w2cs=None,
                                intrinsics=None,
                                rendering_network=None,
                                rendering_projector=None,
                                query_c2w=None,
                                lod=None, occupancy_mask=None,
                                bound_min=[-1, -1, -1], bound_max=[1, 1, 1], meta='', iter_step=0, scale_mat=None,
                                trans_mat=None
                                ):

        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)

        vertices, triangles, fields = func_extract_geometry(
            density_or_sdf_network,
            bound_min, bound_max, resolution=resolution,
            threshold=threshold, device=conditional_volume.device,
            # * 3d feature volume
            conditional_volume=conditional_volume, lod=lod,
            occupancy_mask=occupancy_mask
        )
        

        with torch.no_grad():
            ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask, _, _ = rendering_projector.compute_view_independent(
                torch.tensor(vertices).to(conditional_volume),
                lod=lod,
                # * 3d geometry feature volumes
                geometryVolume=conditional_volume[0],
                geometryVolumeMask=conditional_valid_mask_volume[0],
                sdf_network=density_or_sdf_network,
                # * 2d rendering feature maps
                rendering_feature_maps=feature_maps, # [n_view, 56, 256, 256]
                color_maps=color_maps,
                w2cs=w2cs,
                target_candidate_w2cs=target_candidate_w2cs,
                intrinsics=intrinsics,
                img_wh=[256,256],
                query_img_idx=0,  # the index of the N_views dim for rendering
                query_c2w=query_c2w,
            )


            vertices_color, rendering_valid_mask = rendering_network(
                ren_geo_feats, ren_rgb_feats, ren_ray_diff, ren_mask)
        


        if scale_mat is not None:
            scale_mat_np = scale_mat.cpu().numpy()
            vertices = vertices * scale_mat_np[0][0, 0] + scale_mat_np[0][:3, 3][None]

        if trans_mat is not None: # w2c_ref_inv
            trans_mat_np = trans_mat.cpu().numpy()
            vertices_homo = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)
            vertices = np.matmul(trans_mat_np, vertices_homo[:, :, None])[:, :3, 0]

        vertices_color = np.array(vertices_color.squeeze(0).cpu() * 255, dtype=np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertices_color)
        # os.makedirs(os.path.join(self.base_exp_dir, 'meshes_' + mode, 'lod{:0>1d}'.format(lod)), exist_ok=True)
        # mesh.export(os.path.join(self.base_exp_dir, 'meshes_' + mode, 'lod{:0>1d}'.format(lod),
        #                          'mesh_{:0>8d}_{}_lod{:0>1d}.ply'.format(iter_step, meta, lod)))  
        
        mesh.export(os.path.join(self.base_exp_dir, 'mesh.ply'))