# The codes are partly from IBRNet

import torch
import torch.nn.functional as F
from models.render_utils import sample_ptsFeatures_from_featureMaps, sample_ptsFeatures_from_featureVolume

def safe_l2_normalize(x, dim=None, eps=1e-6):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class Projector():
    """
    Obtain features from geometryVolume and rendering_feature_maps for generalized rendering
    """

    def compute_angle(self, xyz, query_c2w, supporting_c2ws):
        """

        :param xyz: [N_rays, n_samples,3 ]
        :param query_c2w: [1,4,4]
        :param supporting_c2ws: [n,4,4]
        :return:
        """
        N_rays, n_samples, _ = xyz.shape
        num_views = supporting_c2ws.shape[0]
        xyz = xyz.reshape(-1, 3)

        ray2tar_pose = (query_c2w[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2support_pose = (supporting_c2ws[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2support_pose /= (torch.norm(ray2support_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2support_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2support_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, N_rays, n_samples, 4))  # the last dimension (4) is dot-product
        return ray_diff.detach()


    def compute_angle_view_independent(self, xyz, surface_normals, supporting_c2ws):
        """

        :param xyz: [N_rays, n_samples,3 ]
        :param surface_normals: [N_rays, n_samples,3 ]
        :param supporting_c2ws: [n,4,4]
        :return:
        """
        N_rays, n_samples, _ = xyz.shape
        num_views = supporting_c2ws.shape[0]
        xyz = xyz.reshape(-1, 3)

        ray2tar_pose = surface_normals
        ray2support_pose = (supporting_c2ws[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2support_pose /= (torch.norm(ray2support_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2support_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2support_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, N_rays, n_samples, 4))  # the last dimension (4) is dot-product, 
                                                                        # and the first three dimensions is the normalized ray diff vector
        return ray_diff.detach()

    @torch.no_grad()
    def compute_z_diff(self, xyz, w2cs, intrinsics, pred_depth_values):
        """
        compute the depth difference of query pts projected on the image and the predicted depth values of the image
        :param xyz:  [N_rays, n_samples,3 ]
        :param w2cs:  [N_views, 4, 4]
        :param intrinsics: [N_views, 3, 3]
        :param pred_depth_values: [N_views, N_rays, n_samples,1 ]
        :param pred_depth_masks: [N_views, N_rays, n_samples]
        :return:
        """
        device = xyz.device
        N_views = w2cs.shape[0]
        N_rays, n_samples, _ = xyz.shape
        proj_matrix = torch.matmul(intrinsics, w2cs[:, :3, :])

        proj_rot = proj_matrix[:, :3, :3]
        proj_trans = proj_matrix[:, :3, 3:]

        batch_xyz = xyz.permute(2, 0, 1).contiguous().view(1, 3, N_rays * n_samples).repeat(N_views, 1, 1)

        proj_xyz = proj_rot.bmm(batch_xyz) + proj_trans

        # X = proj_xyz[:, 0]
        # Y = proj_xyz[:, 1]
        Z = proj_xyz[:, 2].clamp(min=1e-3)  # [N_views, N_rays*n_samples]
        proj_z = Z.view(N_views, N_rays, n_samples, 1)

        z_diff = proj_z - pred_depth_values  # [N_views, N_rays, n_samples,1 ]

        return z_diff

    def compute(self,
                pts,
                # * 3d geometry feature volumes
                geometryVolume=None,
                geometryVolumeMask=None,
                vol_dims=None,
                partial_vol_origin=None,
                vol_size=None,
                # * 2d rendering feature maps
                rendering_feature_maps=None,
                color_maps=None,
                w2cs=None,
                intrinsics=None,
                img_wh=None,
                query_img_idx=0,  # the index of the N_views dim for rendering
                query_c2w=None,
                pred_depth_maps=None,   # no use here
                pred_depth_masks=None   # no use here
                ):
        """
        extract features of pts for rendering
        :param pts:
        :param geometryVolume:
        :param vol_dims:
        :param partial_vol_origin:
        :param vol_size:
        :param rendering_feature_maps:
        :param color_maps:
        :param w2cs:
        :param intrinsics:
        :param img_wh:
        :param rendering_img_idx: by default, we render the first view of w2cs
        :return:
        """
        device = pts.device
        c2ws = torch.inverse(w2cs)

        if len(pts.shape) == 2:
            pts = pts[None, :, :]

        N_rays, n_samples, _ = pts.shape
        N_views = rendering_feature_maps.shape[0]  # shape (N_views, C, H, W)

        supporting_img_idxs = torch.LongTensor([x for x in range(N_views) if x != query_img_idx]).to(device)
        query_img_idx = torch.LongTensor([query_img_idx]).to(device)

        if query_c2w is None and query_img_idx > -1:
            query_c2w = torch.index_select(c2ws, 0, query_img_idx)
            supporting_c2ws = torch.index_select(c2ws, 0, supporting_img_idxs)
            supporting_w2cs = torch.index_select(w2cs, 0, supporting_img_idxs)
            supporting_rendering_feature_maps = torch.index_select(rendering_feature_maps, 0, supporting_img_idxs)
            supporting_color_maps = torch.index_select(color_maps, 0, supporting_img_idxs)
            supporting_intrinsics = torch.index_select(intrinsics, 0, supporting_img_idxs)

            if pred_depth_maps is not None:
                supporting_depth_maps = torch.index_select(pred_depth_maps, 0, supporting_img_idxs)
                supporting_depth_masks = torch.index_select(pred_depth_masks, 0, supporting_img_idxs)
            # print("N_supporting_views: ", N_views - 1)
            N_supporting_views = N_views - 1
        else:
            supporting_c2ws = c2ws
            supporting_w2cs = w2cs
            supporting_rendering_feature_maps = rendering_feature_maps
            supporting_color_maps = color_maps
            supporting_intrinsics = intrinsics
            supporting_depth_maps = pred_depth_masks
            supporting_depth_masks = pred_depth_masks
            # print("N_supporting_views: ", N_views)
            N_supporting_views = N_views
        # import ipdb; ipdb.set_trace()
        if geometryVolume is not None:
            # * sample feature of pts from 3D feature volume
            pts_geometry_feature, pts_geometry_masks_0 = sample_ptsFeatures_from_featureVolume(
                pts, geometryVolume, vol_dims,
                partial_vol_origin, vol_size)  # [N_rays, n_samples, C], [N_rays, n_samples]

            if len(geometryVolumeMask.shape) == 3:
                geometryVolumeMask = geometryVolumeMask[None, :, :, :]

            pts_geometry_masks_1, _ = sample_ptsFeatures_from_featureVolume(
                pts, geometryVolumeMask.to(geometryVolume.dtype), vol_dims,
                partial_vol_origin, vol_size)  # [N_rays, n_samples, C]

            pts_geometry_masks = pts_geometry_masks_0 & (pts_geometry_masks_1[..., 0] > 0)
        else:
            pts_geometry_feature = None
            pts_geometry_masks = None

        # * sample feature of pts from 2D feature maps
        pts_rendering_feats, pts_rendering_mask = sample_ptsFeatures_from_featureMaps(
            pts, supporting_rendering_feature_maps, supporting_w2cs,
            supporting_intrinsics, img_wh,
            return_mask=True)  # [N_views, C, N_rays, n_samples], # [N_views, N_rays, n_samples]
        # import ipdb; ipdb.set_trace()
        # * size (N_views, N_rays*n_samples, c)
        pts_rendering_feats = pts_rendering_feats.permute(0, 2, 3, 1).contiguous()

        pts_rendering_colors = sample_ptsFeatures_from_featureMaps(pts, supporting_color_maps, supporting_w2cs,
                                                                   supporting_intrinsics, img_wh)
        # * size (N_views, N_rays*n_samples, c)
        pts_rendering_colors = pts_rendering_colors.permute(0, 2, 3, 1).contiguous()

        rgb_feats = torch.cat([pts_rendering_colors, pts_rendering_feats], dim=-1)  # [N_views, N_rays, n_samples, 3+c]


        ray_diff = self.compute_angle(pts, query_c2w, supporting_c2ws)  # [N_views, N_rays, n_samples, 4]
        # import ipdb; ipdb.set_trace()
        if pts_geometry_masks is not None:
            final_mask = pts_geometry_masks[None, :, :].repeat(N_supporting_views, 1, 1) & \
                         pts_rendering_mask  # [N_views, N_rays, n_samples]
        else:
            final_mask = pts_rendering_mask
        # import ipdb; ipdb.set_trace()
        z_diff, pts_pred_depth_masks = None, None
        
        if pred_depth_maps is not None:
            pts_pred_depth_values = sample_ptsFeatures_from_featureMaps(pts, supporting_depth_maps, supporting_w2cs,
                                                                        supporting_intrinsics, img_wh)
            pts_pred_depth_values = pts_pred_depth_values.permute(0, 2, 3,
                                                                  1).contiguous()  # (N_views, N_rays*n_samples, 1)

            # - pts_pred_depth_masks are critical than final_mask,
            # - the ray containing few invalid pts will be treated invalid
            pts_pred_depth_masks = sample_ptsFeatures_from_featureMaps(pts, supporting_depth_masks.float(),
                                                                       supporting_w2cs,
                                                                       supporting_intrinsics, img_wh)
            
            pts_pred_depth_masks = pts_pred_depth_masks.permute(0, 2, 3, 1).contiguous()[:, :, :,
                                   0]  # (N_views, N_rays*n_samples)

            z_diff = self.compute_z_diff(pts, supporting_w2cs, supporting_intrinsics, pts_pred_depth_values)
        # import ipdb; ipdb.set_trace()
        return pts_geometry_feature, rgb_feats, ray_diff, final_mask, z_diff, pts_pred_depth_masks


    def compute_view_independent(   
                                    self,
                                    pts,
                                    # * 3d geometry feature volumes
                                    geometryVolume=None,
                                    geometryVolumeMask=None,
                                    sdf_network=None,
                                    lod=0,
                                    vol_dims=None,
                                    partial_vol_origin=None,
                                    vol_size=None,
                                    # * 2d rendering feature maps
                                    rendering_feature_maps=None,
                                    color_maps=None,
                                    w2cs=None,
                                    target_candidate_w2cs=None,
                                    intrinsics=None,
                                    img_wh=None,
                                    query_img_idx=0,  # the index of the N_views dim for rendering
                                    query_c2w=None,
                                    pred_depth_maps=None,   # no use here
                                    pred_depth_masks=None   # no use here
                                ):
        """
        extract features of pts for rendering
        :param pts:
        :param geometryVolume:
        :param vol_dims:
        :param partial_vol_origin:
        :param vol_size:
        :param rendering_feature_maps:
        :param color_maps:
        :param w2cs:
        :param intrinsics:
        :param img_wh:
        :param rendering_img_idx: by default, we render the first view of w2cs
        :return:
        """
        device = pts.device
        c2ws = torch.inverse(w2cs)

        if len(pts.shape) == 2:
            pts = pts[None, :, :]

        N_rays, n_samples, _ = pts.shape
        N_views = rendering_feature_maps.shape[0]  # shape (N_views, C, H, W)

        supporting_img_idxs = torch.LongTensor([x for x in range(N_views) if x != query_img_idx]).to(device)
        query_img_idx = torch.LongTensor([query_img_idx]).to(device)

        if query_c2w is None and query_img_idx > -1:
            query_c2w = torch.index_select(c2ws, 0, query_img_idx)
            supporting_c2ws = torch.index_select(c2ws, 0, supporting_img_idxs)
            supporting_w2cs = torch.index_select(w2cs, 0, supporting_img_idxs)
            supporting_rendering_feature_maps = torch.index_select(rendering_feature_maps, 0, supporting_img_idxs)
            supporting_color_maps = torch.index_select(color_maps, 0, supporting_img_idxs)
            supporting_intrinsics = torch.index_select(intrinsics, 0, supporting_img_idxs)

            if pred_depth_maps is not None:
                supporting_depth_maps = torch.index_select(pred_depth_maps, 0, supporting_img_idxs)
                supporting_depth_masks = torch.index_select(pred_depth_masks, 0, supporting_img_idxs)
            # print("N_supporting_views: ", N_views - 1)
            N_supporting_views = N_views - 1
        else:
            supporting_c2ws = c2ws
            supporting_w2cs = w2cs
            supporting_rendering_feature_maps = rendering_feature_maps
            supporting_color_maps = color_maps
            supporting_intrinsics = intrinsics
            supporting_depth_maps = pred_depth_masks
            supporting_depth_masks = pred_depth_masks
            # print("N_supporting_views: ", N_views)
            N_supporting_views = N_views
        # import ipdb; ipdb.set_trace()
        if geometryVolume is not None:
            # * sample feature of pts from 3D feature volume
            pts_geometry_feature, pts_geometry_masks_0 = sample_ptsFeatures_from_featureVolume(
                pts, geometryVolume, vol_dims,
                partial_vol_origin, vol_size)  # [N_rays, n_samples, C], [N_rays, n_samples]

            if len(geometryVolumeMask.shape) == 3:
                geometryVolumeMask = geometryVolumeMask[None, :, :, :]

            pts_geometry_masks_1, _ = sample_ptsFeatures_from_featureVolume(
                pts, geometryVolumeMask.to(geometryVolume.dtype), vol_dims,
                partial_vol_origin, vol_size)  # [N_rays, n_samples, C]

            pts_geometry_masks = pts_geometry_masks_0 & (pts_geometry_masks_1[..., 0] > 0)
        else:
            pts_geometry_feature = None
            pts_geometry_masks = None

        # * sample feature of pts from 2D feature maps
        pts_rendering_feats, pts_rendering_mask = sample_ptsFeatures_from_featureMaps(
            pts, supporting_rendering_feature_maps, supporting_w2cs,
            supporting_intrinsics, img_wh,
            return_mask=True)  # [N_views, C, N_rays, n_samples], # [N_views, N_rays, n_samples]

        # * size (N_views, N_rays*n_samples, c)
        pts_rendering_feats = pts_rendering_feats.permute(0, 2, 3, 1).contiguous()

        pts_rendering_colors = sample_ptsFeatures_from_featureMaps(pts, supporting_color_maps, supporting_w2cs,
                                                                   supporting_intrinsics, img_wh)
        # * size (N_views, N_rays*n_samples, c)
        pts_rendering_colors = pts_rendering_colors.permute(0, 2, 3, 1).contiguous()

        rgb_feats = torch.cat([pts_rendering_colors, pts_rendering_feats], dim=-1)  # [N_views, N_rays, n_samples, 3+c]
        
        # import ipdb; ipdb.set_trace()
        
        gradients = sdf_network.gradient(
                                        pts.reshape(-1, 3), # pts.squeeze(0),
                                        geometryVolume.unsqueeze(0), 
                                        lod=lod
                                        ).squeeze()
        
        surface_normals = safe_l2_normalize(gradients, dim=-1) # [npts, 3]
        # input normals
        ren_ray_diff = self.compute_angle_view_independent(
                         xyz=pts,
                         surface_normals=surface_normals,
                         supporting_c2ws=supporting_c2ws   
        )

        # # choose closest target view direction from 32 candidate views
        # # choose the closest source view as view direction instead of the normals vectors
        # pts2src_centers = safe_l2_normalize((supporting_c2ws[:, :3, 3].unsqueeze(1) - pts)) # [N_views, npts, 3]

        # cosine_distance = torch.sum(pts2src_centers * surface_normals, dim=-1, keepdim=True) # [N_views, npts, 1]
        # # choose the largest cosine distance as the view direction
        # max_idx = torch.argmax(cosine_distance, dim=0) # [npts, 1]

        # chosen_view_direction = pts2src_centers[max_idx.squeeze(), torch.arange(pts.shape[1]), :] # [npts, 3]
        # ren_ray_diff = self.compute_angle_view_independent(
        #                  xyz=pts,
        #                  surface_normals=chosen_view_direction,
        #                  supporting_c2ws=supporting_c2ws   
        # )



        # # choose closest target view direction from 8 candidate views
        # # choose the closest source view as view direction instead of the normals vectors
        # target_candidate_c2ws = torch.inverse(target_candidate_w2cs)
        # pts2src_centers = safe_l2_normalize((target_candidate_c2ws[:, :3, 3].unsqueeze(1) - pts)) # [N_views, npts, 3]

        # cosine_distance = torch.sum(pts2src_centers * surface_normals, dim=-1, keepdim=True) # [N_views, npts, 1]
        # # choose the largest cosine distance as the view direction
        # max_idx = torch.argmax(cosine_distance, dim=0) # [npts, 1]

        # chosen_view_direction = pts2src_centers[max_idx.squeeze(), torch.arange(pts.shape[1]), :] # [npts, 3]
        # ren_ray_diff = self.compute_angle_view_independent(
        #                  xyz=pts,
        #                  surface_normals=chosen_view_direction,
        #                  supporting_c2ws=supporting_c2ws   
        # )


        # ray_diff = self.compute_angle(pts, query_c2w, supporting_c2ws)  # [N_views, N_rays, n_samples, 4]
        # import ipdb; ipdb.set_trace()


        # input_directions = safe_l2_normalize(pts)
        # ren_ray_diff = self.compute_angle_view_independent(
        #                  xyz=pts,
        #                  surface_normals=input_directions,
        #                  supporting_c2ws=supporting_c2ws   
        # )

        if pts_geometry_masks is not None:
            final_mask = pts_geometry_masks[None, :, :].repeat(N_supporting_views, 1, 1) & \
                         pts_rendering_mask  # [N_views, N_rays, n_samples]
        else:
            final_mask = pts_rendering_mask
        # import ipdb; ipdb.set_trace()
        z_diff, pts_pred_depth_masks = None, None
        
        if pred_depth_maps is not None:
            pts_pred_depth_values = sample_ptsFeatures_from_featureMaps(pts, supporting_depth_maps, supporting_w2cs,
                                                                        supporting_intrinsics, img_wh)
            pts_pred_depth_values = pts_pred_depth_values.permute(0, 2, 3,
                                                                  1).contiguous()  # (N_views, N_rays*n_samples, 1)

            # - pts_pred_depth_masks are critical than final_mask,
            # - the ray containing few invalid pts will be treated invalid
            pts_pred_depth_masks = sample_ptsFeatures_from_featureMaps(pts, supporting_depth_masks.float(),
                                                                       supporting_w2cs,
                                                                       supporting_intrinsics, img_wh)
            
            pts_pred_depth_masks = pts_pred_depth_masks.permute(0, 2, 3, 1).contiguous()[:, :, :,
                                   0]  # (N_views, N_rays*n_samples)

            z_diff = self.compute_z_diff(pts, supporting_w2cs, supporting_intrinsics, pts_pred_depth_values)
        # import ipdb; ipdb.set_trace()
        return pts_geometry_feature, rgb_feats, ren_ray_diff, final_mask, z_diff, pts_pred_depth_masks
