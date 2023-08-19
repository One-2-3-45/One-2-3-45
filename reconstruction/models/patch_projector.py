"""
Patch Projector
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.render_utils import sample_ptsFeatures_from_featureMaps


class PatchProjector():
    def __init__(self, patch_size):
        self.h_patch_size = patch_size
        self.offsets = build_patch_offset(patch_size)  # the warping patch offsets index

        self.z_axis = torch.tensor([0, 0, 1]).float()

        self.plane_dist_thresh = 0.001

    # * correctness checked
    def pixel_warp(self, pts, imgs, intrinsics,
                   w2cs, img_wh=None):
        """

        :param pts: [N_rays, n_samples, 3]
        :param imgs: [N_views, 3, H, W]
        :param intrinsics: [N_views, 4, 4]
        :param c2ws: [N_views, 4, 4]
        :param img_wh:
        :return:
        """
        if img_wh is None:
            N_views, _, sizeH, sizeW = imgs.shape
            img_wh = [sizeW, sizeH]

        pts_color, valid_mask = sample_ptsFeatures_from_featureMaps(
            pts, imgs, w2cs, intrinsics, img_wh,
            proj_matrix=None, return_mask=True)  # [N_views, c, N_rays, n_samples], [N_views, N_rays, n_samples]

        pts_color = pts_color.permute(2, 3, 0, 1)
        valid_mask = valid_mask.permute(1, 2, 0)

        return pts_color, valid_mask  # [N_rays, n_samples, N_views,  3] , [N_rays, n_samples, N_views]

    def patch_warp(self, pts, uv, normals, src_imgs,
                   ref_intrinsic, src_intrinsics,
                   ref_c2w, src_c2ws, img_wh=None
                   ):
        """

        :param pts: [N_rays, n_samples, 3]
        :param uv : [N_rays, 2]  normalized in (-1, 1)
        :param normals: [N_rays, n_samples, 3]  The normal of pt in world space
        :param src_imgs: [N_src, 3, h, w]
        :param ref_intrinsic: [4,4]
        :param src_intrinsics: [N_src, 4, 4]
        :param ref_c2w: [4,4]
        :param src_c2ws: [N_src, 4, 4]
        :return:
        """
        device = pts.device

        N_rays, n_samples, _ = pts.shape
        N_pts = N_rays * n_samples

        N_src, _, sizeH, sizeW = src_imgs.shape

        if img_wh is not None:
            sizeW, sizeH = img_wh[0], img_wh[1]

        # scale uv from (-1, 1) to (0, W/H)
        uv[:, 0] = (uv[:, 0] + 1) / 2. * (sizeW - 1)
        uv[:, 1] = (uv[:, 1] + 1) / 2. * (sizeH - 1)

        ref_intr = ref_intrinsic[:3, :3]
        inv_ref_intr = torch.inverse(ref_intr)
        src_intrs = src_intrinsics[:, :3, :3]
        inv_src_intrs = torch.inverse(src_intrs)

        ref_pose = ref_c2w
        inv_ref_pose = torch.inverse(ref_pose)
        src_poses = src_c2ws
        inv_src_poses = torch.inverse(src_poses)

        ref_cam_loc = ref_pose[:3, 3].unsqueeze(0)  # [1, 3]
        sampled_dists = torch.norm(pts - ref_cam_loc, dim=-1)  # [N_pts, 1]

        relative_proj = inv_src_poses @ ref_pose
        R_rel = relative_proj[:, :3, :3]
        t_rel = relative_proj[:, :3, 3:]
        R_ref = inv_ref_pose[:3, :3]
        t_ref = inv_ref_pose[:3, 3:]

        pts = pts.view(-1, 3)
        normals = normals.view(-1, 3)

        with torch.no_grad():
            rot_normals = R_ref @ normals.unsqueeze(-1)  # [N_pts, 3, 1]
            points_in_ref = R_ref @ pts.unsqueeze(
                -1) + t_ref  # [N_pts, 3, 1]  points in the reference frame coordiantes system
            d1 = torch.sum(rot_normals * points_in_ref, dim=1).unsqueeze(
                1)  # distance from the plane to ref camera center

            d2 = torch.sum(rot_normals.unsqueeze(1) * (-R_rel.transpose(1, 2) @ t_rel).unsqueeze(0),
                           dim=2)  # distance from the plane to src camera center
            valid_hom = (torch.abs(d1) > self.plane_dist_thresh) & (
                    torch.abs(d1 - d2) > self.plane_dist_thresh) & ((d2 / d1) < 1)

            d1 = d1.squeeze()
            sign = torch.sign(d1)
            sign[sign == 0] = 1
            d = torch.clamp(torch.abs(d1), 1e-8) * sign

            H = src_intrs.unsqueeze(1) @ (
                    R_rel.unsqueeze(1) + t_rel.unsqueeze(1) @ rot_normals.view(1, N_pts, 1, 3) / d.view(1,
                                                                                                        N_pts,
                                                                                                        1, 1)
            ) @ inv_ref_intr.view(1, 1, 3, 3)

            # replace invalid homs with fronto-parallel homographies
            H_invalid = src_intrs.unsqueeze(1) @ (
                    R_rel.unsqueeze(1) + t_rel.unsqueeze(1) @ self.z_axis.to(device).view(1, 1, 1, 3).expand(-1, N_pts,
                                                                                                             -1,
                                                                                                             -1) / sampled_dists.view(
                1, N_pts, 1, 1)
            ) @ inv_ref_intr.view(1, 1, 3, 3)
            tmp_m = ~valid_hom.view(-1, N_src).t()
            H[tmp_m] = H_invalid[tmp_m]

        pixels = uv.view(N_rays, 1, 2) + self.offsets.float().to(device)
        Npx = pixels.shape[1]
        grid, warp_mask_full = self.patch_homography(H, pixels)

        warp_mask_full = warp_mask_full & (grid[..., 0] < (sizeW - self.h_patch_size)) & (
                grid[..., 1] < (sizeH - self.h_patch_size)) & (grid >= self.h_patch_size).all(dim=-1)
        warp_mask_full = warp_mask_full.view(N_src, N_rays, n_samples, Npx)

        grid = torch.clamp(normalize(grid, sizeH, sizeW), -10, 10)

        sampled_rgb_val = F.grid_sample(src_imgs, grid.view(N_src, -1, 1, 2), align_corners=True).squeeze(
            -1).transpose(1, 2)
        sampled_rgb_val = sampled_rgb_val.view(N_src, N_rays, n_samples, Npx, 3)

        warp_mask_full = warp_mask_full.permute(1, 2, 0, 3).contiguous()  # (N_rays, n_samples, N_src, Npx)
        sampled_rgb_val = sampled_rgb_val.permute(1, 2, 0, 3, 4).contiguous()  # (N_rays, n_samples, N_src, Npx, 3)

        return sampled_rgb_val, warp_mask_full

    def patch_homography(self, H, uv):
        N, Npx = uv.shape[:2]
        Nsrc = H.shape[0]
        H = H.view(Nsrc, N, -1, 3, 3)
        hom_uv = add_hom(uv)

        # einsum is 30 times faster
        # tmp = (H.view(Nsrc, N, -1, 1, 3, 3) @ hom_uv.view(1, N, 1, -1, 3, 1)).squeeze(-1).view(Nsrc, -1, 3)
        tmp = torch.einsum("vprik,pok->vproi", H, hom_uv).reshape(Nsrc, -1, 3)

        grid = tmp[..., :2] / torch.clamp(tmp[..., 2:], 1e-8)
        mask = tmp[..., 2] > 0
        return grid, mask


def add_hom(pts):
    try:
        dev = pts.device
        ones = torch.ones(pts.shape[:-1], device=dev).unsqueeze(-1)
        return torch.cat((pts, ones), dim=-1)

    except AttributeError:
        ones = np.ones((pts.shape[0], 1))
        return np.concatenate((pts, ones), axis=1)


def normalize(flow, h, w, clamp=None):
    # either h and w are simple float or N torch.tensor where N batch size
    try:
        h.device

    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    # for grid_sample with align_corners=True
    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res


def build_patch_offset(h_patch_size):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing="ij")[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 2
