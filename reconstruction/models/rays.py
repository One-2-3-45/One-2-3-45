import os, torch
import numpy as np

import torch.nn.functional as F

def build_patch_offset(h_patch_size):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 2


def gen_rays_from_single_image(H, W, image, intrinsic, c2w, depth=None, mask=None):
    """
    generate rays in world space, for image image
    :param H:
    :param W:
    :param intrinsics: [3,3]
    :param c2ws: [4,4]
    :return:
    """
    device = image.device
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H),
                            torch.linspace(0, W - 1, W), indexing="ij")  # pytorch's meshgrid has indexing='ij'
    p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

    # normalized ndc uv coordinates, (-1, 1)
    ndc_u = 2 * xs / (W - 1) - 1
    ndc_v = 2 * ys / (H - 1) - 1
    rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float().to(device)

    intrinsic_inv = torch.inverse(intrinsic)

    p = p.view(-1, 3).float().to(device)  # N_rays, 3
    p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # N_rays, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_rays, 3
    rays_v = torch.matmul(c2w[None, :3, :3], rays_v[:, :, None]).squeeze()  # N_rays, 3
    rays_o = c2w[None, :3, 3].expand(rays_v.shape)  # N_rays, 3

    image = image.permute(1, 2, 0)
    color = image.view(-1, 3)
    depth = depth.view(-1, 1) if depth is not None else None
    mask = mask.view(-1, 1) if mask is not None else torch.ones([H * W, 1]).to(device)
    sample = {
        'rays_o': rays_o,
        'rays_v': rays_v,
        'rays_ndc_uv': rays_ndc_uv,
        'rays_color': color,
        # 'rays_depth': depth,
        'rays_mask': mask,
        'rays_norm_XYZ_cam': p  # - XYZ_cam, before multiply depth
    }
    if depth is not None:
        sample['rays_depth'] = depth

    return sample


def gen_random_rays_from_single_image(H, W, N_rays, image, intrinsic, c2w, depth=None, mask=None, dilated_mask=None,
                                      importance_sample=False, h_patch_size=3):
    """
    generate random rays in world space, for a single image
    :param H:
    :param W:
    :param N_rays:
    :param image: [3, H, W]
    :param intrinsic: [3,3]
    :param c2w: [4,4]
    :param depth: [H, W]
    :param mask: [H, W]
    :return:
    """
    device = image.device

    if dilated_mask is None:
        dilated_mask = mask

    if not importance_sample:
        pixels_x = torch.randint(low=0, high=W, size=[N_rays])
        pixels_y = torch.randint(low=0, high=H, size=[N_rays])
    elif importance_sample and dilated_mask is not None:  # sample more pts in the valid mask regions
        pixels_x_1 = torch.randint(low=0, high=W, size=[N_rays // 4])
        pixels_y_1 = torch.randint(low=0, high=H, size=[N_rays // 4])

        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H),
                                torch.linspace(0, W - 1, W), indexing="ij")  # pytorch's meshgrid has indexing='ij'
        p = torch.stack([xs, ys], dim=-1)  # H, W, 2

        try:
            p_valid = p[dilated_mask > 0]  # [num, 2]
            random_idx = torch.randint(low=0, high=p_valid.shape[0], size=[N_rays // 4 * 3])
        except:
            print("dilated_mask.shape: ", dilated_mask.shape)
            print("dilated_mask valid number", dilated_mask.sum())

            raise ValueError("hhhh")
        p_select = p_valid[random_idx]  # [N_rays//2, 2]
        pixels_x_2 = p_select[:, 0]
        pixels_y_2 = p_select[:, 1]

        pixels_x = torch.cat([pixels_x_1, pixels_x_2], dim=0).to(torch.int64)
        pixels_y = torch.cat([pixels_y_1, pixels_y_2], dim=0).to(torch.int64)

    # - crop patch from images
    offsets = build_patch_offset(h_patch_size).to(device)
    grid_patch = torch.stack([pixels_x, pixels_y], dim=-1).view(-1, 1, 2) + offsets.float()  # [N_pts, Npx, 2]
    patch_mask = (pixels_x > h_patch_size) * (pixels_x < (W - h_patch_size)) * (pixels_y > h_patch_size) * (
            pixels_y < H - h_patch_size)  # [N_pts]
    grid_patch_u = 2 * grid_patch[:, :, 0] / (W - 1) - 1
    grid_patch_v = 2 * grid_patch[:, :, 1] / (H - 1) - 1
    grid_patch_uv = torch.stack([grid_patch_u, grid_patch_v], dim=-1)  # [N_pts, Npx, 2]
    patch_color = F.grid_sample(image[None, :, :, :], grid_patch_uv[None, :, :, :], mode='bilinear',
                                padding_mode='zeros',align_corners=True)[0]  # [3, N_pts, Npx]
    patch_color = patch_color.permute(1, 2, 0).contiguous()

    # normalized ndc uv coordinates, (-1, 1)
    ndc_u = 2 * pixels_x / (W - 1) - 1
    ndc_v = 2 * pixels_y / (H - 1) - 1
    rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float().to(device)

    image = image.permute(1, 2, 0)  # H ,W, C
    color = image[(pixels_y, pixels_x)]  # N_rays, 3

    if mask is not None:
        mask = mask[(pixels_y, pixels_x)]  # N_rays
        patch_mask = patch_mask * mask  # N_rays
        mask = mask.view(-1, 1)
    else:
        mask = torch.ones([N_rays, 1])

    if depth is not None:
        depth = depth[(pixels_y, pixels_x)]  # N_rays
        depth = depth.view(-1, 1)

    intrinsic_inv = torch.inverse(intrinsic)

    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(device)  # N_rays, 3
    p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # N_rays, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_rays, 3
    rays_v = torch.matmul(c2w[None, :3, :3], rays_v[:, :, None]).squeeze()  # N_rays, 3
    rays_o = c2w[None, :3, 3].expand(rays_v.shape)  # N_rays, 3

    sample = {
        'rays_o': rays_o,
        'rays_v': rays_v,
        'rays_ndc_uv': rays_ndc_uv,
        'rays_color': color,
        # 'rays_depth': depth,
        'rays_mask': mask,
        'rays_norm_XYZ_cam': p,  # - XYZ_cam, before multiply depth,
        'rays_patch_color': patch_color,
        'rays_patch_mask': patch_mask.view(-1, 1)
    }

    if depth is not None:
        sample['rays_depth'] = depth

    return sample


def gen_random_rays_of_patch_from_single_image(H, W, N_rays, num_neighboring_pts, patch_size,
                                               image, intrinsic, c2w, depth=None, mask=None):
    """
    generate random rays in world space, for a single image
    sample rays from local patches
    :param H:
    :param W:
    :param N_rays: the number of center rays of patches
    :param image: [3, H, W]
    :param intrinsic: [3,3]
    :param c2w: [4,4]
    :param depth: [H, W]
    :param mask: [H, W]
    :return:
    """
    device = image.device
    patch_radius_max = patch_size // 2

    unit_u = 2 / (W - 1)
    unit_v = 2 / (H - 1)

    pixels_x_center = torch.randint(low=patch_size, high=W - patch_size, size=[N_rays])
    pixels_y_center = torch.randint(low=patch_size, high=H - patch_size, size=[N_rays])

    # normalized ndc uv coordinates, (-1, 1)
    ndc_u_center = 2 * pixels_x_center / (W - 1) - 1
    ndc_v_center = 2 * pixels_y_center / (H - 1) - 1
    ndc_uv_center = torch.stack([ndc_u_center, ndc_v_center], dim=-1).view(-1, 2).float().to(device)[:, None,
                    :]  # [N_rays, 1, 2]

    shift_u, shift_v = torch.rand([N_rays, num_neighboring_pts, 1]), torch.rand(
        [N_rays, num_neighboring_pts, 1])  # uniform distribution of [0,1)
    shift_u = 2 * (shift_u - 0.5)  # mapping to [-1, 1)
    shift_v = 2 * (shift_v - 0.5)

    # - avoid sample points which are too close to center point
    shift_uv = torch.cat([(shift_u * patch_radius_max) * unit_u, (shift_v * patch_radius_max) * unit_v],
                         dim=-1)  # [N_rays, num_npts, 2]
    neighboring_pts_uv = ndc_uv_center + shift_uv  # [N_rays, num_npts, 2]

    sampled_pts_uv = torch.cat([ndc_uv_center, neighboring_pts_uv], dim=1)  # concat the center point

    # sample the gts
    color = F.grid_sample(image[None, :, :, :], sampled_pts_uv[None, :, :, :], mode='bilinear',
                          align_corners=True)[0]  # [3, N_rays, num_npts]
    depth = F.grid_sample(depth[None, None, :, :], sampled_pts_uv[None, :, :, :], mode='bilinear',
                          align_corners=True)[0]  # [1, N_rays, num_npts]

    mask = F.grid_sample(mask[None, None, :, :].to(torch.float32), sampled_pts_uv[None, :, :, :], mode='nearest',
                         align_corners=True).to(torch.int64)[0]  # [1, N_rays, num_npts]

    intrinsic_inv = torch.inverse(intrinsic)

    sampled_pts_uv = sampled_pts_uv.view(N_rays * (1 + num_neighboring_pts), 2)
    color = color.permute(1, 2, 0).contiguous().view(N_rays * (1 + num_neighboring_pts), 3)
    depth = depth.permute(1, 2, 0).contiguous().view(N_rays * (1 + num_neighboring_pts), 1)
    mask = mask.permute(1, 2, 0).contiguous().view(N_rays * (1 + num_neighboring_pts), 1)

    pixels_x = (sampled_pts_uv[:, 0] + 1) * (W - 1) / 2
    pixels_y = (sampled_pts_uv[:, 1] + 1) * (H - 1) / 2
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(device)  # N_rays*num_pts, 3
    p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # N_rays*num_pts, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_rays*num_pts, 3
    rays_v = torch.matmul(c2w[None, :3, :3], rays_v[:, :, None]).squeeze()  # N_rays*num_pts, 3
    rays_o = c2w[None, :3, 3].expand(rays_v.shape)  # N_rays*num_pts, 3

    sample = {
        'rays_o': rays_o,
        'rays_v': rays_v,
        'rays_ndc_uv': sampled_pts_uv,
        'rays_color': color,
        'rays_depth': depth,
        'rays_mask': mask,
        # 'rays_norm_XYZ_cam': p  # - XYZ_cam, before multiply depth
    }

    return sample


def gen_random_rays_from_batch_images(H, W, N_rays, images, intrinsics, c2ws, depths=None, masks=None):
    """

    :param H:
    :param W:
    :param N_rays:
    :param images: [B,3,H,W]
    :param intrinsics: [B, 3, 3]
    :param c2ws: [B, 4, 4]
    :param depths: [B,H,W]
    :param masks: [B,H,W]
    :return:
    """
    assert len(images.shape) == 4

    rays_o = []
    rays_v = []
    rays_color = []
    rays_depth = []
    rays_mask = []
    for i in range(images.shape[0]):
        sample = gen_random_rays_from_single_image(H, W, N_rays, images[i], intrinsics[i], c2ws[i],
                                                   depth=depths[i] if depths is not None else None,
                                                   mask=masks[i] if masks is not None else None)
        rays_o.append(sample['rays_o'])
        rays_v.append(sample['rays_v'])
        rays_color.append(sample['rays_color'])
        if depths is not None:
            rays_depth.append(sample['rays_depth'])
        if masks is not None:
            rays_mask.append(sample['rays_mask'])

    sample = {
        'rays_o': torch.stack(rays_o, dim=0),  # [batch, N_rays, 3]
        'rays_v': torch.stack(rays_v, dim=0),
        'rays_color': torch.stack(rays_color, dim=0),
        'rays_depth': torch.stack(rays_depth, dim=0) if depths is not None else None,
        'rays_mask': torch.stack(rays_mask, dim=0) if masks is not None else None
    }
    return sample


from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def gen_rays_between(c2w_0, c2w_1, intrinsic, ratio, H, W, resolution_level=1):
    device = c2w_0.device

    l = resolution_level
    tx = torch.linspace(0, W - 1, W // l)
    ty = torch.linspace(0, H - 1, H // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(device)  # W, H, 3

    intrinsic_inv = torch.inverse(intrinsic[:3, :3])
    p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    trans = c2w_0[:3, 3] * (1.0 - ratio) + c2w_1[:3, 3] * ratio

    pose_0 = c2w_0.detach().cpu().numpy()
    pose_1 = c2w_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    key_rots = [rot_0, rot_1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)

    c2w = torch.from_numpy(pose).to(device)
    rot = torch.from_numpy(pose[:3, :3]).cuda()
    trans = torch.from_numpy(pose[:3, 3]).cuda()
    rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
    return c2w, rays_o.transpose(0, 1).contiguous().view(-1, 3), rays_v.transpose(0, 1).contiguous().view(-1, 3)
