import numpy as np
import torch


def rigid_transform(xyz, transform):
    """Applies a rigid transform (c2w) to an (N, 3) pointcloud.
    """
    device = xyz.device
    xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1)).to(device)], dim=1)  # (N, 4)
    xyz_t_h = (transform @ xyz_h.T).T  # * checked: the same with the below

    return xyz_t_h[:, :3]


def get_view_frustum(min_depth, max_depth, size, cam_intr, c2w):
    """Get corners of 3D camera view frustum of depth image
    """
    device = cam_intr.device
    im_h, im_w = size
    im_h = int(im_h)
    im_w = int(im_w)
    view_frust_pts = torch.stack([
        (torch.tensor([0, 0, im_w, im_w, 0, 0, im_w, im_w]).to(device) - cam_intr[0, 2]) * torch.tensor(
            [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(device) /
        cam_intr[0, 0],
        (torch.tensor([0, im_h, 0, im_h, 0, im_h, 0, im_h]).to(device) - cam_intr[1, 2]) * torch.tensor(
            [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(device) /
        cam_intr[1, 1],
        torch.tensor([min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(
            device)
    ])
    view_frust_pts = view_frust_pts.type(torch.float32)
    c2w = c2w.type(torch.float32)
    view_frust_pts = rigid_transform(view_frust_pts.T, c2w).T
    return view_frust_pts


def set_pixel_coords(h, w):
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type(torch.float32)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type(torch.float32)  # [1, H, W]
    ones = torch.ones(1, h, w).type(torch.float32)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    return pixel_coords


def get_boundingbox(img_hw, intrinsics, extrinsics, near_fars):
    """
    # get the minimum bounding box of all visual hulls
    :param img_hw:
    :param intrinsics:
    :param extrinsics:
    :param near_fars:
    :return:
    """

    bnds = torch.zeros((3, 2))
    bnds[:, 0] = np.inf
    bnds[:, 1] = -np.inf

    if isinstance(intrinsics, list):
        num = len(intrinsics)
    else:
        num = intrinsics.shape[0]
    # print("num: ", num)
    view_frust_pts_list = []
    for i in range(num):
        if not isinstance(intrinsics[i], torch.Tensor):
            cam_intr = torch.tensor(intrinsics[i])
            w2c = torch.tensor(extrinsics[i])
            c2w = torch.inverse(w2c)
        else:
            cam_intr = intrinsics[i]
            w2c = extrinsics[i]
            c2w = torch.inverse(w2c)
        min_depth, max_depth = near_fars[i][0], near_fars[i][1]
        # todo: check the coresponding points are matched

        view_frust_pts = get_view_frustum(min_depth, max_depth, img_hw, cam_intr, c2w)
        bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
        bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])
        view_frust_pts_list.append(view_frust_pts)
    all_view_frust_pts = torch.cat(view_frust_pts_list, dim=1)

    # print("all_view_frust_pts: ", all_view_frust_pts.shape)
    # distance = torch.norm(all_view_frust_pts, dim=0)
    # print("distance: ", distance)

    # print("all_view_frust_pts_z: ", all_view_frust_pts[2, :])

    center = torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2,
                           (bnds[2, 1] + bnds[2, 0]) / 2))

    lengths = bnds[:, 1] - bnds[:, 0]

    max_length, _ = torch.max(lengths, dim=0)
    radius = max_length / 2

    # print("radius: ", radius)
    return center, radius, bnds
