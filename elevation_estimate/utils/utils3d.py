import numpy as np
import torch


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.concatenate((pts, np.ones([*pts.shape[:-1], 1], dtype=np.float32)), -1)
    else:
        ones = torch.ones([*pts.shape[:-1], 1], dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=-1)
    return pts_hom


def hom_to_cart(pts):
    return pts[..., :-1] / pts[..., -1:]


def canonical_to_camera(pts, pose):
    pts = cart_to_hom(pts)
    pts = pts @ pose.transpose(-1, -2)
    pts = hom_to_cart(pts)
    return pts


def rect_to_img(K, pts_rect):
    from dl_ext.vision_ext.datasets.kitti.structures import Calibration
    pts_2d_hom = pts_rect @ K.t()
    pts_img = Calibration.hom_to_cart(pts_2d_hom)
    return pts_img


def calc_pose(phis, thetas, size, radius=1.2):
    import torch
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    device = torch.device('cuda')
    thetas = torch.FloatTensor(thetas).to(device)
    phis = torch.FloatTensor(phis).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = normalize(centers).squeeze(0)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses
