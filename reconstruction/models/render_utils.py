import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.back_project import cam2pixel


def sample_pdf(bins, weights, n_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''
    device = weights.device

    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)

    # if bins.shape[1] != weights.shape[1]:  # - minor modification, add this constraint
    #     cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Invert CDF
    u = u.contiguous()
    # inds = searchsorted(cdf, u, side='right')
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    # pdb.set_trace()
    return samples


def sample_ptsFeatures_from_featureVolume(pts, featureVolume, vol_dims=None, partial_vol_origin=None, vol_size=None):
    """
    sample feature of pts_wrd from featureVolume, all in world space
    :param pts: [N_rays, n_samples, 3]
    :param featureVolume: [C,wX,wY,wZ]
    :param vol_dims: [3] "3" for dimX, dimY, dimZ
    :param partial_vol_origin: [3]
    :return: pts_feature: [N_rays, n_samples, C]
    :return: valid_mask: [N_rays]
    """

    N_rays, n_samples, _ = pts.shape

    if vol_dims is None:
        pts_normalized = pts
    else:
        # normalized to (-1, 1)
        pts_normalized = 2 * (pts - partial_vol_origin[None, None, :]) / (vol_size * (vol_dims[None, None, :] - 1)) - 1

    valid_mask = (torch.abs(pts_normalized[:, :, 0]) < 1.0) & (
            torch.abs(pts_normalized[:, :, 1]) < 1.0) & (
                         torch.abs(pts_normalized[:, :, 2]) < 1.0)  # (N_rays, n_samples)

    pts_normalized = torch.flip(pts_normalized, dims=[-1])  # ! reverse the xyz for grid_sample

    # ! checked grid_sample, (x,y,z) is for (D,H,W), reverse for (W,H,D)
    pts_feature = F.grid_sample(featureVolume[None, :, :, :, :], pts_normalized[None, None, :, :, :],
                                padding_mode='zeros',
                                align_corners=True).view(-1, N_rays, n_samples)  # [C, N_rays, n_samples]

    pts_feature = pts_feature.permute(1, 2, 0)  # [N_rays, n_samples, C]
    return pts_feature, valid_mask


def sample_ptsFeatures_from_featureMaps(pts, featureMaps, w2cs, intrinsics, WH, proj_matrix=None, return_mask=False):
    """
    sample features of pts from 2d feature maps
    :param pts: [N_rays, N_samples, 3]
    :param featureMaps: [N_views, C, H, W]
    :param w2cs: [N_views, 4, 4]
    :param intrinsics: [N_views, 3, 3]
    :param proj_matrix: [N_views, 4, 4]
    :param HW:
    :return:
    """
    # normalized to (-1, 1)
    N_rays, n_samples, _ = pts.shape
    N_views = featureMaps.shape[0]

    if proj_matrix is None:
        proj_matrix = torch.matmul(intrinsics, w2cs[:, :3, :])

    pts = pts.permute(2, 0, 1).contiguous().view(1, 3, N_rays, n_samples).repeat(N_views, 1, 1, 1)
    pixel_grids = cam2pixel(pts, proj_matrix[:, :3, :3], proj_matrix[:, :3, 3:],
                            'zeros', sizeH=WH[1], sizeW=WH[0])  # (nviews, N_rays, n_samples, 2)

    valid_mask = (torch.abs(pixel_grids[:, :, :, 0]) < 1.0) & (
            torch.abs(pixel_grids[:, :, :, 1]) < 1.00)  # (nviews, N_rays, n_samples)

    pts_feature = F.grid_sample(featureMaps, pixel_grids,
                                padding_mode='zeros',
                                align_corners=True)  # [N_views, C, N_rays, n_samples]

    if return_mask:
        return pts_feature, valid_mask
    else:
        return pts_feature
