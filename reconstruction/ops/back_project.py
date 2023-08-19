import torch
from torch.nn.functional import grid_sample


def back_project_sparse_type(coords, origin, voxel_size, feats, KRcam, sizeH=None, sizeW=None, only_mask=False,
                             with_proj_z=False):
    # - modified version from NeuRecon
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, num_of_views, c)
    :return: mask_volume_all: indicate the voxel of sampled feature volume is valid or not
    dim: (num of voxels, num_of_views)
    '''
    n_views, bs, c, h, w = feats.shape
    device = feats.device

    if sizeH is None:
        sizeH, sizeW = h, w  # - if the KRcam is not suitable for the current feats

    feature_volume_all = torch.zeros(coords.shape[0], n_views, c).to(device)
    mask_volume_all = torch.zeros([coords.shape[0], n_views], dtype=torch.int32).to(device)
    # import ipdb; ipdb.set_trace()
    for batch in range(bs):
        # import ipdb; ipdb.set_trace()
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).to(device)], dim=1)

        # Project grid
        im_p = proj_batch @ rs_grid  # - transform world pts to image UV space
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]

        im_z[im_z >= 0] = im_z[im_z >= 0].clamp(min=1e-6)

        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (sizeW - 1) - 1, 2 * im_y / (sizeH - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        mask = mask.view(n_views, -1)
        mask = mask.permute(1, 0).contiguous()  # [num_pts, nviews]

        mask_volume_all[batch_ind] = mask.to(torch.int32)

        if only_mask:
            return mask_volume_all

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)
        # if features.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()
        features = features.view(n_views, c, -1)
        features = features.permute(2, 0, 1).contiguous()  # [num_pts, nviews, c]

        feature_volume_all[batch_ind] = features

        if with_proj_z:
            im_z = im_z.view(n_views, 1, -1).permute(2, 0, 1).contiguous()  # [num_pts, nviews, 1]
            return feature_volume_all, mask_volume_all, im_z
    # if feature_volume_all.isnan().sum() > 0:
    #     import ipdb; ipdb.set_trace()
    return feature_volume_all, mask_volume_all


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, sizeH=None, sizeW=None, with_depth=False):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 3]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, H, W, 2]
    """
    b, _, h, w = cam_coords.size()
    if sizeH is None:
        sizeH = h
        sizeW = w

    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (sizeW - 1) - 1  # Normalized, -1 if on extreme left,
    # 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (sizeH - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    if with_depth:
        pixel_coords = torch.stack([X_norm, Y_norm, Z], dim=2)  # [B, H*W, 3]
        return pixel_coords.view(b, h, w, 3)
    else:
        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.view(b, h, w, 2)


# * have already checked, should check whether proj_matrix is for right coordinate system and resolution
def back_project_dense_type(coords, origin, voxel_size, feats, proj_matrix, sizeH=None, sizeW=None):
    '''
    Unproject the image fetures to form a 3D (dense) feature volume

    :param coords: coordinates of voxels,
    dim: (batch, nviews, 3, X,Y,Z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (batch size, num of views,  C, H, W)
    :param proj_matrix: projection matrix
    dim: (batch size, num of views, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (batch, nviews, C, X,Y,Z)
    :return: count: number of times each voxel can be seen
    dim: (batch, nviews, 1, X,Y,Z)
    '''

    batch, nviews, _, wX, wY, wZ = coords.shape

    if sizeH is None:
        sizeH, sizeW = feats.shape[-2:]
    proj_matrix = proj_matrix.view(batch * nviews, *proj_matrix.shape[2:])

    coords_wrd = coords * voxel_size + origin.view(batch, 1, 3, 1, 1, 1)
    coords_wrd = coords_wrd.view(batch * nviews, 3, wX * wY * wZ, 1)  # (b*nviews,3,wX*wY*wZ, 1)

    pixel_grids = cam2pixel(coords_wrd, proj_matrix[:, :3, :3], proj_matrix[:, :3, 3:],
                            'zeros', sizeH=sizeH, sizeW=sizeW)  # (b*nviews,wX*wY*wZ, 2)
    pixel_grids = pixel_grids.view(batch * nviews, 1, wX * wY * wZ, 2)

    feats = feats.view(batch * nviews, *feats.shape[2:])  # (b*nviews,c,h,w)

    ones = torch.ones((batch * nviews, 1, *feats.shape[2:])).to(feats.dtype).to(feats.device)

    features_volume = torch.nn.functional.grid_sample(feats, pixel_grids, padding_mode='zeros', align_corners=True)
    counts_volume = torch.nn.functional.grid_sample(ones, pixel_grids, padding_mode='zeros', align_corners=True)

    features_volume = features_volume.view(batch, nviews, -1, wX, wY, wZ)  # (batch, nviews, C, X,Y,Z)
    counts_volume = counts_volume.view(batch, nviews, -1, wX, wY, wZ)
    return features_volume, counts_volume

