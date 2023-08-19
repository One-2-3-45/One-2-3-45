"""
pytorch grid_sample doesn't support second-order derivative
implement custom version
"""

import torch
import torch.nn.functional as F
import numpy as np


def grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


# - checked for correctness
def grid_sample_3d(volume, optical):
    """
    bilinear sampling cannot guarantee continuous first-order gradient
    mimic pytorch grid_sample function
    The 8 corner points of a volume noted as: 4 points (front view); 4 points (back view)
    fnw (front north west) point
    bse (back south east) point
    :param volume: [B, C, X, Y, Z]
    :param optical: [B, x, y, z, 3]
    :return:
    """
    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    mask_x = (ix > 0) & (ix < IW)
    mask_y = (iy > 0) & (iy < IH)
    mask_z = (iz > 0) & (iz < ID)

    mask = mask_x & mask_y & mask_z  # [B, x, y, z]
    mask = mask[:, None, :, :, :].repeat(1, C, 1, 1, 1)  # [B, C, x, y, z]

    with torch.no_grad():
        # back north west
        ix_bnw = torch.floor(ix)
        iy_bnw = torch.floor(iy)
        iz_bnw = torch.floor(iz)

        ix_bne = ix_bnw + 1
        iy_bne = iy_bnw
        iz_bne = iz_bnw

        ix_bsw = ix_bnw
        iy_bsw = iy_bnw + 1
        iz_bsw = iz_bnw

        ix_bse = ix_bnw + 1
        iy_bse = iy_bnw + 1
        iz_bse = iz_bnw

        # front view
        ix_fnw = ix_bnw
        iy_fnw = iy_bnw
        iz_fnw = iz_bnw + 1

        ix_fne = ix_bnw + 1
        iy_fne = iy_bnw
        iz_fne = iz_bnw + 1

        ix_fsw = ix_bnw
        iy_fsw = iy_bnw + 1
        iz_fsw = iz_bnw + 1

        ix_fse = ix_bnw + 1
        iy_fse = iy_bnw + 1
        iz_fse = iz_bnw + 1

    # back view
    bnw = (ix_fse - ix) * (iy_fse - iy) * (iz_fse - iz)  # smaller volume, larger weight
    bne = (ix - ix_fsw) * (iy_fsw - iy) * (iz_fsw - iz)
    bsw = (ix_fne - ix) * (iy - iy_fne) * (iz_fne - iz)
    bse = (ix - ix_fnw) * (iy - iy_fnw) * (iz_fnw - iz)

    # front view
    fnw = (ix_bse - ix) * (iy_bse - iy) * (iz - iz_bse)  # smaller volume, larger weight
    fne = (ix - ix_bsw) * (iy_bsw - iy) * (iz - iz_bsw)
    fsw = (ix_bne - ix) * (iy - iy_bne) * (iz - iz_bne)
    fse = (ix - ix_bnw) * (iy - iy_bnw) * (iz - iz_bnw)

    with torch.no_grad():
        # back view
        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        # front view
        torch.clamp(ix_fnw, 0, IW - 1, out=ix_fnw)
        torch.clamp(iy_fnw, 0, IH - 1, out=iy_fnw)
        torch.clamp(iz_fnw, 0, ID - 1, out=iz_fnw)

        torch.clamp(ix_fne, 0, IW - 1, out=ix_fne)
        torch.clamp(iy_fne, 0, IH - 1, out=iy_fne)
        torch.clamp(iz_fne, 0, ID - 1, out=iz_fne)

        torch.clamp(ix_fsw, 0, IW - 1, out=ix_fsw)
        torch.clamp(iy_fsw, 0, IH - 1, out=iy_fsw)
        torch.clamp(iz_fsw, 0, ID - 1, out=iz_fsw)

        torch.clamp(ix_fse, 0, IW - 1, out=ix_fse)
        torch.clamp(iy_fse, 0, IH - 1, out=iy_fse)
        torch.clamp(iz_fse, 0, ID - 1, out=iz_fse)

    # xxx = volume[:, :, iz_bnw.long(), iy_bnw.long(), ix_bnw.long()]
    volume = volume.view(N, C, ID * IH * IW)
    # yyy = volume[:, :, (iz_bnw * ID + iy_bnw * IW + ix_bnw).long()]

    # back view
    bnw_val = torch.gather(volume, 2,
                           (iz_bnw * ID ** 2 + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(volume, 2,
                           (iz_bne * ID ** 2 + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(volume, 2,
                           (iz_bsw * ID ** 2 + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(volume, 2,
                           (iz_bse * ID ** 2 + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    # front view
    fnw_val = torch.gather(volume, 2,
                           (iz_fnw * ID ** 2 + iy_fnw * IW + ix_fnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fne_val = torch.gather(volume, 2,
                           (iz_fne * ID ** 2 + iy_fne * IW + ix_fne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fsw_val = torch.gather(volume, 2,
                           (iz_fsw * ID ** 2 + iy_fsw * IW + ix_fsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fse_val = torch.gather(volume, 2,
                           (iz_fse * ID ** 2 + iy_fse * IW + ix_fse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (
        # back
            bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
            bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
            bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
            bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W) +
            # front
            fnw_val.view(N, C, D, H, W) * fnw.view(N, 1, D, H, W) +
            fne_val.view(N, C, D, H, W) * fne.view(N, 1, D, H, W) +
            fsw_val.view(N, C, D, H, W) * fsw.view(N, 1, D, H, W) +
            fse_val.view(N, C, D, H, W) * fse.view(N, 1, D, H, W)

    )

    # * zero padding
    out_val = torch.where(mask, out_val, torch.zeros_like(out_val).float().to(out_val.device))

    return out_val


# Interpolation kernel
def get_weight(s, a=-0.5):
    mask_0 = (torch.abs(s) >= 0) & (torch.abs(s) <= 1)
    mask_1 = (torch.abs(s) > 1) & (torch.abs(s) <= 2)
    mask_2 = torch.abs(s) > 2

    weight = torch.zeros_like(s).to(s.device)
    weight = torch.where(mask_0, (a + 2) * (torch.abs(s) ** 3) - (a + 3) * (torch.abs(s) ** 2) + 1, weight)
    weight = torch.where(mask_1,
                         a * (torch.abs(s) ** 3) - (5 * a) * (torch.abs(s) ** 2) + (8 * a) * torch.abs(s) - 4 * a,
                         weight)

    # if (torch.abs(s) >= 0) & (torch.abs(s) <= 1):
    #     return (a + 2) * (torch.abs(s) ** 3) - (a + 3) * (torch.abs(s) ** 2) + 1
    #
    # elif (torch.abs(s) > 1) & (torch.abs(s) <= 2):
    #     return a * (torch.abs(s) ** 3) - (5 * a) * (torch.abs(s) ** 2) + (8 * a) * torch.abs(s) - 4 * a
    # return 0

    return weight


def cubic_interpolate(p, x):
    """
    one dimensional cubic interpolation
    :param p: [N, 4]  (4) should be in order
    :param x: [N]
    :return:
    """
    return p[:, 1] + 0.5 * x * (p[:, 2] - p[:, 0] + x * (
            2.0 * p[:, 0] - 5.0 * p[:, 1] + 4.0 * p[:, 2] - p[:, 3] + x * (
            3.0 * (p[:, 1] - p[:, 2]) + p[:, 3] - p[:, 0])))


def bicubic_interpolate(p, x, y, if_batch=True):
    """
    two dimensional cubic interpolation
    :param p: [N, 4, 4]
    :param x: [N]
    :param y: [N]
    :return:
    """
    num = p.shape[0]

    if not if_batch:
        arr0 = cubic_interpolate(p[:, 0, :], x)  # [N]
        arr1 = cubic_interpolate(p[:, 1, :], x)
        arr2 = cubic_interpolate(p[:, 2, :], x)
        arr3 = cubic_interpolate(p[:, 3, :], x)
        return cubic_interpolate(torch.stack([arr0, arr1, arr2, arr3], dim=-1), y)  # [N]
    else:
        x = x[:, None].repeat(1, 4).view(-1)
        p = p.contiguous().view(num * 4, 4)
        arr = cubic_interpolate(p, x)
        arr = arr.view(num, 4)

        return cubic_interpolate(arr, y)


def tricubic_interpolate(p, x, y, z):
    """
    three dimensional cubic interpolation
    :param p: [N,4,4,4]
    :param x: [N]
    :param y: [N]
    :param z: [N]
    :return:
    """
    num = p.shape[0]

    arr0 = bicubic_interpolate(p[:, 0, :, :], x, y)  # [N]
    arr1 = bicubic_interpolate(p[:, 1, :, :], x, y)
    arr2 = bicubic_interpolate(p[:, 2, :, :], x, y)
    arr3 = bicubic_interpolate(p[:, 3, :, :], x, y)

    return cubic_interpolate(torch.stack([arr0, arr1, arr2, arr3], dim=-1), z)  # [N]


def cubic_interpolate_batch(p, x):
    """
    one dimensional cubic interpolation
    :param p: [B, N, 4]  (4) should be in order
    :param x: [B, N]
    :return:
    """
    return p[:, :, 1] + 0.5 * x * (p[:, :, 2] - p[:, :, 0] + x * (
            2.0 * p[:, :, 0] - 5.0 * p[:, :, 1] + 4.0 * p[:, :, 2] - p[:, :, 3] + x * (
            3.0 * (p[:, :, 1] - p[:, :, 2]) + p[:, :, 3] - p[:, :, 0])))


def bicubic_interpolate_batch(p, x, y):
    """
    two dimensional cubic interpolation
    :param p: [B, N, 4, 4]
    :param x: [B, N]
    :param y: [B, N]
    :return:
    """
    B, N, _, _ = p.shape

    x = x[:, :, None].repeat(1, 1, 4).view(B, N * 4)  # [B, N*4]
    arr = cubic_interpolate_batch(p.contiguous().view(B, N * 4, 4), x)
    arr = arr.view(B, N, 4)
    return cubic_interpolate_batch(arr, y)  # [B, N]


# * batch version cannot speed up training
def tricubic_interpolate_batch(p, x, y, z):
    """
    three dimensional cubic interpolation
    :param p: [N,4,4,4]
    :param x: [N]
    :param y: [N]
    :param z: [N]
    :return:
    """
    N = p.shape[0]

    x = x[None, :].repeat(4, 1)
    y = y[None, :].repeat(4, 1)

    p = p.permute(1, 0, 2, 3).contiguous()

    arr = bicubic_interpolate_batch(p[:, :, :, :], x, y)  # [4, N]

    arr = arr.permute(1, 0).contiguous()  # [N, 4]

    return cubic_interpolate(arr, z)  # [N]


def tricubic_sample_3d(volume, optical):
    """
    tricubic sampling; can guarantee continuous gradient  (interpolation border)
    :param volume: [B, C, ID, IH, IW]
    :param optical: [B, D, H, W, 3]
    :param sample_num:
    :return:
    """

    @torch.no_grad()
    def get_shifts(x):
        x1 = -1 * (1 + x - torch.floor(x))
        x2 = -1 * (x - torch.floor(x))
        x3 = torch.floor(x) + 1 - x
        x4 = torch.floor(x) + 2 - x

        return torch.stack([x1, x2, x3, x4], dim=-1)  # (B,d,h,w,4)

    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = optical.shape

    device = volume.device

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)  # (B,d,h,w)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    ix = ix.view(-1)
    iy = iy.view(-1)
    iz = iz.view(-1)

    with torch.no_grad():
        shifts_x = get_shifts(ix).view(-1, 4)  # (B*d*h*w,4)
        shifts_y = get_shifts(iy).view(-1, 4)
        shifts_z = get_shifts(iz).view(-1, 4)

        perm_weights = torch.ones([N * D * H * W, 4 * 4 * 4]).long().to(device)
        perm = torch.cumsum(perm_weights, dim=-1) - 1  # (B*d*h*w,64)

        perm_z = perm // 16  # [N*D*H*W, num]
        perm_y = (perm - perm_z * 16) // 4
        perm_x = (perm - perm_z * 16 - perm_y * 4)

        shifts_x = torch.gather(shifts_x, 1, perm_x)  # [N*D*H*W, num]
        shifts_y = torch.gather(shifts_y, 1, perm_y)
        shifts_z = torch.gather(shifts_z, 1, perm_z)

        ix_target = (ix[:, None] + shifts_x).long()  # [N*D*H*W, num]
        iy_target = (iy[:, None] + shifts_y).long()
        iz_target = (iz[:, None] + shifts_z).long()

        torch.clamp(ix_target, 0, IW - 1, out=ix_target)
        torch.clamp(iy_target, 0, IH - 1, out=iy_target)
        torch.clamp(iz_target, 0, ID - 1, out=iz_target)

    local_dist_x = ix - ix_target[:, 1]  # ! attention here is [:, 1]
    local_dist_y = iy - iy_target[:, 1 + 4]
    local_dist_z = iz - iz_target[:, 1 + 16]

    local_dist_x = local_dist_x.view(N, 1, D * H * W).repeat(1, C, 1).view(-1)
    local_dist_y = local_dist_y.view(N, 1, D * H * W).repeat(1, C, 1).view(-1)
    local_dist_z = local_dist_z.view(N, 1, D * H * W).repeat(1, C, 1).view(-1)

    # ! attention: IW is correct
    idx_target = iz_target * ID ** 2 + iy_target * IW + ix_target  # [N*D*H*W, num]

    volume = volume.view(N, C, ID * IH * IW)

    out = torch.gather(volume, 2,
                       idx_target.view(N, 1, D * H * W * 64).repeat(1, C, 1))
    out = out.view(N * C * D * H * W, 4, 4, 4)

    # - tricubic_interpolate() is a bit faster than tricubic_interpolate_batch()
    final = tricubic_interpolate(out, local_dist_x, local_dist_y, local_dist_z).view(N, C, D, H, W)  # [N,C,D,H,W]

    return final



if __name__ == "__main__":
    # image = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).view(1, 3, 1, 3)
    #
    # optical = torch.Tensor([0.9, 0.5, 0.6, -0.7]).view(1, 1, 2, 2)
    #
    # print(grid_sample_2d(image, optical))
    #
    # print(F.grid_sample(image, optical, padding_mode='border', align_corners=True))

    from ops.generate_grids import generate_grid

    p = torch.tensor([x for x in range(4)]).view(1, 4).float()

    v = cubic_interpolate(p, torch.tensor([0.5]).view(1))
    # v = bicubic_interpolate(p, torch.tensor([2/3]).view(1) , torch.tensor([2/3]).view(1))

    vsize = 9
    volume = generate_grid([vsize, vsize, vsize], 1)  # [1,3,10,10,10]
    # volume = torch.tensor([x for x in range(1000)]).view(1, 1, 10, 10, 10).float()
    X, Y, Z = 0, 0, 6
    x = 2 * X / (vsize - 1) - 1
    y = 2 * Y / (vsize - 1) - 1
    z = 2 * Z / (vsize - 1) - 1

    # print(volume[:, :, Z, Y, X])

    # volume = volume.view(1, 3, -1)
    # xx = volume[:, :, Z * 9*9 + Y * 9 + X]

    optical = torch.Tensor([-0.6, -0.7, 0.5, 0.3, 0.5, 0.5]).view(1, 1, 1, 2, 3)

    print(F.grid_sample(volume, optical, padding_mode='border', align_corners=True))
    print(grid_sample_3d(volume, optical))
    print(tricubic_sample_3d(volume, optical))
    # target, relative_coords = implicit_sample_3d(volume, optical, 1)
    # print(target)
