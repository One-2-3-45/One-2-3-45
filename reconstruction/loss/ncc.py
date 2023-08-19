import torch
import torch.nn.functional as F
import numpy as np
from math import exp, sqrt


class NCC(torch.nn.Module):
    def __init__(self, h_patch_size, mode='rgb'):
        super(NCC, self).__init__()
        self.window_size = 2 * h_patch_size + 1
        self.mode = mode  # 'rgb' or 'gray'
        self.channel = 3
        self.register_buffer("window", create_window(self.window_size, self.channel))

    def forward(self, img_pred, img_gt):
        """
        :param img_pred: [Npx, nviews, npatch, c]
        :param img_gt: [Npx, npatch, c]
        :return:
        """
        ntotpx, nviews, npatch, channels = img_pred.shape

        patch_size = int(sqrt(npatch))
        patch_img_pred = img_pred.reshape(ntotpx, nviews, patch_size, patch_size, channels).permute(0, 1, 4, 2,
                                                                                                    3).contiguous()
        patch_img_gt = img_gt.reshape(ntotpx, patch_size, patch_size, channels).permute(0, 3, 1, 2)

        return _ncc(patch_img_pred, patch_img_gt, self.window, self.channel)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, std=1.5):
    _1D_window = gaussian(window_size, std).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ncc(pred, gt, window, channel):
    ntotpx, nviews, nc, h, w = pred.shape
    flat_pred = pred.view(-1, nc, h, w)
    mu1 = F.conv2d(flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc)
    mu2 = F.conv2d(gt, window, padding=0, groups=channel).view(ntotpx, nc)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2).unsqueeze(1)  # (ntotpx, 1, nc)

    sigma1_sq = F.conv2d(flat_pred * flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=0, groups=channel).view(ntotpx, 1, 3) - mu2_sq

    sigma1 = torch.sqrt(sigma1_sq + 1e-4)
    sigma2 = torch.sqrt(sigma2_sq + 1e-4)

    pred_norm = (pred - mu1[:, :, :, None, None]) / (sigma1[:, :, :, None, None] + 1e-8)  # [ntotpx, nviews, nc, h, w]
    gt_norm = (gt[:, None, :, :, :] - mu2[:, None, :, None, None]) / (
            sigma2[:, :, :, None, None] + 1e-8)  # ntotpx, nc, h, w

    ncc = F.conv2d((pred_norm * gt_norm).view(-1, nc, h, w), window, padding=0, groups=channel).view(
        ntotpx, nviews, nc)

    return torch.mean(ncc, dim=2)
