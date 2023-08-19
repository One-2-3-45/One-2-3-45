import torch
import torch.nn as nn
from loss.ncc import NCC


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class OcclusionColorLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.025, gama=0.01, occlusion_aware=True, weight_thred=[0.6]):
        super(OcclusionColorLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.occlusion_aware = occlusion_aware
        self.eps = 1e-4

        self.weight_thred = weight_thred
        self.adjuster = ParamAdjuster(self.weight_thred, self.beta)

    def forward(self, pred, gt, weight, mask, detach=False, occlusion_aware=True):
        """

        :param pred: [N_pts, 3]
        :param gt: [N_pts, 3]
        :param weight: [N_pts]
        :param mask: [N_pts]
        :return:
        """
        if detach:
            weight = weight.detach()

        error = torch.abs(pred - gt).sum(dim=-1, keepdim=False)  # [N_pts]
        error = error[mask]

        if not (self.occlusion_aware and occlusion_aware):
            return torch.mean(error), torch.mean(error)

        beta = self.adjuster(weight.mean())

        # weight = weight[mask]
        weight = weight.clamp(0.0, 1.0)
        term1 = self.alpha * torch.mean(weight[mask] * error)
        term2 = beta * torch.log(1 - weight + self.eps).mean()
        term3 = self.gama * torch.log(weight + self.eps).mean()

        return term1 + term2 + term3, term1


class OcclusionColorPatchLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.025, gama=0.015,
                 occlusion_aware=True, type='l1', h_patch_size=3, weight_thred=[0.6]):
        super(OcclusionColorPatchLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.occlusion_aware = occlusion_aware
        self.type = type  # 'l1' or 'ncc' loss
        self.ncc = NCC(h_patch_size=h_patch_size)
        self.eps = 1e-4
        self.weight_thred = weight_thred

        self.adjuster = ParamAdjuster(self.weight_thred, self.beta)

        print("type {} patch_size {}  beta {}  gama {}  weight_thred {}".format(type, h_patch_size, beta, gama,
                                                                                weight_thred))

    def forward(self, pred, gt, weight, mask, penalize_ratio=0.9, detach=False, occlusion_aware=True):
        """

        :param pred: [N_pts, Npx, 3]
        :param gt: [N_pts, Npx, 3]
        :param weight: [N_pts]
        :param mask: [N_pts]
        :return:
        """

        if detach:
            weight = weight.detach()

        if self.type == 'l1':
            error = torch.abs(pred - gt).mean(dim=-1, keepdim=False).sum(dim=-1, keepdim=False)  # [N_pts]
        elif self.type == 'ncc':
            error = 1 - self.ncc(pred[:, None, :, :], gt)[:, 0]  # ncc 1 positive, -1 negative
            error, indices = torch.sort(error)
            mask = torch.index_select(mask, 0, index=indices)
            mask[int(penalize_ratio * mask.shape[0]):] = False  # can help boundaries
        elif self.type == 'ssd':
            error = ((pred - gt) ** 2).mean(dim=-1, keepdim=False).sum(dim=-1, keepdims=False)

        error = error[mask]
        if not (self.occlusion_aware and occlusion_aware):
            return torch.mean(error), torch.mean(error), 0.

        # * weight adjuster
        beta = self.adjuster(weight.mean())

        # weight = weight[mask]
        weight = weight.clamp(0.0, 1.0)

        term1 = self.alpha * torch.mean(weight[mask] * error)
        term2 = beta * torch.log(1 - weight + self.eps).mean()
        term3 = self.gama * torch.log(weight + self.eps).mean()

        return term1 + term2 + term3, term1, beta


class ParamAdjuster(nn.Module):
    def __init__(self, weight_thred, param):
        super(ParamAdjuster, self).__init__()
        self.weight_thred = weight_thred
        self.thred_num = len(weight_thred)
        self.param = param
        self.global_step = 0
        self.statis_window = 100
        self.counter = 0
        self.adjusted = False
        self.adjusted_step = 0
        self.thred_idx = 0

    def reset(self):
        self.counter = 0
        self.adjusted = False

    def adjust(self):
        if (self.counter / self.statis_window) > 0.3:
            self.param = self.param + 0.005
            self.adjusted = True
            self.adjusted_step = self.global_step
            self.thred_idx += 1
            print("adjusted param, now {}".format(self.param))

    def forward(self, weight_mean):
        self.global_step += 1

        if (self.global_step % self.statis_window == 0) and self.adjusted is False:
            self.adjust()
            self.reset()

        if self.thred_idx < self.thred_num:
            if weight_mean < self.weight_thred[self.thred_idx] and (not self.adjusted):
                self.counter += 1

        return self.param
