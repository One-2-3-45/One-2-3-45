import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    def __init__(self, type='l1'):
        super(DepthLoss, self).__init__()
        self.type = type


    def forward(self, depth_pred, depth_gt, mask=None):
            if (depth_gt < 0).sum() > 0:
                # print("no depth loss")
                return torch.tensor(0.0).to(depth_pred.device)
            if mask is not None:
                mask_d = (depth_gt > 0).float()

                mask = mask * mask_d

                mask_sum = mask.sum() + 1e-5
                depth_error = (depth_pred - depth_gt) * mask
                depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth_error.device),
                                    reduction='sum') / mask_sum
            else:
                depth_error = depth_pred - depth_gt
                depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth_error.device),
                                    reduction='mean')
            return depth_loss

def forward(self, depth_pred, depth_gt, mask=None):
        if mask is not None:
            mask_d = (depth_gt > 0).float()

            mask = mask * mask_d

            mask_sum = mask.sum() + 1e-5
            depth_error = (depth_pred - depth_gt) * mask
            depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth_error.device),
                                   reduction='sum') / mask_sum
        else:
            depth_error = depth_pred - depth_gt
            depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error).to(depth_error.device),
                                   reduction='mean')
        return depth_loss

class DepthSmoothLoss(nn.Module):
    def __init__(self):
        super(DepthSmoothLoss, self).__init__()

    def forward(self, disp, img, mask):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        :param disp: [B, 1, H, W]
        :param img: [B, 1, H, W]
        :param mask: [B, 1, H, W]
        :return:
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        grad_disp = (grad_disp_x * mask[:, :, :, :-1]).mean() + (grad_disp_y * mask[:, :, :-1, :]).mean()

        return grad_disp
