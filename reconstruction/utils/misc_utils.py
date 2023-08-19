import os, torch, cv2, re
import numpy as np

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr2 = lambda x: -10. * np.log(x) / np.log(10.)


def get_psnr(imgs_pred, imgs_gt):
    psnrs = []
    for (img, tar) in zip(imgs_pred, imgs_gt):
        psnrs.append(mse2psnr2(np.mean((img - tar.cpu().numpy()) ** 2)))
    return np.array(psnrs)


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def abs_error_numpy(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return np.abs(depth_pred - depth_gt)


def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    err = depth_pred - depth_gt
    return np.abs(err) if type(depth_pred) is np.ndarray else err.abs()


def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.astype('float') if type(depth_pred) is np.ndarray else acc_mask.float()


def to_tensor_cuda(data, device, filter):
    for item in data.keys():

        if item in filter:
            continue

        if type(data[item]) is np.ndarray:
            data[item] = torch.tensor(data[item], dtype=torch.float32, device=device)
        else:
            data[item] = data[item].float().to(device)
    return data


def to_cuda(data, device, filter):
    for item in data.keys():
        if item in filter:
            continue

        data[item] = data[item].float().to(device)
    return data


def tensor_unsqueeze(data, filter):
    for item in data.keys():
        if item in filter:
            continue

        data[item] = data[item][None]
    return data


def filter_keys(dict):
    dict.pop('N_samples')
    if 'ndc' in dict.keys():
        dict.pop('ndc')
    if 'lindisp' in dict.keys():
        dict.pop('lindisp')
    return dict


def sub_selete_data(data_batch, device, idx, filtKey=[],
                    filtIndex=['view_ids_all', 'c2ws_all', 'scan', 'bbox', 'w2ref', 'ref2w', 'light_id', 'ckpt',
                               'idx']):
    data_sub_selete = {}
    for item in data_batch.keys():
        data_sub_selete[item] = data_batch[item][:, idx].float() if (
                item not in filtIndex and torch.is_tensor(item) and item.dim() > 2) else data_batch[item].float()
        if not data_sub_selete[item].is_cuda:
            data_sub_selete[item] = data_sub_selete[item].to(device)
    return data_sub_selete


def detach_data(dictionary):
    dictionary_new = {}
    for key in dictionary.keys():
        dictionary_new[key] = dictionary[key].detach().clone()
    return dictionary_new


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


# from warmup_scheduler import GradualWarmupScheduler
def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)

    else:
        raise ValueError('scheduler not recognized!')

    # if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
    #     scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
    #                                        total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)
    return scheduler


####  pairing  ####
def get_nearest_pose_ids(tar_pose, ref_poses, num_select):
    '''
    Args:
        tar_pose: target pose [N, 4, 4]
        ref_poses: reference poses [M, 4, 4]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''

    dists = np.linalg.norm(tar_pose[:, None, :3, 3] - ref_poses[None, :, :3, 3], axis=-1)

    sorted_ids = np.argsort(dists, axis=-1)
    selected_ids = sorted_ids[:, :num_select]
    return selected_ids
