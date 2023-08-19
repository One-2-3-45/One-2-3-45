import os
import cv2
import numpy as np
import os.path as osp
import imageio
from copy import deepcopy

import loguru
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from ..loftr import LoFTR, default_cfg
from . import plt_utils
from .plotting import make_matching_figure
from .utils3d import rect_to_img, canonical_to_camera, calc_pose


class ElevEstHelper:
    _feature_matcher = None

    @classmethod
    def get_feature_matcher(cls):
        if cls._feature_matcher is None:
            loguru.logger.info("Loading feature matcher...")
            _default_cfg = deepcopy(default_cfg)
            _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
            matcher = LoFTR(config=_default_cfg)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_path = os.path.join(current_dir, "weights/indoor_ds_new.ckpt")
            if not osp.exists(ckpt_path):
                loguru.logger.info("Downloading feature matcher...")
                os.makedirs("weights", exist_ok=True)
                import gdown
                gdown.cached_download(url="https://drive.google.com/uc?id=19s3QvcCWQ6g-N1PrYlDCg-2mOJZ3kkgS",
                                      path=ckpt_path)
            matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
            matcher = matcher.eval().cuda()
            cls._feature_matcher = matcher
        return cls._feature_matcher


def mask_out_bkgd(img_path, dbg=False):
    img = imageio.imread_v2(img_path)
    if img.shape[-1] == 4:
        fg_mask = img[:, :, :3]
    else:
        loguru.logger.info("Image has no alpha channel, using thresholding to mask out background")
        fg_mask = ~(img > 245).all(axis=-1)
        if dbg:
            plt.imshow(plt_utils.vis_mask(img, fg_mask.astype(np.uint8), color=[0, 255, 0]))
            plt.show()
    return fg_mask


def get_feature_matching(img_paths, dbg=False):
    assert len(img_paths) == 4
    matcher = ElevEstHelper.get_feature_matcher()
    feature_matching = {}
    masks = []
    for i in range(4):
        mask = mask_out_bkgd(img_paths[i], dbg=dbg)
        masks.append(mask)
    for i in range(0, 4):
        for j in range(i + 1, 4):
            img0_pth = img_paths[i]
            img1_pth = img_paths[j]
            mask0 = masks[i]
            mask1 = masks[j]
            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
            original_shape = img0_raw.shape
            img0_raw_resized = cv2.resize(img0_raw, (480, 480))
            img1_raw_resized = cv2.resize(img1_raw, (480, 480))

            img0 = torch.from_numpy(img0_raw_resized)[None][None].cuda() / 255.
            img1 = torch.from_numpy(img1_raw_resized)[None][None].cuda() / 255.
            batch = {'image0': img0, 'image1': img1}

            # Inference with LoFTR and get prediction
            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
            mkpts0[:, 0] = mkpts0[:, 0] * original_shape[1] / 480
            mkpts0[:, 1] = mkpts0[:, 1] * original_shape[0] / 480
            mkpts1[:, 0] = mkpts1[:, 0] * original_shape[1] / 480
            mkpts1[:, 1] = mkpts1[:, 1] * original_shape[0] / 480
            keep0 = mask0[mkpts0[:, 1].astype(int), mkpts1[:, 0].astype(int)]
            keep1 = mask1[mkpts1[:, 1].astype(int), mkpts1[:, 0].astype(int)]
            keep = np.logical_and(keep0, keep1)
            mkpts0 = mkpts0[keep]
            mkpts1 = mkpts1[keep]
            mconf = mconf[keep]
            if dbg:
                # Draw visualization
                color = cm.jet(mconf)
                text = [
                    'LoFTR',
                    'Matches: {}'.format(len(mkpts0)),
                ]
                fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
                fig.show()
            feature_matching[f"{i}_{j}"] = np.concatenate([mkpts0, mkpts1, mconf[:, None]], axis=1)

    return feature_matching


def gen_pose_hypothesis(center_elevation):
    elevations = np.radians(
        [center_elevation, center_elevation - 10, center_elevation + 10, center_elevation, center_elevation])  # 45~120
    azimuths = np.radians([30, 30, 30, 20, 40])
    input_poses = calc_pose(elevations, azimuths, len(azimuths))
    input_poses = input_poses[1:]
    input_poses[..., 1] *= -1
    input_poses[..., 2] *= -1
    return input_poses


def ba_error_general(K, matches, poses):
    projmat0 = K @ poses[0].inverse()[:3, :4]
    projmat1 = K @ poses[1].inverse()[:3, :4]
    match_01 = matches[0]
    pts0 = match_01[:, :2]
    pts1 = match_01[:, 2:4]
    Xref = cv2.triangulatePoints(projmat0.cpu().numpy(), projmat1.cpu().numpy(),
                                 pts0.cpu().numpy().T, pts1.cpu().numpy().T)
    Xref = Xref[:3] / Xref[3:]
    Xref = Xref.T
    Xref = torch.from_numpy(Xref).cuda().float()
    reproj_error = 0
    for match, cp in zip(matches[1:], poses[2:]):
        dist = (torch.norm(match_01[:, :2][:, None, :] - match[:, :2][None, :, :], dim=-1))
        if dist.numel() > 0:
            # print("dist.shape", dist.shape)
            m0to2_index = dist.argmin(1)
            keep = dist[torch.arange(match_01.shape[0]), m0to2_index] < 1
            if keep.sum() > 0:
                xref_in2 = rect_to_img(K, canonical_to_camera(Xref, cp.inverse()))
                reproj_error2 = torch.norm(match[m0to2_index][keep][:, 2:4] - xref_in2[keep], dim=-1)
                conf02 = match[m0to2_index][keep][:, -1]
                reproj_error += (reproj_error2 * conf02).sum() / (conf02.sum())

    return reproj_error


def find_optim_elev(elevs, nimgs, matches, K, dbg=False):
    errs = []
    for elev in elevs:
        err = 0
        cam_poses = gen_pose_hypothesis(elev)
        for start in range(nimgs - 1):
            batch_matches, batch_poses = [], []
            for i in range(start, nimgs + start):
                ci = i % nimgs
                batch_poses.append(cam_poses[ci])
            for j in range(nimgs - 1):
                key = f"{start}_{(start + j + 1) % nimgs}"
                match = matches[key]
                batch_matches.append(match)
            err += ba_error_general(K, batch_matches, batch_poses)
        errs.append(err)
    errs = torch.tensor(errs)
    if dbg:
        plt.plot(elevs, errs)
        plt.show()
    optim_elev = elevs[torch.argmin(errs)].item()
    return optim_elev


def get_elev_est(feature_matching, min_elev=30, max_elev=150, K=None, dbg=False):
    flag = True
    matches = {}
    for i in range(4):
        for j in range(i + 1, 4):
            match_ij = feature_matching[f"{i}_{j}"]
            if len(match_ij) == 0:
                flag = False
            match_ji = np.concatenate([match_ij[:, 2:4], match_ij[:, 0:2], match_ij[:, 4:5]], axis=1)
            matches[f"{i}_{j}"] = torch.from_numpy(match_ij).float().cuda()
            matches[f"{j}_{i}"] = torch.from_numpy(match_ji).float().cuda()
    if not flag:
        loguru.logger.info("0 matches, could not estimate elevation")
        return None
    interval = 10
    elevs = np.arange(min_elev, max_elev, interval)
    optim_elev1 = find_optim_elev(elevs, 4, matches, K)

    elevs = np.arange(optim_elev1 - 10, optim_elev1 + 10, 1)
    optim_elev2 = find_optim_elev(elevs, 4, matches, K)

    return optim_elev2


def elev_est_api(img_paths, min_elev=30, max_elev=150, K=None, dbg=False):
    feature_matching = get_feature_matching(img_paths, dbg=dbg)
    if K is None:
        loguru.logger.warning("K is not provided, using default K")
        K = np.array([[280.0, 0, 128.0],
                      [0, 280.0, 128.0],
                      [0, 0, 1]])
    K = torch.from_numpy(K).cuda().float()
    elev = get_elev_est(feature_matching, min_elev, max_elev, K, dbg=dbg)
    return elev
