import os.path as osp
import os
import matplotlib.pyplot as plt
import torch
import cv2
import math

import numpy as np
import tqdm
from cv2 import findContours
from dl_ext.primitive import safe_zip
from dl_ext.timer import EvalTime


def plot_confidence(confidence):
    n = len(confidence)
    plt.plot(np.arange(n), confidence)
    plt.show()


def image_grid(
        images,
        rows=None,
        cols=None,
        fill: bool = True,
        show_axes: bool = False,
        rgb=None,
        show=True,
        label=None,
        **kwargs
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    evaltime = EvalTime(disable=True)
    evaltime('')
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if len(images[0].shape) == 2:
        rgb = False
    if images[0].shape[-1] == 2:
        # flow
        images = [flow_to_image(im) for im in images]
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = int(len(images) ** 0.5)
        cols = math.ceil(len(images) / rows)

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    if len(images) < 50:
        figsize = (10, 10)
    else:
        figsize = (15, 15)
    evaltime('0.5')
    plt.figure(figsize=figsize)
    # fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=figsize)
    if label:
        # fig.suptitle(label, fontsize=30)
        plt.suptitle(label, fontsize=30)
    # bleed = 0
    # fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))
    evaltime('subplots')

    # for i, (ax, im) in enumerate(tqdm.tqdm(zip(axarr.ravel(), images), leave=True, total=len(images))):
    for i in range(len(images)):
        # evaltime(f'{i} begin')
        plt.subplot(rows, cols, i + 1)
        if rgb:
            # only render RGB channels
            plt.imshow(images[i][..., :3], **kwargs)
            # ax.imshow(im[..., :3], **kwargs)
        else:
            # only render Alpha channel
            plt.imshow(images[i], **kwargs)
            # ax.imshow(im, **kwargs)
        if not show_axes:
            plt.axis('off')
            # ax.set_axis_off()
        # ax.set_title(f'{i}')
        plt.title(f'{i}')
        # evaltime(f'{i} end')
    evaltime('2')
    if show:
        plt.show()
    # return fig


def depth_grid(
        depths,
        rows=None,
        cols=None,
        fill: bool = True,
        show_axes: bool = False,
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(depths)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), depths):
        ax.imshow(im)
        if not show_axes:
            ax.set_axis_off()
    plt.show()


def hover_masks_on_imgs(images, masks):
    masks = np.array(masks)
    new_imgs = []
    tids = list(range(1, masks.max() + 1))
    colors = colormap(rgb=True, lighten=True)
    for im, mask in tqdm.tqdm(safe_zip(images, masks), total=len(images)):
        for tid in tids:
            im = vis_mask(
                im,
                (mask == tid).astype(np.uint8),
                color=colors[tid],
                alpha=0.5,
                border_alpha=0.5,
                border_color=[255, 255, 255],
                border_thick=3)
        new_imgs.append(im)
    return new_imgs


def vis_mask(img,
             mask,
             color=[255, 255, 255],
             alpha=0.4,
             show_border=True,
             border_alpha=0.5,
             border_thick=1,
             border_color=None):
    """Visualizes a single binary mask."""
    if isinstance(mask, torch.Tensor):
        from anypose.utils.pn_utils import to_array
        mask = to_array(mask > 0).astype(np.uint8)
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += [alpha * x for x in color]

    if show_border:
        contours, _ = findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours = [c for c in contours if c.shape[0] > 10]
        if border_color is None:
            border_color = color
        if not isinstance(border_color, list):
            border_color = border_color.tolist()
        if border_alpha < 1:
            with_border = img.copy()
            cv2.drawContours(with_border, contours, -1, border_color,
                             border_thick, cv2.LINE_AA)
            img = (1 - border_alpha) * img + border_alpha * with_border
        else:
            cv2.drawContours(img, contours, -1, border_color, border_thick,
                             cv2.LINE_AA)

    return img.astype(np.uint8)


def colormap(rgb=False, lighten=True):
    """Copied from Detectron codebase."""
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]

    if lighten:
        # Make all the colors a little lighter / whiter. This is copied
        # from the detectron visualization code (search for 'w_ratio').
        w_ratio = 0.4
        color_list = (color_list * (1 - w_ratio) + w_ratio)
    return color_list * 255


def vis_layer_mask(masks, save_path=None):
    masks = torch.as_tensor(masks)
    tids = masks.unique().tolist()
    tids.remove(0)
    for tid in tqdm.tqdm(tids):
        show = save_path is None
        image_grid(masks == tid, label=f'{tid}', show=show)
        if save_path:
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            plt.savefig(save_path % tid)
            plt.close('all')


def show(x, **kwargs):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    plt.imshow(x, **kwargs)
    plt.show()


def vis_title(rgb, text, shift_y=30):
    tmp = rgb.copy()
    shift_x = rgb.shape[1] // 2
    cv2.putText(tmp, text,
                (shift_x, shift_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return tmp
