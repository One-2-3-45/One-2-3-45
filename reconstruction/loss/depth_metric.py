import numpy as np


def l1(depth1, depth2):
    """
    Computes the l1 errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        L1(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    diff = depth1 - depth2
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels


def l1_inverse(depth1, depth2):
    """
    Computes the l1 errors between inverses of two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        L1(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    diff = np.reciprocal(depth1) - np.reciprocal(depth2)
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels


def rmse_log(depth1, depth2):
    """
    Computes the root min square errors between the logs of two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        RMSE(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels)


def rmse(depth1, depth2):
    """
    Computes the root min square errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        RMSE(log)

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    diff = depth1 - depth2
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(diff)) / num_pixels)


def scale_invariant(depth1, depth2):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        scale_invariant_distance

    """
    # sqrt(Eq. 3)
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))


def abs_relative(depth_pred, depth_gt):
    """
    Computes relative absolute distance.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth_pred:  depth map prediction
    depth_gt:    depth map ground truth

    Returns:
        abs_relative_distance

    """
    assert (np.all(np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred >= 0) & (depth_gt >= 0)))
    diff = depth_pred - depth_gt
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff) / depth_gt) / num_pixels


def avg_log10(depth1, depth2):
    """
    Computes average log_10 error (Liu, Neural Fields, 2015).
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        abs_relative_distance

    """
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    log_diff = np.log10(depth1) - np.log10(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(log_diff)) / num_pixels


def sq_relative(depth_pred, depth_gt):
    """
    Computes relative squared distance.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth_pred:  depth map prediction
    depth_gt:    depth map ground truth

    Returns:
        squared_relative_distance

    """
    assert (np.all(np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_pred >= 0) & (depth_gt >= 0)))
    diff = depth_pred - depth_gt
    num_pixels = float(diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.square(diff) / depth_gt) / num_pixels


def ratio_threshold(depth1, depth2, threshold):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns:
        percentage of pixels with ratio less than the threshold

    """
    assert (threshold > 0.)
    assert (np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return float(np.sum(np.absolute(log_diff) < np.log(threshold))) / num_pixels


def compute_depth_errors(depth_pred, depth_gt, valid_mask):
    """
    Computes different distance measures between two depth maps.

    depth_pred:           depth map prediction
    depth_gt:             depth map ground truth
    distances_to_compute: which distances to compute

    Returns:
        a dictionary with computed distances, and the number of valid pixels

    """
    depth_pred = depth_pred[valid_mask]
    depth_gt = depth_gt[valid_mask]
    num_valid = np.sum(valid_mask)

    distances_to_compute = ['l1',
                            'l1_inverse',
                            'scale_invariant',
                            'abs_relative',
                            'sq_relative',
                            'avg_log10',
                            'rmse_log',
                            'rmse',
                            'ratio_threshold_1.25',
                            'ratio_threshold_1.5625',
                            'ratio_threshold_1.953125']

    results = {'num_valid': num_valid}
    for dist in distances_to_compute:
        if dist.startswith('ratio_threshold'):
            threshold = float(dist.split('_')[-1])
            results[dist] = ratio_threshold(depth_pred, depth_gt, threshold)
        else:
            results[dist] = globals()[dist](depth_pred, depth_gt)

    return results
