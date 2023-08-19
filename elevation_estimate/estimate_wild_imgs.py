import os.path as osp
from .utils.elev_est_api import elev_est_api

def estimate_elev(root_dir):
    img_dir = osp.join(root_dir, "stage2_8")
    img_paths = []
    for i in range(4):
        img_paths.append(f"{img_dir}/0_{i}.png")
    elev = elev_est_api(img_paths)
    return elev
