from torch.utils.data import Dataset
import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from data.scene import get_boundingbox

from models.rays import gen_rays_from_single_image, gen_random_rays_from_single_image
from kornia import create_meshgrid

def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5 # 1xHxWx2

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


# ! load one ref-image with multiple src-images in camera coordinate system
class BlenderPerView(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=(256, 256), downSample=1.0,
                 split_filepath=None, pair_filepath=None,
                 N_rays=512,
                 vol_dims=[128, 128, 128], batch_size=1,
                 clean_image=False, importance_sample=False, test_ref_views=[],
                 specific_dataset_name = 'GSO'
                 ):

        # print("root_dir: ", root_dir)
        self.root_dir = root_dir
        self.split = split

        self.specific_dataset_name = specific_dataset_name
        self.n_views = n_views
        self.N_rays = N_rays
        self.batch_size = batch_size  # - used for construct new metas for gru fusion training

        self.clean_image = clean_image
        self.importance_sample = importance_sample
        self.test_ref_views = test_ref_views  # used for testing
        self.scale_factor = 1.0
        self.scale_mat = np.float32(np.diag([1, 1, 1, 1.0]))
        assert self.split == 'val' or 'export_mesh', 'only support val or export_mesh'
        # find all subfolders
        main_folder = os.path.join(root_dir, self.specific_dataset_name)
        self.shape_list = [""] # os.listdir(main_folder) # MODIFIED
        self.shape_list.sort()

        # self.shape_list = ['barrel_render']
        # self.shape_list = ["barrel", "bag", "mailbox", "shoe", "chair", "car", "dog", "teddy"] # TO BE DELETED


        self.lvis_paths = []
        for shape_name in self.shape_list:
            self.lvis_paths.append(os.path.join(main_folder, shape_name))

        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'


        # * bounding box for rendering
        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])

        # - used for cost volume regularization
        self.voxel_dims = torch.tensor(vol_dims, dtype=torch.float32)
        self.partial_vol_origin = torch.tensor([-1., -1., -1.], dtype=torch.float32)
        

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor()])



    def load_cam_info(self):
        for vid, img_id in enumerate(self.img_ids):
            intrinsic, extrinsic, near_far = self.intrinsic, np.linalg.inv(self.c2ws[vid]), self.near_far
            self.all_intrinsics.append(intrinsic)
            self.all_extrinsics.append(extrinsic)
            self.all_near_fars.append(near_far)

    def read_mask(self, filename):
        mask_h = cv2.imread(filename, 0)
        mask_h = cv2.resize(mask_h, None, fx=self.downSample, fy=self.downSample,
                            interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask_h, None, fx=0.25, fy=0.25,
                          interpolation=cv2.INTER_NEAREST)

        mask[mask > 0] = 1  # the masks stored in png are not binary
        mask_h[mask_h > 0] = 1

        return mask, mask_h

    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):

        center, radius, bounds = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)

        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()

    def __len__(self):
        # return 8*len(self.lvis_paths)
        return len(self.lvis_paths)

    def __getitem__(self, idx):
        sample = {}
        idx = idx * 8 # to be deleted
        origin_idx = idx
        imgs, depths_h, masks_h = [], [], []  # full size (256, 256)
        intrinsics, w2cs, c2ws, near_fars = [], [], [], []  # record proj-mats between views

        folder_path = self.lvis_paths[idx//8]
        idx = idx % 8 # [0, 7]

        # last subdir name
        shape_name = os.path.split(folder_path)[-1]

        pose_json_path = os.path.join(folder_path, "pose.json")
        with open(pose_json_path, 'r') as f:
            meta = json.load(f)
        
        self.img_ids = list(meta["c2ws"].keys()) # e.g. "view_0", "view_7", "view_0_2_10"
        self.img_wh = (256, 256)
        self.input_poses = np.array(list(meta["c2ws"].values()))
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.array(meta["intrinsics"])
        self.intrinsic = intrinsic
        self.near_far = np.array(meta["near_far"])
        self.near_far[1] = 1.8
        self.define_transforms()
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        
        self.c2ws = []
        self.w2cs = []
        self.near_fars = []
        for image_idx, img_id in enumerate(self.img_ids):
            pose = self.input_poses[image_idx]
            c2w = pose @ self.blender2opencv
            self.c2ws.append(c2w)
            self.w2cs.append(np.linalg.inv(c2w))
            self.near_fars.append(self.near_far)
        self.c2ws = np.stack(self.c2ws, axis=0)
        self.w2cs = np.stack(self.w2cs, axis=0)


        self.all_intrinsics = []  # the cam info of the whole scene
        self.all_extrinsics = []
        self.all_near_fars = []
        self.load_cam_info() 


        # target view
        c2w = self.c2ws[idx]
        w2c = np.linalg.inv(c2w)
        w2c_ref = w2c
        w2c_ref_inv = np.linalg.inv(w2c_ref)

        w2cs.append(w2c @ w2c_ref_inv)
        c2ws.append(np.linalg.inv(w2c @ w2c_ref_inv))

        img_filename = os.path.join(folder_path, 'stage1_8', f'{self.img_ids[idx]}')

        img = Image.open(img_filename)
        img = self.transform(img)  # (4, h, w)
        

        if img.shape[0] == 4:
            img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
        imgs += [img]


        depth_h = torch.ones((img.shape[1], img.shape[2]), dtype=torch.float32)
        depth_h = depth_h.fill_(-1.0)
        mask_h = torch.ones((img.shape[1], img.shape[2]), dtype=torch.int32)


        depths_h.append(depth_h)
        masks_h.append(mask_h)
        
        intrinsic = self.intrinsic
        intrinsics.append(intrinsic)
    

        near_fars.append(self.near_fars[idx])
        image_perm = 0  # only supervised on reference view

        mask_dilated = None


        src_views = range(8, 8 + 8 * 4)

        for vid in src_views:

            img_filename = os.path.join(folder_path, 'stage2_8', f'{self.img_ids[vid]}')
            img = Image.open(img_filename)
            img_wh = self.img_wh

            img = self.transform(img)
            if img.shape[0] == 4:
                img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB

            imgs += [img]
            depth_h = np.ones(img.shape[1:], dtype=np.float32)
            depths_h.append(depth_h)
            masks_h.append(np.ones(img.shape[1:], dtype=np.int32))

            near_fars.append(self.all_near_fars[vid])
            intrinsics.append(self.all_intrinsics[vid])

            w2cs.append(self.all_extrinsics[vid] @ w2c_ref_inv)

        
        # ! estimate scale_mat
        scale_mat, scale_factor = self.cal_scale_mat(
                                                     img_hw=[img_wh[1], img_wh[0]],
                                                     intrinsics=intrinsics, extrinsics=w2cs,
                                                     near_fars=near_fars, factor=1.1
                                                     )


        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_affine_mats = []
        new_depths_h = []
        for intrinsic, extrinsic, near_far, depth in zip(intrinsics, w2cs, near_fars, depths_h):

            P = intrinsic @ extrinsic @ scale_mat
            P = P[:3, :4]
            # - should use load_K_Rt_from_P() to obtain c2w
            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = intrinsic[:3, :3] @ w2c[:3, :4]
            new_affine_mats.append(affine_mat)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1

            new_near_fars.append([0.95 * near, 1.05 * far])
            new_depths_h.append(depth * scale_factor)

        imgs = torch.stack(imgs).float()
        depths_h = np.stack(new_depths_h)
        masks_h = np.stack(masks_h)

        affine_mats = np.stack(new_affine_mats)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), np.stack(
            new_near_fars)
        
        if self.split == 'train':
            start_idx = 0
        else:
            start_idx = 1



        target_w2cs = []
        target_intrinsics = []
        new_target_w2cs = []
        for i_idx in range(8):
            target_w2cs.append(self.all_extrinsics[i_idx] @ w2c_ref_inv)
            target_intrinsics.append(self.all_intrinsics[i_idx])

        for intrinsic, extrinsic in zip(target_intrinsics, target_w2cs):

            P = intrinsic @ extrinsic @ scale_mat
            P = P[:3, :4]
            # - should use load_K_Rt_from_P() to obtain c2w
            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_target_w2cs.append(w2c)
        target_w2cs = np.stack(new_target_w2cs)



        view_ids = [idx] + list(src_views)
        sample['origin_idx'] = origin_idx
        sample['images'] = imgs  # (V, 3, H, W)
        sample['depths_h'] = torch.from_numpy(depths_h.astype(np.float32))  # (V, H, W)
        sample['masks_h'] = torch.from_numpy(masks_h.astype(np.float32))  # (V, H, W)
        sample['w2cs'] = torch.from_numpy(w2cs.astype(np.float32))  # (V, 4, 4)
        sample['c2ws'] = torch.from_numpy(c2ws.astype(np.float32))  # (V, 4, 4)
        sample['target_candidate_w2cs'] = torch.from_numpy(target_w2cs.astype(np.float32))  # (8, 4, 4)
        sample['near_fars'] = torch.from_numpy(near_fars.astype(np.float32))  # (V, 2)
        sample['intrinsics'] = torch.from_numpy(intrinsics.astype(np.float32))[:, :3, :3]  # (V, 3, 3)
        sample['view_ids'] = torch.from_numpy(np.array(view_ids))
        sample['affine_mats'] = torch.from_numpy(affine_mats.astype(np.float32))  # ! in world space

        sample['scan'] = shape_name

        sample['scale_factor'] = torch.tensor(scale_factor)
        sample['img_wh'] = torch.from_numpy(np.array(img_wh))
        sample['render_img_idx'] = torch.tensor(image_perm)
        sample['partial_vol_origin'] = self.partial_vol_origin
        sample['meta'] = str(self.specific_dataset_name) + '_' + str(shape_name) + "_refview" + str(view_ids[0])
        # print("meta: ", sample['meta'])

        # - image to render
        sample['query_image'] = sample['images'][0]
        sample['query_c2w'] = sample['c2ws'][0]
        sample['query_w2c'] = sample['w2cs'][0]
        sample['query_intrinsic'] = sample['intrinsics'][0]
        sample['query_depth'] = sample['depths_h'][0]
        sample['query_mask'] = sample['masks_h'][0]
        sample['query_near_far'] = sample['near_fars'][0]

        sample['images'] = sample['images'][start_idx:]  # (V, 3, H, W)
        sample['depths_h'] = sample['depths_h'][start_idx:]  # (V, H, W)
        sample['masks_h'] = sample['masks_h'][start_idx:]  # (V, H, W)
        sample['w2cs'] = sample['w2cs'][start_idx:]  # (V, 4, 4)
        sample['c2ws'] = sample['c2ws'][start_idx:]  # (V, 4, 4)
        sample['intrinsics'] = sample['intrinsics'][start_idx:]  # (V, 3, 3)
        sample['view_ids'] = sample['view_ids'][start_idx:]
        sample['affine_mats'] = sample['affine_mats'][start_idx:]  # ! in world space

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(w2c_ref_inv)

        # - generate rays
        if ('val' in self.split) or ('test' in self.split):
            sample_rays = gen_rays_from_single_image(
                img_wh[1], img_wh[0],
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=sample['query_depth'],
                mask=sample['query_mask'] if self.clean_image else None)
        else:
            sample_rays = gen_random_rays_from_single_image(
                img_wh[1], img_wh[0],
                self.N_rays,
                sample['query_image'],
                sample['query_intrinsic'],
                sample['query_c2w'],
                depth=sample['query_depth'],
                mask=sample['query_mask'] if self.clean_image else None,
                dilated_mask=mask_dilated,
                importance_sample=self.importance_sample)


        sample['rays'] = sample_rays

        return sample
