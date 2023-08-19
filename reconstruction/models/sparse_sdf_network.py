import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor, SparseTensor
import torchsparse.nn as spnn

from tsparse.modules import SparseCostRegNet
from tsparse.torchsparse_utils import sparse_to_dense_channel
from ops.grid_sampler import grid_sample_3d, tricubic_sample_3d

# from .gru_fusion import GRUFusion
from ops.back_project import back_project_sparse_type
from ops.generate_grids import generate_grid

from inplace_abn import InPlaceABN

from models.embedder import Embedding
from models.featurenet import ConvBnReLU

import pdb
import random

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
    return mean, var


class LatentSDFLayer(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_out=129,
                 d_hidden=128,
                 n_layers=4,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True,
                 activation='softplus',
                 d_conditional_feature=16):
        super(LatentSDFLayer, self).__init__()

        self.d_conditional_feature = d_conditional_feature

        # concat latent code for ench layer input excepting the first layer and the last layer
        dims_in = [d_in] + [d_hidden + d_conditional_feature for _ in range(n_layers - 2)] + [d_hidden]
        dims_out = [d_hidden for _ in range(n_layers - 1)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn = Embedding(in_channels=d_in, N_freqs=multires)  # * include the input
            self.embed_fn_fine = embed_fn
            dims_in[0] = embed_fn.out_channels

        self.num_layers = n_layers
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                in_dim = dims_in[l] + dims_in[0]
            else:
                in_dim = dims_in[l]

            out_dim = dims_out[l]
            lin = nn.Linear(in_dim, out_dim)

            if geometric_init:  # - from IDR code,
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                    # the channels for latent codes are set to 0
                    torch.nn.init.constant_(lin.weight[:, -d_conditional_feature:], 0.0)
                    torch.nn.init.constant_(lin.bias[-d_conditional_feature:], 0.0)

                elif multires > 0 and l == 0:  # the first layer
                    torch.nn.init.constant_(lin.bias, 0.0)
                    # * the channels for position embeddings are set to 0
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    # * the channels for the xyz coordinate (3 channels) for initialized by normal distribution
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # * the channels for position embeddings (and conditional_feature) are initialized to 0
                    torch.nn.init.constant_(lin.weight[:, -(dims_in[0] - 3 + d_conditional_feature):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # the channels for latent code are initialized to 0
                    torch.nn.init.constant_(lin.weight[:, -d_conditional_feature:], 0.0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, latent):
        inputs = inputs
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        # - only for lod1 network can use the pretrained params of lod0 network
        if latent.shape[1] != self.d_conditional_feature:
            latent = torch.cat([latent, latent], dim=1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            # * due to the conditional bias, different from original neus version
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            if 0 < l < self.num_layers - 1:
                x = torch.cat([x, latent], 1)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x


class SparseSdfNetwork(nn.Module):
    '''
    Coarse-to-fine sparse cost regularization network
    return sparse volume feature for extracting sdf
    '''

    def __init__(self, lod, ch_in, voxel_size, vol_dims,
                 hidden_dim=128, activation='softplus',
                 cost_type='variance_mean',
                 d_pyramid_feature_compress=16,
                 regnet_d_out=8, num_sdf_layers=4,
                 multires=6,
                 ):
        super(SparseSdfNetwork, self).__init__()

        self.lod = lod  # - gradually training, the current regularization lod
        self.ch_in = ch_in
        self.voxel_size = voxel_size  # - the voxel size of the current volume
        self.vol_dims = torch.tensor(vol_dims)  # - the dims of the current volume

        self.selected_views_num = 2  # the number of selected views for feature aggregation
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.cost_type = cost_type
        self.d_pyramid_feature_compress = d_pyramid_feature_compress
        self.gru_fusion = None

        self.regnet_d_out = regnet_d_out
        self.multires = multires

        self.pos_embedder = Embedding(3, self.multires)

        self.compress_layer = ConvBnReLU(
            self.ch_in, self.d_pyramid_feature_compress, 3, 1, 1,
            norm_act=InPlaceABN)
        sparse_ch_in = self.d_pyramid_feature_compress * 2

        sparse_ch_in = sparse_ch_in + 16 if self.lod > 0 else sparse_ch_in
        self.sparse_costreg_net = SparseCostRegNet(
            d_in=sparse_ch_in, d_out=self.regnet_d_out)
        # self.regnet_d_out = self.sparse_costreg_net.d_out

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

        self.sdf_layer = LatentSDFLayer(d_in=3,
                                        d_out=self.hidden_dim + 1,
                                        d_hidden=self.hidden_dim,
                                        n_layers=num_sdf_layers,
                                        multires=multires,
                                        geometric_init=True,
                                        weight_norm=True,
                                        activation=activation,
                                        d_conditional_feature=16  # self.regnet_d_out
                                        )

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def aggregate_multiview_features(self, multiview_features, multiview_masks):
        """
        aggregate mutli-view features by compute their cost variance
        :param multiview_features: (num of voxels, num_of_views, c)
        :param multiview_masks: (num of voxels, num_of_views)
        :return:
        """
        num_pts, n_views, C = multiview_features.shape

        counts = torch.sum(multiview_masks, dim=1, keepdim=False)  # [num_pts]

        assert torch.all(counts > 0)  # the point is visible for at least 1 view

        volume_sum = torch.sum(multiview_features, dim=1, keepdim=False)  # [num_pts, C]
        volume_sq_sum = torch.sum(multiview_features ** 2, dim=1, keepdim=False)

        if volume_sum.isnan().sum() > 0:
            import ipdb; ipdb.set_trace()

        del multiview_features

        counts = 1. / (counts + 1e-5)
        costvar = volume_sq_sum * counts[:, None] - (volume_sum * counts[:, None]) ** 2

        costvar_mean = torch.cat([costvar, volume_sum * counts[:, None]], dim=1)
        del volume_sum, volume_sq_sum, counts



        return costvar_mean

    def sparse_to_dense_volume(self, coords, feature, vol_dims, interval, device=None):
        """
        convert the sparse volume into dense volume to enable trilinear sampling
        to save GPU memory;
        :param coords: [num_pts, 3]
        :param feature: [num_pts, C]
        :param vol_dims: [3]  dX, dY, dZ
        :param interval:
        :return:
        """

        # * assume batch size is 1
        if device is None:
            device = feature.device

        coords_int = (coords / interval).to(torch.int64)
        vol_dims = (vol_dims / interval).to(torch.int64)

        # - if stored in CPU, too slow
        dense_volume = sparse_to_dense_channel(
            coords_int.to(device), feature.to(device), vol_dims.to(device),
            feature.shape[1], 0, device)  # [X, Y, Z, C]

        valid_mask_volume = sparse_to_dense_channel(
            coords_int.to(device),
            torch.ones([feature.shape[0], 1]).to(feature.device),
            vol_dims.to(device),
            1, 0, device)  # [X, Y, Z, 1]

        dense_volume = dense_volume.permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # [1, C, X, Y, Z]
        valid_mask_volume = valid_mask_volume.permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # [1, 1, X, Y, Z]

        return dense_volume, valid_mask_volume

    def get_conditional_volume(self, feature_maps, partial_vol_origin, proj_mats, sizeH=None, sizeW=None, lod=0,
                               pre_coords=None, pre_feats=None,
                               ):
        """

        :param feature_maps: pyramid features (B,V,C0+C1+C2,H,W) fused pyramid features
        :param partial_vol_origin: [B, 3]  the world coordinates of the volume origin (0,0,0)
        :param proj_mats: projection matrix transform world pts into image space [B,V,4,4] suitable for original image size
        :param sizeH: the H of original image size
        :param sizeW: the W of original image size
        :param pre_coords: the coordinates of sparse volume from the prior lod
        :param pre_feats: the features of sparse volume from the prior lod
        :return:
        """
        device = proj_mats.device
        bs = feature_maps.shape[0]
        N_views = feature_maps.shape[1]
        minimum_visible_views = np.min([1, N_views - 1])
        # import ipdb; ipdb.set_trace()
        outputs = {}
        pts_samples = []

        # ----coarse to fine----

        # * use fused pyramid feature maps are very important
        if self.compress_layer is not None:
            feats = self.compress_layer(feature_maps[0])
        else:
            feats = feature_maps[0]
        feats = feats[:, None, :, :, :]  # [V, B, C, H, W]
        KRcam = proj_mats.permute(1, 0, 2, 3).contiguous()  # [V, B, 4, 4]
        interval = 1

        if self.lod == 0:
            # ----generate new coords----
            coords = generate_grid(self.vol_dims, 1)[0]
            coords = coords.view(3, -1).to(device)  # [3, num_pts]
            up_coords = []
            for b in range(bs):
                up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
            up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            # * since we only estimate the geometry of input reference image at one time;
            # * mask the outside of the camera frustum
            # import ipdb; ipdb.set_trace()
            frustum_mask = back_project_sparse_type(
                up_coords, partial_vol_origin, self.voxel_size,
                feats, KRcam, sizeH=sizeH, sizeW=sizeW, only_mask=True)  # [num_pts, n_views]
            frustum_mask = torch.sum(frustum_mask, dim=-1) > minimum_visible_views  # ! here should be large
            up_coords = up_coords[frustum_mask]  # [num_pts_valid, 4]

        else:
            # ----upsample coords----
            assert pre_feats is not None
            assert pre_coords is not None
            up_feat, up_coords = self.upsample(pre_feats, pre_coords, 1)

        # ----back project----
        # give each valid 3d grid point all valid 2D features and masks
        multiview_features, multiview_masks = back_project_sparse_type(
            up_coords, partial_vol_origin, self.voxel_size, feats,
            KRcam, sizeH=sizeH, sizeW=sizeW)  # (num of voxels, num_of_views, c), (num of voxels, num_of_views)
                                              # num_of_views = all views
        
        # if multiview_features.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        if self.lod > 0:
            # ! need another invalid voxels filtering
            frustum_mask = torch.sum(multiview_masks, dim=-1) > 1
            up_feat = up_feat[frustum_mask]
            up_coords = up_coords[frustum_mask]
            multiview_features = multiview_features[frustum_mask]
            multiview_masks = multiview_masks[frustum_mask]
        # if multiview_features.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()
        volume = self.aggregate_multiview_features(multiview_features, multiview_masks) # compute variance for all images features
        # import ipdb; ipdb.set_trace()

        # if volume.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()

        del multiview_features, multiview_masks

        # ----concat feature from last stage----
        if self.lod != 0:
            feat = torch.cat([volume, up_feat], dim=1)
        else:
            feat = volume

        # batch index is in the last position
        r_coords = up_coords[:, [1, 2, 3, 0]]

        # if feat.isnan().sum() > 0:
        #     print('feat has nan:', feat.isnan().sum())
        #     import ipdb; ipdb.set_trace()

        sparse_feat = SparseTensor(feat, r_coords.to(
            torch.int32))  # - directly use sparse tensor to avoid point2voxel operations
        # import ipdb; ipdb.set_trace()
        feat = self.sparse_costreg_net(sparse_feat)

        dense_volume, valid_mask_volume = self.sparse_to_dense_volume(up_coords[:, 1:], feat, self.vol_dims, interval,
                                                                      device=None)  # [1, C/1, X, Y, Z]

        # if dense_volume.isnan().sum() > 0:
        #     import ipdb; ipdb.set_trace()


        outputs['dense_volume_scale%d' % self.lod] = dense_volume # [1, 16, 96, 96, 96]
        outputs['valid_mask_volume_scale%d' % self.lod] = valid_mask_volume # [1, 1, 96, 96, 96]
        outputs['visible_mask_scale%d' % self.lod] = valid_mask_volume # [1, 1, 96, 96, 96]
        outputs['coords_scale%d' % self.lod] = generate_grid(self.vol_dims, interval).to(device)
        # import ipdb; ipdb.set_trace()
        return outputs

    def sdf(self, pts, conditional_volume, lod):
        num_pts = pts.shape[0]
        device = pts.device
        pts_ = pts.clone()
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])
        # import ipdb; ipdb.set_trace()
        sampled_feature = grid_sample_3d(conditional_volume, pts)  # [1, c, 1, 1, num_pts]
        sampled_feature = sampled_feature.view(-1, num_pts).permute(1, 0).contiguous().to(device)

        sdf_pts = self.sdf_layer(pts_, sampled_feature)

        outputs = {}
        outputs['sdf_pts_scale%d' % lod] = sdf_pts[:, :1]
        outputs['sdf_features_pts_scale%d' % lod] = sdf_pts[:, 1:]
        outputs['sampled_latent_scale%d' % lod] = sampled_feature

        return outputs

    @torch.no_grad()
    def sdf_from_sdfvolume(self, pts, sdf_volume, lod=0):
        num_pts = pts.shape[0]
        device = pts.device
        pts_ = pts.clone()
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])

        sdf = torch.nn.functional.grid_sample(sdf_volume, pts, mode='bilinear', align_corners=True,
                                              padding_mode='border')
        sdf = sdf.view(-1, num_pts).permute(1, 0).contiguous().to(device)

        outputs = {}
        outputs['sdf_pts_scale%d' % lod] = sdf

        return outputs

    @torch.no_grad()
    def get_sdf_volume(self, conditional_volume, mask_volume, coords_volume, partial_origin):
        """

        :param conditional_volume: [1,C, dX,dY,dZ]
        :param mask_volume: [1,1, dX,dY,dZ]
        :param coords_volume: [1,3, dX,dY,dZ]
        :return:
        """
        device = conditional_volume.device
        chunk_size = 10240

        _, C, dX, dY, dZ = conditional_volume.shape
        conditional_volume = conditional_volume.view(C, dX * dY * dZ).permute(1, 0).contiguous()
        mask_volume = mask_volume.view(-1)
        coords_volume = coords_volume.view(3, dX * dY * dZ).permute(1, 0).contiguous()

        pts = coords_volume * self.voxel_size + partial_origin  # [dX*dY*dZ, 3]

        sdf_volume = torch.ones([dX * dY * dZ, 1]).float().to(device)

        conditional_volume = conditional_volume[mask_volume > 0]
        pts = pts[mask_volume > 0]
        conditional_volume = conditional_volume.split(chunk_size)
        pts = pts.split(chunk_size)

        sdf_all = []
        for pts_part, feature_part in zip(pts, conditional_volume):
            sdf_part = self.sdf_layer(pts_part, feature_part)[:, :1]
            sdf_all.append(sdf_part)

        sdf_all = torch.cat(sdf_all, dim=0)
        sdf_volume[mask_volume > 0] = sdf_all
        sdf_volume = sdf_volume.view(1, 1, dX, dY, dZ)
        return sdf_volume

    def gradient(self, x, conditional_volume, lod):
        """
        return the gradient of specific lod
        :param x:
        :param lod:
        :return:
        """
        x.requires_grad_(True)
        # import ipdb; ipdb.set_trace()
        with torch.enable_grad():
            output = self.sdf(x, conditional_volume, lod)
        y = output['sdf_pts_scale%d' % lod]

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # ! Distributed Data Parallel doesn’t work with torch.autograd.grad()
        # ! (i.e. it will only work if gradients are to be accumulated in .grad attributes of parameters).
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


def sparse_to_dense_volume(coords, feature, vol_dims, interval, device=None):
    """
    convert the sparse volume into dense volume to enable trilinear sampling
    to save GPU memory;
    :param coords: [num_pts, 3]
    :param feature: [num_pts, C]
    :param vol_dims: [3]  dX, dY, dZ
    :param interval:
    :return:
    """

    # * assume batch size is 1
    if device is None:
        device = feature.device

    coords_int = (coords / interval).to(torch.int64)
    vol_dims = (vol_dims / interval).to(torch.int64)

    # - if stored in CPU, too slow
    dense_volume = sparse_to_dense_channel(
        coords_int.to(device), feature.to(device), vol_dims.to(device),
        feature.shape[1], 0, device)  # [X, Y, Z, C]

    valid_mask_volume = sparse_to_dense_channel(
        coords_int.to(device),
        torch.ones([feature.shape[0], 1]).to(feature.device),
        vol_dims.to(device),
        1, 0, device)  # [X, Y, Z, 1]

    dense_volume = dense_volume.permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # [1, C, X, Y, Z]
    valid_mask_volume = valid_mask_volume.permute(3, 0, 1, 2).contiguous().unsqueeze(0)  # [1, 1, X, Y, Z]

    return dense_volume, valid_mask_volume


class SdfVolume(nn.Module):
    def __init__(self, volume, coords=None, type='dense'):
        super(SdfVolume, self).__init__()
        self.volume = torch.nn.Parameter(volume, requires_grad=True)
        self.coords = coords
        self.type = type

    def forward(self):
        return self.volume


class FinetuneOctreeSdfNetwork(nn.Module):
    '''
    After obtain the conditional volume from generalized network;
    directly optimize the conditional volume
    The conditional volume is still sparse
    '''

    def __init__(self, voxel_size, vol_dims,
                 origin=[-1., -1., -1.],
                 hidden_dim=128, activation='softplus',
                 regnet_d_out=8,
                 multires=6,
                 if_fitted_rendering=True,
                 num_sdf_layers=4,
                 ):
        super(FinetuneOctreeSdfNetwork, self).__init__()

        self.voxel_size = voxel_size  # - the voxel size of the current volume
        self.vol_dims = torch.tensor(vol_dims)  # - the dims of the current volume

        self.origin = torch.tensor(origin).to(torch.float32)

        self.hidden_dim = hidden_dim
        self.activation = activation

        self.regnet_d_out = regnet_d_out

        self.if_fitted_rendering = if_fitted_rendering
        self.multires = multires
        # d_in_embedding = self.regnet_d_out if self.pos_add_type == 'latent' else 3
        # self.pos_embedder = Embedding(d_in_embedding, self.multires)

        # - the optimized parameters
        self.sparse_volume_lod0 = None
        self.sparse_coords_lod0 = None

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

        self.sdf_layer = LatentSDFLayer(d_in=3,
                                        d_out=self.hidden_dim + 1,
                                        d_hidden=self.hidden_dim,
                                        n_layers=num_sdf_layers,
                                        multires=multires,
                                        geometric_init=True,
                                        weight_norm=True,
                                        activation=activation,
                                        d_conditional_feature=16  # self.regnet_d_out
                                        )

        # - add mlp rendering when finetuning
        self.renderer = None

        d_in_renderer = 3 + self.regnet_d_out + 3 + 3
        self.renderer = BlendingRenderingNetwork(
            d_feature=self.hidden_dim - 1,
            mode='idr',  # ! the view direction influence a lot
            d_in=d_in_renderer,
            d_out=50,  # maximum 50 images
            d_hidden=self.hidden_dim,
            n_layers=3,
            weight_norm=True,
            multires_view=4,
            squeeze_out=True,
        )

    def initialize_conditional_volumes(self, dense_volume_lod0, dense_volume_mask_lod0,
                                       sparse_volume_lod0=None, sparse_coords_lod0=None):
        """

        :param dense_volume_lod0: [1,C,dX,dY,dZ]
        :param dense_volume_mask_lod0: [1,1,dX,dY,dZ]
        :param dense_volume_lod1:
        :param dense_volume_mask_lod1:
        :return:
        """

        if sparse_volume_lod0 is None:
            device = dense_volume_lod0.device
            _, C, dX, dY, dZ = dense_volume_lod0.shape

            dense_volume_lod0 = dense_volume_lod0.view(C, dX * dY * dZ).permute(1, 0).contiguous()
            mask_lod0 = dense_volume_mask_lod0.view(dX * dY * dZ) > 0

            self.sparse_volume_lod0 = SdfVolume(dense_volume_lod0[mask_lod0], type='sparse')

            coords = generate_grid(self.vol_dims, 1)[0]  # [3, dX, dY, dZ]
            coords = coords.view(3, dX * dY * dZ).permute(1, 0).to(device)
            self.sparse_coords_lod0 = torch.nn.Parameter(coords[mask_lod0], requires_grad=False)
        else:
            self.sparse_volume_lod0 = SdfVolume(sparse_volume_lod0, type='sparse')
            self.sparse_coords_lod0 = torch.nn.Parameter(sparse_coords_lod0, requires_grad=False)

    def get_conditional_volume(self):
        dense_volume, valid_mask_volume = sparse_to_dense_volume(
            self.sparse_coords_lod0,
            self.sparse_volume_lod0(), self.vol_dims, interval=1,
            device=None)  # [1, C/1, X, Y, Z]

        # valid_mask_volume = self.dense_volume_mask_lod0

        outputs = {}
        outputs['dense_volume_scale%d' % 0] = dense_volume
        outputs['valid_mask_volume_scale%d' % 0] = valid_mask_volume

        return outputs

    def tv_regularizer(self):
        dense_volume, valid_mask_volume = sparse_to_dense_volume(
            self.sparse_coords_lod0,
            self.sparse_volume_lod0(), self.vol_dims, interval=1,
            device=None)  # [1, C/1, X, Y, Z]

        dx = (dense_volume[:, :, 1:, :, :] - dense_volume[:, :, :-1, :, :]) ** 2  # [1, C/1, X-1, Y, Z]
        dy = (dense_volume[:, :, :, 1:, :] - dense_volume[:, :, :, :-1, :]) ** 2  # [1, C/1, X, Y-1, Z]
        dz = (dense_volume[:, :, :, :, 1:] - dense_volume[:, :, :, :, :-1]) ** 2  # [1, C/1, X, Y, Z-1]

        tv = dx[:, :, :, :-1, :-1] + dy[:, :, :-1, :, :-1] + dz[:, :, :-1, :-1, :]  # [1, C/1, X-1, Y-1, Z-1]

        mask = valid_mask_volume[:, :, :-1, :-1, :-1] * valid_mask_volume[:, :, 1:, :-1, :-1] * \
               valid_mask_volume[:, :, :-1, 1:, :-1] * valid_mask_volume[:, :, :-1, :-1, 1:]

        tv = torch.sqrt(tv + 1e-6).mean(dim=1, keepdim=True) * mask
        # tv = tv.mean(dim=1, keepdim=True) * mask

        assert torch.all(~torch.isnan(tv))

        return torch.mean(tv)

    def sdf(self, pts, conditional_volume, lod):

        outputs = {}

        num_pts = pts.shape[0]
        device = pts.device
        pts_ = pts.clone()
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])

        sampled_feature = grid_sample_3d(conditional_volume, pts)  # [1, c, 1, 1, num_pts]
        sampled_feature = sampled_feature.view(-1, num_pts).permute(1, 0).contiguous()
        outputs['sampled_latent_scale%d' % lod] = sampled_feature

        sdf_pts = self.sdf_layer(pts_, sampled_feature)

        lod = 0
        outputs['sdf_pts_scale%d' % lod] = sdf_pts[:, :1]
        outputs['sdf_features_pts_scale%d' % lod] = sdf_pts[:, 1:]

        return outputs

    def color_blend(self, pts, position, normals, view_dirs, feature_vectors, img_index,
                    pts_pixel_color, pts_pixel_mask, pts_patch_color=None, pts_patch_mask=None):

        return self.renderer(torch.cat([pts, position], dim=-1), normals, view_dirs, feature_vectors,
                             img_index, pts_pixel_color, pts_pixel_mask,
                             pts_patch_color=pts_patch_color, pts_patch_mask=pts_patch_mask)

    def gradient(self, x, conditional_volume, lod):
        """
        return the gradient of specific lod
        :param x:
        :param lod:
        :return:
        """
        x.requires_grad_(True)
        output = self.sdf(x, conditional_volume, lod)
        y = output['sdf_pts_scale%d' % 0]

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)

        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    @torch.no_grad()
    def prune_dense_mask(self, threshold=0.02):
        """
        Just gradually prune the mask of dense volume to decrease the number of sdf network inference
        :return:
        """
        chunk_size = 10240
        coords = generate_grid(self.vol_dims_lod0, 1)[0]  # [3, dX, dY, dZ]

        _, dX, dY, dZ = coords.shape

        pts = coords.view(3, -1).permute(1,
                                         0).contiguous() * self.voxel_size_lod0 + self.origin[None, :]  # [dX*dY*dZ, 3]

        # dense_volume = self.dense_volume_lod0()  # [1,C,dX,dY,dZ]
        dense_volume, _ = sparse_to_dense_volume(
            self.sparse_coords_lod0,
            self.sparse_volume_lod0(), self.vol_dims_lod0, interval=1,
            device=None)  # [1, C/1, X, Y, Z]

        sdf_volume = torch.ones([dX * dY * dZ, 1]).float().to(dense_volume.device) * 100

        mask = self.dense_volume_mask_lod0.view(-1) > 0

        pts_valid = pts[mask].to(dense_volume.device)
        feature_valid = dense_volume.view(self.regnet_d_out, -1).permute(1, 0).contiguous()[mask]

        pts_valid = pts_valid.split(chunk_size)
        feature_valid = feature_valid.split(chunk_size)

        sdf_list = []

        for pts_part, feature_part in zip(pts_valid, feature_valid):
            sdf_part = self.sdf_layer(pts_part, feature_part)[:, :1]
            sdf_list.append(sdf_part)

        sdf_list = torch.cat(sdf_list, dim=0)

        sdf_volume[mask] = sdf_list

        occupancy_mask = torch.abs(sdf_volume) < threshold  # [num_pts, 1]

        # - dilate
        occupancy_mask = occupancy_mask.float()
        occupancy_mask = occupancy_mask.view(1, 1, dX, dY, dZ)
        occupancy_mask = F.avg_pool3d(occupancy_mask, kernel_size=7, stride=1, padding=3)
        occupancy_mask = occupancy_mask > 0

        self.dense_volume_mask_lod0 = torch.logical_and(self.dense_volume_mask_lod0,
                                                        occupancy_mask).float()  # (1, 1, dX, dY, dZ)


class BlendingRenderingNetwork(nn.Module):
    def __init__(
            self,
            d_feature,
            mode,
            d_in,
            d_out,
            d_hidden,
            n_layers,
            weight_norm=True,
            multires_view=0,
            squeeze_out=True,
    ):
        super(BlendingRenderingNetwork, self).__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedder = None
        if multires_view > 0:
            self.embedder = Embedding(3, multires_view)
            dims[0] += (self.embedder.out_channels - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.color_volume = None

        self.softmax = nn.Softmax(dim=1)

        self.type = 'blending'

    def sample_pts_from_colorVolume(self, pts):
        device = pts.device
        num_pts = pts.shape[0]
        pts_ = pts.clone()
        pts = pts.view(1, 1, 1, num_pts, 3)  # - should be in range (-1, 1)

        pts = torch.flip(pts, dims=[-1])

        sampled_color = grid_sample_3d(self.color_volume, pts)  # [1, c, 1, 1, num_pts]
        sampled_color = sampled_color.view(-1, num_pts).permute(1, 0).contiguous().to(device)

        return sampled_color

    def forward(self, position, normals, view_dirs, feature_vectors, img_index,
                pts_pixel_color, pts_pixel_mask, pts_patch_color=None, pts_patch_mask=None):
        """

        :param position: can be 3d coord or interpolated volume latent
        :param normals:
        :param view_dirs:
        :param feature_vectors:
        :param img_index: [N_views], used to extract corresponding weights
        :param pts_pixel_color: [N_pts, N_views, 3]
        :param pts_pixel_mask: [N_pts, N_views]
        :param pts_patch_color: [N_pts, N_views, Npx, 3]
        :return:
        """
        if self.embedder is not None:
            view_dirs = self.embedder(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([position, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([position, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([position, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'no_points':
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_points_no_view_dir':
            rendering_input = torch.cat([normals, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)  # [n_pts, d_out]

        ## extract value based on img_index
        x_extracted = torch.index_select(x, 1, img_index.long())

        weights_pixel = self.softmax(x_extracted)  # [n_pts, N_views]
        weights_pixel = weights_pixel * pts_pixel_mask
        weights_pixel = weights_pixel / (
                torch.sum(weights_pixel.float(), dim=1, keepdim=True) + 1e-8)  # [n_pts, N_views]
        final_pixel_color = torch.sum(pts_pixel_color * weights_pixel[:, :, None], dim=1,
                                      keepdim=False)  # [N_pts, 3]

        final_pixel_mask = torch.sum(pts_pixel_mask.float(), dim=1, keepdim=True) > 0  # [N_pts, 1]

        final_patch_color, final_patch_mask = None, None
        # pts_patch_color  [N_pts, N_views, Npx, 3]; pts_patch_mask  [N_pts, N_views, Npx]
        if pts_patch_color is not None:
            N_pts, N_views, Npx, _ = pts_patch_color.shape
            patch_mask = torch.sum(pts_patch_mask, dim=-1, keepdim=False) > Npx - 1  # [N_pts, N_views]

            weights_patch = self.softmax(x_extracted)  # [N_pts, N_views]
            weights_patch = weights_patch * patch_mask
            weights_patch = weights_patch / (
                    torch.sum(weights_patch.float(), dim=1, keepdim=True) + 1e-8)  # [n_pts, N_views]

            final_patch_color = torch.sum(pts_patch_color * weights_patch[:, :, None, None], dim=1,
                                          keepdim=False)  # [N_pts, Npx, 3]
            final_patch_mask = torch.sum(patch_mask, dim=1, keepdim=True) > 0  # [N_pts, 1]  at least one image sees

        return final_pixel_color, final_pixel_mask, final_patch_color, final_patch_mask
