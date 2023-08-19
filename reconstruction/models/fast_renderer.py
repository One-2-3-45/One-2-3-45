import torch
import torch.nn.functional as F
import torch.nn as nn
from icecream import ic


# - neus: use sphere-tracing to speed up depth maps extraction
# This code snippet is heavily borrowed from IDR.
class FastRenderer(nn.Module):
    def __init__(self):
        super(FastRenderer, self).__init__()

        self.sdf_threshold = 5e-5
        self.line_search_step = 0.5
        self.line_step_iters = 1
        self.sphere_tracing_iters = 10
        self.n_steps = 100
        self.n_secant_steps = 8

        # - use sdf_network to inference sdf value or directly interpolate sdf value from precomputed sdf_volume
        self.network_inference = False

    def extract_depth_maps(self, rays_o, rays_d, near, far, sdf_network, conditional_volume):
        with torch.no_grad():
            curr_start_points, network_object_mask, acc_start_dis = self.get_intersection(
                rays_o, rays_d, near, far,
                sdf_network, conditional_volume)

        network_object_mask = network_object_mask.reshape(-1)

        return network_object_mask, acc_start_dis

    def get_intersection(self, rays_o, rays_d, near, far, sdf_network, conditional_volume):
        device = rays_o.device
        num_pixels, _ = rays_d.shape

        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(rays_o, rays_d, near, far, sdf_network, conditional_volume)

        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().to(device)
        if sampler_mask.sum() > 0:
            # sampler_min_max = torch.zeros((num_pixels, 2)).to(device)
            # sampler_min_max[sampler_mask, 0] = acc_start_dis[sampler_mask]
            # sampler_min_max[sampler_mask, 1] = acc_end_dis[sampler_mask]

            # ray_sampler(self, rays_o, rays_d, near, far, sampler_mask):
            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(rays_o,
                                                                                rays_d,
                                                                                acc_start_dis,
                                                                                acc_end_dis,
                                                                                sampler_mask,
                                                                                sdf_network,
                                                                                conditional_volume
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask][:, None]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask][:, None]

        # print('----------------------------------------------------------------')
        # print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
        #       .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(),
        #               sampler_mask.sum()))
        # print('----------------------------------------------------------------')

        return curr_start_points, network_object_mask, acc_start_dis

    def sphere_tracing(self, rays_o, rays_d, near, far, sdf_network, conditional_volume):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        device = rays_o.device

        unfinished_mask_start = (near < far).reshape(-1).clone()
        unfinished_mask_end = (near < far).reshape(-1).clone()

        # Initialize start current points
        curr_start_points = rays_o + rays_d * near
        acc_start_dis = near.clone()

        # Initialize end current points
        curr_end_points = rays_o + rays_d * far
        acc_end_dis = far.clone()

        # Initizlize min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).to(device)

        if self.network_inference:
            sdf_func = sdf_network.sdf
        else:
            sdf_func = sdf_network.sdf_from_sdfvolume

        next_sdf_start[unfinished_mask_start] = sdf_func(
            curr_start_points[unfinished_mask_start],
            conditional_volume, lod=0, gru_fusion=False)['sdf_pts_scale%d' % 0]

        next_sdf_end = torch.zeros_like(acc_end_dis).to(device)
        next_sdf_end[unfinished_mask_end] = sdf_func(curr_end_points[unfinished_mask_end],
                                                     conditional_volume, lod=0, gru_fusion=False)[
            'sdf_pts_scale%d' % 0]

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).to(device)
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).to(device)
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold).reshape(-1)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold).reshape(-1)

            if (
                    unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = rays_o + acc_start_dis * rays_d
            curr_end_points = rays_o + acc_end_dis * rays_d

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).to(device)
            if unfinished_mask_start.sum() > 0:
                next_sdf_start[unfinished_mask_start] = sdf_func(curr_start_points[unfinished_mask_start],
                                                                 conditional_volume, lod=0, gru_fusion=False)[
                    'sdf_pts_scale%d' % 0]

            next_sdf_end = torch.zeros_like(acc_end_dis).to(device)
            if unfinished_mask_end.sum() > 0:
                next_sdf_end[unfinished_mask_end] = sdf_func(curr_end_points[unfinished_mask_end],
                                                             conditional_volume, lod=0, gru_fusion=False)[
                    'sdf_pts_scale%d' % 0]

            not_projected_start = (next_sdf_start < 0).reshape(-1)
            not_projected_end = (next_sdf_end < 0).reshape(-1)
            not_proj_iters = 0

            while (
                    not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                if not_projected_start.sum() > 0:
                    acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * \
                                                          curr_sdf_start[not_projected_start]
                    curr_start_points[not_projected_start] = (rays_o + acc_start_dis * rays_d)[not_projected_start]

                    next_sdf_start[not_projected_start] = sdf_func(
                        curr_start_points[not_projected_start],
                        conditional_volume, lod=0, gru_fusion=False)['sdf_pts_scale%d' % 0]

                if not_projected_end.sum() > 0:
                    acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * \
                                                      curr_sdf_end[
                                                          not_projected_end]
                    curr_end_points[not_projected_end] = (rays_o + acc_end_dis * rays_d)[not_projected_end]

                    # Calc sdf

                    next_sdf_end[not_projected_end] = sdf_func(
                        curr_end_points[not_projected_end],
                        conditional_volume, lod=0, gru_fusion=False)['sdf_pts_scale%d' % 0]

                # Update mask
                not_projected_start = (next_sdf_start < 0).reshape(-1)
                not_projected_end = (next_sdf_end < 0).reshape(-1)
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis).reshape(-1)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis).reshape(-1)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, rays_o, rays_d, near, far, sampler_mask, sdf_network, conditional_volume):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''
        device = rays_o.device
        num_pixels, _ = rays_d.shape
        sampler_pts = torch.zeros(num_pixels, 3).to(device).float()
        sampler_dists = torch.zeros(num_pixels).to(device).float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).to(device).view(1, -1)

        pts_intervals = near + intervals_dist * (far - near)
        points = rays_o[:, None, :] + pts_intervals[:, :, None] * rays_d[:, None, :]

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        if self.network_inference:
            sdf_func = sdf_network.sdf
        else:
            sdf_func = sdf_network.sdf_from_sdfvolume

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf_func(pnts,
                                        conditional_volume, lod=0, gru_fusion=False)['sdf_pts_scale%d' % 0])
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).to(device).float().reshape(
            (1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~net_surface_pts
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx,
                                                          :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][
                torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]

            cam_loc_secant = rays_o[mask_intersect_idx[secant_pts]]
            ray_directions_secant = rays_d[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant,
                                        sdf_network, conditional_volume)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant[:,
                                                                           None] * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, rays_o, rays_d, sdf_network, conditional_volume):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        if self.network_inference:
            sdf_func = sdf_network.sdf
        else:
            sdf_func = sdf_network.sdf_from_sdfvolume

        z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = rays_o + z_pred[:, None] * rays_d
            sdf_mid = sdf_func(p_mid,
                               conditional_volume, lod=0, gru_fusion=False)['sdf_pts_scale%d' % 0].reshape(-1)
            ind_low = (sdf_mid > 0).reshape(-1)
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred  # 1D tensor

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''
        device = sdf.device
        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).to(device)
        steps = torch.empty(n).uniform_(0.0, 1.0).to(device)
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist
