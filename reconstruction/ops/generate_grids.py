import torch


def generate_grid(n_vox, interval):
    """
    generate grid
    if 3D volume, grid[:,:,x,y,z]  = (x,y,z)
    :param n_vox:
    :param interval:
    :return:
    """
    with torch.no_grad():
        # Create voxel grid
        grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
        grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2], indexing="ij"))  # 3 dx dy dz
        # ! don't create tensor on gpu; imbalanced gpu memory in ddp mode
        grid = grid.unsqueeze(0).type(torch.float32)  # 1 3 dx dy dz

    return grid


if __name__ == "__main__":
    import torch.nn.functional as F
    grid = generate_grid([5, 6, 8], 1)

    pts = 2 * torch.tensor([1, 2, 3]) / (torch.tensor([5, 6, 8]) - 1) - 1
    pts = pts.view(1, 1, 1, 1, 3)

    pts = torch.flip(pts, dims=[-1])

    sampled = F.grid_sample(grid, pts, mode='nearest')

    print(sampled)
