import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def point_cloud_to_voxel_grid(point_cloud: torch.Tensor, grid_range, voxel_size, voxel_point_max, normalize=True):
    '''
    point_cloud: (B, N, 3)
    '''
    grid_range = grid_range
    grid_range = torch.tensor(grid_range)
    grid_range = grid_range.to(point_cloud.device)

    # 计算体素网格各个维度的数量
    dims = ((grid_range[1] - grid_range[0]) / voxel_size).int()

    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]

    # 将点云中的每个点分配到相应的体素中
    voxels = torch.zeros((batch_size, dims[0], dims[1], dims[2]),
                         dtype=torch.long, device=point_cloud.device)
    for i in range(batch_size):
        indices = ((point_cloud[i] - grid_range[0]) / voxel_size).long()
        mask = (indices >= 0) & (indices < dims)
        indices = indices[mask.all(dim=1)]
        unique_indices, counts = torch.unique(indices, return_counts=True, dim=0)
        voxels[i, unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = counts

    # 对voxels进行归一化
    if normalize:
        voxels = torch.where(voxels > voxel_point_max, torch.ones_like(voxels, dtype=torch.float32) *
                             voxel_point_max, voxels.to(torch.float32))
        normalized_voxels = voxels / voxel_point_max
        normalized_voxels = normalized_voxels.unsqueeze(1)
        return normalized_voxels
    else:
        return voxels


def depth_to_points(depth: torch.Tensor, fx=1000.0, fy=850.0, cx=320.0, cy=240.0):
    '''
    depth: depth map, (h, w)
    fx, fy: Camera inter-parameter, set to (1000.0,  850.0) temporarily
    cx, cy: Set as image center, (320.0, 240.0)
    '''
    h, w = depth.shape
    device = depth.device
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w))
    rows = rows.to(device)
    cols = cols.to(device)

    # limit the range of depth
    valid_mask = depth < 6.0
    valid_mask = valid_mask.flatten()

    x_c = (cols - cx) * depth / fx
    x_c = x_c.flatten()
    y_c = (rows - cy) * depth / fy
    y_c = y_c.flatten()
    z_c = depth
    z_c = z_c.flatten()
    points_c = torch.stack((x_c[valid_mask], y_c[valid_mask], z_c[valid_mask]), axis=1)
    points_l = points_c[:, [2, 0, 1]]   # covert image coordinate to lidar coordinate
    return points_c, points_l


def points_to_bev(points: torch.Tensor, x_range, y_range, voxel_size):
    '''
    points: (B, N, 3)
    x_range: bev range in x axis
    y_range: bev range in y axis
    voxel_size: the size of bev grid
    '''
    device = points.device
    B, N, _ = points.shape
    # 确定体素范围
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_size = int((x_max - x_min) / voxel_size)
    y_size = int((y_max - y_min) / voxel_size)
    bev_shape = (x_size, y_size)

    bev_res_list = []
    for i in range(B):
        # 将点云的xy坐标缩放到BEV图像大小内部
        points_xy = points[i, :, :2]
        points_xy[:, 0] -= x_min  # move x axis
        points_xy[:, 1] -= y_min  # move y axis
        points_xy /= voxel_size
        points_xy = points_xy.type(torch.int32)
        points_xy[:, 0] = points_xy[:, 0].clamp(0, bev_shape[0] - 1)
        points_xy[:, 1] = points_xy[:, 1].clamp(0, bev_shape[1] - 1)

        # 创建BEV图像
        bev = torch.zeros(bev_shape, dtype=torch.float32, device=device)

        # 将点的坐标转换为一维索引
        indices = points_xy[:, 0] * bev_shape[1] + points_xy[:, 1]
        # 使用 torch.unique 函数获取唯一的索引值和对应的计数
        unique_indices, counts = torch.unique(indices, return_counts=True)

        for i, index in enumerate(unique_indices):
            x = torch.div(index, bev_shape[0], rounding_mode='trunc')
            y = index % bev_shape[0]
            bev[x, y] = counts[i]
        bev_res_list.append(bev.unsqueeze(0))
    bev_res = torch.stack(bev_res_list, dim=0)

    # normalize
    mean = torch.Tensor([0.5])
    std = torch.Tensor([0.5])
    normalize = transforms.Normalize(mean, std)
    bev_res = normalize(bev_res)
    return bev_res


# def set_requires_grad(module: nn.Module, requires_grad: bool):
#     for p in module.parameters():
#         p.requires_grad_(requires_grad)
#
#
# def compute_distance_transform(mask: torch.Tensor):
#     image_size = mask.shape[-1]
#     distance_transform = torch.stack([
#         torch.from_numpy(cv2.distanceTransform(
#             (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
#         ) / (image_size / 2))
#         for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
#     ]).unsqueeze(1).clip(0, 1).to(mask.device)
#     return distance_transform
#
#
# def default(x, d):
#     return d if x is None else x
#
#
# def get_num_points(x: Pointclouds, /):
#     return x.points_padded().shape[1]
#
#
# def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
#     """Custom beta schedule"""
#     betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
#     warmup_frac = 0.3
#     warmup_time = int(num_train_timesteps * warmup_frac)
#     warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
#     warmup_time = min(warmup_time, num_train_timesteps)
#     betas[:warmup_time] = warmup_steps[:warmup_time]
#     return betas
