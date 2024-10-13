import torch.nn as nn
import torch
import torchvision.transforms as transforms
import numpy as np

from .model_utils import depth_to_points, points_to_bev
from .bev_encoding_model import BevEncode


class DepthConditionModel(nn.Module):
    def __init__(self, grid_range=None, voxel_size=None, bev_encoding=False, model_path=None):
        super(DepthConditionModel, self).__init__()
        self.x_range = (grid_range[0][0], grid_range[1][0])
        self.y_range = (grid_range[0][1], grid_range[1][1])
        self.z_range = (grid_range[0][2], grid_range[1][2])
        self.fx, self.fy = (1000.0, 850.0)
        self.cx, self.cy = (320.0, 240.0)
        self.voxel_size = voxel_size
        self.voxel_grid_shape = ((grid_range[1][0] - grid_range[0][0]) / voxel_size,
                                 (grid_range[1][1] - grid_range[0][1]) / voxel_size,
                                 (grid_range[1][2] - grid_range[0][2]) / voxel_size)
        self.bev_encoding = bev_encoding
        if self.bev_encoding:
            self.bev_encoding_model = BevEncode(80, 1)
        self.mean = torch.ones(80) * 0.5
        self.std = torch.ones(80) * 0.5
        self.model_path = model_path
        if self.model_path is not None:
            checkpoint = torch.load(self.model_path)
            self.load_state_dict(checkpoint['model_state_dict'])

    def filter_out_of_range(self, points):
        device = points.device

        # 定义范围
        x_range = torch.tensor(self.x_range, device=device)
        y_range = torch.tensor(self.y_range, device=device)
        z_range = torch.tensor(self.z_range, device=device)

        # 找到不在指定范围内的点的索引
        out_of_range_indices = torch.logical_or(torch.logical_or(
            torch.logical_or(points[:, 0] < x_range[0], points[:, 0] > x_range[1]),
            torch.logical_or(points[:, 1] < y_range[0], points[:, 1] > y_range[1])),
            torch.logical_or(points[:, 2] < z_range[0], points[:, 2] > z_range[1]))

        # 根据索引获取在指定范围内的点
        filtered_points = points[~out_of_range_indices]
        return filtered_points

    def voxel_pooling(self, points, voxel_index):
        device = points.device
        Dx, Dy, Dz = self.voxel_grid_shape
        Dx, Dy, Dz = int(Dx), int(Dy), int(Dz)

        # 创建空的voxel
        voxel = torch.zeros((4, Dx, Dy, Dz), dtype=torch.float32, device=device)

        # 将点的voxel index转换为一维索引
        indices = voxel_index[:, 0] * (Dy * Dz) + voxel_index[:, 1] * Dz + voxel_index[:, 2]
        # 使用 torch.unique 函数获取唯一的索引值和对应的计数
        unique_indices, counts = torch.unique(indices, return_counts=True)

        for i, index in enumerate(unique_indices):
            z = index % Dz
            y = (torch.div(index, Dz, rounding_mode='trunc')) % Dy
            x = torch.div(index, Dy*Dz, rounding_mode='trunc')
            points_id = torch.nonzero(indices == index, as_tuple=False).flatten()
            points_pool_xyz = torch.sum(points[points_id], dim=0)
            points_pool_n = counts[i]
            voxel[:3, x, y, z] = points_pool_xyz
            voxel[3, x, y, z] = points_pool_n
        return voxel

    def normalize(self, inputs):
        normalize = transforms.Normalize(self.mean, self.std)
        normalized_inputs = normalize(inputs)
        return normalized_inputs

    def forward_with_bev(self, inputs: torch.Tensor):
        """
        The inputs have size (B, C, Dx, Dy, Dz). The timesteps t can be either
        continuous or discrete.
        TODO: This model has a sort of U-Net-like structure I think,
        which is why it first goes down and then up in terms of resolution (?)
        """
        B, H, W = inputs.shape
        bev_res_list = []
        for i in range(B):
            depth = inputs[i, :]
            _, points = depth_to_points(depth, self.fx, self.fy, self.cx, self.cy)
            bev = points_to_bev(points, self.x_range, self.y_range, self.voxel_size)
            bev_res_list.append(bev.unsqueeze(0))
        outputs = torch.cat(bev_res_list, dim=0)

        # normalize
        mean = torch.mean(outputs, dim=[1, 2], keepdim=True)
        std = torch.std(outputs, dim=[1, 2], keepdim=True)
        normalized_outputs = (outputs - mean) / std
        return normalized_outputs.unsqueeze(1)

    def forward_with_bev_encoding(self, inputs: torch.Tensor):
        B, H, W = inputs.shape
        bev_res_list = []
        for i in range(B):
            depth = inputs[i, :]
            _, points = depth_to_points(depth, self.fx, self.fy, self.cx, self.cy)
            # 去除(0, 0, 0)点和超范围点
            points = points[torch.norm(points, dim=1) != 0.0]
            points = self.filter_out_of_range(points)

            # 计算点在体素网格中的索引
            i = torch.div((points[:, 0] - self.x_range[0]), self.voxel_size, rounding_mode='trunc').long()
            j = torch.div((points[:, 1] - self.y_range[0]), self.voxel_size, rounding_mode='trunc').long()
            k = torch.div((points[:, 2] - self.z_range[0]), self.voxel_size, rounding_mode='trunc').long()
            voxel_index = torch.stack([i, j, k], dim=1) # (N, 3), N -> point number

            # voxel pooling
            voxel = self.voxel_pooling(points, voxel_index) # (4, 15, 15, 10)

            # voxel to bev
            bev = torch.cat(voxel.unbind(dim=-1), 0)    # (40, 15, 15)

            bev_res_list.append(bev.unsqueeze(0))
        bev_res = torch.cat(bev_res_list, dim=0)    # (B, 40, 15, 15)

        # normalize
        bev_res = self.normalize(bev_res)

        # bev encoding
        outputs = self.bev_encoding_model(bev_res)
        return outputs

    def forward(self, inputs: torch.Tensor):
        if self.bev_encoding:
            if self.model_path is not None:
                with torch.no_grad():
                    outputs, features = self.forward_with_bev_encoding(inputs)
            else:
                outputs, features = self.forward_with_bev_encoding(inputs)
            return outputs, features
        else:
            return self.forward_with_bev(inputs)



if __name__ == '__main__':
    import cv2
    from visualization.vis_utils import plot_voxel, plot_point_cloud
    import matplotlib.pyplot as plt
    # model = DepthConditionModel()
    # lidar_points = np.load('../lidar_pc_sample.npy')
    # depth = cv2.imread('../depth_sample.png', cv2.IMREAD_UNCHANGED)
    # depth = 0.001 * depth
    # fx, fy = (1000.0, 850.0)
    # cx, cy = (320.0, 240.0)
    # _, points = depth_to_points(depth, fx, fy, cx, cy)
    # print(points.shape)
    # plot_point_cloud(lidar_points[0, :])
    # plot_point_cloud(points)
    # bev = points_to_bev(points, voxel_size=0.2)
    # plt.imshow(bev, cmap='jet')
    # plt.show()
    # bev_lidar = points_to_bev(lidar_points[0, :], voxel_size=0.2)
    # plt.imshow(bev_lidar, cmap='jet')
    # plt.show()

    model = DepthConditionModel(grid_range=((0, -3.2, -2), (6.4, 3.2, 2)), voxel_size=0.2, bev_encoding=True)
    depth = cv2.imread('../depth_sample.png', cv2.IMREAD_UNCHANGED)
    depth = 0.001 * depth
    input = torch.from_numpy(depth)
    input = input.type(torch.float32)
    input = input.unsqueeze(0)
    output = model(input)
    print(output.shape)
    bev = output[0, :]
    bev = bev.numpy()
    plt.imshow(bev, cmap='jet')
    plt.show()
