import inspect
from typing import Optional
from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
import numpy as np

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler

from model.point_cloud_model import PointCloudModel


KPT_ENCODING = {
    0: [0, 0, 0, 0, 1, 0],
    1: [0, 0, 0, 1, 0, 0],
    2: [0, 0, 0, 1, 1, 0],
    3: [0, 0, 1, 0, 0, 0],
    4: [0, 0, 1, 0, 1, 0],
    5: [0, 0, 1, 1, 0, 0],
    6: [0, 0, 1, 1, 1, 0],
    7: [0, 1, 0, 0, 0, 0],
    8: [0, 1, 0, 0, 1, 0],
    9: [0, 1, 0, 1, 0, 0],
    10: [0, 1, 0, 1, 1, 0],
    11: [0, 1, 1, 0, 0, 0],
    12: [0, 1, 1, 0, 1, 0],
    13: [0, 1, 1, 1, 0, 0],
    14: [0, 1, 1, 1, 1, 0],
    15: [1, 0, 0, 0, 0, 0],
    16: [1, 0, 0, 0, 1, 0]
}


class ConditionalPointCloudDiffusionModel(nn.Module):
    def __init__(
        self,
        grid_range: (int) = ((0, -3.2, -2), (6.4, 3.2, 2)),
        voxel_size: float = 0.4,
        voxel_point_max: int = 40,
        voxel_num: int = 60,
        timestep_num: int = 500,
    ):
        super().__init__()
        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {'num_train_timesteps': timestep_num}
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(
            # in_channels=6,
            in_channels=12,
            out_channels=3,
            voxel_point_max=voxel_point_max,
            voxel_num=voxel_num,
            embed_dim=14,
            # embed_dim=8,
        )

        # x_0 transformation
        self.weight = 3.0
        self.bias = 0.0

        # noise
        self.noise_mean = 0.0
        self.noise_std = 1.0

        # Voxelization
        self.grid_range = grid_range
        self.voxel_size = voxel_size
        self.voxel_point_max = voxel_point_max
        self.voxel_num = voxel_num

    def get_kpt_condition(self, vl_ids, kpts):
        '''
        vl_ids: (N, 3)
        kpts: (n, 3)
        '''
        N, _ = vl_ids.shape
        device = kpts.device
        kpt_condition = torch.zeros((N, 6), device=device)

        kpts = kpts[:, torch.tensor([2, 0, 1])]  # (z, x, y) -> (x, y, z)
        distances = torch.cdist(kpts, vl_ids)
        min_distances, min_indices = torch.min(distances, dim=1)

        for j, id in enumerate(min_indices):
            kpt_encode = KPT_ENCODING[j]
            kpt_encode[-1] = float(min_distances[j])
            if torch.all(kpt_condition[id] == 0).item():
                kpt_condition[id] = torch.tensor(kpt_encode, device=device)
            else:
                kpt_condition[id] = (kpt_condition[id] + torch.tensor(kpt_encode, device=device)) / 2
        return kpt_condition

    def get_invoxel_points_with_id(self, pc, kpts):
        '''
        pc: (B, N, 3)
        '''
        B, N, _ = pc.shape
        b, n, _ = kpts.shape
        device = pc.device

        # 计算体素网格各个维度的数量
        grid_range = torch.tensor(self.grid_range, device=device)
        voxel_dims = ((grid_range[1] - grid_range[0]) / self.voxel_size).int()

        batch_pts_list, batch_vl_ids_list, batch_kpt_condition_list = [], [], []
        for i in range(B):
            kpts_batch = kpts[i]    # (n, 3)
            pc_batch = pc[i]    # (N, 3)
            # 过滤补齐的（0，0，0）
            pc_batch = pc_batch[torch.any(pc_batch != 0, dim=1)]

            vl_indices = ((pc_batch - grid_range[0]) / self.voxel_size).long()
            mask = (vl_indices >= 0) & (vl_indices < voxel_dims)
            vl_indices = vl_indices[mask.all(dim=1)]
            unique_vl_indices, counts = torch.unique(vl_indices, return_counts=True, dim=0)
            # 根据 counts 进行降序排列
            sort = torch.argsort(counts, descending=True)
            counts = counts[sort]
            unique_vl_indices = unique_vl_indices[sort]
            pt_list, vl_id_list = [], []
            for e, j in enumerate(unique_vl_indices):
                vl_id = grid_range[0] + (j * self.voxel_size + self.voxel_size / 2)
                # 遍历体素
                mask = torch.eq(vl_indices, j).all(dim=1)
                pt_indices = torch.nonzero(mask).squeeze()
                pt_num = counts[e]
                if pt_num > self.voxel_point_max:
                    random_pick = torch.randperm(pt_num)[:self.voxel_point_max]
                    pt_indices = pt_indices[random_pick]
                    pt = pc_batch[pt_indices, :]   # (50, 3)
                else:
                    repeat_num = int(torch.ceil(self.voxel_point_max/pt_num))
                    pt_indices = pt_indices.repeat(repeat_num)[:self.voxel_point_max]
                    pt = pc_batch[pt_indices, :]   # (50, 3)
                pt_list.append(pt)
                vl_id_list.append(vl_id)
                if e >= (self.voxel_num - 1):
                    break
            pts = torch.stack(pt_list, dim=0)   # (50, 40, 3)
            vl_ids = torch.stack(vl_id_list, dim=0) # (50, 3)
            batch_pts_list.append(pts)
            batch_vl_ids_list.append(vl_ids)

            # 找到与kpts最近的vl_id，并对该vl_id进行kpt condition编码 (4+1)
            kpt_condition = self.get_kpt_condition(vl_ids, kpts_batch)
            batch_kpt_condition_list.append(kpt_condition)
        batch_pts = torch.stack(batch_pts_list, dim=0)  # (B, 50, 40, 3)
        batch_vl_ids = torch.stack(batch_vl_ids_list, dim=0)    # (B, 50, 3)
        batch_kpt_condition = torch.stack(batch_kpt_condition_list, dim=0)  # (B, 50, 6)
        return batch_pts, batch_vl_ids, batch_kpt_condition

    def get_invoxel_points_with_id_without_kpt(self, pc):
        '''
        pc: (B, N, 3)
        '''
        B, N, _ = pc.shape
        device = pc.device

        # 计算体素网格各个维度的数量
        grid_range = torch.tensor(self.grid_range, device=device)
        voxel_dims = ((grid_range[1] - grid_range[0]) / self.voxel_size).int()

        batch_pts_list, batch_vl_ids_list = [], []
        for i in range(B):
            pc_batch = pc[i]    # (N, 3)
            # 过滤补齐的（0，0，0）
            pc_batch = pc_batch[torch.any(pc_batch != 0, dim=1)]
            max_coordinate = abs(torch.max(pc_batch.view(-1)).item())
            min_coordinate = abs(torch.min(pc_batch.view(-1)).item())
            base = max(max_coordinate, min_coordinate)
            pc_batch = pc_batch/base
            vl_indices = ((pc_batch - grid_range[0]) / self.voxel_size).long()
            mask = (vl_indices >= 0) & (vl_indices < voxel_dims)
            vl_indices = vl_indices[mask.all(dim=1)]
            unique_vl_indices, counts = torch.unique(vl_indices, return_counts=True, dim=0)
            # 根据 counts 进行降序排列
            sort = torch.argsort(counts, descending=True)
            counts = counts[sort]
            unique_vl_indices = unique_vl_indices[sort]
            pt_list, vl_id_list = [], []
            for e, j in enumerate(unique_vl_indices):
                vl_id = grid_range[0] + (j * self.voxel_size + self.voxel_size / 2)
                # 遍历体素
                mask = torch.eq(vl_indices, j).all(dim=1)
                pt_indices = torch.nonzero(mask).squeeze()
                pt_num = counts[e]
                if pt_num > self.voxel_point_max:
                    random_pick = torch.randperm(pt_num)[:self.voxel_point_max]
                    pt_indices = pt_indices[random_pick]
                    pt = pc_batch[pt_indices, :]   # (50, 3)
                else:
                    repeat_num = int(torch.ceil(self.voxel_point_max/pt_num))
                    pt_indices = pt_indices.repeat(repeat_num)[:self.voxel_point_max]
                    pt = pc_batch[pt_indices, :]   # (50, 3)
                pt_list.append(pt)
                vl_id_list.append(vl_id)
                if e >= (self.voxel_num - 1):
                    break
            pts = torch.stack(pt_list, dim=0)   # (50, 40, 3)
            vl_ids = torch.stack(vl_id_list, dim=0) # (50, 3)
            batch_pts_list.append(pts)
            batch_vl_ids_list.append(vl_ids)

            # 找到与kpts最近的vl_id，并对该vl_id进行kpt condition编码 (4+1)
            # kpt_condition = self.get_kpt_condition(vl_ids, kpts_batch)
            # batch_kpt_condition_list.append(kpt_condition)
        batch_pts = torch.stack(batch_pts_list, dim=0)  # (B, 50, 40, 3)
        batch_vl_ids = torch.stack(batch_vl_ids_list, dim=0)    # (B, 50, 3)
        # batch_kpt_condition = torch.stack(batch_kpt_condition_list, dim=0)  # (B, 50, 6)
        return batch_pts, batch_vl_ids


    @torch.no_grad()
    def forward_sample(
        self,
        batch_data: dict,
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 500,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = 100,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):
        pc = batch_data['input_lidar']
        kpt = batch_data['keypoints']

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Sample voxel noise from pc
        B, N, C = pc.shape
        device = pc.device
        invl_pts, vl_id, kpt_condition = self.get_invoxel_points_with_id(pc, kpt)
        B, V, P, _ = invl_pts.shape # (B, 50, 40, 3)
        vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
        vl_id = vl_id.expand(B, V, P, 3)    # (B, V, P, 3)
        kpt_condition = kpt_condition.unsqueeze(2)  # (B, V, 1, 6)
        kpt_condition = kpt_condition.expand(B, V, P, 6)    # (B, V, P, 6)

        x_t = torch.randn_like(invl_pts)   # (B, V, P, 3)
        std = x_t.std(dim=2, unbiased=False)
        x_t = x_t / std.view(B, V, 1, -1) * self.noise_std

        # Set time steps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        for i, t in enumerate(progress_bar):
            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1) # (B, V, P, 3+3)
            # keypoint condition
            x_t_input = torch.cat([x_t_input, kpt_condition], dim=-1)   # (B, V, P, 6+6)
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        # Convert output back into point cloud
        output = (x_t - self.bias) / self.weight + vl_id
        output = output.reshape(B, V*P, -1)

        if return_all_outputs:
            all_outputs = [(o-self.bias)/self.weight+vl_id for o in all_outputs]
            all_outputs = [o.reshape(B, V*P, -1) for o in all_outputs]

        batch_data['pc_diffusion_output'] = output  # (B, V*P, 3)
        batch_data['pc_diffusion_all_outputs'] = all_outputs    # [(B, V*P, 3)]
        batch_data['pc_diffusion_input'] = invl_pts.reshape(B, V*P, -1) # (B, V*P, 3)
        return batch_data

    def forward_train(
            self,
            batch_data: dict,
            use_kpt = True,
            return_intermediate_steps: bool = False
    ):
        if use_kpt:
            pc = batch_data['input_lidar']
            kpt = batch_data['keypoints']

            B, N, _ = pc.shape
            b, n, _ = kpt.shape
            device = pc.device

            invl_pts, vl_id, kpt_condition = self.get_invoxel_points_with_id(pc, kpt)
            B, V, P, _ = invl_pts.shape # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)    # (B, V, P, 3)
            kpt_condition = kpt_condition.unsqueeze(2)  # (B, V, 1, 6)
            kpt_condition = kpt_condition.expand(B, V, P, 6)    # (B, V, P, 6)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)   # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)    # (B, V, P, 3)


            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)   # (B, V, P, 3+3)
            # keypoint condition
            x_t_input = torch.cat([x_t_input, kpt_condition], dim=-1)  # (B, V, P, 6+6)

            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)   # (B, V, P, 3)

            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss
        else:
            pc = batch_data['input_lidar']
            B, N, _ = pc.shape
            device = pc.device

            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)    # (B, V, P, 3)


            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)   # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)    # (B, V, P, 3)


            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)   # (B, V, P, 3+3)


            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)   # (B, V, P, 3)

            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

