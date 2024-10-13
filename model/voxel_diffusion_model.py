import inspect
from typing import Optional
from pytorch3d.structures import Pointclouds
import torchvision.models
from tqdm import tqdm
from torch.nn import MultiheadAttention
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
import timm
from .vae_impl import VAE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from torchvision.models.resnet import resnet50
from .point_cloud_model import PointCloudModel
from .pvcnn_model import PVCNNModel
from .conv3d_model import Conv3DModel
from .depth_condition_model import DepthConditionModel
from .vae_impl import vae_loss_function
from .model_utils import point_cloud_to_voxel_grid
from .feature_model import FeatureModel
from pytorch3d.renderer import (
    FoVOrthographicCameras, look_at_view_transform, RasterizationSettings,
    PointsRasterizer, PointsRenderer, AlphaCompositor, PerspectiveCameras, PointsRasterizationSettings
)
def calculate_noise_std(signal_std, snr_db):
    # 信噪比（SNR）转换为线性尺度
    snr_linear = 10 ** (snr_db / 10)
    # 计算噪声标准差
    noise_std = signal_std / torch.sqrt(torch.tensor(snr_linear))
    return noise_std


def get_projection(pc, rasterizer, rasterized_condition):
    B, C, H, W, device = *rasterized_condition.shape, rasterized_condition.device
    N = pc.shape[1]
    fragments = rasterizer(Pointclouds(pc))  # (B, H, W, R)
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]
    R = 10
    # Reshape local features to (B, H, W, R, C)
    local_features = rasterized_condition.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * N, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, N, C)

    # import pdb; pdb.set_trace()
    return local_features_proj


def get_dist_condition(device, rasterizer, point_clouds, B, N):
    fragments = rasterizer(point_clouds)
    fragments_idx = fragments.idx.long()
    points_to_dist = torch.full((B*N, 1), -1, device=device).float()
    points_to_e_dist = torch.full((B*N, 1), -1, device=device).float()
    nearest_visible = (fragments_idx[:, :, :, 0] > -1)
    dist = fragments.dists[:, :, :, 0]
    e_dist = fragments.zbuf[:, :, :, 0]
    # nearest_visible = (fragments_idx[:, :, :, 0] > -1)
    nearest_points_to_dist = fragments_idx[:, :, :, 0][nearest_visible]    # get the valid nearest point of the pixel
    # points_to_dist[nearest_points_to_dist] = dist.unsqueeze(dim=1)[nearest_points_to_dist]
    points_to_dist[nearest_points_to_dist] = dist.unsqueeze(dim=-1)[nearest_visible]
    # points_to_e_dist[nearest_points_to_dist] = e_dist.unsqueeze(dim=1)[nearest_points_to_dist]
    points_to_e_dist[nearest_points_to_dist] = e_dist.unsqueeze(dim=-1)[nearest_visible]

    points_to_dist = points_to_dist.reshape(B, N, 1)
    points_to_e_dist = points_to_e_dist.reshape(B, N, 1)
    dist_condition = torch.cat((points_to_dist, points_to_e_dist), dim=-1)
    return dist_condition


class ConditionalVoxelDiffusionModel(nn.Module):
    def __init__(
        self,
        beta_start: float = 1e-5,
        beta_end: float = 8e-3,
        beta_schedule: str = 'linear',
        point_cloud_model: str = 'pvcnn',
        point_cloud_model_embed_dim: int = 64,
        voxel: bool = False,
        grid_range: [int] = [],
        voxel_size: float = 0.0,
        voxel_point_max: int = 0,
        depth_condition_model_path: str = None,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {'num_train_timesteps': 100}
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(

            embed_dim=point_cloud_model_embed_dim,
            in_channels=3,
            out_channels=3,
        )

        # Create conv3d model for processing voxel at each diffusion step
        self.conv3d_model = Conv3DModel(
            in_channels=2,
            out_channels=1,
        )

        # Create depth condition model
        self.bev_encoding = True
        self.depth_condition_model = DepthConditionModel(
            grid_range = grid_range,
            voxel_size = voxel_size,
            bev_encoding = self.bev_encoding,
            model_path = depth_condition_model_path,
        )


        # Normalization
        self.norm_mean = 1.0
        self.norm_std = 1.0

        # Voxelization
        self.voxel = voxel
        if self.voxel:
            self.grid_range = grid_range
            self.voxel_size = voxel_size
            self.voxel_point_max = voxel_point_max

    def denormalize(self, x: Tensor):
        x = x * self.norm_std + self.norm_mean
        return x

    def normalize(self, x: Tensor):
        self.norm_mean = torch.mean(x, dim=(0, 1))
        self.norm_std = torch.std(x, dim=(0, 1))
        x = (x - self.norm_mean) / self.norm_std
        return x

    def voxel_to_point_cloud(self, voxel):
        B, C, Dx, Dy, Dz = voxel.shape
        voxel = torch.relu(voxel)
        if B != 1:
            raise ValueError("Batch size must be 1 in sample forward")
        voxel = voxel.cpu()
        mask = voxel.squeeze()*self.voxel_point_max

        # 定义网格的维度
        grid_size = (Dx, Dy, Dz)

        # 定义实际点云坐标范围
        [[x_min, y_min, z_min], [x_max, y_max, z_max]] = self.grid_range
        x_range = torch.arange(x_min, x_max+1, (x_max-x_min+1)/Dx)
        y_range = torch.arange(y_min, y_max+1, (y_max-y_min+1)/Dx)
        z_range = torch.arange(z_min, z_max+1, (z_max-z_min+1)/Dx)


        # 随机生成点云坐标
        point_clouds = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    # 获取当前网格中可以存在的点云数量
                    num_points = int(mask[i, j, k])
                    if num_points > 0:
                        # print((i, j, k), (num_points))

                        # 在当前网格中随机生成 num_points 个点云坐标
                        points = torch.rand(num_points, 3) * self.voxel_size + \
                                 torch.stack((x_range[i], y_range[j], z_range[k]), dim=0)
                        # print(points)
                        point_clouds.append(points)

        # 将点云列表转换成张量
        point_clouds = torch.cat(point_clouds, dim=0).unsqueeze(0)
        return point_clouds

    def get_input_with_depth_conditioning(self, x_t, depth_bev_condition, depth_condition=None):
        '''
        x_t: (B, C, Dx, Dy, Dz)
        depth_bev_condition: (B, 1, Dx, Dy)
        depth_condition: (B, C, Dx, Dy)
        '''
        B, C, Dx, Dy, Dz = x_t.shape
        depth_bev_condition = torch.unsqueeze(depth_bev_condition, dim=4)  # (B, 1, Dx, Dy, 1)
        depth_bev_condition_exp = depth_bev_condition.expand(B, 1, Dx, Dy, Dz)  # (B, 1, Dx, Dy, Dz)
        condition = torch.mul(x_t, depth_bev_condition_exp)  # (B, 1, Dx, Dy, Dz)
        if depth_condition is not None:
            C_d = depth_condition.shape[1]
            depth_condition = torch.unsqueeze(depth_condition, dim=4)   # (B, C_d, Dx, Dy, 1)
            depth_condition_exp = depth_condition.expand(B, C_d, Dx, Dy, Dz)
            condition = torch.cat((condition, depth_condition_exp), dim=1)  # (B, C_d+1, Dx, Dy, Dz)
        x_t_input = torch.cat((x_t, condition), dim=1)  # (B, 2, Dx, Dy, Dz) or (B, C_d+2, Dx, Dy, Dz)
        return x_t_input

    @torch.no_grad()
    def forward_sample(
        self,
        pc: Tensor,
        depth: Tensor = None,
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = 10,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):
        """
        由于每个batch的点云数量不一致，forward_sample时，batch size只能为1
        """

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Sample noise
        if self.voxel:
            # Convert point could to voxel
            vl = point_cloud_to_voxel_grid(pc, self.grid_range, self.voxel_size, self.voxel_point_max)
            B, C, Dx, Dy, Dz = vl.shape
            device = vl.device
            x_t = torch.randn_like(vl, device=device)   # （B, C, Dx, Dy, Dz）
            x_t = (x_t - x_t.mean()) / x_t.std()
        else:
            B, N, D = pc.shape
            device = pc.device
            x_t = torch.randn(B, N, D, device=device)

        # Depth condition (BEV format)
        if self.bev_encoding:
            # depth_bev_condition -> bev output; depth_condition -> bev feature
            depth_bev_condition, _ = self.depth_condition_model(depth)
            depth_condition = None
        else:
            depth_bev_condition = self.depth_condition_model(depth)
            depth_condition = None

        # Set timesteps
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

        if not self.voxel:
            x_t = self.normalize(x_t)
        for i, t in enumerate(progress_bar):
            if self.voxel:
                # Conditioning
                x_t_input = self.get_input_with_depth_conditioning(x_t, depth_bev_condition, depth_condition)   # (B, 2, Dx, Dy, Dz) or (B, 258, Dx, Dy, Dz)

                noise_pred = self.conv3d_model(x_t_input, t.reshape(1).expand(B))
            else:
                x_t_input = x_t
                noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))

            # TODO: step 会导致分布变化
            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
            # print(torch.mean(x_t))
            # print(torch.var(x_t))

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        if self.voxel:
            (w, b) = (2, -1)
            vl = (x_t - b) / w
            output = self.voxel_to_point_cloud(vl)
        else:
            output = self.denormalize(x_t)

        if return_all_outputs:
            if self.voxel:
                (w, b) = (2, -1)
                all_outputs = [self.voxel_to_point_cloud((o-b)/w) for o in all_outputs]
            else:
                all_outputs = [self.denormalize(o) for o in all_outputs]
        
        return (output, all_outputs) if return_all_outputs else output

    def forward_train(
            self,
            pc: Tensor,
            depth: Tensor = None,
            return_intermediate_steps: bool = False
    ):
        if self.voxel:
            # Convert point could to voxel
            vl = point_cloud_to_voxel_grid(pc, self.grid_range, self.voxel_size, self.voxel_point_max)
            B, C, Dx, Dy, Dz = vl.shape
            x_0 = vl.reshape(B, C, -1).permute(0, 2, 1)
            (w, b) = (2, -1)
            x_0 = w * x_0 + b
            B, N, D = x_0.shape
        else:
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape

        device = x_0.device

        # Sample random noise ~ N(0, 1)
        noise = torch.randn_like(x_0)
        noise = (noise - noise.mean()) / noise.std()
        # print(torch.mean(noise))
        # print(torch.var(noise))

        # Depth condition (BEV format)
        if self.bev_encoding:
            # depth_bev_condition -> bev output; depth_condition -> bev feature
            depth_bev_condition, depth_condition = self.depth_condition_model(depth)
            depth_condition = None
        else:
            depth_bev_condition = self.depth_condition_model(depth)
            depth_condition = None

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                 device=device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # Forward
        if self.voxel:
            x_t_ = x_t.permute(0, 2, 1).reshape(B, C, Dx, Dy, Dz)

            # Conditioning
            x_t_input = self.get_input_with_depth_conditioning(x_t_, depth_bev_condition, depth_condition)

            noise_pred = self.conv3d_model(x_t_input, timestep) # (32, 258, 32, 20)
            noise_pred = noise_pred.reshape(B, -1, Dx*Dy*Dz).permute(0, 2, 1)
        else:
            x_t_input = x_t
            noise_pred = self.point_cloud_model(x_t_input, timestep)
        # print('noise_pred -------------')
        # print(torch.mean(noise_pred, dim=1))
        # print(torch.var(noise_pred, dim=1))

        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss


class PhysicalEncoder(nn.Module):
    def __init__(self, channel=2):
        super(PhysicalEncoder, self).__init__()
        self.vae1 = VAE()
        self.vae2 = VAE()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conx = torch.nn.Conv1d(1, 2, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bnx = nn.BatchNorm1d(2)
        self.fc = torch.nn.Linear(1024,1024)

    def forward(self, x):
        recon_batch1, mu1, logvar1 = self.vae1(x[:, :, 0])
        vae_loss1 = vae_loss_function(recon_batch1, x[:, :, 0], mu1, logvar1)
        recon_batch2, mu2, logvar2 = self.vae2(x[:, :, 1])
        vae_loss2 = vae_loss_function(recon_batch2, x[:, :, 1], mu2, logvar2)

        # xy = x[:, :, 0].unsqueeze(-1)
        # xy = xy.transpose(2, 1)
        # xy = F.relu(self.bnx(self.conx(xy)))
        #
        # x = x.transpose(2, 1)
        # z = x[:, 1, :].unsqueeze(1)
        # x = torch.concat((xy, z), dim=1)

        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.transpose(2,1)
        x = self.fc(x)
        return x, vae_loss1, vae_loss2, mu1, mu2

class VaePhysicalEncoder(nn.Module):
    def __init__(self, channel=3):
        super(VaePhysicalEncoder, self).__init__()
        self.coordinate_dim = 20
        self.vae1 = VAE()
        self.vae2 = VAE()
        self.conx = torch.nn.Conv1d(1, self.coordinate_dim, 1)
        self.cony = torch.nn.Conv1d(1, self.coordinate_dim, 1)
        self.conz = torch.nn.Conv1d(1, self.coordinate_dim, 1)
        self.conv1 = torch.nn.Conv1d(self.coordinate_dim*3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bnx = nn.BatchNorm1d(20)
        self.bny = nn.BatchNorm1d(20)
        self.bnz = nn.BatchNorm1d(20)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(1024, 1024)
        self.recon_fc = nn.Linear(1, 2)

    def forward(self, x):
        xy = x[:, :, 0]
        z = x[:, :, 1]
        recon_batch1, mu1, logvar1 = self.vae1(xy)
        vae_loss1 = vae_loss_function(recon_batch1, x[:, :, 0], mu1, logvar1)
        recon_batch2, mu2, logvar2 = self.vae2(z)
        vae_loss2 = vae_loss_function(recon_batch2, x[:, :, 1], mu2, logvar2)
        xy = xy.unsqueeze(-1)
        xy_reconstruct = self.recon_fc(xy)
        x, y = xy_reconstruct[:, :, 0], xy_reconstruct[:, :, 1]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        x = F.relu(self.bnx(self.conx(x)))
        y = F.relu(self.bny(self.cony(y)))
        z = F.relu(self.bnz(self.conz(z)))
        xyz = torch.cat((x, y, z), dim=1)

        xyz = F.relu(self.bn1(self.conv1(xyz)))
        xyz = F.relu(self.bn2(self.conv2(xyz)))
        xyz = F.relu(self.bn3(self.conv3(xyz)))
        xyz = xyz.transpose(2,1)
        xyz = self.fc(xyz)
        return xyz, vae_loss1, vae_loss2, mu1, mu2

class MaskEncoder(nn.Module):
    def __init__(self, channel=3):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=3)
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x = self.avg_pool(x.float())
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        return x


class ConditionalSegVoxelDiffusionModel(nn.Module):
    def __init__(
            self,
            point_cloud_model_type: str = 'pvcnnplusplus',
            point_cloud_model_embed_dim: int = 64,
            voxel: bool = False,
            voxel_num = 0,
            grid_range: [int] = [],
            voxel_size: float = 0.0,
            voxel_point_max: int = 0,
            timestep_num = 0,
            simple_point_in_chan = 0,
            **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {'num_train_timesteps': timestep_num}
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            'pndm': PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference
        self.voxel = voxel
        self.text_hidden_dim = 768
        # x_0 transformation
        self.weight = 3.0
        self.bias = 0.0
        self.use_vae = False
        self.vae_count = 0
        camera_distance = 2.0
        object_position = torch.tensor([0.0, 0.0, 0.0])
        camera_position = object_position + torch.tensor([0.0, 0.0, -camera_distance])

        # 计算相机的旋转矩阵 R 和平移矩阵 T
        R, T = look_at_view_transform(20, 10, 0)
        # R, T = look_at_view_transform(camera_position, object_position, torch.tensor([0.0, 1.0, 0.0]))

        # 设置相机参数
        cameras = FoVOrthographicCameras(device=torch.device("cuda:0"), R=R, T=T, znear=0.1)
        # cameras = FoVOrthographicCameras(device=torch.device("cpu"), R=R, T=T, znear=0.1)

        # 设置光栅化参数
        raster_settings = PointsRasterizationSettings(
            image_size=224,
            radius=0.05,
            points_per_pixel=10,

        )


        # 创建点云光栅化器
        self.rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

        # 创建点云渲染器
        self.renderer = PointsRenderer(rasterizer=self.rasterizer, compositor=AlphaCompositor())
        self.physical_encoder = VaePhysicalEncoder()

        # 文本特征聚合器
        self.text_aggregator = nn.Linear(self.text_hidden_dim, 768)
        self.mask_encoder = MaskEncoder()

        # noise
        self.noise_mean = 0.0
        self.noise_std = 1.0
        if self.voxel:
            self.grid_range = grid_range
            self.voxel_size = voxel_size
            self.voxel_point_max = voxel_point_max
            self.voxel_num = voxel_num
        # self.img_encoder = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.img_encoder = torchvision.models.vit_b_16(pretrained=True)
        self.img_encoder.heads.head = nn.Identity()

        self.rasterized_encoder = torchvision.models.vit_b_16(pretrained=True)
        self.rasterized_encoder.heads.head = nn.Identity()
        # self.img_encoder = resnet50(pretrained=True)
        # self.img_encoder.fc = nn.Linear(in_features=self.img_encoder.fc.in_features, out_features=3000)
        # self.img_encoder.fc = nn.Linear(in_features=self.img_encoder.fc.in_features, out_features=3072)

        for param in self.img_encoder.parameters():
            param.requires_grad = False
        for param in self.rasterized_encoder.parameters():
            param.requires_grad = False


        self.downmlp = nn.Linear(in_features=768, out_features=64)
        self.num_heads = 4
        self.embed_dim = 64
        self.attention = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.LN = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.conv = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        # self.point_cloud_model = PVCNNModel(model_type=point_cloud_model_type, in_channels=3)
        self.point_cloud_model = PVCNNModel(model_type=point_cloud_model_type, in_channels=simple_point_in_chan)
        self.fc = nn.Linear(in_features=64, out_features=3)

        # Create conv3d model for processing voxel at each diffusion step
        self.conv3d_model = Conv3DModel(
            in_channels=2,
            out_channels=1,
        )

        # Create depth condition model
        self.bev_encoding = True

        # Normalization
        self.norm_mean = 1.0
        self.norm_std = 1.0


    def denormalize(self, x: Tensor):
        x = x * self.norm_std + self.norm_mean
        return x

    def normalize(self, x: Tensor):
        self.norm_mean = torch.mean(x, dim=(0, 1))
        self.norm_std = torch.std(x, dim=(0, 1))
        x = (x - self.norm_mean) / self.norm_std
        return x

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

    def voxel_to_point_cloud(self, voxel):
        B, C, Dx, Dy, Dz = voxel.shape
        voxel = torch.relu(voxel)
        if B != 1:
            raise ValueError("Batch size must be 1 in sample forward")
        voxel = voxel.cpu()
        mask = voxel.squeeze() * self.voxel_point_max

        # 定义网格的维度
        grid_size = (Dx, Dy, Dz)

        # 定义实际点云坐标范围
        [[x_min, y_min, z_min], [x_max, y_max, z_max]] = self.grid_range
        x_range = torch.arange(x_min, x_max + 1, (x_max - x_min + 1) / Dx)
        y_range = torch.arange(y_min, y_max + 1, (y_max - y_min + 1) / Dx)
        z_range = torch.arange(z_min, z_max + 1, (z_max - z_min + 1) / Dx)

        # 随机生成点云坐标
        point_clouds = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    # 获取当前网格中可以存在的点云数量
                    num_points = int(mask[i, j, k])
                    if num_points > 0:
                        # print((i, j, k), (num_points))

                        # 在当前网格中随机生成 num_points 个点云坐标
                        points = torch.rand(num_points, 3) * self.voxel_size + \
                                 torch.stack((x_range[i], y_range[j], z_range[k]), dim=0)
                        # print(points)
                        point_clouds.append(points)

        # 将点云列表转换成张量
        point_clouds = torch.cat(point_clouds, dim=0).unsqueeze(0)
        return point_clouds

    def get_input_with_depth_conditioning(self, x_t, depth_bev_condition, depth_condition=None):
        '''
        x_t: (B, C, Dx, Dy, Dz)
        depth_bev_condition: (B, 1, Dx, Dy)
        depth_condition: (B, C, Dx, Dy)
        '''
        B, C, Dx, Dy, Dz = x_t.shape
        depth_bev_condition = torch.unsqueeze(depth_bev_condition, dim=4)  # (B, 1, Dx, Dy, 1)
        depth_bev_condition_exp = depth_bev_condition.expand(B, 1, Dx, Dy, Dz)  # (B, 1, Dx, Dy, Dz)
        condition = torch.mul(x_t, depth_bev_condition_exp)  # (B, 1, Dx, Dy, Dz)
        if depth_condition is not None:
            C_d = depth_condition.shape[1]
            depth_condition = torch.unsqueeze(depth_condition, dim=4)  # (B, C_d, Dx, Dy, 1)
            depth_condition_exp = depth_condition.expand(B, C_d, Dx, Dy, Dz)
            condition = torch.cat((condition, depth_condition_exp), dim=1)  # (B, C_d+1, Dx, Dy, Dz)
        x_t_input = torch.cat((x_t, condition), dim=1)  # (B, 2, Dx, Dy, Dz) or (B, C_d+2, Dx, Dy, Dz)
        return x_t_input

    @torch.no_grad()
    def forward_sample(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt=None,
            tokenizer=None,
            text_encoder=None,
            # Optional overrides
            scheduler: Optional[str] = 'ddpm',
            # Inference parameters
            num_inference_steps: Optional[int] = 100,
            eta: Optional[float] = 0.0,  # for DDIM
            # Whether to return all the intermediate steps in generation
            return_sample_every_n_steps: int = 10,
            # Whether to disable tqdm
            disable_tqdm: bool = False,
            mode = None
    ):
        """
        由于每个batch的点云数量不一致，forward_sample时，batch size只能为1
        """

        # Get scheduler from mapping, or use self.scheduler if None
        # seg = seg.permute([0, 3, 1, 2])
        # seg_condition = self.img_encoder(seg)
        # seg_condition = seg_condition.unsqueeze(-1)

        if self.voxel:
            scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

            # Sample noise
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            device = pc.device
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 50, 40, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)
            x_t = torch.randn_like(invl_pts)  # (B, V, P, 3)
            std = x_t.std(dim=2, unbiased=False)
            x_t = x_t / std.view(B, V, 1, -1) * self.noise_std


            # Set timesteps
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

                x_t_input = x_t
                x_t_input = torch.cat([x_t_input, vl_id], dim=-1)  # (B, V, P, 3+3)

                x_t_input = x_t_input.view(B, V * P, -1)
                x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
                # Forward
                noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))  # (B, V, P, 3)
                noise_pred = noise_pred.view(B, V, P, -1)
                # TODO: step 会导致分布变化
                # Step
                x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
                # print(torch.mean(x_t))
                # print(torch.var(x_t))

                # Append to output list if desired
                if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                    all_outputs.append(x_t)

            # Convert output back into a point cloud, undoing normalization and scaling

            output = (x_t - self.bias) / self.weight + vl_id
            output = output.reshape(B, V * P, -1)

            if return_all_outputs:
                all_outputs = [(o - self.bias) / self.weight + vl_id for o in all_outputs]
                all_outputs = [o.reshape(B, V * P, -1) for o in all_outputs]

            return (output, all_outputs) if return_all_outputs else output
        else:
            # max_length = 77
            # encoded_input = tokenizer.batch_encode_plus(
            #     prompt,
            #     add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
            #     padding='max_length',  # 填充到最大长度
            #     max_length=max_length,  # 最大序列长度
            #     return_attention_mask=True,  # 返回注意力掩码
            #     truncation=True,
            #     return_tensors='pt'  # 返回PyTorch张量
            # )
            B, N, D = pc.shape
            device = pc.device

            # encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
            # output = text_encoder(**encoded_input)
            # text_emb = output['last_hidden_state']

            # text_emb = torch.mean(text_emb, dim=1)
            # text_emb = text_emb.repeat(1, 4)
            # text_emb = text_emb.unsqueeze(dim=-1)

            # text_emb = text_emb.repeat(1, 1, 4)
            _, H, W, _ = seg.shape
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)

            # signal_std = torch.std(seg_condition)
            # signal_mean = torch.mean(seg_condition)
            # # random_snr = random.uniform(0.1, 20)
            # snr = 20
            # noise_std = calculate_noise_std(signal_std, snr)
            # noise_cn = torch.normal(0, noise_std.detach(), seg_condition.size())
            # seg_condition = seg_condition + noise_cn.to(device)

            seg_condition = seg_condition.unsqueeze(-1)
            seg_condition = seg_condition.permute(0, 2, 1)
            # seg_condition = seg_condition.expand(-1, N, -1)
            seg_condition = seg_condition.repeat(1, N, 1)
            scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

            # Sample noise


            x_t = torch.randn(B, N, D, device=device)

            # Set timesteps
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


            # seg_condition = self.downmlp(seg_condition)
            x_t = self.normalize(x_t)
            rgb = torch.ones(B, N, 3).to(device)
            point_clouds = Pointclouds(points=pc, features=rgb)
            rasterized = self.renderer(point_clouds)

            if mode == 'dist':
                # # distance condition
                dist_condition = get_dist_condition(device=device, rasterizer=self.rasterizer, point_clouds=point_clouds, B=B, N=N)
                dist_condition, vae_loss1, vae_loss2, mu1, mu2 = self.physical_encoder(dist_condition)

                # # vae encoding (need dist_condition)
                # mu1, logvar1 = self.physical_encoder.vae1.encode(dist_condition[:, :, 0])
                # mu2, logvar2 = self.physical_encoder.vae2.encode(dist_condition[:, :, 1])
                # mu1 = mu1.unsqueeze(1).repeat(1, N, 1)
                # mu2 = mu2.unsqueeze(1).repeat(1, N, 1)
                # dist_condition = torch.cat((dist_condition, mu1), dim=-1)
                # dist_condition = torch.cat((dist_condition, mu2), dim=-1)


            if mode == 'proj':
                # signal_std = torch.std(rasterized)
                # signal_mean = torch.mean(rasterized)
                # # snr = random.uniform(0, 20)
                # snr = 15
                # noise_std = calculate_noise_std(signal_std, snr)
                # noise_cn = torch.normal(0, noise_std.detach(), rasterized.size())
                # rasterized = rasterized + noise_cn.to(device)
                #
                rasterized = rasterized[:, :, :, :3].permute([0, 3, 1, 2])
                x = self.rasterized_encoder._process_input(rasterized)
                n = x.shape[0]
                # Expand the class token to the full batch
                batch_class_token = self.rasterized_encoder.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                rasterized_condition = self.rasterized_encoder.encoder(x)
                rasterized_condition = rasterized_condition[:, 1:, :].reshape(-1,
                                                                              H // self.rasterized_encoder.patch_size,
                                                                              W // self.rasterized_encoder.patch_size,
                                                                              self.rasterized_encoder.hidden_dim)
                rasterized_condition = rasterized_condition.permute([0, 3, 1, 2])
                rasterized_condition = F.interpolate(rasterized_condition, size=(H, W), mode='bilinear',
                                                     align_corners=False)
                projection_condition = get_projection(pc, self.rasterizer, rasterized_condition)
            if mode == 'raster':
                # raster condition
                rgb = torch.ones(B, N, 3).to(device)
                point_clouds = Pointclouds(points=pc, features=rgb)
                rasterized = self.renderer(point_clouds)
                rasterized = rasterized[:, :, :, :3].permute([0, 3, 1, 2])
                rasterized_condition = self.rasterized_encoder(rasterized).unsqueeze(1).repeat(1, N, 1)

            for i, t in enumerate(progress_bar):
                x_t_input = x_t

                x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
                if mode=='raster':
                    x_t_input = torch.cat((x_t_input, rasterized_condition), dim=-1)
                elif mode=='dist':
                    x_t_input = torch.cat((x_t_input, dist_condition), dim=-1)
                elif mode=='proj':
                    x_t_input = torch.cat((x_t_input, projection_condition), dim=-1)

                # # x_t_input = torch.cat((x_t_input, text_emb),dim=-1)
                noise_pred, feature = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))



                # # feature = feature.permute(0, 2, 1)
                #
                # attn_output, attn_output_weights = self.attention(feature, seg_condition, seg_condition)
                # attn_output = self.LN(attn_output)
                # # attn_output = attn_output.view(B*N, -1)
                # # attn_output = self.fc(attn_output)
                # # attn_output = attn_output.view(B, N, 3)
                # attn_output = attn_output.permute(0, 2, 1)
                # attn_output = self.conv(attn_output)
                # x = attn_output.permute(0, 2, 1)
                # noise_pred = noise_pred + x


                # TODO: step 会导致分布变化
                # Step
                x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
                # print(torch.mean(x_t))
                # print(torch.var(x_t))

                # Append to output list if desired
                if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                    all_outputs.append(x_t)

            # Convert output back into a point cloud, undoing normalization and scaling
            output = self.denormalize(x_t)

            if return_all_outputs:

                all_outputs = [self.denormalize(o) for o in all_outputs]

            return (output, all_outputs) if return_all_outputs else output




    def forward_train(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt = None,
            tokenizer = None,
            text_encoder = None,
            return_intermediate_steps: bool = False
    ):

        if self.voxel:
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            B, N, _ = pc.shape
            device = pc.device
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(-1)
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)  # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)  # (B, V, P, 3)

            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)  # (B, V, P, 3+3)
            x_t_input = x_t_input.view(B, V*P, -1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)  # (B, V, P, 3)
            noise_pred = noise_pred.view(B, V, P, -1)
            # Check
            # if not noise_pred.shape == noise.shape:
            #     raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

        else:
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape
            device = x_0.device

            # max_length = 77
            # encoded_input = tokenizer.batch_encode_plus(
            #     prompt,
            #     add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
            #     padding='max_length',  # 填充到最大长度
            #     max_length=max_length,  # 最大序列长度
            #     return_attention_mask=True,  # 返回注意力掩码
            #     truncation=True,
            #     return_tensors='pt'  # 返回PyTorch张量
            # )
            # encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
            # output = text_encoder(**encoded_input)
            # text_emb = output['last_hidden_state'][:, 0]
            # text_emb = self.text_aggregator(text_emb)
            # text_emb = text_emb.unsqueeze(1).repeat(1, N, 1)

            # seg_mask = (seg == 0.0).permute([0, 3, 1, 2])
            # seg_mask = seg_mask.int() * 1
            # mask_emb = self.mask_encoder(seg_mask)
            # mask_emb = mask_emb.contiguous().view(B, -1)
            # mask_emb = mask_emb.unsqueeze(1).repeat(1, N, 1)

            # Sample random noise ~ N(0, 1)
            noise = torch.randn_like(x_0)
            noise = (noise - noise.mean()) / noise.std()
            # print(torch.mean(noise))
            # print(torch.var(noise))

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)



            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)

            # seg_condition
            x_t_input = x_t
            _, H, W, _ = seg.shape
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            # seg_condition = seg_condition.unsqueeze(1).expand(-1, N, -1)
            seg_condition = seg_condition.unsqueeze(1).repeat(1, N, 1)

            # # dist_condition
            # rgb = torch.ones(B, N, 3).to(device)
            # point_clouds = Pointclouds(points=pc, features=rgb)
            # fragments = self.rasterizer(point_clouds)
            # fragments_idx = fragments.idx.long()
            # visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
            # points_to_visible_pixels = fragments_idx[visible_pixels]
            # dist_condition = get_dist_condition(device=device,rasterizer=self.rasterizer,point_clouds=point_clouds,B=B, N=N)
            # dist_condition, vae_loss1, vae_loss2, mu1, mu2 = self.physical_encoder(dist_condition)

            # # vae encoding (need dist_condition)
            # mu1, logvar1 = self.physical_encoder.vae1.encode(dist_condition[:, :, 0])
            # mu2, logvar2 = self.physical_encoder.vae2.encode(dist_condition[:, :, 1])
            # mu1 = mu1.unsqueeze(1).repeat(1, N, 1)
            # mu2 = mu2.unsqueeze(1).repeat(1, N, 1)
            # dist_condition = torch.cat((dist_condition, mu1), dim=-1)
            # dist_condition = torch.cat((dist_condition, mu2), dim=-1)


            # rasterization condition
            # rasterized = self.renderer(point_clouds)
            # rasterized = rasterized[:, :, :, :3].permute([0, 3, 1, 2])
            # rasterized_condition = self.img_encoder(rasterized).unsqueeze(1).repeat(1, N, 1)

            # signal_std = torch.std(rasterized)
            # signal_mean = torch.mean(rasterized)
            # snr = random.uniform(0, 20)
            # # snr = 20
            # noise_std = calculate_noise_std(signal_std, snr)
            # noise_cn = torch.normal(0, noise_std.detach(), rasterized.size())
            # rasterized = rasterized + noise_cn.to(device)
            # rasterized_condition = self.rasterized_encoder(rasterized)
            # rasterized_condition = rasterized_condition.unsqueeze(1).repeat(1, N, 1)

            # x = self.rasterized_encoder._process_input(rasterized)
            # n = x.shape[0]
            # # Expand the class token to the full batch
            # batch_class_token = self.rasterized_encoder.class_token.expand(n, -1, -1)
            # x = torch.cat([batch_class_token, x], dim=1)
            # rasterized_condition = self.rasterized_encoder.encoder(x)
            # rasterized_condition = rasterized_condition[:, 1:, :].reshape(-1, H//self.rasterized_encoder.patch_size, W//self.rasterized_encoder.patch_size, self.rasterized_encoder.hidden_dim)
            # rasterized_condition = rasterized_condition.permute([0, 3, 1, 2])
            # rasterized_condition = F.interpolate(rasterized_condition, size=(H, W), mode='bilinear',
            #                              align_corners=False)
            # projection_condition = get_projection(pc, self.rasterizer, rasterized_condition)


            # signal_std = torch.std(seg_condition)
            # signal_mean = torch.mean(seg_condition)
            # random_snr = random.uniform(0, 20)
            # noise_std = calculate_noise_std(signal_std, random_snr)
            # noise_cn = torch.normal(0, noise_std.detach(), seg_condition.size())
            # seg_condition = seg_condition + noise_cn.to(device)

            # seg_condition = seg_condition.unsqueeze(-1)
            # seg_condition = seg_condition.permute(0, 2, 1)

            # # concat method
            # seg_condition = seg_condition.permute(0, 2, 1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, rasterized_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, dist_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, projection_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, text_emb), dim=-1)
            # x_t_input = torch.cat((x_t_input, mask_emb), dim=-1)
            noise_pred, feature = self.point_cloud_model(x_t_input, timestep)

            # attention method
            # noise_pred, feature = self.point_cloud_model(x_t_input, timestep)
            # # feature = feature.permute(0, 2, 1)
            # # attn_output, attn_output_weights = self.attention(feature, text_emb, text_emb)
            # seg_condition = self.downmlp(seg_condition)
            # attn_output, attn_output_weights = self.attention(feature, seg_condition, seg_condition)
            # attn_output = self.LN(attn_output)
            # # attn_output = attn_output.view(B*N, -1)
            # # attn_output = self.fc(attn_output)
            # # attn_output = attn_output.view(B, N, 3)
            # attn_output = attn_output.permute(0, 2, 1)
            # attn_output = self.conv(attn_output)
            # x = attn_output.permute(0, 2, 1)
            # noise_pred = noise_pred + x



            # print('noise_pred -------------')
            # print(torch.mean(noise_pred, dim=1))
            # print(torch.var(noise_pred, dim=1))

            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)
            # if self.use_vae:
            #     return loss, vae_loss1, vae_loss2
            return loss



    def forward_train_raster(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt = None,
            tokenizer = None,
            text_encoder = None,
            return_intermediate_steps: bool = False
    ):

        if self.voxel:
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            B, N, _ = pc.shape
            device = pc.device
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(-1)
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)  # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)  # (B, V, P, 3)

            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)  # (B, V, P, 3+3)
            x_t_input = x_t_input.view(B, V*P, -1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)  # (B, V, P, 3)
            noise_pred = noise_pred.view(B, V, P, -1)
            # Check
            # if not noise_pred.shape == noise.shape:
            #     raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

        else:
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape
            device = x_0.device


            # Sample random noise ~ N(0, 1)
            noise = torch.randn_like(x_0)
            noise = (noise - noise.mean()) / noise.std()
            # print(torch.mean(noise))
            # print(torch.var(noise))

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)



            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)

            # seg_condition
            x_t_input = x_t
            _, H, W, _ = seg.shape
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(1).repeat(1, N, 1)

            rgb = torch.ones(B, N, 3).to(device)
            point_clouds = Pointclouds(points=pc, features=rgb)


            # rasterization condition
            rasterized = self.renderer(point_clouds)
            rasterized = rasterized[:, :, :, :3].permute([0, 3, 1, 2])
            rasterized_condition = self.rasterized_encoder(rasterized).unsqueeze(1).repeat(1, N, 1)

            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            x_t_input = torch.cat((x_t_input, rasterized_condition), dim=-1)
            # print(x_t_input.shape)
            noise_pred, feature = self.point_cloud_model(x_t_input, timestep)



            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)
            # if self.use_vae:
            #     return loss, vae_loss1, vae_loss2
            return loss


    def forward_train_seg(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt = None,
            tokenizer = None,
            text_encoder = None,
            return_intermediate_steps: bool = False
    ):

        if self.voxel:
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            B, N, _ = pc.shape
            device = pc.device
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(-1)
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)  # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)  # (B, V, P, 3)

            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)  # (B, V, P, 3+3)
            x_t_input = x_t_input.view(B, V*P, -1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)  # (B, V, P, 3)
            noise_pred = noise_pred.view(B, V, P, -1)
            # Check
            # if not noise_pred.shape == noise.shape:
            #     raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

        else:
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape
            device = x_0.device


            # Sample random noise ~ N(0, 1)
            noise = torch.randn_like(x_0)
            noise = (noise - noise.mean()) / noise.std()
            # print(torch.mean(noise))
            # print(torch.var(noise))

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)



            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)

            # seg_condition
            x_t_input = x_t
            _, H, W, _ = seg.shape
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(1).repeat(1, N, 1)
            #
            # rgb = torch.ones(B, N, 3).to(device)
            # point_clouds = Pointclouds(points=pc, features=rgb)
            #
            #
            # # rasterization condition
            # rasterized = self.renderer(point_clouds)
            # rasterized = rasterized[:, :, :, :3].permute([0, 3, 1, 2])
            # rasterized_condition = self.rasterized_encoder(rasterized).unsqueeze(1).repeat(1, N, 1)

            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, rasterized_condition), dim=-1)
            # print(x_t_input.shape)
            noise_pred, feature = self.point_cloud_model(x_t_input, timestep)



            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)
            # if self.use_vae:
            #     return loss, vae_loss1, vae_loss2
            return loss

    def forward_train_dist(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt = None,
            tokenizer = None,
            text_encoder = None,
            return_intermediate_steps: bool = False
    ):

        if self.voxel:
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            B, N, _ = pc.shape
            device = pc.device
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(-1)
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)  # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)  # (B, V, P, 3)

            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)  # (B, V, P, 3+3)
            x_t_input = x_t_input.view(B, V*P, -1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)  # (B, V, P, 3)
            noise_pred = noise_pred.view(B, V, P, -1)
            # Check
            # if not noise_pred.shape == noise.shape:
            #     raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

        else:
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape
            device = x_0.device



            # Sample random noise ~ N(0, 1)
            noise = torch.randn_like(x_0)
            noise = (noise - noise.mean()) / noise.std()
            # print(torch.mean(noise))
            # print(torch.var(noise))

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)



            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)

            # seg_condition
            x_t_input = x_t
            _, H, W, _ = seg.shape
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(1).repeat(1, N, 1)

            # dist_condition
            rgb = torch.ones(B, N, 3).to(device)
            point_clouds = Pointclouds(points=pc, features=rgb)
            fragments = self.rasterizer(point_clouds)
            fragments_idx = fragments.idx.long()
            visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
            points_to_visible_pixels = fragments_idx[visible_pixels]
            dist_condition = get_dist_condition(device=device,rasterizer=self.rasterizer,point_clouds=point_clouds,B=B, N=N)
            dist_condition, vae_loss1, vae_loss2, mu1, mu2 = self.physical_encoder(dist_condition)

            # # vae encoding (need dist_condition)
            # mu1, logvar1 = self.physical_encoder.vae1.encode(dist_condition[:, :, 0])
            # mu2, logvar2 = self.physical_encoder.vae2.encode(dist_condition[:, :, 1])
            # mu1 = mu1.unsqueeze(1).repeat(1, N, 1)
            # mu2 = mu2.unsqueeze(1).repeat(1, N, 1)
            # dist_condition = torch.cat((dist_condition, mu1), dim=-1)
            # dist_condition = torch.cat((dist_condition, mu2), dim=-1)



            # # concat method
            # seg_condition = seg_condition.permute(0, 2, 1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, rasterized_condition), dim=-1)
            x_t_input = torch.cat((x_t_input, dist_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, projection_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, text_emb), dim=-1)
            # x_t_input = torch.cat((x_t_input, mask_emb), dim=-1)
            # print(x_t_input.shape)
            noise_pred, feature = self.point_cloud_model(x_t_input, timestep)



            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)
            if self.vae_count < 150:
                loss = loss + vae_loss1 + vae_loss2
                self.vae_count = self.vae_count + 1
            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)
            # if self.use_vae:
            #     return loss, vae_loss1, vae_loss2
            return loss

    def forward_train_proj(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt = None,
            tokenizer = None,
            text_encoder = None,
            return_intermediate_steps: bool = False
    ):

        if self.voxel:
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            B, N, _ = pc.shape
            device = pc.device
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(-1)
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)  # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)  # (B, V, P, 3)

            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)  # (B, V, P, 3+3)
            x_t_input = x_t_input.view(B, V*P, -1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)  # (B, V, P, 3)
            noise_pred = noise_pred.view(B, V, P, -1)
            # Check
            # if not noise_pred.shape == noise.shape:
            #     raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

        else:
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape
            device = x_0.device



            # Sample random noise ~ N(0, 1)
            noise = torch.randn_like(x_0)
            noise = (noise - noise.mean()) / noise.std()
            # print(torch.mean(noise))
            # print(torch.var(noise))

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)



            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)

            # seg_condition
            x_t_input = x_t
            _, H, W, _ = seg.shape
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(1).repeat(1, N, 1)

            # dist_condition
            rgb = torch.ones(B, N, 3).to(device)
            point_clouds = Pointclouds(points=pc, features=rgb)
            # fragments = self.rasterizer(point_clouds)
            # fragments_idx = fragments.idx.long()
            # visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
            # points_to_visible_pixels = fragments_idx[visible_pixels]
            # dist_condition = get_dist_condition(device=device,rasterizer=self.rasterizer,point_clouds=point_clouds,B=B, N=N)
            # dist_condition, vae_loss1, vae_loss2, mu1, mu2 = self.physical_encoder(dist_condition)

            # # vae encoding (need dist_condition)
            # mu1, logvar1 = self.physical_encoder.vae1.encode(dist_condition[:, :, 0])
            # mu2, logvar2 = self.physical_encoder.vae2.encode(dist_condition[:, :, 1])
            # mu1 = mu1.unsqueeze(1).repeat(1, N, 1)
            # mu2 = mu2.unsqueeze(1).repeat(1, N, 1)
            # dist_condition = torch.cat((dist_condition, mu1), dim=-1)
            # dist_condition = torch.cat((dist_condition, mu2), dim=-1)


            # rasterization condition
            rasterized = self.renderer(point_clouds)
            rasterized = rasterized[:, :, :, :3].permute([0, 3, 1, 2])
            # rasterized_condition = self.img_encoder(rasterized).unsqueeze(1).repeat(1, N, 1)

            # signal_std = torch.std(rasterized)
            # signal_mean = torch.mean(rasterized)
            # snr = random.uniform(0, 20)
            # # snr = 20
            # noise_std = calculate_noise_std(signal_std, snr)
            # noise_cn = torch.normal(0, noise_std.detach(), rasterized.size())
            # rasterized = rasterized + noise_cn.to(device)
            # rasterized_condition = self.rasterized_encoder(rasterized)
            # rasterized_condition = rasterized_condition.unsqueeze(1).repeat(1, N, 1)

            x = self.rasterized_encoder._process_input(rasterized)
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.rasterized_encoder.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            rasterized_condition = self.rasterized_encoder.encoder(x)
            rasterized_condition = rasterized_condition[:, 1:, :].reshape(-1, H//self.rasterized_encoder.patch_size, W//self.rasterized_encoder.patch_size, self.rasterized_encoder.hidden_dim)
            rasterized_condition = rasterized_condition.permute([0, 3, 1, 2])
            rasterized_condition = F.interpolate(rasterized_condition, size=(H, W), mode='bilinear',
                                         align_corners=False)
            projection_condition = get_projection(pc, self.rasterizer, rasterized_condition)


            # signal_std = torch.std(seg_condition)
            # signal_mean = torch.mean(seg_condition)
            # random_snr = random.uniform(0, 20)
            # noise_std = calculate_noise_std(signal_std, random_snr)
            # noise_cn = torch.normal(0, noise_std.detach(), seg_condition.size())
            # seg_condition = seg_condition + noise_cn.to(device)

            # seg_condition = seg_condition.unsqueeze(-1)
            # seg_condition = seg_condition.permute(0, 2, 1)

            # # concat method
            # seg_condition = seg_condition.permute(0, 2, 1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, rasterized_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, dist_condition), dim=-1)
            x_t_input = torch.cat((x_t_input, projection_condition), dim=-1)
            # x_t_input = torch.cat((x_t_input, text_emb), dim=-1)
            # x_t_input = torch.cat((x_t_input, mask_emb), dim=-1)
            # print(x_t_input.shape)
            noise_pred, feature = self.point_cloud_model(x_t_input, timestep)





            # print('noise_pred -------------')
            # print(torch.mean(noise_pred, dim=1))
            # print(torch.var(noise_pred, dim=1))

            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)
            # if self.use_vae:
            #     return loss, vae_loss1, vae_loss2
            return loss



class ConditionalClipSegVoxelDiffusionModel(nn.Module):
    def __init__(
            self,
            beta_start: float = 1e-5,
            beta_end: float = 8e-3,
            beta_schedule: str = 'linear',
            point_cloud_model: str = 'pvcnn',
            point_cloud_model_embed_dim: int = 64,
            voxel: bool = False,
            voxel_num = 0,
            grid_range: [int] = [],
            voxel_size: float = 0.0,
            voxel_point_max: int = 0,
            timestep_num = 0,
            depth_condition_model_path: str = None,
            **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {'num_train_timesteps': timestep_num}
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            'pndm': PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference
        self.voxel = voxel
        # x_0 transformation
        self.weight = 3.0
        self.bias = 0.0

        # noise
        self.noise_mean = 0.0
        self.noise_std = 1.0
        if self.voxel:
            self.grid_range = grid_range
            self.voxel_size = voxel_size
            self.voxel_point_max = voxel_point_max
            self.voxel_num = voxel_num
        self.img_encoder = resnet50(pretrained=False)
        # self.img_encoder.fc = nn.Linear(in_features=self.img_encoder.fc.in_features, out_features=3000)
        self.img_encoder.fc = nn.Linear(in_features=self.img_encoder.fc.in_features, out_features=3072)
        self.num_heads = 4
        self.embed_dim = 3072
        self.attention = MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.LN = nn.LayerNorm(normalized_shape=self.embed_dim)
        self.conv = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        # Create point cloud model for processing point cloud at each diffusion step
        # self.point_cloud_model = PointCloudModel(
        #     embed_dim=point_cloud_model_embed_dim,
        #     in_channels=3,
        #     out_channels=3,
        # )

        self.point_cloud_model = PVCNNModel(model_type='pvcnnplusplus')

        # Create conv3d model for processing voxel at each diffusion step
        self.conv3d_model = Conv3DModel(
            in_channels=2,
            out_channels=1,
        )

        # Create depth condition model
        self.bev_encoding = True
        # self.depth_condition_model = DepthConditionModel(
        #     grid_range=grid_range,
        #     voxel_size=voxel_size,
        #     bev_encoding=self.bev_encoding,
        #     model_path=depth_condition_model_path,
        # )

        # Normalization
        self.norm_mean = 1.0
        self.norm_std = 1.0




    def denormalize(self, x: Tensor):
        x = x * self.norm_std + self.norm_mean
        return x

    def normalize(self, x: Tensor):
        self.norm_mean = torch.mean(x, dim=(0, 1))
        self.norm_std = torch.std(x, dim=(0, 1))
        x = (x - self.norm_mean) / self.norm_std
        return x

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

    def voxel_to_point_cloud(self, voxel):
        B, C, Dx, Dy, Dz = voxel.shape
        voxel = torch.relu(voxel)
        if B != 1:
            raise ValueError("Batch size must be 1 in sample forward")
        voxel = voxel.cpu()
        mask = voxel.squeeze() * self.voxel_point_max

        # 定义网格的维度
        grid_size = (Dx, Dy, Dz)

        # 定义实际点云坐标范围
        [[x_min, y_min, z_min], [x_max, y_max, z_max]] = self.grid_range
        x_range = torch.arange(x_min, x_max + 1, (x_max - x_min + 1) / Dx)
        y_range = torch.arange(y_min, y_max + 1, (y_max - y_min + 1) / Dx)
        z_range = torch.arange(z_min, z_max + 1, (z_max - z_min + 1) / Dx)

        # 随机生成点云坐标
        point_clouds = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    # 获取当前网格中可以存在的点云数量
                    num_points = int(mask[i, j, k])
                    if num_points > 0:
                        # print((i, j, k), (num_points))

                        # 在当前网格中随机生成 num_points 个点云坐标
                        points = torch.rand(num_points, 3) * self.voxel_size + \
                                 torch.stack((x_range[i], y_range[j], z_range[k]), dim=0)
                        # print(points)
                        point_clouds.append(points)

        # 将点云列表转换成张量
        point_clouds = torch.cat(point_clouds, dim=0).unsqueeze(0)
        return point_clouds

    def get_input_with_depth_conditioning(self, x_t, depth_bev_condition, depth_condition=None):
        '''
        x_t: (B, C, Dx, Dy, Dz)
        depth_bev_condition: (B, 1, Dx, Dy)
        depth_condition: (B, C, Dx, Dy)
        '''
        B, C, Dx, Dy, Dz = x_t.shape
        depth_bev_condition = torch.unsqueeze(depth_bev_condition, dim=4)  # (B, 1, Dx, Dy, 1)
        depth_bev_condition_exp = depth_bev_condition.expand(B, 1, Dx, Dy, Dz)  # (B, 1, Dx, Dy, Dz)
        condition = torch.mul(x_t, depth_bev_condition_exp)  # (B, 1, Dx, Dy, Dz)
        if depth_condition is not None:
            C_d = depth_condition.shape[1]
            depth_condition = torch.unsqueeze(depth_condition, dim=4)  # (B, C_d, Dx, Dy, 1)
            depth_condition_exp = depth_condition.expand(B, C_d, Dx, Dy, Dz)
            condition = torch.cat((condition, depth_condition_exp), dim=1)  # (B, C_d+1, Dx, Dy, Dz)
        x_t_input = torch.cat((x_t, condition), dim=1)  # (B, 2, Dx, Dy, Dz) or (B, C_d+2, Dx, Dy, Dz)
        return x_t_input

    @torch.no_grad()
    def forward_sample(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt=None,
            processor=None,
            text_encoder=None,
            # Optional overrides
            scheduler: Optional[str] = 'ddpm',
            # Inference parameters
            num_inference_steps: Optional[int] = 100,
            eta: Optional[float] = 0.0,  # for DDIM
            # Whether to return all the intermediate steps in generation
            return_sample_every_n_steps: int = 10,
            # Whether to disable tqdm
            disable_tqdm: bool = False,
    ):
        """
        由于每个batch的点云数量不一致，forward_sample时，batch size只能为1
        """

        # Get scheduler from mapping, or use self.scheduler if None
        # seg = seg.permute([0, 3, 1, 2])
        # seg_condition = self.img_encoder(seg)
        # seg_condition = seg_condition.unsqueeze(-1)

        if self.voxel:
            scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

            # Sample noise
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            device = pc.device
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 50, 40, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)
            x_t = torch.randn_like(invl_pts)  # (B, V, P, 3)
            std = x_t.std(dim=2, unbiased=False)
            x_t = x_t / std.view(B, V, 1, -1) * self.noise_std


            # Set timesteps
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

                x_t_input = x_t
                x_t_input = torch.cat([x_t_input, vl_id], dim=-1)  # (B, V, P, 3+3)

                x_t_input = x_t_input.view(B, V * P, -1)
                # x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
                # Forward
                noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))  # (B, V, P, 3)
                noise_pred = noise_pred.view(B, V, P, -1)
                # TODO: step 会导致分布变化
                # Step
                x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
                # print(torch.mean(x_t))
                # print(torch.var(x_t))

                # Append to output list if desired
                if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                    all_outputs.append(x_t)

            # Convert output back into a point cloud, undoing normalization and scaling

            output = (x_t - self.bias) / self.weight + vl_id
            output = output.reshape(B, V * P, -1)

            if return_all_outputs:
                all_outputs = [(o - self.bias) / self.weight + vl_id for o in all_outputs]
                all_outputs = [o.reshape(B, V * P, -1) for o in all_outputs]

            return (output, all_outputs) if return_all_outputs else output
        else:
            # max_length = 95
            # encoded_input = tokenizer.batch_encode_plus(
            #     prompt,
            #     add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
            #     padding='max_length',  # 填充到最大长度
            #     max_length=max_length,  # 最大序列长度
            #     return_attention_mask=True,  # 返回注意力掩码
            #     truncation=True,
            #     return_tensors='pt'  # 返回PyTorch张量
            # )
            # encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
            # output = text_encoder(**encoded_input)
            # text_emb = output['last_hidden_state']
            # text_emb = torch.mean(text_emb, dim=1)
            #
            # text_emb = text_emb.repeat(1, 4)
            # text_emb = text_emb.unsqueeze(dim=-1)

            inputs = processor(text=prompt, return_tensors="pt", padding=True, max_length=77, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to('cuda:0')
            with torch.no_grad():
                text_features = text_encoder.get_text_features(**inputs)
            text_features = text_features.repeat(1, 6).unsqueeze(-1)
            text_emb = text_features.permute(0,2,1)

            scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

            # Sample noise

            B, N, D = pc.shape
            device = pc.device
            x_t = torch.randn(B, N, D, device=device)

            # Set timesteps
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


            x_t = self.normalize(x_t)
            for i, t in enumerate(progress_bar):
                x_t_input = x_t
                # x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
                # x_t_input = torch.cat((x_t_input, text_features),dim=-1)
                noise_pred, feature = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))
                feature = feature.permute(0, 2, 1)
                attn_output, attn_output_weights = self.attention(feature, text_emb, text_emb)
                attn_output = self.LN(attn_output)
                # attn_output = attn_output.view(B*N, -1)
                # attn_output = self.fc(attn_output)
                # attn_output = attn_output.view(B, N, 3)
                attn_output = self.conv(attn_output)
                x = attn_output.permute(0, 2, 1)
                noise_pred = noise_pred + x
                # TODO: step 会导致分布变化
                # Step
                x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
                # print(torch.mean(x_t))
                # print(torch.var(x_t))

                # Append to output list if desired
                if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                    all_outputs.append(x_t)

            # Convert output back into a point cloud, undoing normalization and scaling
            output = self.denormalize(x_t)

            if return_all_outputs:

                all_outputs = [self.denormalize(o) for o in all_outputs]

            return (output, all_outputs) if return_all_outputs else output

    def forward_train(
            self,
            pc: Tensor,
            seg: Tensor = None,
            prompt = None,
            processor = None,
            text_encoder = None,
            return_intermediate_steps: bool = False
    ):

        if self.voxel:
            min_values = pc.min(dim=1)[0]
            max_values = pc.max(dim=1)[0]
            min_values = min_values.unsqueeze(-2)
            max_values = max_values.unsqueeze(-2)
            pc = (pc - min_values) / (max_values - min_values)
            pc = pc * 2 - 1

            B, N, _ = pc.shape
            device = pc.device
            seg = seg.permute([0, 3, 1, 2])
            seg_condition = self.img_encoder(seg)
            seg_condition = seg_condition.unsqueeze(-1)
            invl_pts, vl_id = self.get_invoxel_points_with_id_without_kpt(pc)
            B, V, P, _ = invl_pts.shape  # (B, 40, 50, 3)
            vl_id = vl_id.unsqueeze(2)  # (B, V, 1, 3)
            vl_id = vl_id.expand(B, V, P, 3)  # (B, V, P, 3)

            x_0 = invl_pts - vl_id  # (B, V, P, 3)
            x_0 = self.weight * x_0 + self.bias

            # Sample random noise N(0, 1)
            noise = torch.randn_like(x_0)  # (B, V, P, 3)
            std = noise.std(dim=2, unbiased=False)
            noise = noise / std.view(B, V, 1, -1) * self.noise_std

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)  # (B, V, P, 3)

            # Conditioning
            # voxel id condition
            x_t_input = torch.cat([x_t, vl_id], dim=-1)  # (B, V, P, 3+3)
            x_t_input = x_t_input.view(B, V*P, -1)
            x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)
            # Forward
            noise_pred = self.point_cloud_model(x_t_input, timestep)  # (B, V, P, 3)
            noise_pred = noise_pred.view(B, V, P, -1)
            # Check
            # if not noise_pred.shape == noise.shape:
            #     raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss

        else:

            # max_length = 95
            # encoded_input = tokenizer.batch_encode_plus(
            #     prompt,
            #     add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
            #     padding='max_length',  # 填充到最大长度
            #     max_length=max_length,  # 最大序列长度
            #     return_attention_mask=True,  # 返回注意力掩码
            #     truncation=True,
            #     return_tensors='pt'  # 返回PyTorch张量
            # )
            # encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
            # output = text_encoder(**encoded_input)
            # text_emb = output['last_hidden_state']
            # text_emb = torch.mean(text_emb, dim=1)
            #
            # text_emb = text_emb.repeat(1, 4)
            # text_emb = text_emb.unsqueeze(dim=-1)

            inputs = processor(text=prompt, return_tensors="pt", padding=True, max_length=77, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to('cuda:0')
            with torch.no_grad():
                text_emb = text_encoder.get_text_features(**inputs)
            text_emb = text_emb.repeat(1, 6).unsqueeze(-1)
            text_emb = text_emb.permute(0,2,1)
            # Normalize point cloud
            x_0 = self.normalize(pc)
            B, N, D = x_0.shape
            device = x_0.device
            # Sample random noise ~ N(0, 1)
            noise = torch.randn_like(x_0)
            noise = (noise - noise.mean()) / noise.std()
            # print(torch.mean(noise))
            # print(torch.var(noise))

            # Sample random timesteps for each point_cloud
            timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,),
                                     device=device, dtype=torch.long)

            # Add noise to points
            x_t = self.scheduler.add_noise(x_0, noise, timestep)

            # Forward
            x_t_input = x_t

            # seg = seg.permute([0,3,1,2])
            # seg_condition = self.img_encoder(seg)
            # seg_condition = seg_condition.unsqueeze(-1)
            # x_t_input = torch.cat((x_t_input, seg_condition), dim=-1)

            # x_t_input = torch.cat((x_t_input, text_features), dim=-1)
            noise_pred, feature = self.point_cloud_model(x_t_input, timestep)
            feature = feature.permute(0, 2, 1)
            attn_output, attn_output_weights = self.attention(feature, text_emb, text_emb)
            attn_output = self.LN(attn_output)
            # attn_output = attn_output.view(B*N, -1)
            # attn_output = self.fc(attn_output)
            # attn_output = attn_output.view(B, N, 3)
            attn_output = self.conv(attn_output)
            x = attn_output.permute(0, 2, 1)
            noise_pred = noise_pred + x
            # print('noise_pred -------------')
            # print(torch.mean(noise_pred, dim=1))
            # print(torch.var(noise_pred, dim=1))

            # Check
            if not noise_pred.shape == noise.shape:
                raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Whether to return intermediate steps
            if return_intermediate_steps:
                return loss, (x_0, x_t, noise, noise_pred)

            return loss




if __name__ == '__main__':
    from visualization.vis_utils import plot_voxel, plot_point_cloud
    model = ConditionalVoxelDiffusionModel(voxel=True, grid_range=[[0, -3, -2], [6, 3, 2]], voxel_size=0.2,
                                                voxel_point_max=5)
    point_cloud = np.load('../lidar_pc_sample.npy')
    plot_point_cloud(point_cloud[0, :])
    point_cloud = torch.from_numpy(point_cloud)
    voxels = model.point_cloud_to_voxel_grid(point_cloud)
    print(voxels.shape)
    voxel = voxels[0, 0, :]
    plot_voxel(voxel.numpy())
    pc = model.voxel_to_point_cloud(voxel.unsqueeze(0).unsqueeze(0))
    plot_point_cloud(pc[0, :].numpy())

    # model = ConditionalPointCloudDiffusionModel(voxel=True, grid_range=[[0, -3, -2], [6, 3, 2]], voxel_size=0.2,
    #                                             voxel_point_max=5)
    # voxels = torch.ones((1, 1, 30, 30, 20))
    # point_cloud = model.voxel_to_point_cloud(voxels)
    # print(point_cloud.shape)
    # from visualization.vis_utils import plot_point_cloud
    # plot_point_cloud(point_cloud[0, :])

    # # 设置随机种子以确保结果可重复性
    # torch.manual_seed(0)
    #
    # # 定义网格的维度
    # grid_size = (30, 30, 20)
    #
    # # 定义每个格子中点云的数量
    # num_points = 5
    #
    # mask = (torch.rand(30, 30, 20) * 5).int()
    #
    # # 随机生成点云坐标
    # point_clouds = []
    # for i in range(grid_size[0]):
    #     for j in range(grid_size[1]):
    #         for k in range(grid_size[2]):
    #             # 获取当前网格中可以存在的点云数量
    #             num_points = mask[i, j, k]
    #
    #             # 在当前网格中随机生成 num_points 个点云坐标
    #             points = torch.rand(num_points, 3) + torch.tensor([i, j, k])
    #             point_clouds.append(points)
    #
    # # 将点云列表转换成张量
    # point_clouds = torch.cat(point_clouds, dim=0)
    #
    # # 打印整个网格生成的点云张量的形状和前几个点的坐标
    # print("整个网格生成的点云张量的形状:", point_clouds.shape)
    # print("前几个点的坐标:")
    # print(point_clouds[:10])
    # from visualization.vis_utils import plot_point_cloud
    # plot_point_cloud(point_clouds)
