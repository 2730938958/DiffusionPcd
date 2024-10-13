import torch
from pytorch3d.renderer import (
    FoVOrthographicCameras, look_at_view_transform, RasterizationSettings,
    PointsRasterizer, PointsRenderer, AlphaCompositor, PerspectiveCameras, PointsRasterizationSettings
)
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
# 创建点云数据
device = torch.device('cuda:0')
# pointcloud = np.load("pointcloud.npz")
camera_distance = 2.0
object_position = torch.tensor([0.0, 0.0, 0.0])
camera_position = object_position + torch.tensor([0.0, 0.0, -camera_distance])

# 计算相机的旋转矩阵 R 和平移矩阵 T
R, T = look_at_view_transform(1, 10, 0)
# R, T = look_at_view_transform(camera_position, object_position, torch.tensor([0.0, 1.0, 0.0]))

# 设置相机参数
# fov = 60.0  # 视场角度
cameras = PerspectiveCameras(device=torch.device("cuda:0"), R=R, T=T)

# 设置光栅化参数
raster_settings = PointsRasterizationSettings(
    image_size=224,
    # blur_radius=0.05,
    radius=0.05,
    points_per_pixel=10,
    # faces_per_pixel=3
)
# raster_settings.radius=0.05
# raster_settings.points_per_pixel=10
# raster_settings.max_points_per_bin=10
compositor = AlphaCompositor()

# 创建点云光栅化器
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

# 创建点云渲染器
renderer = PointsRenderer(rasterizer=rasterizer,compositor=compositor)

# 将点云转换为合适的格式
# point_clouds = torch.stack([points], dim=0)


import os
import argparse
import math
import sys
import datetime
import time
import numpy as np
import yaml
from shapenet.dataset import ShapeNet, make_shapenet_loader
import torch
from typing import Any, Iterable
import ast
from transformers import BertTokenizer, BertModel
# from visualization.visualize import visualization_shapenet_voxel_diffusion, dict_to_device
# from model.voxel_diffusion_model import ConditionalSegVoxelDiffusionModel
from utils import get_optimizer, MetricLogger, SmoothedValue
from transformers import CLIPTextModel, CLIPTokenizer

def calculate_noise_std(signal_std, snr_db):
    # 信噪比（SNR）转换为线性尺度
    snr_linear = 10 ** (snr_db / 10)
    # 计算噪声标准差
    noise_std = signal_std / torch.sqrt(torch.tensor(snr_linear))
    return noise_std
def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--text_encoder_type", default='SD', type=str, help="BERT OR SD")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='outputs/{}_PC_DIFFUSION'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    # parser.add_argument("--depth_condition_model_path", default='outputs/20240213_143050_depth_condition/ckpt/epoch_1')
    args = parser.parse_args()
    return args


args = get_args()

# dataset_root = args.dataset_root
with open(args.config_file, 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)
train_path = config['shapenet']['train_path']
train_dataset = ShapeNet(train_path, 'train')
val_dataset = ShapeNet(train_path, 'val')
rng_generator = torch.manual_seed(config['init_rand_seed'])
train_bs, val_bs = config['train_loader']['batch_size'], config['validation_loader']['batch_size']
num_workers = config['num_workers']
train_loader = make_shapenet_loader(train_dataset, train_bs, num_workers, rng_generator)
val_loader = make_shapenet_loader(val_dataset, val_bs, num_workers, rng_generator)
device = torch.device('cuda:0')

# Set model, optimizer, device
grid_range = ast.literal_eval(config['model']['grid_range'])
voxel_size = config['model']['voxel_size']
voxel_point_max = config['model']['voxel_point_max']
voxel_num = config['model']['voxel_num']
timestep_num = config['model']['time_step_num']

# # model.to(device)
# text_encoder.to(device)
# text_encoder.eval()
# optimizer, scheduler = get_optimizer(config, model)

# TODO: Visualization before train
# visualization(train_loader, model, device)

# TODO: Codes for training (and saving models)
if args.output_dir is not None:
    os.makedirs(args.output_dir)
epoch = 0
step = 0
min_val_loss = 10000000
min_val_epoch = 0
while True:

    # Train progress bar
    log_header = f'Epoch: [{epoch}]'
    metric_logger = MetricLogger(delimiter="  ", save_log_path=os.path.join(args.output_dir, 'log.txt'))
    metric_logger.add_meter('step', SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    progress_bar: Iterable[Any] = metric_logger.log_every(train_loader, int(config['run']['print_step_freq']),
                                                          header=log_header)
    val_progress_bar: Iterable[Any] = metric_logger.log_every(val_loader, int(config['run']['print_step_freq']),
                                                          header=log_header)



    # Train
    for i, batch_data in enumerate(progress_bar):
        # dict_to_device(batch_data, device)

        # verts = batch_data['points'][0].to(device).float()
        #
        # # rgb = torch.Tensor(pointcloud['rgb']).to(device)
        # rgb = torch.ones(8, 3072, 4).to(device)
        # # point_clouds = Pointclouds(points=[verts])
        # point_clouds = Pointclouds(points=verts, features=[rgb])
        # # images = renderer(point_clouds)
        # # 执行光栅化


        verts = batch_data['points'].to(device).float()

        # rgb = torch.Tensor(pointcloud['rgb']).to(device)
        rgb = torch.ones(8, 3072, 3).to(device)
        # point_clouds = Pointclouds(points=[verts])
        point_clouds = Pointclouds(points=verts, features=rgb)

        fragments = rasterizer(point_clouds)
        fragments_idx = fragments.idx.long()
        visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
        points_to_visible_pixels = fragments_idx[visible_pixels]

        local_features = torch.ones(8, 768, 224, 224).to(device)
        B, N, _ = batch_data['points'].shape
        R = 10
        C = 768
        local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

        # Get local features corresponding to visible points
        local_features_proj = torch.zeros(B * N, C, device=device)
        local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
        local_features_proj = local_features_proj.reshape(B, N, C)

        points_to_dist = torch.full((B*N, 1), -1, device=device).float()
        points_to_e_dist = torch.full((B*N, 1), -1, device=device).float()
        nearest_visible = (fragments_idx[:, :, :, 0] > -1)
        dist = fragments.dists[:, :, :, 0][nearest_visible]
        e_dist = fragments.zbuf[:, :, :, 0][nearest_visible]
        nearest_visible = (fragments_idx[:, :, :, 0] > -1)
        nearest_points_to_dist = fragments_idx[:, :, :, 0][nearest_visible]
        points_to_dist[nearest_points_to_dist] = dist.unsqueeze(dim=1)[nearest_points_to_dist]
        points_to_e_dist[nearest_points_to_dist] = e_dist.unsqueeze(dim=1)[nearest_points_to_dist]

        points_to_dist = points_to_dist.reshape(B, N, 1)
        points_to_e_dist = points_to_e_dist.reshape(B, N, 1)
        dist_condition = torch.cat((points_to_dist,points_to_e_dist), dim=-1)

        images = renderer(point_clouds)
        signal_std = torch.std(images)
        signal_mean = torch.mean(images)
        # random_snr = random.uniform(0.1, 20)
        snr = 10
        noise_std = calculate_noise_std(signal_std, snr)
        noise_cn = torch.normal(0, noise_std.detach(), images.size())
        images = images + noise_cn.to(device)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(images[0, ..., :3].cpu().numpy())
        images = images[0, ..., :3].cpu().numpy()
        images[images > 1] = 1
        images[images < 0] = 0
        plt.imsave(f'vis/test{i}.png', images)
        # break
    # break

