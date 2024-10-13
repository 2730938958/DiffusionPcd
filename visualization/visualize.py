import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random

from visualization.vis_utils import plot_frame, plot_voxel
from model.model_utils import point_cloud_to_voxel_grid

def visualize_input(batch_data, sample_idx, save_dir=None, show=True):

    modality = batch_data['modality']

    for mod in modality:
        mod_key = 'input_{}'.format(mod)
        mod_data = batch_data[mod_key]
        B = mod_data.shape[0]
        if B > 1:
            sample_data = mod_data[sample_idx, :]
        else:
            sample_data = mod_data[0, :]
        sample_data = sample_data.cpu().numpy()
        plot_frame(sample_data, mod, save_dir, name=mod_key, show=show)
    pass


def visualize_shapenet_input(batch_data, sample_idx, save_dir=None, show=True):
    mod_data = batch_data['points']
    B = mod_data.shape[0]
    if B > 1:
        sample_data = mod_data[sample_idx, :]
    else:
        sample_data = mod_data[0, :]
    sample_data = sample_data.cpu().numpy()
    # max_values = sample_data.max(axis=0)
    # min_values = sample_data.min(axis=0)
    # sample_data = ((sample_data - min_values) / (max_values - min_values)) * (3 - (-3)) + (-3)

    sample_data = sample_data * 3
    plot_frame(sample_data, 'lidar', save_dir, name='pcd', show=show)
    plot_frame(batch_data['seg_img'][sample_idx], 'seg', save_dir, name='rgb', show=show)
    pass


def visualize_diffu_forward(outputs, sample_idx, save_dir=None, show=True):
    for idx, out in enumerate(outputs):
        sample_data = out[sample_idx, :]
        sample_data = sample_data.cpu().numpy()
        plot_frame(sample_data, 'lidar', save_dir, name='diffu_{}'.format(idx), show=show)
    pass


def visualize_output_voxel(output, save_dir, grid_range, voxel_size, voxel_point_max):
    voxel = point_cloud_to_voxel_grid(output, grid_range, voxel_size, voxel_point_max)
    voxel = voxel[0, 0, :].cpu().numpy()
    plot_voxel(voxel, save_dir, name='output')


# def dict_to_device(batch_data, device):
#     batch_data['input_lidar'] = batch_data['input_lidar'].to(device)


def dict_to_device(batch_data, device):
    keys = list(batch_data.keys())
    for k in keys:
        if k != 'prompt':
            batch_data[k] = batch_data[k].float().to(device)

def data_to_device(batch_data, device):
    # keys = list(batch_data.keys())
    # for k in keys:
    #     batch_data[k] = batch_data[k].to(device)
    batch_data['output'] = batch_data['output'].to(device)
    batch_data['keypoints'] = batch_data['keypoints'].to(device)

    modality = batch_data['modality']
    for mod in modality:
        mod_name = 'input_{}'.format(mod)
        batch_data[mod_name] = batch_data[mod_name].to(device)


def visualization_eval(dataloader, model, device):
    for batch_idx, batch_data in enumerate(dataloader):
        data_to_device(batch_data, device)

        B = batch_data['input_lidar'].shape[0]
        if B == 1:
            sample_idx = 0
        else:
            sample_idx = random.randint(0, B - 1)

        # visualize the input
        visualize_input(batch_data, sample_idx)

        # diffusion forward
        output, all_outputs = model.forward_sample(batch_data['input_lidar'], return_sample_every_n_steps=100)
        # visualize the forward process
        visualize_diffu_forward(all_outputs, sample_idx)


def visualization_voxel_diffusion(batch_data, model, grid_range, voxel_size, voxel_point_max, save_dir=None):
    sample_idx = 0

    # visualize the input
    visualize_input(batch_data, sample_idx, save_dir)

    # diffusion forward
    pc = batch_data['input_lidar'][:1, :]
    depth = batch_data['input_depth'][:1, :]
    output, all_outputs = model.forward_sample(pc, depth)
    # visualize the forward process (point cloud)
    visualize_diffu_forward(all_outputs, sample_idx, save_dir)
    # visualize the output (voxel)
    visualize_output_voxel(output, save_dir, grid_range, voxel_size, voxel_point_max)

    return output


def visualization_shapenet_voxel_diffusion(batch_data, model, grid_range, voxel_size, voxel_point_max, save_dir=None, tokenizer = None, text_encoder = None):
    sample_idx = 0

    # visualize the input
    visualize_shapenet_input(batch_data, sample_idx, save_dir)

    # diffusion forward


    prompt = batch_data['prompt'][:1]
    pc = batch_data['points'][:1, :]
    img = batch_data['seg_img'][:1, :]
    output, all_outputs = model.forward_sample(pc, img, prompt, tokenizer, text_encoder)
    # visualize the forward process (point cloud)
    visualize_diffu_forward(all_outputs, sample_idx, save_dir)
    # visualize the output (voxel)
    # visualize_output_voxel(output, save_dir, grid_range, voxel_size, voxel_point_max)

    return output


def visualization_pc_diffusion(batch_data, save_dir=None, show=True):
    output = batch_data['pc_diffusion_output']  # tensor (B, N, 3)
    all_outputs = batch_data['pc_diffusion_all_outputs']    # [tensor] (B, N 3)
    os.makedirs(save_dir, exist_ok=True)

    visualize_input(batch_data, sample_idx=0, save_dir=save_dir, show=show)

    for idx, out in enumerate(all_outputs):
        sample_data = out[0, :]
        sample_data = sample_data.cpu().numpy()
        plot_frame(sample_data, 'lidar', save_dir, name='diffu_{}'.format(idx), show=show)

    if 'adjust_output' in batch_data.keys():
        output = batch_data['adjust_output']
        sample_data = output[0, :]
        sample_data = sample_data.detach().cpu().numpy()
        plot_frame(sample_data, 'lidar', save_dir, name='adjust', show=show)

    if 'filter_output' in batch_data.keys():
        output = batch_data['filter_output']
        sample_data = output[0, :]
        sample_data = sample_data.detach().cpu().numpy()
        plot_frame(sample_data, 'lidar', save_dir, name='fliter', show=show)
    return output[0, :].detach().cpu().numpy()


def visualization_pc_adjust(batch_data, save_dir=None, show=True):
    before_adjust = batch_data['pc_diffusion_output']  # tensor (B, N, 3)
    after_adjust = batch_data['adjust_output']  # tensor (B, N, 3)
    os.makedirs(save_dir, exist_ok=True)

    visualize_input(batch_data, sample_idx=0, save_dir=save_dir)

    before_adjust_sample = before_adjust[0, :].detach().cpu().numpy()
    after_adjust_sample = after_adjust[0, :].detach().cpu().numpy()
    plot_frame(before_adjust_sample, 'lidar', save_dir, name='before_adjust', show=show)
    plot_frame(after_adjust_sample, 'lidar', save_dir, name='after_adjust', show=show)
    return after_adjust_sample
