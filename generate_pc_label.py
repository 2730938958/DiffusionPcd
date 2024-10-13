import os
import argparse
import math
import sys
import datetime
import time
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import ast
from typing import Any, Iterable, List, Optional
from tqdm import tqdm
from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error
import shutil
from visualization.visualize import visualization_pc_diffusion, data_to_device
from model.pointcloud_diffusion_model import ConditionalPointCloudDiffusionModel
from model.adjust_model import AdjustModel
from model.postprocess import PostProcess
from utils import MetricLogger
from metrics import calulate_metrics



def gen_pc_diffusion(gen_result, scene, subject, action, idx, gen_save_dir, gen_set):
    B, _, _ = gen_result.shape
    for i in range(B):
        action_path = f"{scene[i]}/{subject[i]}/{action[i]}"
        folder_path = f"{gen_save_dir}/{action_path}/lidar"
        gen_set.add(action_path)
        index = idx[i]+1
        file_name = f"{folder_path}/frame{index:0>3}.bin"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        numpy_data = gen_result[i].cpu().numpy().astype(np.float64)
        with open(file_name, 'wb') as f:
            f.write(numpy_data.tobytes())
    return gen_set


def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='configs/config_eval_A01.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='generated', type=str)
    parser.add_argument('--pc_diffusion_model_path', default='/hy-tmp/ImpoMan/output/pc_diffusion_50_40_20240303_001905', type=str)
    parser.add_argument('--pc_adjust_model_path', default='/hy-tmp/ImpoMan/output/pc_adjust_50_40_20240303_102908', type=str)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # Set model, device
    grid_range = ast.literal_eval(config['model']['grid_range'])
    voxel_size = config['model']['voxel_size']
    voxel_point_max = config['model']['voxel_point_max']
    voxel_num = config['model']['voxel_num']
    timestep_num = config['model']['time_step_num']
    pc_diffusion_model = ConditionalPointCloudDiffusionModel(grid_range=grid_range, voxel_size=voxel_size,
                                                voxel_point_max=voxel_point_max, voxel_num=voxel_num,
                                                timestep_num=timestep_num)
    pc_adjust_model = AdjustModel()
    device = torch.device(config['model']['device'])

    # load pc diffusion model param
    pc_diffusion_model_param = torch.load(args.pc_diffusion_model_path)
    pc_diffusion_model.load_state_dict(pc_diffusion_model_param['model_state_dict'])
    print('load pc diffusion model from {}'.format(args.pc_diffusion_model_path))
    pc_adjust_model_param = torch.load(args.pc_adjust_model_path)
    pc_adjust_model.load_state_dict(pc_adjust_model_param['model_state_dict'])
    print('load pc adjust model from {}'.format(args.pc_adjust_model_path))
    pc_diffusion_model.to(device)
    pc_adjust_model.to(device)

    filter_threshold = float(config['model']['filter_threshold'])
    postprocess = PostProcess(threshold=filter_threshold)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    gen_set = set()
    gen_save_dir = args.output_dir
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(val_loader)):
            data_to_device(batch_data, device)
            pc_diffusion_model.eval()
            pc_adjust_model.eval()


            batch_data = pc_diffusion_model.forward_sample(batch_data)
            batch_data = pc_adjust_model.forward_sample(batch_data)
            batch_data = postprocess.filter_overlap_points(batch_data)

            gen_set = gen_pc_diffusion(batch_data['filter_output'], batch_data['scene'], batch_data['subject'], batch_data['action'], batch_data['idx'], gen_save_dir, gen_set)

    for actions in sorted(gen_set):
        mmfi_gt_path = f"{dataset_root}/{actions}/ground_truth.npy"
        gen_gt_path = f"{gen_save_dir}/{actions}/ground_truth.npy"
        shutil.copy(mmfi_gt_path, gen_gt_path)


if __name__ == '__main__':
    main()


