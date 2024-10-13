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

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error

from visualization.visualize import visualization_pc_diffusion, data_to_device
from model.pointcloud_diffusion_model import ConditionalPointCloudDiffusionModel
from model.adjust_model import AdjustModel
from model.postprocess import PostProcess
from utils import MetricLogger
from metrics import calulate_metrics


def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='experiments/Eval_A01', type=str)
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
        output_save_dir = os.path.join(args.output_dir, 'outputs')

    # Eval
    log_header = 'Evaluate'
    metric_eval_logger = MetricLogger(delimiter="  ", save_log_path=os.path.join(args.output_dir, 'log.txt'))
    progress_bar_eval: Iterable[Any] = metric_eval_logger.log_every(val_loader, int(config['run']['print_step_freq']),
                                                                    header=log_header)
    metric_CD_list, metric_EMD_list, metric_F_Score_list = [], [], []
    for i, batch_data in enumerate(progress_bar_eval):
        data_to_device(batch_data, device)
        pc_diffusion_model.eval()
        pc_adjust_model.eval()

        # Forward
        batch_data = pc_diffusion_model.forward_sample(batch_data)
        batch_data = pc_adjust_model.forward_sample(batch_data)
        batch_data = postprocess.filter_overlap_points(batch_data)

        # Metrics
        metrics = calulate_metrics(batch_data['filter_output'], batch_data['input_lidar'])
        metric_CD_list.append(metrics['metric_CD'])
        metric_EMD_list.append(metrics['metric_EMD'])
        metric_F_Score_list.append(metrics['metric_F_Score'])

        # Logging
        eval_log_dict = {
            'metric_CD': metrics['metric_CD'],
            'metric_EMD': metrics['metric_EMD'],
            'metric_F_Score': metrics['metric_F_Score'],
        }
        metric_eval_logger.update(**eval_log_dict)

        # Visualize the output
        if (i % int(config['run']['print_step_freq'])) == 0:
            visual_save_dir = os.path.join(args.output_dir, 'visual', 'batch_{}'.format(i))
            visualization_pc_diffusion(batch_data, visual_save_dir, show=False)

    metric_dict = {
        'metric_CD': float(np.mean(metric_CD_list)),
        'metric_EMD': float(np.mean(metric_EMD_list)),
        'metric_F_Score': float(np.mean(metric_F_Score_list))
    }
    print('Mean of metrics: {}'.format(str(metric_dict)))

if __name__ == '__main__':
    main()


