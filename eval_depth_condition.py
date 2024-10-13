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
from typing import Any, Iterable, List, Optional

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error

from visualization.visualize import visualization_eval, visualization_train, data_to_device
from model.depth_condition_model import DepthConditionModel
from utils import get_optimizer, MetricLogger, SmoothedValue
from model.model_utils import points_to_bev, depth_to_points
from visualization.vis_utils import plot_bev



def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--dataset_root", default='/data/szy4017/data/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='outputs/{}_depth_condition'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--ckpt_path', default='outputs/20240209_122243_depth_condition/ckpt/epoch_3', type=str)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # create model and load checkpoints
    model = DepthConditionModel(grid_range = ((0, -3.2, -2), (6.4, 3.2, 2)),
                                voxel_size = 0.2,
                                bev_encoding = True,
                                model_path=args.ckpt_path)
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    x_range = (0, 6.4)
    y_range = (-3.2, 3.2)
    voxel_size = 0.2
    device = torch.device('cuda:0')
    model.to(device)

    if args.output_dir is not None:
        os.makedirs(args.output_dir)

    # Train dataset
    print('Visualize the prediction in train dataset')
    for i, batch_data in enumerate(train_loader):
        data_to_device(batch_data, device)

        model.eval()

        # Forward
        pred, _ = model.forward(batch_data['input_depth'])
        label = points_to_bev(batch_data['input_lidar'], x_range, y_range, voxel_size)
        input = batch_data['input_depth'][0, :]
        _, input = depth_to_points(input)
        input = points_to_bev(input.unsqueeze(0), x_range, y_range, voxel_size)

        visual_save_dir = os.path.join(args.output_dir, 'visual')
        os.makedirs(visual_save_dir, exist_ok=True)
        input_visual = input[0, 0, :].detach().cpu().numpy()
        plot_bev(input_visual, visual_save_dir, 'train_input_{}'.format(i))
        pred_visual = pred[0, 0, :].detach().cpu().numpy()
        plot_bev(pred_visual, visual_save_dir, 'train_pred_{}'.format(i))
        label_visual = label[0, 0, :].detach().cpu().numpy()
        plot_bev(label_visual, visual_save_dir, 'train_label_{}'.format(i))

        if i > 10:
            break

    # Val dataset
    print('Visualize the prediction in val dataset')
    for i, batch_data in enumerate(val_loader):
        data_to_device(batch_data, device)

        model.eval()

        # Forward
        pred, _ = model.forward(batch_data['input_depth'])
        label = points_to_bev(batch_data['input_lidar'], x_range, y_range, voxel_size)
        input = batch_data['input_depth'][0, :]
        _, input = depth_to_points(input)
        input = points_to_bev(input.unsqueeze(0), x_range, y_range, voxel_size)

        visual_save_dir = os.path.join(args.output_dir, 'visual')
        os.makedirs(visual_save_dir, exist_ok=True)
        input_visual = input[0, 0, :].detach().cpu().numpy()
        plot_bev(input_visual, visual_save_dir, 'val_input_{}'.format(i))
        pred_visual = pred[0, 0, :].detach().cpu().numpy()
        plot_bev(pred_visual, visual_save_dir, 'val_pred_{}'.format(i))
        label_visual = label[0, 0, :].detach().cpu().numpy()
        plot_bev(label_visual, visual_save_dir, 'val_label_{}'.format(i))


        if i > 10:
            break



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    main()



