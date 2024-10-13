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

from visualization.visualize import visualization_voxel_diffusion, data_to_device
from model.voxel_diffusion_model import ConditionalVoxelDiffusionModel
from utils import get_optimizer, MetricLogger, SmoothedValue
from model.model_utils import points_to_bev, depth_to_points
from visualization.visualize import visualize_input
from visualization.vis_utils import plot_bev



def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--dataset_root", default='/data/szy4017/data/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='outputs/{}_VOXEL_DIFFUSION'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--ckpt_path', default='outputs/20240212_220536_VOXEL_DIFFUSION/ckpt/epoch_6', type=str)
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
    grid_range = ((0, -3.2, -2), (6.4, 3.2, 2))
    voxel_size = 0.2
    voxel_point_max = 5
    model = ConditionalVoxelDiffusionModel(voxel=True,
                                           grid_range=grid_range,
                                           voxel_size=voxel_size,
                                           voxel_point_max=voxel_point_max)
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda:0')
    model.to(device)
    print('load checkpoint from {}'.format(args.ckpt_path))

    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    output_save_dir = os.path.join(args.output_dir, 'outputs')
    os.makedirs(output_save_dir, exist_ok=True)

    # Train dataset
    print('Visualize the prediction in train dataset')
    for i, batch_data in enumerate(train_loader):
        data_to_device(batch_data, device)

        model.eval()

        # visualize
        visual_save_dir = os.path.join(args.output_dir, 'visual', 'trainset_{}'.format(i))
        os.makedirs(visual_save_dir, exist_ok=True)
        output = visualization_voxel_diffusion(batch_data, model, grid_range, voxel_size, voxel_point_max, visual_save_dir)

        # save output
        print('ouput pc shape = {}'.format(output.shape))
        output_save_path = os.path.join(output_save_dir, 'trainset_{}.npy'.format(i))
        output_pc = output.numpy()
        np.save(output_save_path, output_pc)

        if i > 10:
            break

    # Val dataset
    print('Visualize the prediction in val dataset')
    for i, batch_data in enumerate(val_loader):
        data_to_device(batch_data, device)

        model.eval()

        # visualize
        visual_save_dir = os.path.join(args.output_dir, 'visual', 'valset_{}'.format(i))
        os.makedirs(visual_save_dir, exist_ok=True)
        output = visualization_voxel_diffusion(batch_data, model, grid_range, voxel_size, voxel_point_max, visual_save_dir)

        # save output
        print('ouput pc shape = {}'.format(output.shape))
        output_save_path = os.path.join(output_save_dir, 'valset_{}.npy'.format(i))
        output_pc = output.numpy()
        np.save(output_save_path, output_pc)


        if i > 10:
            break



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    main()



