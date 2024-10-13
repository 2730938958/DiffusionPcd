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
from typing import Any, Iterable, List, Optional
import ast
from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error

from visualization.visualize import visualization_voxel_diffusion, data_to_device
from model.voxel_diffusion_model import ConditionalVoxelDiffusionModel, ConditionalSegVoxelDiffusionModel
from utils import get_optimizer, MetricLogger, SmoothedValue



def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='outputs/{}_VOXEL_DIFFUSION'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument("--depth_condition_model_path", default='outputs/20240213_143050_depth_condition/ckpt/epoch_1')
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    train_path = config['shapenet']['train_path']
    train_dataset = ShapeNet(train_path, 'val')
    val_dataset = ShapeNet(train_path, 'val')
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_bs, val_bs = config['train_loader']['batch_size'], config['validation_loader']['batch_size']
    num_workers = config['num_workers']
    train_loader = make_shapenet_loader(train_dataset, train_bs, num_workers, rng_generator)
    val_loader = make_shapenet_loader(val_dataset, val_bs, num_workers, rng_generator)

    # Set model, optimizer, device
    grid_range = ast.literal_eval(config['model']['grid_range'])
    voxel_size = config['model']['voxel_size']
    voxel_point_max = config['model']['voxel_point_max']
    voxel_num = config['model']['voxel_num']
    timestep_num = config['model']['time_step_num']

    # Set model, optimizer and device
    # grid_range = ((0, -3.2, -2), (6.4, 3.2, 2))
    # voxel_size = 0.2
    # voxel_point_max = 5
    model = ConditionalVoxelDiffusionModel(voxel=True,
                                           grid_range=grid_range,
                                           voxel_size=voxel_size,
                                           voxel_point_max=voxel_point_max,
                                           depth_condition_model_path=args.depth_condition_model_path)


    device = torch.device('cuda:0')
    model.to(device)
    optimizer = get_optimizer(config, model)

    # TODO: Visualization before train
    # visualization(train_loader, model, device)

    # TODO: Codes for training (and saving models)
    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    epoch = 0
    step = 0
    while True:

        # Train progress bar
        log_header = f'Epoch: [{epoch}]'
        metric_logger = MetricLogger(delimiter="  ", save_log_path=os.path.join(args.output_dir, 'log.txt'))
        metric_logger.add_meter('step', SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar: Iterable[Any] = metric_logger.log_every(train_loader, int(config['run']['print_step_freq']),
                                                              header=log_header)

        # Train
        for i, batch_data in enumerate(progress_bar):
            data_to_device(batch_data, device)

            model.train()

            # Forward
            loss = model.forward_train(batch_data['input_lidar'], batch_data['input_depth'])


            # Backward
            loss.backward()

            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # Exit if loss was NaN
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Logging
            log_dict = {
                'lr': optimizer.param_groups[0]["lr"],
                'step': step,
                'train_loss': loss_value,
            }
            metric_logger.update(**log_dict)

            # End training after the desired number of steps/epochs
            if step >= int(config['run']['max_steps']) or epoch > int(config['run']['num_epoch']):
                print(f'Ending training at: {datetime.datetime.now()}')

                time.sleep(5)
                return

        # Save the checkpoint and visualize the output
        if args.output_dir is not None and epoch % int(config['run']['ckpt_epoch_freq']) == 0 and epoch > 0:
            ckpt_save_dir = os.path.join(args.output_dir, 'ckpt')
            os.makedirs(ckpt_save_dir, exist_ok=True)
            model.eval()
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(ckpt_save_dir, 'epoch_{}'.format(epoch)))
            print('Saving checpoint at {}'.format(os.path.join(ckpt_save_dir, 'epoch_{}'.format(epoch))))

            visual_save_dir = os.path.join(args.output_dir, 'visual', 'epoch_{}'.format(epoch))
            os.makedirs(visual_save_dir, exist_ok=True)
            visualization_voxel_diffusion(batch_data, model, grid_range, voxel_size, voxel_point_max, visual_save_dir)


        # Epoch complete, log it and continue training
        epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=device)
        print(f'{log_header}  Average stats --', metric_logger)

    # TODO: Codes for test (if)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    main()



