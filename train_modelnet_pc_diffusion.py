import os
import argparse
import math
import sys
import datetime
import time
import numpy as np
import yaml
import torch
import ast
from typing import Any, Iterable, List, Optional
from loader import make_loader
from arguments import rpmnet_train_arguments
from data_loader.datasets import get_train_datasets
from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error

from visualization.visualize import visualization_pc_diffusion, data_to_device, dict_to_device
from model.pointcloud_diffusion_model import ConditionalPointCloudDiffusionModel
from utils import get_optimizer, MetricLogger, SmoothedValue
from metrics import calulate_metrics



def get_args():
    parser = rpmnet_train_arguments()
    # parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='experiments/Train_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = get_train_datasets(args)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_bs, val_bs = config['train_loader']['batch_size'], config['validation_loader']['batch_size']
    num_workers = config['num_workers']
    train_loader, val_loader = make_loader(train_dataset, val_dataset, train_bs, val_bs, num_workers, rng_generator)


    # train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    # val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # Set model, optimizer, device
    grid_range = ast.literal_eval(config['model']['grid_range'])
    voxel_size = config['model']['voxel_size']
    voxel_point_max = config['model']['voxel_point_max']
    voxel_num = config['model']['voxel_num']
    timestep_num = config['model']['time_step_num']
    model = ConditionalPointCloudDiffusionModel(grid_range=grid_range, voxel_size=voxel_size,
                                                voxel_point_max=voxel_point_max, voxel_num=voxel_num,
                                                timestep_num=timestep_num)
    optimizer, scheduler = get_optimizer(config, model)
    device = torch.device(config['model']['device'])
    model.to(device)

    # Train model, eval model and save model
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_save_dir = os.path.join(args.output_dir, 'ckpt')
        os.makedirs(ckpt_save_dir, exist_ok=True)
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

        # Train model
        for i, batch_data in enumerate(progress_bar):
            dict_to_device(batch_data, device)
            model.train()

            # Forward
            loss = model.forward_train(batch_data, False)

            # Backward
            loss.backward()

            # Step optimizer
            optimizer.step()
            scheduler.step()
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


        # Eval
        metric_eval_logger = MetricLogger(delimiter="  ", save_log_path=os.path.join(args.output_dir, 'log.txt'))
        progress_bar_eval: Iterable[Any] = metric_eval_logger.log_every(val_loader, int(config['run']['print_step_freq']),
                                                                        header=log_header)
        metric_CD_list, metric_EMD_list, metric_F_Score_list = [], [], []
        for i, batch_data in enumerate(progress_bar_eval):
            data_to_device(batch_data, device)
            model.eval()

            # Forward
            batch_data = model.forward_sample(batch_data)

            # Metrics
            metrics = calulate_metrics(batch_data['pc_diffusion_output'], batch_data['pc_diffusion_input'])
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
            if i > 4:
                break
        metric_dict = {
            'metric_CD': float(np.mean(metric_CD_list)),
            'metric_EMD': float(np.mean(metric_EMD_list)),
            'metric_F_Score': float(np.mean(metric_F_Score_list))
        }
        print('Mean of metrics: {}'.format(str(metric_dict)))


        # Visualize the output and save the checkpoint
        if args.output_dir is not None and epoch % int(config['run']['ckpt_epoch_freq']) == 0:
            visual_save_dir = os.path.join(args.output_dir, 'visual', 'epoch_{}'.format(epoch))
            visualization_pc_diffusion(batch_data, visual_save_dir, show=True)
            print('Saving outputs visualization at {}'.format(visual_save_dir))


            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            ckpt_save_path = os.path.join(ckpt_save_dir, 'epoch_{}'.format(epoch))
            torch.save(checkpoint, ckpt_save_path)
            print('Saving checpoint at {}'.format(ckpt_save_path))

        # Epoch complete, log it and continue training
        epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=device)
        print(f'{log_header}  Average stats --', metric_logger)


if __name__ == '__main__':
    main()