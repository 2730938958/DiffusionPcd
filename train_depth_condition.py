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
    parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='outputs/{}_depth_condition'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
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

    model = DepthConditionModel(grid_range = ((0, -3.2, -2), (6.4, 3.2, 2)),
                                voxel_size = 0.2,
                                bev_encoding = True,
                                model_path='outputs/20240213_143050_depth_condition/ckpt/epoch_1')
    # model = DepthConditionModel(grid_range = ((0, -3.2, -2), (6.4, 3.2, 2)),
    #                             voxel_size = 0.2,
    #                             bev_encoding = True)
    x_range = (0, 6.4)
    y_range = (-3.2, 3.2)
    voxel_size = 0.2
    device = torch.device('cuda:0')
    model.to(device)
    optimizer = get_optimizer(config, model)

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
            pred, _ = model.forward(batch_data['input_depth'])
            label = points_to_bev(batch_data['input_lidar'], x_range, y_range, voxel_size)

            # visual_save_dir = os.path.join(args.output_dir, 'visual', 'epoch_{}'.format(epoch))
            # os.makedirs(visual_save_dir, exist_ok=True)
            # pred_visual = pred[0, 0, :].detach().cpu().numpy()
            # plot_bev(pred_visual, visual_save_dir, 'pred_{}'.format(i))
            # label_visual = label[0, 0, :].detach().cpu().numpy()
            # plot_bev(label_visual, visual_save_dir, 'label_{}'.format(i))

            loss = F.mse_loss(pred, label)

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
        # if True:
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
            model.eval()
            pred, _ = model.forward(batch_data['input_depth'])
            label = points_to_bev(batch_data['input_lidar'], x_range, y_range, voxel_size)
            input = batch_data['input_depth'][0, :]
            _, input = depth_to_points(input)
            input = points_to_bev(input.unsqueeze(0), x_range, y_range, voxel_size)

            input_visual = input[0, 0, :].detach().cpu().numpy()
            plot_bev(input_visual, visual_save_dir, 'input')
            pred_visual = pred[0, 0, :].detach().cpu().numpy()
            plot_bev(pred_visual, visual_save_dir, 'pred')
            label_visual = label[0, 0, :].detach().cpu().numpy()
            plot_bev(label_visual, visual_save_dir, 'label')

        # Epoch complete, log it and continue training
        epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=device)
        print(f'{log_header}  Average stats --', metric_logger)

    # TODO: Codes for test (if)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    main()



