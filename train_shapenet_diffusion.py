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
from visualization.visualize import visualization_shapenet_voxel_diffusion, dict_to_device
from model.voxel_diffusion_model import ConditionalSegVoxelDiffusionModel
from utils import get_optimizer, MetricLogger, SmoothedValue
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import Adam

def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    # parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--text_encoder_type", default='SD', type=str, help="BERT OR SD")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--output_dir", default='outputs/{}_PC_DIFFUSION'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    return args


def main():
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
    mode = 'dist'
    in_chan_dict = {
        'seg': 771,
        'raster': 1539,
        'dist': 1795, # 2595 vae / 1795
        'proj': 1539
    }
    model = ConditionalSegVoxelDiffusionModel(voxel=False,
                                           voxel_num=voxel_num,
                                           grid_range=grid_range,
                                           voxel_size=voxel_size,
                                           voxel_point_max=voxel_point_max,
                                           timestep_num = timestep_num,
                                           simple_point_in_chan=in_chan_dict[mode])

    text_encoder = None
    if args.text_encoder_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('/hy-tmp/bert-base-uncased')
        text_encoder = BertModel.from_pretrained("/hy-tmp/bert-base-uncased")
    elif args.text_encoder_type == 'SD':
        tokenizer = CLIPTokenizer.from_pretrained('/hy-tmp/sd-textencoder', subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("/hy-tmp/sd-textencoder", subfolder="text_encoder", use_safetensors=True)


    text_encoder.to(device)
    text_encoder.eval()

    model.to(device)
    optimizer, scheduler = get_optimizer(config, model)
    vae_optimizer1 = Adam(model.physical_encoder.vae1.parameters(), lr=1e-3)
    vae_optimizer2 = Adam(model.physical_encoder.vae2.parameters(), lr=1e-3)
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
            dict_to_device(batch_data, device)

            model.train()

            # Forward
            if mode == 'seg':
                loss = model.forward_train_seg(batch_data['points'], batch_data['seg_img'], batch_data['prompt'],
                                                    tokenizer, text_encoder)
            elif mode == 'raster':
                loss = model.forward_train_raster(batch_data['points'], batch_data['seg_img'], batch_data['prompt'],
                                                      tokenizer, text_encoder)
            if mode == 'dist':
                loss = model.forward_train_dist(batch_data['points'], batch_data['seg_img'], batch_data['prompt'],
                                                    tokenizer, text_encoder)
            elif mode == 'proj':
                loss = model.forward_train_proj(batch_data['points'], batch_data['seg_img'], batch_data['prompt'],
                                                    tokenizer, text_encoder)
            # loss, vae_loss1, vae_loss2 = model.forward_train(batch_data['points'], batch_data['seg_img'], batch_data['prompt'], tokenizer, text_encoder)
            # loss = model.forward_train_proj(batch_data['points'], batch_data['seg_img'], batch_data['prompt'], tokenizer, text_encoder)
            # loss = model.forward_train(batch_data['points'], batch_data['seg_img'], batch_data['prompt'])

            # Backward
            loss.backward()

            # Step optimizer
            optimizer.step()
            optimizer.zero_grad()

            # vae_loss1.backward()
            # vae_optimizer1.step()
            # vae_optimizer1.zero_grad()
            #
            # vae_loss2.backward()
            # vae_optimizer2.step()
            # vae_optimizer2.zero_grad()

            step += 1

            # Exit if loss was NaN
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                # print("Vae Loss is {}, stopping training".format(vae_loss1.item()+vae_loss2.item()))
                sys.exit(1)

            # Logging
            log_dict = {
                'lr': optimizer.param_groups[0]["lr"],
                'step': step,
                'train_loss': loss_value,
                # 'vae_loss': vae_loss1.item()+vae_loss2.item()
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
            val_loss_value = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_progress_bar):
                    dict_to_device(batch_data, device)
                    if mode == 'dist':
                        val_loss = model.forward_train_dist(batch_data['points'], batch_data['seg_img'], batch_data['prompt'], tokenizer, text_encoder)
                    elif mode == 'raster':
                        val_loss = model.forward_train_raster(batch_data['points'], batch_data['seg_img'], batch_data['prompt'], tokenizer, text_encoder)
                    elif mode == 'proj':
                        val_loss = model.forward_train_proj(batch_data['points'], batch_data['seg_img'], batch_data['prompt'], tokenizer, text_encoder)
                    # val_loss = model.forward_train(batch_data['points'], batch_data['seg_img'], batch_data['prompt'])
                    val_loss_value += val_loss.item() * val_bs
            val_loss_value = val_loss_value / len(val_loader)
            print(f'Validation loss: {val_loss_value}')
            if val_loss_value < min_val_loss:
                min_val_loss = val_loss_value
                min_val_epoch = epoch

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(ckpt_save_dir, '{}_{}_epoch_{}'.format(mode, timestep_num, epoch)))
                print('Saving checkpoint at {}'.format(os.path.join(ckpt_save_dir, 'epoch_{}'.format(epoch))))

            # visual_save_dir = os.path.join(args.output_dir, 'visual', 'epoch_{}'.format(epoch))
            # os.makedirs(visual_save_dir, exist_ok=True)
            # visualization_shapenet_voxel_diffusion(batch_data, model, grid_range, voxel_size, voxel_point_max, visual_save_dir)


        # Epoch complete, log it and continue training
        epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=device)
        print(f'{log_header}  Average stats --', metric_logger)
        print(f'min val loss: {min_val_loss}, min_val_epoch: {min_val_epoch}')



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()



