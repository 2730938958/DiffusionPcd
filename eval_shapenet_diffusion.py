import os
import argparse
import datetime
import numpy as np
import yaml
from shapenet.dataset import ShapeNet, make_shapenet_loader
import torch
from typing import Any, Iterable
import ast
from metrics import calulate_metrics
from visualization.visualize import dict_to_device
from model.voxel_diffusion_model import ConditionalSegVoxelDiffusionModel
from utils import MetricLogger
from transformers import BertTokenizer, BertModel
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    # parser.add_argument("--dataset_root", default='/hy-tmp/mmfi', type=str, help="Root of Dataset")
    parser.add_argument("--config_file", default='config.yaml', type=str, help="Configuration YAML file")
    parser.add_argument("--text_encoder_type", default='SD', type=str, help="BERT OR SD")
    parser.add_argument("--output_dir", default='outputs/{}_PC_DIFFUSION'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    val_cate_list = ['test']
    mode = 'dist'
    in_chan_dict = {
        'seg': 0,
        'dist': 1795, # 2595 vae
        'raster': 1539,
        'proj': 1539
    }
    for cate in val_cate_list:
        # cate = 'mug'
        with open(args.config_file, 'r') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        train_path = config['shapenet']['train_path']
        val_dataset = ShapeNet(train_path, cate)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_bs, val_bs = config['train_loader']['batch_size'], config['validation_loader']['batch_size']
        num_workers = config['num_workers']
        val_loader = make_shapenet_loader(val_dataset, val_bs, num_workers, rng_generator)
        device = torch.device('cuda:0')

        # if args.text_encoder_type == 'BERT':
        #     tokenizer = BertTokenizer.from_pretrained('/hy-tmp/bert-base-uncased')
        #     text_encoder = BertModel.from_pretrained("/hy-tmp/bert-base-uncased")
        # elif args.text_encoder_type == 'SD':
        #     tokenizer = CLIPTokenizer.from_pretrained('/hy-tmp/sd-textencoder', subfolder="tokenizer")
        #     text_encoder = CLIPTextModel.from_pretrained("/hy-tmp/sd-textencoder", subfolder="text_encoder", use_safetensors=True)



        # text_encoder.to(device)
        # text_encoder.eval()

        # Set model, optimizer, device
        grid_range = ast.literal_eval(config['model']['grid_range'])
        voxel_size = config['model']['voxel_size']
        voxel_point_max = config['model']['voxel_point_max']
        voxel_num = config['model']['voxel_num']
        timestep_num = config['model']['time_step_num']
        model = ConditionalSegVoxelDiffusionModel(voxel=False,
                                               voxel_num=voxel_num,
                                               grid_range=grid_range,
                                               voxel_size=voxel_size,
                                               voxel_point_max=voxel_point_max,
                                               timestep_num = timestep_num,
                                               simple_point_in_chan=in_chan_dict[mode])

        model.load_state_dict(torch.load('/hy-tmp/DiffusionPcd/outputs/20240906_074847_PC_DIFFUSION/ckpt/dist_200_epoch_2')['model_state_dict'])

        model.to(device)
        log_header = 'Evaluate'
        if args.output_dir is not None:
            os.makedirs(args.output_dir+'_'+cate, exist_ok=True)

        metric_eval_logger = MetricLogger(delimiter="  ", save_log_path=os.path.join(args.output_dir+'_'+cate, 'log.txt'))
        progress_bar_eval: Iterable[Any] = metric_eval_logger.log_every(val_loader, int(config['run']['print_step_freq']),
                                                                        header=log_header)
        metric_CD_list, metric_EMD_list, metric_F_Score_list = [], [], []
        for i, batch_data in enumerate(progress_bar_eval):
            dict_to_device(batch_data, device)
            model.eval()

            # Forward
            pc = batch_data['points']
            img = batch_data['seg_img']
            prompt = batch_data['prompt']
            # output, all_outputs = model.forward_sample(pc, img, prompt, tokenizer, text_encoder)
            output, all_outputs = model.forward_sample(pc, img, prompt, mode=mode)

            # Metrics
            metrics = calulate_metrics(output, batch_data['points'])
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

        metric_dict = {
            'metric_CD': float(np.mean(metric_CD_list)),
            'metric_EMD': float(np.mean(metric_EMD_list)),
            'metric_F_Score': float(np.mean(metric_F_Score_list))
        }
        print('Mean of metrics: {}'.format(str(metric_dict)))

if __name__ == '__main__':
    main()
