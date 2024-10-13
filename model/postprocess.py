import numpy as np
import torch
import torch.nn.functional as F

from model.adjust_model import AdjustModel

class PostProcess:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.scale = 1 / self.threshold
        self.adjust_model = AdjustModel()

    def filter_overlap_points(self, batch_data):
        pc = batch_data['adjust_output']
        B, N, C = pc.shape
        device = pc.device

        scaled_pc = pc * self.scale
        scaled_pc = scaled_pc.long()

        num_list = []
        filter_pc_list = []
        for b in range(B):
            sample_pc = scaled_pc[b]
            _, unique_id = torch.unique(sample_pc, return_inverse=True, dim=0)
            unique_id, _ = torch.unique(unique_id, return_inverse=True, dim=0)
            filter_pc = pc[b, unique_id, :]
            num_list.append(int(filter_pc.shape[0]))
            filter_pc_list.append(filter_pc)

        num_max = max(num_list)
        filter_padding_pc_list = []
        for fpc in filter_pc_list:
            if fpc.shape[0] < num_max:
                pad_length = num_max - fpc.size(0)
                filter_padding_pc = F.pad(fpc, (0, 0, 0, pad_length), value=0)
                filter_padding_pc_list.append(filter_padding_pc.unsqueeze(0))
            else:
                filter_padding_pc_list.append(fpc.unsqueeze(0))
        output = torch.cat(filter_padding_pc_list, dim=0)
        batch_data['filter_output'] = output
        return batch_data


if __name__ == '__main__':
    from visualization.vis_utils import plot_point_cloud
    points = np.load('../outputs/20240228_205421_PC_DIFFUSION_eval_v4_3.0_0.0/outputs/trainset_0.npy')[0]
    postprocess = PostProcess()
    post_points = postprocess.filter_overlap_points(points)
    plot_point_cloud(points)
    plot_point_cloud(post_points)