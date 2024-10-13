import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet.pointnet_sem_seg import get_model

class AdjustModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AdjustModel, self).__init__()
        self.model = get_model(out_channels)

    def forward_sample(self, batch_data):
        pc = batch_data['pc_diffusion_output']
        B, N, C = pc.shape

        x = pc.transpose(2, 1)  # (B, 3, N)
        output = self.model(x)
        output = output.transpose(2, 1) # (B, N, 3)
        batch_data['adjust_output'] = output
        return batch_data

    def forward_train(self, batch_data):
        pc = batch_data['pc_diffusion_output']
        gt = batch_data['pc_diffusion_input']
        B, N, C = pc.shape

        x = pc.transpose(2, 1)  # (B, 3, N)
        pred = self.model(x)
        pred = pred.transpose(2, 1) # (B, N, 3)
        loss = F.mse_loss(pred, gt)
        return loss