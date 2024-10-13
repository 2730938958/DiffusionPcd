import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, num_point):
        super(get_model, self).__init__()
        self.timestep_embed_dim = 4
        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, self.timestep_embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.timestep_embed_dim, self.timestep_embed_dim),
        )
        self.sa1 = PointNetSetAbstraction(num_point, 0.1, 10, 10 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(num_point, 0.2, 10, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(num_point, 0.4, 10, 128 + 3, [128, 128, 256], False)
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, input, timestep):
        B, N, C = input.shape   # (B, N, 6)
        device = input.device
        # Embed timesteps
        t_emb = get_timestep_embedding(self.timestep_embed_dim, timestep, device)   # (B, 4)
        t_emb = self.embedf(t_emb).unsqueeze(1).expand(-1, N, -1)   # (B, N, 4)
        input = torch.cat([input, t_emb], dim=-1)   # (B, N, 10)
        input = input.permute(0, 2, 1)  # (B, 10, N)

        l0_points = input   # (B, 10, N)
        l0_xyz = input[:,:3,:]  # (B, 3, N)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, N), (B, 64, N)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, N), (B, 128, N)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, N), (B, 256, N)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, N)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)   # (B, 128, N)

        x = self.drop1(torch.nn.functional.leaky_relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)   # (B, 3, N)
        x = x.permute(0, 2, 1)  # (B, N, 3)
        return x


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))