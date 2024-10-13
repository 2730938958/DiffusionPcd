import torch.nn as nn
import torch
import numpy as np



class PointCloudModel(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, voxel_point_max=70, voxel_num=60, time_embed_dim=2, embed_dim=8):
        super(PointCloudModel, self).__init__()
        self.time_embed_dim = time_embed_dim
        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.fc1 = nn.Linear(voxel_point_max*embed_dim, voxel_point_max*embed_dim*2)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv1d(embed_dim*2, embed_dim*2, 1)
        self.bn2 = nn.BatchNorm1d(embed_dim*2)
        self.relu3 = nn.LeakyReLU(0.1, inplace=True)

        self.fc2 = nn.Linear(voxel_point_max*embed_dim*2, voxel_point_max*embed_dim*4)
        self.relu4 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Conv1d(embed_dim * 4, 3, 1)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        inputs: (B, V, P, 6)
        t: (B,)
        """
        B, V, P, C = inputs.shape
        # Embed timesteps
        t_emb = get_timestep_embedding(self.time_embed_dim, t, inputs.device).float()
        t_emb = self.embedf(t_emb)[:, None, None, :].expand(-1, V, P, -1)   # (B, V, P, 2)

        feat = inputs.view(B, V*P, -1) # (B, V*P, 6)
        feat = feat.permute(0, 2, 1)    # (B, 6, V*P)
        feat = self.conv1(feat)
        feat = self.bn1(feat)
        feat = self.relu1(feat) # (B, 6, V*P)

        feat = feat.reshape(B, -1, V, P) # (B, 6, V, P)
        feat = feat.permute(0, 2, 3, 1) # (B, V, P, 6)
        x1 = torch.cat([feat, t_emb], dim=-1)    # (B, V, P, 8)
        x1 = x1.reshape(B, V, -1) # (B, V, P*8)
        x1 = self.fc1(x1)
        x1 = self.relu2(x1) # (B, V, P*16)

        x2 = x1.reshape(B, V, P, -1)    # (B, V, P, 16)
        x2 = x2.view(B, V*P, -1)  # (B, V*P, 16)
        x2 = x2.permute(0, 2, 1)  # (B, 16, V*P)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu3(x2) # (B, 16, V*P)

        x3 = x2.reshape(B, -1, V, P) # (B, 6, V, P)
        x3 = x3.permute(0, 2, 3, 1) # (B, V, P, 32)
        x3 = x3.reshape(B, V, -1) # (B, V, P*16)
        x3 = self.fc2(x3)
        x3 = self.relu4(x3) # (B, V, P*32)

        x4 = x3.reshape(B, V, P, -1)    # (B, V, P, 32)
        x4 = x4.view(B, V*P, -1)  # (B, V*P, 32)
        x4 = x4.permute(0, 2, 1)  # (B, 32, V*P)
        pred = self.conv3(x4)   # (B, 3, V*P)
        pred = pred.reshape(B, -1, V, P) # (B, 3, V, P)
        pred = pred.permute(0, 2, 3, 1) # (B, V, P, 3)

        return pred


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / half_dim
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


if __name__ == '__main__':
    model = PointCloudModel()
    input = torch.randn(8, 50, 40, 6)
    t = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
    output = model(input, t)
    print(output.shape)  # should be (8, 1, 30, 30, 20)