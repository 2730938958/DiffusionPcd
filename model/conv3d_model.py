import torch.nn as nn
import torch
import numpy as np



class Conv3DModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, embed_dim=8):
        super(Conv3DModel, self).__init__()
        self.embed_dim = embed_dim
        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(embed_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(embed_dim*2, embed_dim*4, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(embed_dim*4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(embed_dim*4, embed_dim*8, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(embed_dim*8)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(embed_dim*8, out_channels, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        The inputs have size (B, C, Dx, Dy, Dz). The timesteps t can be either
        continuous or discrete.
        TODO: This model has a sort of U-Net-like structure I think,
        which is why it first goes down and then up in terms of resolution (?)
        """
        B, C, Dx, Dy, Dz = inputs.shape
        # Embed timesteps
        t_emb = get_timestep_embedding(self.embed_dim, t, inputs.device).float()
        t_emb = self.embedf(t_emb)[:, :, None, None, None].expand(-1, -1, Dx, Dy, Dz)

        feat = self.conv1(inputs)
        feat = self.bn1(feat)
        feat = self.relu1(feat)

        x = self.conv2(torch.cat([feat, t_emb], dim=1))
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        pred = self.conv4(x)
        # pred = self.relu4(pred)

        return pred


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


if __name__ == '__main__':
    model = Conv3DModel()
    input = torch.randn(8, 1, 30, 30, 20)
    t = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
    output = model(input, t)
    print(output.shape)  # should be (8, 1, 30, 30, 20)