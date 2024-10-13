import torch
from torch import nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(128+256, 256, scale_factor=2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        '''
        x: (B, 40, 30, 30)
        '''
        x = self.conv1(x)   # (B, 64, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x) # (B, 64, 32, 32)
        x1 = self.layer2(x) # (B, 128, 16, 16)
        x = self.layer3(x1)  # (B, 256, 8, 8)
        feature = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True) # (B, 256, 32, 32)

        output = self.up1(x, x1) # (B, 256, 16, 16)
        output = self.up2(output) # (B, 1, 32, 32)

        return output, feature