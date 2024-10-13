import torch
import torch.nn as nn
class PoseVectorizer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PoseVectorizer, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)         # 第二个全连接层
        self.fc3 = nn.Linear(64, output_size)  # 输出层
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x