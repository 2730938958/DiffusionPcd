import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)

class CoAttention(nn.Module):
    def __init__(self, pose_embed_size, pc_embed_size, hidden_size, n_voxel=50, n_point=40):
        super(CoAttention, self).__init__()
        self.pose_embed_size = pose_embed_size
        self.pc_embed_size = pc_embed_size
        self.hidden_size = hidden_size
        # 协同注意力权重矩阵
        self.W_t = nn.Parameter(torch.randn(hidden_size, pose_embed_size))
        self.W_v = nn.Parameter(torch.randn(hidden_size, pc_embed_size))
        # self.W_c = nn.Parameter(torch.randn(pose_embed_size+pc_embed_size, hidden_size * 2))
        self.W_c = nn.Parameter(torch.randn(n_voxel*n_point, pose_embed_size+pc_embed_size))
        # 用于生成注意力分数的线性层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)



    def forward(self, pose_embeds, pc_embeds):
        # pose_embeds: (batch_size, pose_seq_len, pose_embed_size)
        # pc_embeds: (batch_size, num_regions, pc_embed_size)
        batch_size, pose_seq_len, _ = pose_embeds.size()
        _, num_regions, _ = pc_embeds.size()

        pose_attn_weights = torch.matmul(self.W_t, pose_embeds.permute(0, 2, 1))  # (batch_size, hidden_size, pose_seq_len)
        pc_attn_weights = torch.matmul(self.W_v, pc_embeds.permute(0, 2, 1))  # (batch_size, hidden_size, num_regions)

        pose_attn_scores = self.fc1(torch.tanh(pose_attn_weights).permute(0, 2, 1)).squeeze(2)  # (batch_size, pose_seq_len)
        pc_attn_scores = self.fc2(torch.tanh(pc_attn_weights).permute(0, 2, 1)).squeeze(2)  # (batch_size, num_regions)

        pose_attn_weights = F.softmax(pose_attn_scores, dim=1)  # (batch_size, pose_seq_len)
        pc_attn_weights = F.softmax(pc_attn_scores, dim=1)  # (batch_size, num_regions)

        attended_pose = torch.bmm(pose_embeds.permute(0, 2, 1), pose_attn_weights).squeeze(2)  # (batch_size, pose_embed_size)
        attended_pc = torch.bmm(pc_embeds.permute(0, 2, 1), pc_attn_weights).squeeze(2)  # (batch_size, pc_embed_size)

        combined_embed = torch.cat((attended_pose, attended_pc), 1)  # (batch_size, pose_embed_size + pc_embed_size)

        final_embed = swish(self.fc3(torch.matmul(self.W_c, combined_embed).squeeze(2)))  # (batch_size, hidden_size)
        return final_embed

def main():
    # Example usage
    batch_size = 16
    num_pc_features = 2000
    num_pose_features = 17
    pc_features_dim = 64
    pose_features_dim = 3
    hidden_dim = 64

    # Random input features
    pc_features = torch.randn(batch_size, num_pc_features, pc_features_dim)
    pose_features = torch.randn(batch_size, num_pose_features, pose_features_dim)

    # Create CoAttention module
    co_attention = CoAttention(pose_features_dim, pc_features_dim, hidden_dim, 50, 40)

    # Apply co-attention
    combined_features = co_attention(pose_features, pc_features)
    print(combined_features.size())  # Should be (batch_size, 1, pc_features_dim + pose_features_dim)

if __name__ == '__main__':
    main()