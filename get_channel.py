import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelSelfAttention(nn.Module):
    def __init__(self, num_channels=129, embed_dim=128):
        super(ChannelSelfAttention, self).__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Các trọng số học được để tạo Q, K, V
        self.W_q = nn.Linear(500, embed_dim)  # Query projection
        self.W_k = nn.Linear(500, embed_dim)  # Key projection
        self.W_v = nn.Linear(500, embed_dim)  # Value projection

        self.scale = embed_dim ** 0.5  # Scaling factor

    def forward(self, x):
        """
        x: Tensor có shape (batch_size, 129, 500)
        """
        batch_size = x.shape[0]

        # Tính Q, K, V: shape (batch_size, 129, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Tính Attention scores: shape (batch_size, 129, 129)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # Áp dụng Softmax

        # Tính đầu ra Attention: (batch_size, 129, embed_dim)
        attn_output = torch.matmul(attn_weights, V)

        return attn_output, attn_weights

def get_channel_important(data_file, top = 40):
# data_file = 'D:/dataset/Position_task_with_dots_synchronised_min.npz'
    with np.load(data_file) as f:  # Load the data array
        x = f['EEG']
        # self.trainY = f['labels']
    x = np.concatenate((x[:9127, :, :], x[9523:, :, :]), axis=0)
    attention_layer = ChannelSelfAttention(num_channels=129, embed_dim=128)
    length_data = x.shape[0]
    all_matrix = []
    for i in range(length_data):
        x1 = x[i].T
        # x1 = torch.tensor(x1)
        x1 = torch.from_numpy(x1).float()
        attn_output, attn_weights = attention_layer(x1)
        attn_weights = attn_weights.detach().cpu().numpy()  # Chuyển về numpy
        all_matrix.append(attn_weights)
    all_matrixs = np.asarray(all_matrix)
    all_matrix = np.mean(all_matrixs, axis=0)
    channel_importance = all_matrix.sum(axis=1)  # Tổng theo hàng, shape (129,)

    # Lấy index của 40 channel có tổng Attention cao nhất
    top_40_idx = np.argsort(channel_importance)[-top:]  # Chọn 40 channel lớn nhất

    top_40_idx.sort()
    top_40_idx = top_40_idx.tolist()
    return top_40_idx

if __name__ == "__main__":
    list_channel = []
    for i in range(100):
        print(i)
        top = 40
        data_file = '/data/oanh/Position_task_with_dots_synchronised_min.npz'
        channel_import = get_channel_important(data_file, top)
        # print(channel_import)
        list_channel.append(channel_import)
    merged_unique = list(set().union(*list_channel))
    print(merged_unique)