import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import numpy as np
import random
from helper_functions import split
from EEGEyeNet import EEGEyeNetDataset
# from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from torch_geometric.data import Data, DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import SupConLoss
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_add_pool, GATConv, ChebConv, GCN2Conv, ARMAConv, GATv2Conv
import csv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_correlation_matrix1(data):
    correlation_matrices = np.corrcoef(data.T)
    # mean_correlation_matrix = np.mean(correlation_matrices, axis=0)
    return correlation_matrices

def compute_correlation_matrix(data):
    correlation_matrices = [np.corrcoef(data[i].T) for i in range(data.shape[0])]
    mean_correlation_matrix = np.mean(correlation_matrices, axis=0)
    return mean_correlation_matrix

def create_graph_from_correlation0(correlation_matrix):
    num_channels = correlation_matrix.shape[0]
    edge_index = []
    edge_attr = []


    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                value = sorted([i, j])
                edge_index.append(value)


    edge_index = np.asarray(edge_index)
    edge_index = np.unique(edge_index, axis = 0)

    for idx in edge_index:
        m, n = idx
        edge_attr.append([correlation_matrix[m,n]])

    # edge_attr = np.asarray(edge_attr)


    return edge_index





def create_graph_from_correlation(data, edge_index, correlation_matrix, threshold=0.86):

    edge_index = np.asarray(edge_index)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    if edge_index.size(1) == 0:
        raise ValueError("Không có cạnh nào được tạo ra. Kiểm tra ma trận tương quan và ngưỡng.")

    return edge_index


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
#
#
# class EEGDataset(Dataset):
#     def __init__(self, eeg, coords, labels):
#         self.eeg = eeg
#         self.coords = coords
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.eeg)
#
#     def __getitem__(self, idx):
#         signal = self.eeg[idx]  # [129, 500]
#         label = self.labels[idx]
#         coord = self.coords[idx]
#
#         # Normalize từng sample
#         signal = (signal - signal.mean()) / (signal.std() + 1e-6)
#
#         return torch.tensor(signal), torch.tensor(label), torch.tensor(coord)
# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r=4, alpha=1.0):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(out_features, in_features))
#         self.r = r
#         self.alpha = alpha
#         self.A = nn.Parameter(torch.randn(r, in_features))
#         self.B = nn.Parameter(torch.randn(out_features, r))
#
#     def forward(self, x):
#         base = F.linear(x, self.weight)
#         # self.A = torch.permute(self.A, (0, 2, 1))
#         # print(x.shape)
#         # print(self.A.T.shape)
#         lora = F.linear(x, self.A)
#         lora = F.linear(lora, self.B)
#         return base + self.alpha * lora


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=40, hidden_size=500, proj_dim=512, num_layers=2):
        super().__init__()
        # self.lstm = nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=False,
        #     bidirectional=True
        # )
        self.linear0 = nn.Sequential(torch.nn.Linear(500, 128, bias=True), nn.ReLU(),
                                     # torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(128, 256, bias=True), nn.ReLU(),
                                     # torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(256, 500, bias=True))
        self.conv11 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv21 = nn.Conv1d(in_channels=64, out_channels=input_size, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm1d(input_size)
        dim_all = 500
        depth = 2  # best=8
        heads = 8  # best=8s
        dim_head = 8
        mlp_dim = 8
        dropout = 0.1
        drop_path = 0.1

        self.layers1 = nn.Sequential(*[])
        # self.layers = nn.Sequential(*[])

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        for _ in range(depth):
            self.layers1.append(nn.Sequential(*[
                PreNorm(dim_all, Attention(dim_all, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim_all, FeedForward(dim_all, mlp_dim, dropout=dropout))
            ]))
        self.lstm =  nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=False, bidirectional=True)
        # self.projector = nn.Sequential(
        #     nn.Linear(input_size * 2*hidden_size, proj_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_dim, proj_dim)
        # )

        # self.conv1 = GCNConv(dim_all * 2, dim_all, improved=True, cached=True, normalize=False)
        #
        # self.conv2 = GCNConv(dim_all, dim_all * 2, improved=True, cached=True, normalize=False)
        #
        # self.conv2_bn = BatchNorm(dim_all * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1 = GATv2Conv(500 * 2, 128, heads=4, concat=True, dropout=0.)
        self.conv2 = GATv2Conv(512, 250, heads=4, concat=True, dropout=0.)
        # self.conv2 = GATv2Conv(512, 500 * 2, heads=4, concat=False, dropout=0.)



        self.conv2_bn = BatchNorm(dim_all * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # self.projection = nn.Sequential(
        #     LoRALinear(dim_all * 2, 128),
        #     nn.ReLU(),
        #     LoRALinear(128, dim_all*2)
        # )

    def forward(self, x, edge_index, batch):
        # x: [B, C=129, T=500] → transpose thành [B, T, C]
        # x = x.transpose(1, 2)  # [B, 500, 129]
        # print(x.shape)
        batch_size = x.shape[0]
        x = self.linear0(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv11(x))
        x = F.leaky_relu(self.bn11(self.conv21(x)))
        for atten, ff in self.layers1:
            x_ = atten(x)
            x = x + self.drop_path(x_)
            x = x + self.drop_path(ff(x))
        x, (hn, _) = self.lstm(x)  # hn: [num_layers*2, B, hidden]

        x = x.contiguous().view(-1, x.size(-1))
        # x_c = x.clone()
        x = F.relu(self.conv1(x, edge_index=edge_index))
        # x = F.dropout(x, p=0.1, training=self.training)
        # x = F.leaky_relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        # x = F.leaky_relu(self.conv3(x, edge_index, edge_weight=edge_attr))
        # x = F.leaky_relu(self.conv2(x, edge_index))
        # x = F.leaky_relu(self.conv2_bn(self.conv2(x, edge_index, edge_weight=edge_attr) + x))
        x = F.relu(self.conv2_bn(self.conv2(x, edge_index=edge_index)))

        # x = F.leaky_relu(self.conv2_bn(self.conv4(x, edge_index, edge_weight=edge_attr)))
        # x = F.leaky_relu(global_add_pool(x, batch=batch))

        x = global_add_pool(x, batch=batch)
        # print(output.shape)
        # Ghép hướng tiến và lùi ở layer cuối cùng
        # h_forward = hn[-2]  # [B, hidden]
        # h_backward = hn[-1]
        # h_final = torch.cat([h_forward, h_backward], dim=1)  # [B, hidden*2]
        # # print(h_final.shape)
        # a
        # x = x.view(batch_size, -1, x.size(-1))

        # z = self.projector(output)  # [B, proj_dim]
        # x = self.projection(x)

        return F.normalize(x, dim=1)
        # return F.normalize(x, p=2, dim=2)

class RegressionHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(1000, 1000, bias=True), nn.ReLU(),
        #                     # torch.nn.Dropout(p=0.1),
        #                     torch.nn.Linear(1000, 512, bias=True), nn.ReLU(),
        #                     torch.nn.Linear(512, 2, bias=True))

        # self.mlp =  nn.Sequential(
        #     nn.Linear(1000, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # )

        # self.mlp = nn.Sequential(
        #     nn.Linear(1000, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)#1305,1423#best
        # )
        self.mlp = nn.Sequential(
            # torch.nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, 2)  # 1305,1423
        )
        self.mlp.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

    def forward(self, x, edge_index, batch):
        with torch.no_grad():
            features_ = self.encoder(x, edge_index, batch)
        features = features_.view(features_.size(0), -1)
        return self.mlp(features), F.normalize(features_, dim=1)


def sliding_window(data, window_size=32, stride=24):
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return windows
def supervised_contrastive_loss(features, labels, temperature=0.07):#0.07

    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    loss = -mean_log_prob_pos.mean()
    return loss



EEGEyeNet = EEGEyeNetDataset('/data/oanh/Position_task_with_dots_synchronised_min.npz')
train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:,0],0.7,0.15,0.15)  # indices for the training set

print('Before remove: ', train_indices.shape)
indices_to_remove = list(range(9127, 9522))
train_indices = np.delete(train_indices, indices_to_remove, axis=0)
print('After remove: ', train_indices.shape)
data_get_edge = np.delete(EEGEyeNet.trainX, indices_to_remove, axis=0)
data_get_cluster = np.delete(EEGEyeNet.trainY, indices_to_remove, axis=0)
correlation_matrix = compute_correlation_matrix(data_get_edge)

print('correlation_matrix: ',correlation_matrix.shape)

edge_index_00 = create_graph_from_correlation0(correlation_matrix)

edge_index_0 = edge_index_00
data = EEGEyeNet.trainX[:, :, :]
labels = EEGEyeNet.trainY[:, 1:]
eye_coords = labels
num_clusters = 25
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# labels_true = kmeans.fit_predict(eye_coords)
labels_true = np.load('/home/oem/oanh/GCN/fillter_by_cluster/contrastive_learning/labels_true_12k.npy')
# np.save("labels_true.npy", labels_true)
# Chuẩn bị danh sách các đối tượng Data của PyG
graph_data_list = []

print('----Number of edge is: ', edge_index_0.shape)


for i in range(data.shape[0]):
    x = torch.tensor(data[i], dtype=torch.float)  # Mỗi kênh là một nút trong đồ thị
    # print(labels[i])
    # print(x.shape)

    correlation_matrix = compute_correlation_matrix1(x)

    # print('correlation_matrix: ', correlation_matrix.shape)
    # a
    edge_index = create_graph_from_correlation(x, edge_index_0, correlation_matrix, 0.86)
    # print(edge_index)

    # edge_index, edge_attr = create_graph_from_correlation_test(correlation_matrix, 0.86)

    # print('correlation_matrix: ', correlation_matrix.shape)
    # a
    # edge_index, edge_attr = create_graph_from_correlation(x, correlation_matrix, 0.86)
    # print(edge_index.shape)


    graph_data = Data(x=x.T, edge_index=edge_index,edge_att=labels_true[i], y=labels[i])
    graph_data_list.append(graph_data)
    # print(graph_data_list[0].x.shape)
train_data, val_data, test_data = [graph_data_list[index] for index in train_indices], [graph_data_list[index] for
                                                                                        index in val_indices], [graph_data_list[index] for index in test_indices]
eeg_dataset_train = sliding_window(train_data, 128, 128)
eeg_dataset_val = sliding_window(val_data, 128, 128)
eeg_dataset_test = sliding_window(test_data, 128, 128)

# train = Subset(EEGEyeNet, indices=train_indices)
# val = Subset(EEGEyeNet, indices=val_indices)
# test = Subset(EEGEyeNet, indices=test_indices)

train_loader = DataLoader(eeg_dataset_train, batch_size=1, shuffle=False)
val_loader = DataLoader(eeg_dataset_val, batch_size=1, shuffle=False)
test_loader = DataLoader(eeg_dataset_test, batch_size=1, shuffle=False)

# x = np.concatenate((x[:9127, :, :], x[9523:, :, :]), axis=0)
# eeg_data = np.random.randn(N, C, T).astype(np.float32)
# eeg_data = np.transpose(x, (0, 2, 1)).astype(np.float32)



encoder = BiLSTMEncoder().cuda()
lr0 = 0.0003

# lr0 = 0.0007
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr0)
# train_dataset = EEGDataset(eeg_data, eye_coords, labels)
batch_size = 64
# train_loader = DataLoader(train, batch_size=batch_size)
# val_loader = DataLoader(val, batch_size=batch_size)
# test_loader = DataLoader(test, batch_size=batch_size)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
torch.cuda.empty_cache()
scheduler_0 = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
criterion = SupConLoss(temperature=0.007)

# for param in encoder.encoder.parameters():
#     param.requires_grad = False

set_seed(42)
for epoch in range(100):
    encoder.train()
    total_loss = 0
    for  i, batch  in tqdm(enumerate(train_loader)):
        batch = Batch.from_data_list(batch)
        num_graphs = batch.num_graphs
        # print(batch.x.shape[0])
        # print(num_graphs)
        num_nodes_per_graph = batch.x.shape[0] // num_graphs
        x = batch.x.view(num_graphs, num_nodes_per_graph, -1).to(device)
        batch = batch.to(device)
        # y_batch = torch.tensor(np.asarray(batch.y)).to(device).float()
        label = torch.tensor(np.asarray(batch.edge_att)).to(device).float()
        out = encoder(x,  batch.edge_index, batch.batch)  # [B, D]
        # print(out.shape, label.shape)
        loss = supervised_contrastive_loss(out, label)
        # loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    print(f"Epoch {epoch} - SupCon Loss: {total_loss / len(train_loader):.4f}")
    value_train_loss = total_loss / len(train_loader)
    # scheduler_0.step(train_loss)
    encoder.eval()
    total_val_loss = 0
    with torch.no_grad():
        for  i, batch  in tqdm(enumerate(test_loader)):
            batch = Batch.from_data_list(batch)
            num_graphs = batch.num_graphs
            # print(batch.x.shape[0])
            # print(num_graphs)
            num_nodes_per_graph = batch.x.shape[0] // num_graphs
            x = batch.x.view(num_graphs, num_nodes_per_graph, -1).to(device)
            batch = batch.to(device)
            # y_batch = torch.tensor(np.asarray(batch.y)).to(device).float()
            label = torch.tensor(np.asarray(batch.edge_att)).to(device).float()
            out = encoder(x,  batch.edge_index, batch.batch)  # [B, D]
            # print(out.shape, label.shape)
            loss = supervised_contrastive_loss(out, label)
            # loss = criterion(out, label)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            total_val_loss += loss.item()

    print(f"====Epoch {epoch} - SupCon Val Loss: {total_val_loss / len(test_loader):.4f}")
    valid_loss_encoder = total_val_loss / len(test_loader)
    scheduler_0.step(value_train_loss)




lr = 0.0005
set_seed(42)
reg_model = RegressionHead(encoder).cuda()
optimizer = torch.optim.Adam(reg_model.parameters(), lr=lr)
mse = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
loss_all = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


param = count_parameters(reg_model)
param1 = count_parameters(encoder)

print('Number of parameters: ', param + param1)
# print('Number of parameters: ', param2)

for epoch in range(100):
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    loss_3 = []
    loss_3.append(epoch)
    reg_model.train()
    total_loss = 0
    epoch_loss = 0
    for i, batch in tqdm(enumerate(train_loader)):
        # Chuyển batch vào thiết bị (CPU hoặc GPU)
        # print(batch)
        batch = Batch.from_data_list(batch)

        # print(batch)

        num_graphs = batch.num_graphs
        # print(batch.x.shape[0])
        # print(num_graphs)
        num_nodes_per_graph = batch.x.shape[0] // num_graphs
        x = batch.x.view(num_graphs, num_nodes_per_graph, -1).to(device)
        batch = batch.to(device)
        label = torch.tensor(np.asarray(batch.edge_att)).to(device).float()

        y_batch = torch.tensor(np.asarray(batch.y)).to(device).float()

        optimizer.zero_grad()
        # edge_index = edge_index.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # edge_attr = edge_attr.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        outputs, feature = reg_model(x, batch.edge_index, batch.batch)

        # feature = encoder(x, batch.edge_index, batch.batch)

        y_batch = y_batch.view(-1, 2)
        outputs = outputs.view(-1, 2)
        loss = mse(outputs, y_batch)
        loss_cluster = supervised_contrastive_loss(feature, label)
        loss = loss + loss_cluster
        # loss_encoder = supervised_contrastive_loss(feature, y_batch)
        # loss = loss_mse*0.6 + loss_encoder*0.4
        # print(outputs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()


    # print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")

    print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_loader)}")
    train_loss = epoch_loss / len(train_loader)
    loss_3.append(train_loss)
    # Đánh giá mô hình
    val_loss = 0
    reg_model.eval()
    with torch.no_grad():

        for i, batch in enumerate(val_loader):
            batch = Batch.from_data_list(batch)
            num_graphs = batch.num_graphs
            num_nodes_per_graph = batch.x.shape[0] // num_graphs
            x = batch.x.view(num_graphs, num_nodes_per_graph, -1).to(device)
            batch = batch.to(device)
            label = torch.tensor(np.asarray(batch.edge_att)).to(device).float()
            y_batch = torch.tensor(np.asarray(batch.y)).to(device)
            outputs, feature = reg_model(x, batch.edge_index, batch.batch)
            y_batch = y_batch.view(-1, 2)
            outputs = outputs.view(-1, 2)
            loss = mse(outputs, y_batch)
            loss_cluster = supervised_contrastive_loss(feature, label)
            loss = loss + loss_cluster
            val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}")
        valid_loss = val_loss / len(val_loader)
        loss_3.append(valid_loss)

        # Đánh giá mô hình
    test_loss = 0
    reg_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = Batch.from_data_list(batch)
            num_graphs = batch.num_graphs
            num_nodes_per_graph = batch.x.shape[0] // num_graphs
            x = batch.x.view(num_graphs, num_nodes_per_graph, -1).to(device)
            batch = batch.to(device)
            y_batch = torch.tensor(np.asarray(batch.y)).to(device)
            outputs, _ = reg_model(x, batch.edge_index, batch.batch)
            y_batch = y_batch.view(-1, 2)
            outputs = outputs.view(-1, 2)
            loss = mse(outputs, y_batch)
            test_loss += loss.item()


        print(f"==================Epoch {epoch + 1}, Test Loss: {test_loss / len(test_loader)}")
        test_loss_value = test_loss / len(test_loader)
        loss_3.append(test_loss_value)
        loss_all.append(loss_3)
        test1_loss = test_loss_value
        if epoch == 0:
            loss_min = test1_loss
        # else:
        if loss_min > test1_loss or epoch ==99 :
        # if epoch == 99 :
        #     loss_all.append(loss_3)
            loss_min = test1_loss
            loss = int(loss_min)
            torch.save(reg_model.state_dict(),
                       '/home/oem/oanh/GCN/fillter_by_cluster/contrastive_learning/weights/2_loss_proposal_biLSTM_GAT_{}_{}_length_128_SupCon_40channel.pt'.format(epoch, loss))

    # print(f"=============Epoch {epoch} - Test MSE Loss: {total_loss_test / len(test_loader):.4f}")
    scheduler.step(valid_loss)
num_channels = 40
fields = ['Epoch_channel_{}'.format(str(num_channels)), 'Train_losses', 'Val_losses', 'Test_losses']
# epochs = range(n_epoch)
with open('csv/Supcon_best_284.csv'.format(batch_size), 'a') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(loss_all)


