import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.bmm(adj, x) 
        return self.linear(x)

def get_adjacency_matrix(num_joints=24):
    edges = [
        (0, 1), (1, 2), (2, 3),     # spine
        (0, 4), (4, 5), (5, 6),     # left arm
        (0, 7), (7, 8), (8, 9),     # right arm
        (2, 10), (10, 11), (11, 12),  # left leg
        (2, 13), (13, 14), (14, 15),  # right leg
        (2, 16), (16, 17),           # neck/head
        (6, 18), (9, 19), (12, 20), (15, 21), (17, 22), (17, 23),  # leaf nodes
    ]
    adj = torch.eye(num_joints)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    # Normalize
    deg = torch.sum(adj, dim=1, keepdim=True)
    adj = adj / deg
    return adj 


class JointsGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, joint_num=24):
        super().__init__()
        self.adj = get_adjacency_matrix(joint_num)
        self.gru = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gcn1 = GCNLayer(hidden_dim * 2, 128)
        self.gcn2 = GCNLayer(128, 64)
        self.head = nn.Linear(64, 3)  

        self.joint_num = joint_num
        self.adj = get_adjacency_matrix(joint_num)  # fixed adj

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, T, _ = x.shape
        h, _ = self.gru(x)  # (B, T, 2H)
        h = h[:, -1, :] 

        # Repeat hidden vector per joint
        h_joint = h.unsqueeze(1).repeat(1, self.joint_num, 1) 

        # GCN
        A = self.adj.to(x.device).unsqueeze(0).repeat(B, 1, 1)
        out = F.relu(self.gcn1(h_joint, A))
        out = F.relu(self.gcn2(out, A))

        out = self.head(out)
        return out
