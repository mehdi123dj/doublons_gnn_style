# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:30:39 2022

@author: remit
"""


import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GATConv, Linear, GCNConv, to_hetero, SAGEConv
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import torch.nn.functional as F


class GNN_link_classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs_layer = torch.nn.ModuleList()
        # self.lins = torch.nn.ModuleDict()
        self.bns = torch.nn.ModuleList()
        # self.mlps = torch.nn.ModuleDict()
        for i in range(num_layers):
            conv2 = SAGEConv((-1, -1), hidden_channels[i])
            conv1 = GATConv((-1, -1), hidden_channels[i])
            convs = torch.nn.ModuleList()
            convs.append(conv1)
            convs.append(conv2)
            self.convs_layer.append(convs)
            BN = torch.nn.BatchNorm1d(hidden_channels[i])
            self.bns.append(BN)
        #self.lin = Linear(hidden_channels[num_layers-1], out_channels)
        self.mlp = nn.Linear(hidden_channels[num_layers-1]*2, 1)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            conv_layer = self.convs_layer[i]
            BN = self.bns[i]
            for elem in conv_layer:
                x = elem(x, edge_index)
            x = BN(x)
        #out = self.lin(x)
        #softmax = nn.Softmax(dim=0)
        T1, T2 = edge_index[0], edge_index[1]
        h_s = torch.index_select(x, 0, T1)
        # print(h_s.size())
        h_d = torch.index_select(x, 0, T2)
        edge_emb = self.mlp(torch.cat([h_s, h_d], 1))
        edge_emb = torch.reshape(edge_emb, (-1,))
        #print(edge_emb)
        #edge_emb = softmax(edge_emb)
        #print(edge_emb)
        # print(torch.reshape(edge_emb,(-1,)).size())
        return edge_emb


def train_link_classifier(model, train_loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        edge_label = data.edge_label
        #print(data)
        out = model(data.x, data.edge_index)
        #print(out)
        #print(edge_label)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(out.size())
        # print(edge_label.size())
        total_loss += float(loss)*edge_label.size(0)
        total_examples += edge_label.size(0)
        
    return total_loss / total_examples

@torch.no_grad()
def test_link_classifier(model, test_loader, device):
    model.eval()
    roc_auc = recall = precision = 0
    m = nn.Sigmoid()
    for data in test_loader:
        data = data.to(device)
        edge_label = data.edge_label
        out = (m(model(data.x, data.edge_index))>0.5).float()
        roc_auc += roc_auc_score(edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())
        recall += recall_score(edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())
        precision += precision_score(edge_label.detach().cpu().numpy(), out.detach().cpu().numpy())
    return roc_auc/len(test_loader.dataset), recall/len(test_loader.dataset), precision/len(test_loader.dataset)