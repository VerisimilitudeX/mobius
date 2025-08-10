#!/usr/bin/env python3
"""
gnn_integration.py

Purpose:
  Provides a compact Graph Convolutional Network (GCN) building block intended
  for import and use by training scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, X, A):
        # X: Node features [num_nodes, in_features]
        # A: Adjacency matrix [num_nodes, num_nodes]
        support = self.linear(X)
        out = torch.matmul(A, support)
        return out

class GNNIntegration(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNNIntegration, self).__init__()
        self.gcn1 = SimpleGCNLayer(num_features, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, X, A):
        x = self.gcn1(X, A)
        x = F.relu(x)
        x = self.gcn2(x, A)
        x = F.relu(x)
        # Global average pooling over nodes
        x = x.mean(dim=0)
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    print("This module provides GNN layers/classes. Import and use within your training code.")
