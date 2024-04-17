import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pygnn

class Net():
    def __init__(self, input, hidden_dim, output) -> None:
        self.layer1 = pygnn.GCN(input, hidden_dim)
        self.layer2 = pygnn.GCN(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.layer3 = pygnn.GCN(hidden_dim,hidden_dim)
        self.layer4 = pygnn.GCN(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.layer5 = pygnn.GCN(hidden_dim, hidden_dim)
        self.layer6 = pygnn.GCN(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()

        self.layer7 = pygnn.GCN(hidden_dim, output)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x_1 = self.layer1(x)
        x_1 = self.layer2(x_1)

        x_2 = self.layer3(x)
        x_2 = self.layer4(x_2)

        x_3 = self.layer5(x)
        x_3 = self.layer6(x_3)

        features = torch.cat(torch.tensor([x_1, x_2, x_3]), dim = 1)

        x = self.layer7(features)

        return x


    def loss():
        return nn.CrossEntropyLoss()

    def optimizer(self, model, lr = 1e-3):    
        return torch.optim.SGD(model.parameters(), lr=lr) #params, lr, momentum 