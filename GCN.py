import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric
from torch_geometric.data import pyg_data
import torch_geometric.nn as pyg_nn

from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

BATCHE_SIZE = 256

L.seed_everything(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj_mat):
        num_neighbours = adj_mat.sum(dim=1, keepdims=True)
        x = self.projection(x)
        x = torch.bmm(adj_mat, x)

        return x
    

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads = 1, concat_heads = True, alpha = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        if self.concat_heads:
            assert out_channels % num_heads = 0, "Num of out features must be a multiple of the count of heads"
            out_channels = out_channels // num_heads

        self.projections = nn.Linear(in_channels, out_channels * num_heads)
        self.a = nn.Parameter(Tensor(num_heads, 2 * out_channels))
        self.leaky_relu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.projections.weight.data, gain = 1.414)
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)

    def forward(self, x, adj_mat, print_attn_probs=False):
        batch_size, num_nodes = x.size(0), x.size(1)
        
        x = self.projections(x)
        x = x.view(batch_size, num_nodes, self.num_heads, -1)

        edges = adj_mat.nonzero(as_tuple=False)
        x_flat = x.view(batch_size * num_nodes, self.num_heads, -1)
        edge_idx_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_idx_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat([
            torch.index_select(input=x_flat, index=edge_idx_row, dim=0),
            torch.index_select(input=x_flat, index=edge_idx_col, dim=0)
        ], dim=-1,)

        attn_logits = torch.einsum("bhd,hc->bh", a_input, self.a)
        attn_logits = self.leaky_relu(attn_logits)

        attn_mat = attn_logits.new_zeros(adj_mat.shape + (self.num_heads,)).fill_(-9e15)
        attn_mat[adj_mat[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_mat, dim=2)
        if print_attn_probs:
            print("Attention probs: ", attn_probs.permute(0, 3, 1, 2))
        x = torch.einsum("bijh,bjhc->bihc", attn_probs, x)

        if self.concat_heads:
            x = x.reshape(batch_size, num_nodes, -1)
        else:
            x = x.mean(dim=2)
        
        return x
    

gnn_layer_by_name = { "GCN": pyg_nn.GCNConv, "GAT": pyg_nn.GATConv, "GraphConv" : pyg_nn.GraphConv}

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 2, layer_name = "GCN", dp_rate = 0.1, **kwargs,):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layer = []
        c_in, c_out = in_channels, hidden_channels

        for l_idx in range(num_layers - 1):
            layers += [gnn_layer(in_channels=c_in, out_channels=c_out, **kwargs),
                       nn.ReLU(),
                       nn.Dropout(dp_rate),]
            c_in = hidden_channels

            layers += [gnn_layer(in_channels=c_in, out_channels=out_channels, **kwargs)]
            self.layers = nn.ModuleList(layers)

        def forward(self, x, adj_mat):
            for layer in self.layers:
                x = layer(x, adj_mat)
            return x

    def forward(self, x, adj_mat):
        for layer in self.layers:
            if isinstance(layer, pyg_nn.MessagePassing):
                x = layer(x, adj_mat)
            else:
                x = layer(x)
        
        return x


class NodeLevelGNN(L.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data, mode= "train"):
        x, adj_mat = data.x, data.adj_mat

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Invalid mode"

        loss = self.loss_fn(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()

        return loss, acc
    
    def config_optimizer(self):
        #ToDo - Testing SGD
        # return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-3)
        return optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode = "train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode = "val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode = "test")
        self.log("test_acc", acc)

