import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing,  global_add_pool
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, c_sigma, c_u):
        super().__init__()
        self.conv = GCNConv(input_dim, output_dim)
        self.c_u = c_u
        self.fc = nn.Linear(input_dim, output_dim)
        self.coef = np.sqrt(c_sigma / output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.c_u * self.conv(x, edge_index))
        out = self.coef * x
        return out


class GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config["num_layers"]
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.c_sigma = config["c_sigma"]
        self.c_u = config["c_u"]
        self.blocks = self.make_blocks()

    def make_blocks(self):
        blocks = []
        for i in range(self.num_layers):
            if i == 0:
                blocks.append(Block(self.input_dim, self.hidden_dim, self.c_sigma, self.c_u))
            elif i == self.num_layers - 1:
                blocks.append(Block(self.hidden_dim, self.output_dim, self.c_sigma, self.c_u))
            else:
                blocks.append(Block(self.hidden_dim, self.hidden_dim, self.c_sigma, self.c_u))
        return nn.ModuleList(blocks)

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.c_u * inputs
        # print(x.shape)
        for i in range(self.num_layers):
            x = self.blocks[i](x, edge_index)
            # print(x.shape)

        x = global_add_pool(x, batch)
        # print(x.shape)

        return x

class GNNClassifier(GNN):
    def __init__(self, config):
        super().__init__(config)
        # MLP for classification
        self.mlp = nn.Linear(config["output_dim"], config["n_class"])
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch):
        x = super().forward(x, edge_index, batch)
        out = self.mlp(x)
        return out

    def get_loss(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        target = data.y

        out = self.forward(x,edge_index, batch)
        loss = self.loss_func(out, target)
        return loss

    def predict(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        out = self.forward(x,edge_index, batch)
        probs = F.softmax(out)
        result = torch.argmax(probs, dim=1)
        return result





