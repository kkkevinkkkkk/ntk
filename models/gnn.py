import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing,  global_add_pool
from torch_geometric.utils import add_self_loops, degree

class BlockConv(MessagePassing):
    def __init__(self, in_channels, out_channels, c_sigma=2, c_u=1):
        super(BlockConv, self).__init__(aggr='add')  # "Add" aggregation.
        # self.lin = torch.nn.Linear(in_channels, out_channels)
        self.c_u = c_u
        self.fc = nn.Linear(in_channels, out_channels)
        self.coef = np.sqrt(c_sigma / out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # print("message input", x_j.shape, edge_index.shape)
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        # row, col = edge_index
        # deg = degree(row, size[0], dtype=x_j.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # out = norm.view(-1, 1) * x_j
        # print("out put of message", out.shape)
        return  x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        x = self.c_u * aggr_out
        x = F.relu(self.fc(x))
        aggr_out = self.coef * x
        return aggr_out


# class Block(nn.Module):
#     def __init__(self, input_dim, output_dim, c_sigma, c_u):
#         super().__init__()
#         self.conv = BlockConv(input_dim, output_dim)
#         self.c_u = c_u
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.coef = np.sqrt(c_sigma / output_dim)
#
#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         x = self.fc(self.conv(x, edge_index))
#         x = F.relu(self.c_u * x)
#         out = self.coef * x
#         return out

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
                blocks.append(BlockConv(self.input_dim, self.hidden_dim, self.c_sigma, self.c_u))
            elif i == self.num_layers - 1:
                blocks.append(BlockConv(self.hidden_dim, self.output_dim, self.c_sigma, self.c_u))
            else:
                blocks.append(BlockConv(self.hidden_dim, self.hidden_dim, self.c_sigma, self.c_u))
        return nn.ModuleList(blocks)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.blocks[i](x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = global_add_pool(x, batch)

        return x


class GNNSim(GNN):
    def __init__(self,config):
        super().__init__(config)

        def weights_init_normal(m):
            '''Takes in a module and initializes all linear layers with weight
               values taken from a normal distribution.'''

            classname = m.__class__.__name__
            # for every Linear layer in a model
            if classname.find('Linear') != -1:
                y = m.in_features
                # m.weight.data shoud be taken from a normal distribution
                # m.weight.data.normal_(0.0, 1 / np.sqrt(y))
                m.weight.data.normal_(0.0, 1)
                # m.bias.data should be 0
                m.bias.data.fill_(0)
        # self.apply(weights_init_normal)
        self.readout = nn.Parameter(torch.randn(self.output_dim))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for i in range(self.num_layers):
            x = self.blocks[i](x, edge_index)

        x = global_add_pool(x, batch)

        # loss = torch.mean(torch.matmul(x, self.readout) / (self.output_dim ** 0.5))
        loss = torch.mean(torch.matmul(x, self.readout))
        return loss


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
        probs = F.softmax(out, dim=1)
        # print(probs)
        result = torch.argmax(probs, dim=1)
        return result





