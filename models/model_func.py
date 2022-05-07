import torch
import numpy as np

def get_weights_list(model):
    weights = []
    for name, weight in model.named_parameters():
        weights.append(weight.detach().cpu().numpy().copy())
    return weights

def calculate_weights_diff(weights1, weights2):
    diffs = []
    for weight1, weight2 in zip(weights1, weights2):
        diffs.append(np.linalg.norm(weight2 - weight1)/np.linalg.norm(weight2))
    return diffs

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
#         m.weight.data.normal_(0.0, 1)
        # m.bias.data should be 0
        m.bias.data.fill_(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clone_grads(net):
    d = {}
    for name, p in net.named_parameters():
        if p.grad is not None:
            d[name] = p.grad.clone().detach().cpu()
#     d = torch.cat([d[k].reshape(-1) for k in d], dim=0)
    return d

def paramdot(d1, d2):

    ans = sum(
        torch.dot(d1[k].reshape(-1), d2[k].reshape(-1))
        for k in d1)

    return ans