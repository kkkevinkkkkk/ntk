from models import GNNClassifier
import torch
import numpy as np
import util


random_state = 1
max_epoch = 100
batch_size = 100

torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
np.random.seed(random_state)


def train(model, dataloader):
    for batch in dataloader:
        pass



optimizer = torch.optim.SGD(model.parameters(),lr = 0.1)
for epoch in range(max_epoch):

