import scipy as sp
import numpy as np
import torch
from torch import nn


class SimpleRNNClassifier(nn.Module):
    def __init__(self, indim, statedim, outdim=1, nonlin=torch.erf, varu=1, varw=1, varb=0, varv=1,
                 avgpool=False, debug=False):
        super().__init__()
        self.varu = varu
        self.varw = varw
        self.varb = varb
        self.varv = varv
        self.nonlin = nonlin
        self.avgpool = avgpool
        self.debug = debug
        self.W = nn.Parameter(torch.randn(statedim, statedim))
        self.U = nn.Parameter(torch.randn(indim, statedim))
        self.b = nn.Parameter(torch.randn(statedim))
        self.v = nn.Parameter(torch.randn(statedim, outdim))
        self.randomize()

    def forward(self, inp, initstate=0):
        '''
        Input:
            inp: (batchsize, seqlen, indim)
        Output:
            out: (batchsize, outdim)
        '''
        indim = self.U.shape[0]
        statedim = self.U.shape[1]
        embed = torch.einsum(
            'ijk,kl->ijl', inp, self.U) / np.sqrt(indim) + self.b
        seqlen = inp.shape[1]
        state = initstate
        self._states = []
        self.hs = []
        for i in range(seqlen):
            h = embed[:, i] + state
            state = self.nonlin(h)
            self._states.append(state)
            if self.debug:
                state.retain_grad()
                # _states[i] = s^{i+1}
                self.hs.append(h)
            if i < seqlen - 1:
                state = state @ self.W / np.sqrt(statedim)
            else:
                if self.avgpool:
                    meanstate = sum(self._states) / len(self._states)
                    return meanstate @ self.v / np.sqrt(statedim)
                else:
                    return state @ self.v / np.sqrt(statedim)

    def randomize(self, varu=None, varw=None, varb=None, varv=None):
        varu = varu or self.varu
        varw = varw or self.varw
        varb = varb or self.varb
        varv = varv or self.varv
        with torch.no_grad():
            self.W.normal_(std=np.sqrt(varw))
            self.U.normal_(std=np.sqrt(varu))
            self.v.normal_(std=np.sqrt(varv))
            if varb > 0:
                self.b.normal_(std=np.sqrt(varb))
            else:
                self.b.zero_()


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, bidirectional=False,
                 dropout_rate=0):
        super().__init__()

        self.rnn1 = nn.RNN(input_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                           dropout=dropout_rate, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        #         self.alpha = 1 / np.sqrt(hidden_dim)
        #         self.alpha = 1 / hidden_dim
        self.alpha = 1

        #         self.apply(weights_init_normal)
        self.readout = nn.Parameter(torch.randn(hidden_dim), requires_grad=False)

    def forward(self, x):
        output, hidden = self.rnn1(x)
        hidden = hidden[-1]
        hidden = self.alpha * hidden

        prediction = self.fc(hidden)
        prediction = self.alpha * prediction
        # prediction = [batch size, output dim]
        return prediction

    def similar_loss(self, x):
        output, hidden = self.rnn1(x)
        hidden = hidden[-1]
        loss = torch.mean(torch.matmul(hidden, self.readout))
        return loss


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        #         self.lstm = nn.RNN(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
        #                             dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        #         embedded = self.dropout(self.embedding(ids))
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            #             hidden = self.dropout(hidden)
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]

        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction