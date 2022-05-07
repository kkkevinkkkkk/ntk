import scipy as sp
import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d, d_in, depth, temp=1, vu=1, vw=1, vv=1, vb=0):
        super().__init__()
        self.d_in = d_in
        self.d = d
        self.depth = depth
        self.temp = temp
        self.vu = vu
        self.vw = vw
        self.vb = vb

        def paramwrap(l):
            return nn.ParameterList([nn.Parameter(p) for p in l])

        self.Us = paramwrap([
            torch.randn(d, d) * np.sqrt(vu)
            for _ in range(depth - 1)])
        self.W1s = paramwrap([torch.randn(d, d) * np.sqrt(vw)
                              for _ in range(depth)])
        self.W2s = paramwrap([torch.randn(d, d) * np.sqrt(vw)
                              for _ in range(depth)])
        self.embedding = nn.Parameter(
            torch.randn(d_in, d) * np.sqrt(vu))
        self.readout = nn.Parameter(
            torch.randn(d) * np.sqrt(vv))
        self.reset_data()

    def forward(self, seq):
        '''
        Input:
            seq: seqlen x tokensize array, for any seqlen and tokensize
        Output:
            out: seqlen x self.d_in array, for the same seqlen as input
        '''
        d = self.d
        d_in = self.d_in
        self.xs.append(seq)
        inseq = seq @ self.embedding / d_in ** 0.5
        inseq.retain_grad()
        self.ks.append(inseq)
        for l in range(self.depth):
            if l > 0:
                inseq = inseq @ self.Us[l - 1] / d ** 0.5
                inseq.retain_grad()
                self.ks.append(inseq)
            # self attn
            gram = inseq @ inseq.T / inseq.shape[1]
            weights = torch.softmax(gram / self.temp, dim=1)
            self.As.append(weights)
            # weights @ inseq gives vectors returned by attention
            # inseq + weights @ inseq is the residual connection
            post_attn = self.layernorm(inseq + weights @ inseq)
            post_attn.retain_grad()
            self.zs.append(post_attn)
            # self.post_attn = post_attn

            # FF
            inseq = post_attn @ self.W1s[l] / d ** 0.5
            inseq.retain_grad()
            self.gs.append(inseq)
            inseq = torch.relu(inseq) @ self.W2s[l] / d ** 0.5
            inseq.retain_grad()
            self.hs.append(inseq)
            inseq = self.layernorm(inseq + post_attn)
            inseq.retain_grad()
            self.xs.append(inseq)
        return (inseq @ self.readout / d ** 0.5).mean()

    def reset_data(self):
        self.xs = []
        self.ks = []
        self.ys = []
        self.zs = []
        self.gs = []
        self.hs = []
        self.As = []

    def layernorm(self, seq):
        '''inplace layernorm
        Input:
            seq: seqlen x tokensize array, for any seqlen and tokensize
        Output:
            out: seqlen x tokensize array
                Means and standard deviation computed over the `tokensize` dimension
        '''
        seq = seq - torch.mean(seq, dim=1, keepdim=True)
        seq = seq / torch.std(seq, dim=1, keepdim=True)
        return seq


class TransformerCuda(nn.Module):
    def __init__(self, d, d_in, depth, temp=1, vu=1, vw=1, vv=1, vb=0):
        super().__init__()
        self.d_in = d_in
        self.d = d
        self.depth = depth
        self.temp = temp
        self.vu = vu
        self.vw = vw
        self.vb = vb

        def paramwrap(l):
            return nn.ParameterList([nn.Parameter(p) for p in l])

        self.Us = paramwrap([
            torch.randn(d, d) * np.sqrt(vu).item()
            for _ in range(depth - 1)])

        nn.Parameter(torch.tensor(torch.randn(d, d) * np.sqrt(vw).item()))
        self.W1s = paramwrap([torch.randn(d, d) * np.sqrt(vw).item()
                              for _ in range(depth)])
        self.W2s = paramwrap([torch.randn(d, d) * np.sqrt(vw).item()
                              for _ in range(depth)])
        self.embedding = nn.Parameter(
            torch.randn(d_in, d) * np.sqrt(vu).item())
        self.readout = nn.Parameter(
            torch.randn(d) * np.sqrt(vv).item())

    def forward(self, seq):
        '''
        Input:
            seq: seqlen x tokensize array, for any seqlen and tokensize
        Output:
            out: seqlen x self.d_in array, for the same seqlen as input
        '''
        d = self.d
        d_in = self.d_in

        inseq = torch.matmul(seq, self.embedding) / d_in ** 0.5

        for l in range(self.depth):
            if l > 0:
                inseq = torch.matmul(inseq, self.Us[l - 1]) / d ** 0.5

            # self attn
            gram = torch.matmul(inseq, torch.transpose(inseq, 1, 2)) / inseq.shape[2]
            weights = torch.softmax(gram / self.temp, dim=2)

            # weights @ inseq gives vectors returned by attention
            # inseq + weights @ inseq is the residual connection
            post_attn = self.layernorm(inseq + torch.matmul(weights, inseq))

            # FF
            inseq = torch.matmul(post_attn, self.W1s[l]) / d ** 0.5
            inseq = torch.matmul(F.relu(inseq), self.W2s[l]) / d ** 0.5
            inseq = self.layernorm(inseq + post_attn)

            inseq = torch.mean(inseq, dim=1)
            out = torch.matmul(inseq, self.readout) / d ** 0.5

        return out

    def layernorm(self, seq):
        '''inplace layernorm
        Input:
            seq: seqlen x tokensize array, for any seqlen and tokensize
        Output:
            out: seqlen x tokensize array
                Means and standard deviation computed over the `tokensize` dimension
        '''
        seq = seq - torch.mean(seq, dim=2, keepdim=True)
        seq = seq / torch.std(seq, dim=2, keepdim=True)
        return seq


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = np.zeros((max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i] = pos

        #         pe = pe.unsqueeze(0)
        self.pe = np.expand_dims(pe, axis=0)

    #         self.register_buffer('pe', pe)
    #         print(pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * 0.5
        # add constant to embedding
        seq_len = x.shape[1]

        x = x + self.pe[:, :seq_len]
        return x


pos_encoder = PositionalEncoder(d_model=1, max_seq_len=50)
# class Transformer(nn.Module):
#     # d_model : number of features
#     def __init__(self,feature_size=8,num_layers=1,dropout=0):
#         super(Transformer, self).__init__()
#         self.linear = nn.Linear(1,feature_size)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.decoder = nn.Linear(feature_size,1)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def forward(self, src):
#         device = src.device
#         src = self.linear(src)
#         mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
#         # mask = self._generate_square_subsequent_mask(len(src)).to(device)
# #         mask = mask.repeat(src.shape[0],1,1)
#         output = self.transformer_encoder(src,mask)
#         output = torch.mean(output, dim=1)
#         output = self.decoder(output)
#         return output