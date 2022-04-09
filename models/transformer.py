import scipy as sp
import numpy as np
import torch
from torch import nn


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