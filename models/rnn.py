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