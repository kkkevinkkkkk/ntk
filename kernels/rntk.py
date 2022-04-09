import scipy as sp
import numpy as np
import torch

from utils import clone_grads, paramdot, VErf3, VDerErf3


def rnnntk_batch(ingram, Vphi3, Vderphi3,
                 varw=1, varu=1, varb=0, varv=1, avgpool=False):
    ''' Compute the RNN-NTK over a batch of sequences of the same length
    Inputs:
        `ingram`: dimension-normalized Gram matrix between all tokens across all input sequences
            of shape [batchsize, batchsize, seqlen, seqlen]
        `Vphi3`: V-transform of nonlin that takes in 3 input arrays (cov, var1, var2)
        `Vderphi3`: V-transform of nonlin derivative that takes in 3 input arrays (cov, var1, var2)
        `varw`: variance of state-to-state weights
        `varu`: variance of input-to-state weights
        `varb`: variance of biases
        `varv`: variance of output weights
        `avgpool`: if True, output is the average of all states multiplied by output weights.
            Otherwise, output is just the last state multiplied by output weigths.
    Outputs:
        a dictionary of kernels
        output['ntk'] gives the NTK
        '''
    seqlen = ingram.shape[-1]
    batchsize = ingram.shape[0]
    # hcov[ia, jb] = < h^(i+2,a), h^(j+2,b) >
    hcov = np.zeros(ingram.shape)
    # hhcov[ia, jb] = < \tilde h^(i+1,a), \tilde h^(j+1,b) >
    hhcov = np.zeros(ingram.shape)
    hhcov[..., 0, :] = varu * ingram[..., 0, :] + varb
    hhcov[..., :, 0] = varu * ingram[..., :, 0] + varb

    # fill in zeroed entries
    def reflect(t):
        return np.where(t == 0, np.moveaxis(t, [0, 2], [1, 3]), t)

    def hhcov_prep(i=None, b=0):
        if i is None:
            d = np.einsum('aaii->ai', hhcov)
            return np.broadcast_arrays(
                hhcov,
                d.reshape(batchsize, 1, seqlen, 1),
                d.reshape(1, batchsize, 1, seqlen)
            )
        return np.broadcast_arrays(
            hhcov[..., i, b:i + 1],
            np.diag(hhcov[..., i, i]).reshape(batchsize, 1, 1),
            np.einsum('aaii->ai', hhcov[..., b:i + 1, b:i + 1]).reshape(1, batchsize, i + 1 - b)
        )

    def Vderphi(mat):
        d = np.diag(mat)
        return Vderphi3(mat, d.reshape(-1, 1), d.reshape(1, -1))

    for i in range(0, seqlen):
        hcov[..., i, :i + 1] = varw * Vphi3(*hhcov_prep(i))
        if i < seqlen - 1:
            hhcov[..., i + 1, 1:i + 2] = hcov[..., i, :i + 1] + varu * ingram[..., i + 1, 1:i + 2] + varb
    hhcov = reflect(hhcov)
    hcov = reflect(hcov)
    scov = varw ** -1 * hcov

    if not avgpool:
        dhcov = np.zeros([batchsize, batchsize, seqlen + 1])
        dhcov[..., -1] = varv
        for i in range(seqlen - 1, -1, -1):
            dhcov[..., i] = varw * Vderphi(hhcov[..., i, i]) * dhcov[..., i + 1]
        dhcov /= varw
        dhcov = dhcov[..., :-1]

        buf = np.einsum('abii->abi', ingram) + 1
        buf[..., 1:] += np.einsum('abii->abi', scov[..., :-1, :-1])
        ntk = np.einsum('abi,abi->ab', dhcov, buf)
        ntk += scov[..., -1, -1]
        return ntk

    # dscov[ia, jb] = <ds^(i+1,a), ds^(j+1,b)>
    dscov = np.zeros(ingram.shape)
    dscov[..., :, -1] = dscov[..., -1, :] = varv
    for i in range(seqlen - 1, 0, -1):
        dscov[..., i - 1, :i] = varw * Vderphi3(*hhcov_prep(i, 1)) * dscov[..., i, 1:i + 1] + varv
    dscov = reflect(dscov)

    # dhcov[ia, jb] = <d\tilde h^(i+1,a), d\tilde h^(j+1,b)>
    dhcov = Vderphi3(*hhcov_prep()) * dscov

    buf = ingram + 1
    buf[..., 1:, 1:] += scov[..., :-1, :-1]
    ntk = np.einsum('abij,abij->ab', dhcov, buf)
    ntk += np.sum(scov, axis=(-1, -2))
    return dict(ntk=ntk / seqlen ** 2, dscov=dscov, scov=scov, hcov=hcov, hhcov=hhcov)