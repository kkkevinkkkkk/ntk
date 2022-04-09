import scipy as sp
import numpy as np
import torch

from utils import VReLU, VStep, paramdot, clone_grads, getCor


def blockdiag(C):
    M, T, _, _ = C.shape
    Z = np.zeros([M, M])
    np.fill_diagonal(Z, 1)
    return Z[:, None, :, None] * C

class TNTK():
    def __init__(self):
        pass

    def getkers(self, C, depth, vw=1, vu=1, vv=1):
        M, T, _, _ = C.shape
        C = C.reshape(M * T, M * T)
        Cxs = [C]
        Cks = []
        As = []
        Cys = []
        Czs = []
        Deltazs = []
        Cgs = []
        Chs = []
        Deltaxs = []

        Dxs = [vv / T ** 2 * np.ones_like(C)]
        Dhs = []
        Dgs = []
        Dzs = []
        Dys = []
        Dks = []

        for i in range(depth):
            Cks += [vu * Cxs[-1]]
            A = blockdiag(
                sp.special.softmax(Cks[-1].reshape(M, T, M, T), axis=-1)
            ).reshape(M * T, M * T)
            As += [A]
            AI = A + np.eye(A.shape[0])
            Cys += [AI @ Cks[-1] @ AI.T]
            Czs += [getCor(Cys[-1])]
            Deltazs += [np.diag(Cys[-1]) ** -0.5]
            Cgs += [vw * Czs[-1]]
            Chs += [vw * VReLU(Cgs[-1])]
            Cxs += [getCor(Chs[-1] + Czs[-1])]
            Deltaxs += [np.diag(Chs[-1] + Czs[-1]) ** -0.5]

        for i in range(depth):
            Dhs += [Deltaxs[-1 - i][:, None] * Dxs[-1] * Deltaxs[-1 - i][None, :]]
            Dgs += [vw * Dhs[-1] * VStep(Cgs[-1 - i])]
            Dzs += [vw * Dgs[-1] + Dhs[-1]]
            Dys += [Deltazs[-1 - i][:, None] * Dzs[-1] * Deltazs[-1 - i][None, :]]
            AI = As[-1 - i] + np.eye(As[-1].shape[0])
            Dks += [AI.T @ Dys[-1] @ AI]
            Dxs += [vu ** 2 * Dks[-1]]

        # Dgs[0] = Dg^L
        # Dgs[L-1] = Dg^1
        # Cgs[0] = Cgs^1
        # Cgs[L-1] = Cgs^L
        # Dxs[0] = Dx^L
        # Dxs[L] = Dx^0
        # Cxs[0] = Cx^0
        # Cxs[L] = Cx^L

        assert len(Cxs) == depth + 1

        ntk = np.mean(Cxs[-1].reshape(M, T, M, T), axis=(1, 3))

        def om(C, D):
            return np.einsum('ijkl,ijkl->ik',
                             C.reshape(M, T, M, T),
                             D.reshape(M, T, M, T))

        for l in range(1, depth + 1):
            # Dks[-1-l+1] = Dk^l
            ntk += om(Dks[-1 - l + 1], Cxs[l - 1])
            ntk += om(Dgs[-1 - l + 1], Czs[l - 1])
            ntk += om(Dhs[-1 - l + 1], VReLU(Cgs[l - 1]))

        return dict(
            Cxs=Cxs,
            Cks=Cks,
            As=As,
            Cys=Cys,
            Czs=Czs,
            Deltazs=Deltazs,
            Cgs=Cgs,
            Chs=Chs,
            Deltaxs=Deltaxs,

            Dxs=Dxs,
            Dhs=Dhs,
            Dgs=Dgs,
            Dzs=Dzs,
            Dys=Dys,
            Dks=Dks,

            ntk=ntk
        )