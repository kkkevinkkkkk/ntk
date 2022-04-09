import torch
import numpy as np
import tqdm

# def NTK_theory_vs_sim(inpseqs, infntk, varw, varu, varb, avgpool,
#                       nonlin=torch.erf,
#                       log2widthmin=6, log2widthmax=15, nseeds=10):
#     if isinstance(inpseqs, np.ndarray):
#         inpseqs = torch.from_numpy(inpseqs).float()
#     widths = 2**np.arange(log2widthmin, log2widthmax)
#     mysimcovs = {}
#     for width in tqdm(widths):
#         mysimcovs[width] = np.stack([
#             simrnn_ntk(inpseqs, width,
#                        nonlin, varw, varu, varb,
#                        seed=seed, avgpool=avgpool, debug=False)['ntk']
#             for seed in range(nseeds)])
#     frobs = []
#     infntknorm = np.linalg.norm(infntk)
#     for width in widths:
#         _frobs = np.sum((mysimcovs[width] - infntk)**2,
#                         axis=(1, 2)) / infntknorm**2
#         for f in _frobs:
#             frobs.append(dict(
#                 relfrob=np.sqrt(f),
#                 width=width
#             ))
#     return pd.DataFrame(frobs)