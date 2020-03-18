import math
import numpy as np
import torch

def bitdecomp(x, n_bits):
    mods = []
    for _ in range(n_bits):
        mods.append(x % 2)
        x = x / 2

    bitrep = torch.stack(mods, dim=-1).byte()
    
    return bitrep

def bitrecon(x, b_source, b_target):
    bitrep = x.view(b_source * len(x) // b_target, b_target)
    exp = torch.ShortTensor([2**e for e in range(b_target)]).view(1, b_target)
    recon = (bitrep.short() * exp).sum(dim=1).view(-1)

    return recon

def numpack(n, n_bits):
    flat = n.view(-1)
    if len(flat) % 8 > 0:
        flat = torch.cat((flat, flat.new_zeros(8 - len(flat) % 8)))
    
    bitrep = bitdecomp(flat, n_bits)
    uint8rep = bitrecon(bitrep, n_bits, 8)

    return uint8rep.byte()

def unpack(p, n_bits, size=None):
    bitrep = bitdecomp(p, 8)
    recon = bitrecon(bitrep, 8, n_bits).short()

    if size is not None:
        nelements = np.prod(size)
        recon = recon[:nelements].view(size)

    return recon

if __name__ == '__main__':
    idx_high = 128
    a = torch.randint(low=0, high=idx_high, size=(4, 3)).long()
    p = numpack(a, int(math.log2(idx_high)))
    r = unpack(p, int(math.log2(idx_high)), a.size())
    diff = (a.short() - r).float().norm()

    print('Reconstruction error: {:.2f}'.format(diff))

