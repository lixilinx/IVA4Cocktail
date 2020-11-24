import itertools
import torch

offset = 1e-10 # to avoid things like log(0), /0, etc.

def is_dist(Ps, Py):
    """Symmetric Itakura–Saito distance between scaling ambiguity removed spectra Ps and Py with shape [time, batch, mic, bin]"""
    r = torch.sqrt( (Ps + offset)/(Py + offset) ) # take sqrt to reduce the dynamic range
    s = torch.sqrt(torch.sum(1/r, dim=[0, 3], keepdim=True)/torch.sum(r, dim=[0, 3], keepdim=True)) # optimal scaling on r
    r = s*r # scaling ambiguity removed ratio
    err =  r + 1/r - 2 # [time, batch, mic, bin]
    return torch.mean(err, dim=[0, 3]) # [batch, mic]

def di_pi_is_dist(Ps, Py):
    """Delay and permutation invariant average symmetric Itakura–Saito distance between scaling ambiguity removed spectra Ps and Py with shape [time, batch, mic, bin].
    The delay search is necessary whenever Ps and Py have different length. """
    Ls, Ly = Ps.shape[0], Py.shape[0]
    N = Ps.shape[2]
    
    Dist = []
    for tau in range(Ls - Ly + 1):
        Pz = Ps[tau : tau+Ly]
        dist = []
        for p in itertools.permutations(range(N)):
            dist.append(is_dist(Pz, Py[:,:, p]))
        Dist.append(torch.stack(dist))
    Dist = torch.stack(Dist) # [delay, permutation, batch, mic]
    Dist, _ = torch.min(Dist, dim=0) # [permutation, batch, mic]
    Dist = torch.mean(Dist, dim=2) # [permutation, batch]
    Dist, _ = torch.min(Dist, dim=0) # [batch]
    return torch.mean(Dist)


def coh(S, Y):
    """Absolute coherence between normalized STFT results S and Y with shape [time, batch, mic, bin, 2]"""
    Cr = torch.mean(S[:,:,:,:,0]*Y[:,:,:,:,0] + S[:,:,:,:,1]*Y[:,:,:,:,1], dim=0) # Re{E[conj(S) * Y]}
    Ci = torch.mean(S[:,:,:,:,0]*Y[:,:,:,:,1] - S[:,:,:,:,1]*Y[:,:,:,:,0], dim=0) # Im{E[conj(S) * Y]}
    C = torch.sqrt(Cr*Cr + Ci*Ci) # [batch, mic, bin]
    return torch.mean(C, dim=2) # [batch, mic]

def di_pi_coh(S, Y):
    """Delay and permutation invariant average absolute coherence between STFT results S and Y with shape [time, batch, mic, bin, 2].
    The delay search is necessary whenever S and Y have different length."""
    Ls, Ly = S.shape[0], Y.shape[0]
    N = S.shape[2]
    
    Y = Y*torch.rsqrt(torch.mean(Y[:,:,:,:,0:1]*Y[:,:,:,:,0:1] + Y[:,:,:,:,1:2]*Y[:,:,:,:,1:2], dim=0, keepdim=True) + offset) # normalize Y
    Cs = []
    for tau in range(Ls - Ly + 1):
        Z = S[tau : tau+Ly]
        Z = Z*torch.rsqrt(torch.mean(Z[:,:,:,:,0:1]*Z[:,:,:,:,0:1] + Z[:,:,:,:,1:2]*Z[:,:,:,:,1:2], dim=0, keepdim=True) + offset) # normalize Z
        cs = []
        for p in itertools.permutations(range(N)):
            cs.append(coh(Z, Y[:,:, p]))
        Cs.append(torch.stack(cs))
    Cs = torch.stack(Cs) # [delay, permutation, batch, mic]
    Cs, _ = torch.max(Cs, dim=0) # [permutation, batch, mic]
    Cs = torch.mean(Cs, dim=2) # [permutation, batch]
    Cs, _ = torch.max(Cs, dim=0) # [batch]
    return torch.mean(Cs)