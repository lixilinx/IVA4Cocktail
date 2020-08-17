import torch
import config


def wfft( x ):
    # windowed fft
    # x: [batch, mic, time]
    x = config.win_a.view(1, 1, config.fft_size) * x
    X = torch.rfft(x, 1)
    return X[:, :, 1:-1, :] # discard the first and last bins; [batch, mic, bin, 2]


def wifft( X ):
    # windowed ifft
    # X: [batch, mic, bin, 2]
    B, M, _, _ = X.shape
    z = torch.zeros(B, M, 1, 2, device = X.device)
    X = torch.cat([z, X, z], dim = 2) # fill the first and last bins with zeros
    x = torch.irfft(X, 1, signal_sizes = [config.fft_size])
    x = config.win_s.view(1, 1, config.fft_size) * x
    return x # [batch, mic, time]


def stft( x ):
    # short-time Fourier transform
    # x: [batch, mic, time]
    X = []
    t = 0
    while t + config.fft_size <= x.shape[2]:
        X.append( wfft( x[:,:, t : t + config.fft_size] ) )
        t += config.hop_size
        
    return torch.stack( X ) # [time, batch, mic, bin, 2]


def istft( X ):
    # inverse short-time Fourier transform
    # X: [time, batch, mic, bin, 2]
    T, B, M, _, _ = X.shape
    x = torch.zeros( B, M, config.fft_size + (T - 1) * config.hop_size, device = X.device )
    t = 0
    for Xt in X:
        x[:,:, t : t + config.fft_size] += wifft( Xt )
        t += config.hop_size
        
    return x # [batch, mic, time]