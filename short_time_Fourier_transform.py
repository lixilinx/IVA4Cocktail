import math
import scipy.io as sio
import torch

try: # use new FFT API
    import torch.fft
    def rfft(x):
        return torch.view_as_real(torch.fft.rfft(x))
    def irfft(X):
        return torch.fft.irfft(torch.view_as_complex(X))
except: # use the old FFT API
    def rfft(x):
        return torch.rfft(x, 1)
    def irfft(X):
        return torch.irfft(X, 1, signal_sizes = [2*(X.shape[-2] - 1)])
        

pi = math.acos(-1.0)

def coswin( N ):
    """
    Cosine window.
    Input: N is the window length.   
    Output: w is the window. Use hop size len(w)/2 and rectangle window for synthesis if needed.  
    """
    w = torch.arange(0, N, dtype=torch.float)
    w = w - 0.5*N + 0.5
    w = 0.5 + 0.5*torch.cos(2*pi*w/N)
    return w

def pre_def_win(fft_size, hop_size):
    """
    Load pre-defined windows (if exists)   
    Inputs: fft_size is the FFT size; hop_size is the hop size.    
    Output: w is my pre-defined window (the same for analysis and synthesis). See
    https://ieeexplore.ieee.org/document/8304771 for design details.
    """
    try:
        mat_contents = sio.loadmat(''.join(['win_', str(fft_size), '_', str(hop_size), '.mat'])) 
    except:
        print('Error: cannot find the pre-defined window')
        return []
          
    w = torch.FloatTensor(mat_contents['win'][0]) # analysis and synthesis windows are the same
    return w


def wfft( x, win_a ):
    """
    Windowed FFT.    
    Inputs: x has shape [batch, mic, time]; win_a is the analysis window.    
    Output: X has shape [batch, mic, bin, 2]. Note that the first and last bins are discarded. 
    """
    x = win_a[None, None, :] * x
    X = rfft(x) # [batch, mic, bin, 2]
    return X[:, :, 1:-1, :] # [batch, mic, bin, 2]; discard the first and last bins (real valued, not very useful in practice) 


def wifft( X, win_s ):
    """Windowed IFFT.  
    Inputs: X has shape [batch, mic, bin, 2]; win_s is the synthesis window.   
    Output: x has shape [batch, mic, time]. """
    B, M, _, _ = X.shape
    z = torch.zeros(B, M, 1, 2, device = X.device)
    X = torch.cat([z, X, z], dim = 2) # fill the first and last bins with zeros
    x = irfft(X)
    x = win_s[None, None, :] * x # [batch, mic, time]
    return x 


def stft( x, win_a, hop_size, bfr=None ):
    """ Short-time Fourier transform.     
    Inputs: x has shape [batch, mic, time]; win_a is the analysis window;
    hop_size is the hop size; bfr is the input buffer for STFT (fft_size - hop_size samples).    
    Outputs: X has shape [time, batch, mic, bin, 2]; bfr is the updated input buffer."""
    fft_size = len(win_a)
    z = x if (bfr is None) else torch.cat([bfr, x], dim=2)
        
    X = []
    t = 0
    while t + fft_size <= z.shape[2]:
        X.append( wfft( z[:,:, t:t+fft_size], win_a ) )
        t += hop_size
        
    X = torch.stack( X )   # [time, batch, mic, bin, 2] 
    bfr = z[:,:, t : t+fft_size-hop_size]
    return X, bfr 


def istft( X, win_s, hop_size, ola_bfr=None ):
    """Inverse short-time Fourier transform.     
    Inputs: X has shape [time, batch, mic, bin, 2]; win_s is the synthesis window; 
    hop_size is the hop size for STFT; ola_bfr is the overlap-add buffer state.   
    Outputs: x has shape [batch, mic, time]; ola_bfr is the updated buffer (fft_size samples). 
    """
    _, B, M, _, _ = X.shape
    z = torch.zeros(B, M, hop_size, device = X.device) 
    if ola_bfr is None:
        ola_bfr = torch.zeros(B, M, len(win_s), device=X.device)
         
    xs = []
    for Xt in X:
        ola_bfr = ola_bfr + wifft( Xt, win_s )
        xs.append( ola_bfr[:,:, :hop_size] )
        ola_bfr = torch.cat([ola_bfr[:,:, hop_size:], z], dim=2)
        
    x = torch.cat(xs, dim=2) # [batch, mic, time]
    return x, ola_bfr 