import os
import random
import math
import itertools
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import config
import short_time_Fourier_transform as F
import preconditioned_stochastic_gradient_descent as psgd 


class WavLoader: # read speech wave files randomly from a folder
    def __init__(self, folder):
        # put all the wave (here, speech) files under a folder 
        self.wav_list = []        
        
        def list_all_wavs( f ):
            lst = os.listdir( f )
            for i in range(len(lst)):
                if os.path.isdir( ''.join([f, '/', lst[i]]) ):
                    list_all_wavs( ''.join([f, '/', lst[i]]) )
                elif lst[i].lower().endswith('.wav'):
                    self.wav_list.append( ''.join([f, '/', lst[i]]) )
                    
        list_all_wavs( folder )
        
    def get_rand_wav( self ):
        i = random.randint(0, len(self.wav_list) - 1)
        fs, data = wavfile.read( self.wav_list[i] )
        if fs != config.fs:
            return []
        elif len(data.shape)==1: # mono
            return data
        else: # get a random channel
            return data[:, random.randint(0, data.shape[1] - 1)]
        

class MixerGenerator: # generate speech mixtures with random mixing matrices
    def __init__(self, wavloader, B, M, L, T, p):
        self.wavloader = wavloader
        self.B = B # batch size
        self.M  = M # number of microphones
        self.L = L # (mixing filter length - 1)//2
        self.T = T # signal length
        self.p = p # probability of sudden mixing path change
        self.srcs = torch.zeros(B, M, T + 2 * L) # sources
        self.wavs = [np.zeros(0) for _ in range(B * M)] # wave file reading buffer
        self.As = torch.randn(2*L+1, B, M, M) # mixing filter matrices
        
    def get_mixtures( self ):
        # make sure that the wave file reading buffer has enough samples
        for i in range(len(self.wavs)):
            while len(self.wavs[i]) <= self.T:
                wav = self.wavloader.get_rand_wav( )
                wav = (wav + np.random.rand(len(wav)) - 0.5)/32768.0 # de-quantization
                self.wavs[i] = np.concatenate([self.wavs[i], wav])
                
        # update self.srcs: replace T old samples with new ones
        new_samples = np.stack([wav[:self.T] for wav in self.wavs])
        new_samples = torch.FloatTensor( new_samples.reshape(self.B, self.M, self.T) )
        self.srcs = torch.cat([self.srcs[:,:, self.T : ], new_samples], dim = 2)
        
        # convolutional mixing
        x = torch.zeros(self.B, self.M, self.T)
        for i in range(2*self.L + 1):
            x += torch.bmm(self.As[i], self.srcs[:,:, i:i+self.T]) / (1.0 + math.fabs(i - self.L))
         
        # update self.wavs buffer: discard T used samples
        for i in range(len(self.wavs)): 
            self.wavs[i] = self.wavs[i][self.T:]
        
        # suddenly change mixing filter matrices with probability p
        for i in range(self.B):
            if random.uniform(0, 1) < self.p:
                self.As[:, i] = torch.randn(2*self.L + 1, self.M, self.M)
            
        # return sources and mixtures 
        return self.srcs[:,:, self.L : self.L + self.T], x
                

def grad_nll( x, h ):
    # gradient of the negative_log_likelihood of x w.r.t. x
    # x: the raw features
    # first dim is batch size    
       
    # prepare the features
    power_x = 2.0 * torch.mean(x * x, dim = 1, keepdim = True) + 1e-7 # for 16 bits pcm, set an offset about fft_size/32768/32768
    logE = torch.log(power_x)
    x = x * torch.rsqrt(power_x) # normalized x
    g = 1.0*x # the gradient in IVA
    
    x = torch.log(x[:, 0::2]*x[:, 0::2] + x[:, 1::2]*x[:, 1::2] + 1e-7)

    # 1st layer
    x = torch.tanh(torch.cat([x, logE, h], dim = 1) @ W1[:-1] + W1[-1:])   
    # keep the states
    h = x[:, :h.shape[1]]
    
    # 2nd layer
    x = torch.tanh(x @ W2[:-1] + W2[-1:])
    
    # 3rd layer
    x = x @ W3[:-1] + W3[-1:]
    x = -torch.log(torch.sigmoid( -x ))
    
    return g * x.repeat_interleave(2, dim=1), h


def iva( Xs, Wr, Wi, h, lr ):
    # IVA on Xs
    # Xs: [time, batch, mic, bin, 2]
    # Wr, Wi: real and imag parts of IVA filters, [batch*bin, mic, mic]
    # lr is the learning rate for IVA
    
    _, B, M, K, _ = Xs.shape
    Ys = [] # will have shape [time, batch, mic, bin, 2]
    for X in Xs:
        # X: [batch, mic, bin, 2]
        # calculate Y = W*X
        X = torch.transpose(X, 1, 2) # now, [batch, bin, mic, 2]
        Xr = X[:,:,:, 0:1].reshape(-1, M, 1) # real part of X, [batch*bin, mic, 1]
        Xi = X[:,:,:, 1:2].reshape(-1, M, 1) # imag part of X, [batch*bin, mic, 1]
        Y = torch.cat([torch.bmm(Wr, Xr) - torch.bmm(Wi, Xi),                    
                       torch.bmm(Wr, Xi) + torch.bmm(Wi, Xr)], dim = 2) # [batch*bin, mic, 2]
        
        # calculate G = - d log p(Y) / d Y
        Y_bbm = Y.view(B, -1, M, 2) # Y in [batch, bin, mic, 2] format
        Y_bmb = torch.transpose(Y_bbm, 1, 2) # Y in [batch, mic, bin, 2] format
        G, h = grad_nll( Y_bmb.reshape(B*M, -1), h ) # gradient in [batch*mic, bin*2] format
        G_bmb = G.view(B, M, -1, 2) # gradient in [batch, mic, bin, 2] format
        G_bbm = torch.transpose(G_bmb, 1, 2) # gradient in [batch, bin, mic, 2] format
        
        # update IVA filter coefficients
        # basically, it is W <-- W - mu*(G*Y^H - I)*W
        # But, there are multiple variations
        Gr = G_bbm[:,:,:, 0:1].reshape(-1, M, 1) # real part of G with shape [batch*bin, mic, 1]
        Gi = G_bbm[:,:,:, 1:2].reshape(-1, M, 1) # imag part of G with shape [batch*bin, mic, 1]
        Yh_r = torch.transpose(Y[:,:, 0:1], 1, 2) # real part of Y^H, [batch*bin, 1, mic]
        Yh_i = -torch.transpose(Y[:,:, 1:2], 1, 2) # imag part of Y^H, [batch*bin, 1, mic]
        # this is normalized natural gradient descent
        GYhmI_r = torch.bmm(Gr, Yh_r) - torch.bmm(Gi, Yh_i) - Eyes # real parat of G*Y^H - I
        GYhmI_i = torch.bmm(Gr, Yh_i) + torch.bmm(Gi, Yh_r) # imag part of G*Y^H - I   
        norm_gain = torch.rsqrt(torch.sum( GYhmI_r*GYhmI_r + GYhmI_i*GYhmI_i, dim = [1, 2] ) - (config.num_mic - 2)) # a normalization gain: 1/norm( G*Y^H - I ). See the paper on 2-norm estimation
        Wr, Wi = [Wr - lr * (torch.bmm(GYhmI_r, Wr) - torch.bmm(GYhmI_i, Wi)) * norm_gain.view(-1, 1, 1), 
                  Wi - lr * (torch.bmm(GYhmI_r, Wi) + torch.bmm(GYhmI_i, Wr)) * norm_gain.view(-1, 1, 1),]
        
        # append Y
        Ys.append(Y)
    # prepare Ys
    Ys = torch.stack( Ys ) # [time, batch*bin, mic, 2]
    Ys = Ys.view(-1, B, K, M, 2) # [time, batch, bin, mic, 2]
    Ys = torch.transpose(Ys, 2, 3) # [time, batch, mic, bin, 2], same as Xs
    return Ys, Wr, Wi, h
 
       
def coherence(S, Y):
    # calculate the coherence between STFT results S and Y
    # S, Y: [time, batch, mic, bin, 2]
    Sr = S[:,:,:,:, 0]
    Si = S[:,:,:,:, 1]
    Yr = Y[:,:,:,:, 0]
    Yi = Y[:,:,:,:, 1]
    Cr = torch.mean(Sr*Yr + Si*Yi, dim = 0) # real part of S*conj(Y)
    Ci = torch.mean(-Sr*Yi + Si*Yr, dim = 0) # imag part of S*conj(Y)
    C2 = (Cr*Cr + Ci*Ci)/(torch.mean(Sr*Sr + Si*Si, dim=0))/(torch.mean(Yr*Yr + Yi*Yi, dim=0))
    return torch.mean(torch.sqrt(C2), dim=[1, 2]) # [batch]


def pi_coherence(S, Y):
    # permutation invariant coherence between STFT results S and Y
    # S, Y: [time, batch, mic, bin, 2]
    M = S.shape[2]
    Cs = []
    for p in itertools.permutations(range(M)):
        Cs.append( coherence(S, Y[:,:, p]) )
    Cs = torch.stack(Cs)
    maxCs, _ = torch.max(Cs, dim = 0)
    return torch.mean(maxCs)
     
    
# begin the training here
device = config.device
B, M, K = config.batch_size, config.num_mic, config.fft_size//2 - 1
wavloader = WavLoader('//10.0.0.60/hdd0/corpus/librivox')
mixergenerator = MixerGenerator(wavloader, B, M, config.L, config.hop_size*config.num_frame, 0.02)

# this Eyes will be useful
Eyes = torch.eye(M, device=device).view(1, M, M).repeat(B * K, 1, 1)

# the model coefficients for gradient calculation
# A simple example with 3 layers.
dim_h = 0
dim_f = 512
h = torch.zeros(B*M, dim_h, device = device)
W1 = torch.cat([torch.randn(K, dim_f)/K**0.5, # feedforward part on x
                torch.zeros(1, dim_f), # feedforward part on delta_logE
                torch.zeros(dim_h, dim_f),  # feedback part on hidden states
                torch.zeros(1, dim_f)]).to(device) # bias
W2 = torch.cat([torch.randn(dim_f, dim_f)/dim_f**0.5, # 2nd layer
                torch.zeros(1, dim_f)]).to(device)
W3 = torch.cat([torch.randn(dim_f, K)/dim_f**0.5, # 3rd layer
                torch.ones(1, K)]).to(device)

Ws = [W1, W2, W3]
for W in Ws:
    W.requires_grad = True
    
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws] # preconditioners for SGD
IVA_Wr, IVA_Wi = Eyes, 0*Eyes

# buffers for wfft
s_bfr = torch.zeros(B, M, config.fft_size)
x_bfr = torch.zeros(B, M, config.fft_size)

Cs = [] # coherences
bi, num_iter, lr = 0, 20000, 1e-2
for bi in range(num_iter):
    srcs, xs = mixergenerator.get_mixtures( )
    S, Xs = [], []
    for i in range(config.num_frame):
        s_bfr = torch.cat([s_bfr[:,:, config.hop_size:], srcs[:,:, i*config.hop_size : (i+1)*config.hop_size]], dim = 2)
        x_bfr = torch.cat([x_bfr[:,:, config.hop_size:],   xs[:,:, i*config.hop_size : (i+1)*config.hop_size]], dim = 2)
        S.append( F.wfft( s_bfr ) )
        Xs.append( F.wfft( x_bfr ) )
        
    S = torch.stack( S ).to(device)
    Xs = torch.stack( Xs ).to(device)
       
    Ys, IVA_Wr, IVA_Wi, h = iva( Xs, IVA_Wr.detach(), IVA_Wi.detach(), h.detach(), 0.01 )
    avg_c = pi_coherence(S, Ys) # average coherence between sources and outputs

    # Preconditioned SGD optimizer. 
    Q_update_gap = max(math.floor(math.log10(bi + 1)), 1)
    if bi % Q_update_gap == 0: # update preconditioner less frequently 
        grads = grad(avg_c, Ws, create_graph=True)     
        v = [torch.randn(W.shape, device=device) for W in Ws]
        Hv = grad(grads, Ws, v)      
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
    else:
        grads = grad(avg_c, Ws)
        
    with torch.no_grad():
        pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
        for i in range(len(Ws)):
            Ws[i] += lr*pre_grads[i] # actaully, gradient ascent
        
    Cs.append( avg_c.item() )
    print('coherence: {0:1.2f}'.format(avg_c.item()))
        
    if (bi + 1) % 100 == 0:
        scipy.io.savemat('grad_fnn_mdl.mat',                               
                         {'W1': Ws[0].cpu().detach().numpy(),                                
                          'W2': Ws[1].cpu().detach().numpy(),                                
                          'W3': Ws[2].cpu().detach().numpy()})
        
plt.plot(Cs)

# check the first two separated outputs
ys = F.istft( Ys.cpu().detach() ).numpy()
wavfile.write('sample_out.wav', 16000, np.stack([ys[0,0], ys[0,1]], axis=1)/np.max(np.abs(ys[0])) )