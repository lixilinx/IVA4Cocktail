import os
import random
import numpy as np
import scipy.io.wavfile as wavfile
import torch

#lpc = [1.0,  -1.23,   0.37] # lpc for speech pre-emphasis
#lpc = [1.0, -0.95, 0.0]
lpc = []

sampling_rate = 16000 # sampling rate
min_wav_len = sampling_rate//10 # minimum wave file length in number of samples

class WavLoader: 
    """Read WAV files randomly from a given folder """
    def __init__(self, folder):
        """folder is the place with WAV files"""
        self.wav_list = [] # a list of all wav files       
        
        def list_all_wavs( f ): # f is the folder
            lst = os.listdir( f )
            for i in range(len(lst)):
                if os.path.isdir( ''.join([f, '/', lst[i]]) ):
                    list_all_wavs( ''.join([f, '/', lst[i]]) )
                elif lst[i].lower().endswith('.wav'):
                    self.wav_list.append( ''.join([f, '/', lst[i]]) )
                    
        list_all_wavs( folder )
        
        
    def get_rand_wav( self ): 
        """Read a random wav file, and do necessary pre-processing (de-quantization, DC removal; re-scaling, ...)"""
        i = random.randint(0, len(self.wav_list) - 1)
        fs, data = wavfile.read( self.wav_list[i] )
        if fs != sampling_rate or data.shape[0] < min_wav_len:
            return [] # invalid reading 
        elif len(data.shape)==1: # mono
            s = data
        else: # get a random channel
            s = data[:, random.randint(0, data.shape[1] - 1)]
        
        if s.dtype != np.dtype('float32'): # then, 1)de-quantization for PCM readings; 2) rescaling to range [-1, 1)
            if s.dtype == np.dtype('int16'):
                s = (s + np.random.rand(len(s)))/32768.0 # 16 bits PCM files
            elif s.dtype == np.dtype('int32'):
                s = (s + np.random.rand(len(s)))/2147483648.0 # 32 bits PCM files
            elif s.dtype == np.dtype('uint8'):
                s = (s + np.random.rand(len(s)))/256.0 # 8 bits PCM files, the DC=0.5 will be removed later
            
        s = s - np.mean(s) # remove any DC
        if lpc:
            s = lpc[0]*s[2:] + lpc[1]*s[1:-1] + lpc[2]*s[:-2] # optional lpc pre-emphasis
        return s
        

class MixerGenerator: 
    """Generate artificial speech mixtures with random mixing matrices"""
    def __init__(self, wavloader, B, M, L, T, p):
        """wavloader: WavLoader object; B: batch size; M: number of mics; L: (len(mixing_filter)-1)/2; T: len(mixtures); p: Prb(mixing condition change)"""
        self.wavloader = wavloader
        self.B = B # batch size
        self.M  = M # number of microphones
        self.L = L # (mixing filter length - 1)//2
        self.T = T # length of mixtures
        self.p = p # probability of sudden mixing path change
        self.srcs = torch.zeros(B, M, T + 2 * L) # sources, the extra 2*L samples are states for convolution 
        self.wavs = [np.zeros(0) for _ in range(B * M)] # wave file reading buffer
        self.As = torch.randn(2*L+1, B, M, M)/(2*L+1)**0.5 # mixing filter matrices
        
    def get_mixtures( self ):
        """Return a set of sources and their mixtures."""
        # make sure that the wave file reading buffer has enough samples
        for i in range(len(self.wavs)):
            while len(self.wavs[i]) <= self.T:
                wav = self.wavloader.get_rand_wav( )
                self.wavs[i] = np.concatenate([self.wavs[i], wav])
                
        # update self.srcs: replace T old samples with new ones
        new_samples = np.stack([wav[:self.T] for wav in self.wavs])
        new_samples = torch.FloatTensor( new_samples.reshape(self.B, self.M, self.T) )
        self.srcs = torch.cat([self.srcs[:,:, self.T : ], new_samples], dim = 2)
        
        # convolutive mixing
        x = torch.zeros(self.B, self.M, self.T)
        for i in range(2*self.L + 1):
            x += torch.bmm(self.As[i], self.srcs[:,:, i:i+self.T])
         
        # update self.wavs buffer: discard T used samples
        for i in range(len(self.wavs)): 
            self.wavs[i] = self.wavs[i][self.T:]
        
        # suddenly change mixing filter matrices with probability p
        for i in range(self.B):
            if random.uniform(0, 1) < self.p:
                self.As[:, i] = torch.randn(2*self.L + 1, self.M, self.M)/(2*self.L+1)**0.5
            
        # return sources and mixtures 
        ##return self.srcs[:,:, self.L : self.L + self.T], x
        return self.srcs, x # note: return all source samples including the extra 2*L states