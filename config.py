import scipy.io as sio
import torch

device = torch.device('cuda')
num_mic = 4 # number of mics

fs = 16000 # sampling rate
fft_size = 512  # fft size
hop_size = 160   # hop size in STFT
batch_size = 64 # batch size
num_frame = 128 # number of frame per chunk
L = 16   # (mixing filter length - 1)//2

mat_contents = sio.loadmat('fb_512_512_160.mat') 
win_a = torch.FloatTensor(mat_contents['win_a'][0]) # analysis window for STFT
win_s = torch.FloatTensor(mat_contents['win_s'][0]) # synthesis window for iSTFT