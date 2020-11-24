import torch

device = torch.device('cuda') # cpu or cuda

iva_fft_size = 1024 # the FFT size for IVA
iva_hop_size = 320 # the hop size of STFT for IVA
iva_lr = 1e-2 # the learning rate for IVA

num_mic = 5 # number of mics
batch_size = 50 # batch size
num_frame = 110 # number of frames for losses calculation
Lh = int(iva_fft_size**0.5)   # (mixing filter length - 1)//2
prb_mix_change = 0.01 # probability of sudden mixing condition changing 
wav_dir = '/hdd0/corpus/librivox' # dir saving the training speeches

circ_prior = {'num_layer':3, 'dim_h':128} # settings for circular source prior
assert circ_prior['num_layer'] >= 2

use_spectra_dist_loss = False # set to True to use the spectral distance loss in pdf training

psgd_setting = {'num_iter':10000, 'lr':0.01} # settings for the preconditioned SGD optimizer