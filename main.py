import math
import scipy.io
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import preconditioned_stochastic_gradient_descent as psgd 
import config
import short_time_Fourier_transform as stft
import artificial_mixture_generator 
import dnn_source_priors 
import losses

if config.src_prior['circ']:
    prior = dnn_source_priors.circ_prior
    initW = dnn_source_priors.circp_Ws_init
else:
    prior = dnn_source_priors.nonc_prior
    initW = dnn_source_priors.noncp_Ws_init


def iva( Xs, W, hs, Ws_prior, lr ):
    """
    IVA.
    Inputs: Xs has shape [time, batch, mic, bin, 2]; W is the filter with shape [batch, bin, mic, mic, 2];
    hs are the hidden states of source model; Ws_prior are the source model coefficients;  
    lr is the learning rate for IVA. 
    Outputs: Ys has shape [time, batch, mic, bin, 2]; updated W with shape [batch, bin, mic, mic, 2]; updated hidden states.
    """   
    _, B, M, K, _ = Xs.shape
    Ys = [] # will have shape [time, batch, mic, bin, 2]
    for X in Xs: # X: [batch, mic, bin, 2]
        # calculate Y = W*X
        Xp = X.permute([0, 2, 1, 3]) # [batch, bin, mic, 2]
        Ypr = torch.matmul(W[:,:,:,:,0], Xp[:,:,:,0:1]) - torch.matmul(W[:,:,:,:,1], Xp[:,:,:,1:2]) # [batch, bin, mic, 1]
        Ypi = torch.matmul(W[:,:,:,:,0], Xp[:,:,:,1:2]) + torch.matmul(W[:,:,:,:,1], Xp[:,:,:,0:1])
        Y = ( torch.cat([Ypr, Ypi], dim=3) ).permute(0,2,1,3) # [batch, mic, bin, 2]
        Ys.append(Y)
        
        # calculate gradient G = - d log p(Y) / d Y
        G, hs = prior(Y.reshape(B*M, K, 2), hs, Ws_prior)
        G = G.view(B, M, K, 2) # [batch, mic, bin, 2] 
        
        # natural gradient descent: W <-- (1+lr)*W - lr*G*Y^H*W
        normlr = lr*torch.rsqrt(2 - 2*torch.sum(G*Y, dim=[1,3]) + torch.sum(G*G, dim=[1,3])*torch.sum(Y*Y, dim=[1,3])) # normalized learning rate
        YphWr = torch.matmul(Ypr.transpose(2,3), W[:,:,:,:,0]) + torch.matmul(Ypi.transpose(2,3), W[:,:,:,:,1])
        YphWi = torch.matmul(Ypr.transpose(2,3), W[:,:,:,:,1]) - torch.matmul(Ypi.transpose(2,3), W[:,:,:,:,0])
        Gp = G.permute([0, 2, 1, 3]) # [batch, bin, mic, 2]
        W = (1 + normlr[:,:,None,None,None])*W - normlr[:,:,None,None,None]*torch.stack([torch.matmul(Gp[:,:,:,0:1], YphWr) - torch.matmul(Gp[:,:,:,1:2], YphWi),
                                                                                         torch.matmul(Gp[:,:,:,0:1], YphWi) + torch.matmul(Gp[:,:,:,1:2], YphWr)], dim=4)
        
    Ys = torch.stack( Ys )  # [time, batch, mic, bin, 2], the same as Xs
    return Ys, W, hs
 

def main( ): 
    """DNN speech prior training"""
    device = config.device               
    wavloader = artificial_mixture_generator.WavLoader(config.wav_dir)
    mixergenerator = artificial_mixture_generator.MixerGenerator(wavloader, 
                                                                 config.batch_size, 
                                                                 config.num_mic, 
                                                                 config.Lh, 
                                                                 config.iva_hop_size*config.num_frame, 
                                                                 config.prb_mix_change)
    
    Ws = [W.to(device) for W in initW(config.iva_fft_size//2-1,                                                     
                                      config.src_prior['num_layer'],  
                                      config.src_prior['num_state'], 
                                      config.src_prior['dim_h'])]
    hs = [torch.zeros(config.batch_size*config.num_mic, config.src_prior['num_state']).to(device) for _ in range(config.src_prior['num_layer'] - 1)]
    for W in Ws:
        W.requires_grad = True
        
    # W_iva initialization
    W_iva = ( 100.0*torch.randn(config.batch_size, config.iva_fft_size//2-1, config.num_mic, config.num_mic, 2) ).to(device)
    
    # STFT window for IVA
    win_iva = stft.pre_def_win(config.iva_fft_size, config.iva_hop_size)
    
    # preconditioners for the source prior neural network optimization
    Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws] # preconditioners for SGD
    
    # buffers for STFT
    s_bfr = torch.zeros(config.batch_size, config.num_mic, config.iva_fft_size - config.iva_hop_size)
    x_bfr = torch.zeros(config.batch_size, config.num_mic, config.iva_fft_size - config.iva_hop_size)
    
    # buffer for overlap-add synthesis and reconstruction loss calculation
    ola_bfr = torch.zeros(config.batch_size, config.num_mic, config.iva_fft_size).to(device)
    xtr_bfr = torch.zeros(config.batch_size, config.num_mic, config.iva_fft_size - config.iva_hop_size) # extra buffer for reconstruction loss calculation

    
    Loss, lr = [], config.psgd_setting['lr']
    for bi in range(config.psgd_setting['num_iter']):
        srcs, xs = mixergenerator.get_mixtures( )

        Ss, s_bfr = stft.stft(srcs[:,:, config.Lh : -config.Lh], win_iva, config.iva_hop_size, s_bfr)
        Xs, x_bfr = stft.stft(xs,                                win_iva, config.iva_hop_size, x_bfr)
        Ss, Xs = Ss.to(device), Xs.to(device)

        Ys, W_iva, hs = iva(Xs, W_iva.detach(), [h.detach() for h in hs], Ws, config.iva_lr)
        
        # loss calculation    
        coherence = losses.di_pi_coh(Ss, Ys)
        loss = 1.0 - coherence
        if config.use_spectra_dist_loss:
            spectra_dist = losses.di_pi_is_dist(Ss[:,:,:,0]*Ss[:,:,:,0] + Ss[:,:,:,1]*Ss[:,:,:,1],                                       
                                                Ys[:,:,:,0]*Ys[:,:,:,0] + Ys[:,:,:,1]*Ys[:,:,:,1])
            loss = loss + spectra_dist   
            
        if config.reconstruction_loss_fft_sizes:
            srcs = torch.cat([xtr_bfr, srcs], dim=2)
            ys, ola_bfr = stft.istft(Ys, win_iva.to(device), config.iva_hop_size, ola_bfr.detach())
            for fft_size in config.reconstruction_loss_fft_sizes:
                win = stft.coswin(fft_size)
                Ss, _ = stft.stft(srcs, win,            fft_size//2)
                Ys, _ = stft.stft(ys,   win.to(device), fft_size//2)
                Ss = Ss.to(device)
                coherence = losses.di_pi_coh(Ss, Ys)
                loss = loss + 1.0 - coherence
                if config.use_spectra_dist_loss:
                    spectra_dist = losses.di_pi_is_dist(Ss[:,:,:,0]*Ss[:,:,:,0] + Ss[:,:,:,1]*Ss[:,:,:,1],                  
                                                        Ys[:,:,:,0]*Ys[:,:,:,0] + Ys[:,:,:,1]*Ys[:,:,:,1])
                    loss = loss + spectra_dist
            
            xtr_bfr = srcs[:,:, -(config.iva_fft_size - config.iva_hop_size):]
    
        Loss.append(loss.item())
        if config.use_spectra_dist_loss:
            print('Loss: {}; coherence: {}; spectral_distance: {}'.format(loss.item(), coherence.item(), spectra_dist.item()))
        else:
            print('Loss: {}; coherence: {}'.format(loss.item(), coherence.item()))
            
        # Preconditioned SGD optimizer for source prior network optimization 
        Q_update_gap = max(math.floor(math.log10(bi + 1)), 1)
        if bi % Q_update_gap == 0: # update preconditioner less frequently 
            grads = grad(loss, Ws, create_graph=True)     
            v = [torch.randn(W.shape, device=device) for W in Ws]
            Hv = grad(grads, Ws, v)      
            with torch.no_grad():
                Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
        else:
            grads = grad(loss, Ws)
            
        with torch.no_grad():
            pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
            for i in range(len(Ws)):
                Ws[i] -= lr*pre_grads[i]
                
        if bi == int(0.9*config.psgd_setting['num_iter']):
            lr *= 0.1
        if (bi+1)%1000 == 0 or bi+1 == config.psgd_setting['num_iter']:
            scipy.io.savemat('src_prior.mat', dict([('W'+str(i), W.cpu().detach().numpy()) for (i, W) in enumerate(Ws)]))
            
    plt.plot(Loss)
    
if __name__ == '__main__':
    main()