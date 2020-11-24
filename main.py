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


def iva( Xs, W, Ws_circp, lr ):
    """
    IVA.
    Inputs: Xs has shape [time, batch, mic, bin, 2]; W is the filter with shape [batch, bin, mic, mic, 2];
    Ws_circp is the circular source prior model coefficients; lr is the learning rate for IVA. 
    Outputs: Ys has shape [time, batch, mic, bin, 2]; and updated W with shape [batch, bin, mic, mic, 2].
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
        G = dnn_source_priors.circ_prior(Y.reshape(B*M, K, 2), Ws_circp).view(B, M, K, 2) # [batch, mic, bin, 2] 
        
        # calculate the natural gradient: (G*Y^H - I)*W
        Gp = G.permute([0, 2, 1, 3]) # [batch, bin, mic, 2]
        NGr =  torch.matmul(Gp[:,:,:,0:1], Ypr.transpose(2,3)) + torch.matmul(Gp[:,:,:,1:2], Ypi.transpose(2,3)) # [batch, bin, mic, mic]
        NGi = -torch.matmul(Gp[:,:,:,0:1], Ypi.transpose(2,3)) + torch.matmul(Gp[:,:,:,1:2], Ypr.transpose(2,3))
        NGr = NGr - torch.eye(M, device=W.device)[None, None, :,:]
        norm_gain = torch.rsqrt(torch.sum(NGr*NGr + NGi*NGi, dim=[2,3]) - (M - 2)) # a normalization gain: 1/norm( G*Y^H - I ). See the paper on 2-norm estimation
        NG = torch.stack([torch.matmul(NGr, W[:,:,:,:,0]) - torch.matmul(NGi, W[:,:,:,:,1]),
                          torch.matmul(NGr, W[:,:,:,:,1]) + torch.matmul(NGi, W[:,:,:,:,0])], dim=4)
        
        # update IVA filter coefficients
        W = W - lr*norm_gain[:,:,None,None,None]*NG
        
    Ys = torch.stack( Ys )  # [time, batch, mic, bin, 2], the same as Xs
    return Ys, W
 

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
    
    Ws = [W.to(device) for W in dnn_source_priors.circp_Ws_init(config.iva_fft_size//2-1,                  
                                                                config.circ_prior['num_layer'],              
                                                                config.circ_prior['dim_h'])]
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
    
    Loss, lr = [], config.psgd_setting['lr']
    for bi in range(config.psgd_setting['num_iter']):
        srcs, xs = mixergenerator.get_mixtures( )

        Ss, s_bfr = stft.stft(srcs[:,:, config.Lh : -config.Lh], win_iva, config.iva_hop_size, s_bfr)
        Xs, x_bfr = stft.stft(xs,                                win_iva, config.iva_hop_size, x_bfr)
        Ss, Xs = Ss.to(device), Xs.to(device)
        
        Ys, W_iva = iva(Xs, W_iva.detach(), Ws, config.iva_lr)
        
        # loss calculation    
        coherence = losses.di_pi_coh(Ss, Ys)
        loss = 1.0 - coherence
        if config.use_spectra_dist_loss:
            spectra_dist = losses.di_pi_is_dist(Ss[:,:,:,0]*Ss[:,:,:,0] + Ss[:,:,:,1]*Ss[:,:,:,1],                                       
                                                Ys[:,:,:,0]*Ys[:,:,:,0] + Ys[:,:,:,1]*Ys[:,:,:,1])
            loss = loss + spectra_dist      
    
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
            
    plt.plot(Loss)
    scipy.io.savemat('circ_src_prior.mat', dict([('W'+str(i), W.cpu().detach().numpy()) for (i, W) in enumerate(Ws)]))
    
if __name__ == '__main__':
    main()