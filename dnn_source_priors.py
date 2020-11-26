import torch

offset = 1e-10 # to avoid things like log(0), /0, etc.

def circ_prior( S, hs, Ws ):
    """Circular source prior, pdf p(S) only depends on |S|^2.    
    Inputs: S is the STFT representation with shape [batch, bin, 2]; hs are the hidden states; Ws are the network coefficients.    
    Output: - d log p(S) / d conj(S); updated hidden states
    """   
    # prepare the input features
    avgP = 2.0*torch.mean(S*S, dim=[1, 2])
    normS = S * torch.rsqrt(avgP + offset)[:, None, None] # normalized S
    logP = torch.log(avgP + offset)
    # this is the input features
    x = torch.log(normS[:,:,0]*normS[:,:,0] + normS[:,:,1]*normS[:,:,1] + offset) # log normalized power spectra
    x = torch.cat([x, logP[:, None]], dim=1)
    
    # pass through the hidden layers
    new_hs = []
    for (i, W) in enumerate(Ws[:-1]):
        x = torch.tanh(torch.cat([x, hs[i]], dim=1) @ W[:-1] + W[-1:])
        new_hs.append(x[:, :hs[i].shape[1]])
        
    # pass through the output layer
    x = x @ Ws[-1][:-1] + Ws[-1][-1:]
    x = -torch.log(torch.sigmoid( -x ))
    
    return normS * x[:,:, None], new_hs


def circp_Ws_init(K, num_layer, num_state, dim_h):
    """Circular source prior neural network initializer.
    K: number of bins; num_layer: number of layers; num_state: number of hidden states; dim_h: dimension of hidden layers (all the same)"""
    Ws = []
    # input layer
    W = torch.cat([torch.randn(K, dim_h)/K**0.5, # feedforward part on normalized spectra
                   torch.zeros(1, dim_h), # feedforward part on logP
                   torch.zeros(num_state, dim_h), # on hidden states
                   torch.zeros(1, dim_h)]) # bias
    Ws.append(W)
    # hidden layers
    for _ in range(num_layer-2):
        W = torch.cat([torch.randn(dim_h, dim_h)/dim_h**0.5, 
                       torch.zeros(num_state, dim_h), # on hidden states
                       torch.zeros(1, dim_h)]) 
        Ws.append(W)
    # output layer    
    W = torch.cat([torch.randn(dim_h, K)/dim_h**0.5,
                   torch.ones(1, K)])
    Ws.append(W)
    
    return Ws



def nonc_prior( S, hs, Ws ):
    """non-Circular source prior   
    Inputs: S is the STFT representation with shape [batch, bin, 2]; hs are the hidden states; Ws are the network coefficients.    
    Output: - d log p(S) / d conj(S); updated hidden states.
    """   
    # prepare the input features
    avgP = torch.mean(S*S, dim=[1, 2])
    normS = S * torch.rsqrt(avgP + offset)[:, None, None] # normalized S
    logP = torch.log(avgP + offset)
    # this is the input features
    x = torch.cat([normS[:,:,0], normS[:,:,1], logP[:, None]], dim=1)
    
    # pass through the hidden layers
    new_hs = []
    for (i, W) in enumerate(Ws[:-1]):
        x = torch.tanh(torch.cat([x, hs[i]], dim=1) @ W[:-1] + W[-1:])
        new_hs.append(x[:, :hs[i].shape[1]])
        
    # pass through the output layer
    x = x @ Ws[-1][:-1] + Ws[-1][-1:]
    
    K = S.shape[1]
    gain = -torch.log(torch.sigmoid( -x[:, :K] ))

    return gain[:,:,None] * normS + torch.stack([x[:, K:2*K], x[:, 2*K:]], dim=2), new_hs


def noncp_Ws_init(K, num_layer, num_state, dim_h):
    """non-Circular source prior neural network initializer.
    K: number of bins; num_layer: number of layers; num_state: number of hidden states; dim_h: dimension of hidden layers (all the same)"""
    Ws = []
    # input layer
    W = torch.cat([torch.randn(2*K, dim_h)/(2*K)**0.5, # feedforward part on normalized spectra
                   torch.zeros(1, dim_h), # feedforward part on logP
                   torch.zeros(num_state, dim_h), # on hidden states
                   torch.zeros(1, dim_h)]) # bias
    Ws.append(W)
    # hidden layers
    for _ in range(num_layer-2):
        W = torch.cat([torch.randn(dim_h, dim_h)/dim_h**0.5, 
                       torch.zeros(num_state, dim_h), # on hidden states
                       torch.zeros(1, dim_h)]) 
        Ws.append(W)
    # output layer   
    W = torch.cat([torch.cat([torch.randn(dim_h, K)/dim_h**0.5, torch.ones(1, K)], dim=0),
                   torch.zeros(dim_h+1, 2*K)], dim=1)
    Ws.append(W)
    
    return Ws