# [IVA4Cocktail](https://arxiv.org/abs/2008.11273)

Speech density estimation for multichannel convolutive speech separation. I use independent vector analysis (IVA) as the separation framework. The code can also be used for music or other source separations. Please check the report for details.

Unlike the currently popular end-to-end black box speech separation methods, here, how the mixtures are separated is a well formulated optimization problem. Our focus is to learn a fine neural network density model for speech by optimization certain proxy objectives. Our resultant density model can be used in different scenarios, i.e., online or batch separation, different number of sources, artificial or realistic mixtures, etc., while an end-to-end model is typically trained for one very specific scenario.         

## Speech Density Estimation

You need to correctly set the path parameter for WavLoader in 'main.py' to train the density model. 

Set dim_h = 0 to get a feedforward neural network (FNN) density model, and 1 <= dim_h <= dim_f to get a recurrent neural network (RNN) density model. Redefine grad_nll to train a different density model.  

Even with a memoryless density model, the whole signal process graph still defines a deep recurrent network due to the way to update the separation matrices. Momentum and first order optimization methods seem perform not very well (might not with the best hyperparameters in my trials). A [second order method](https://ieeexplore.ieee.org/document/7875097) saves the hyperparameter tuning effort. 

## Separation Performance Comparisons

I use [this code](https://www.mathworks.com/matlabcentral/fileexchange/5116-room-impulse-response-generator) to generate room impulse response (RIR). It is light and fast. But, as the original image source method, it cannot simulate fractional delays. So, I generate the RIR with higher sampling rate, and then decimate to the correct sampling rate. 

You need to correctly set the path in 'generate_mixtures.m' to run the performance comparison simulations. Sample FNN and RNN density models prepared as in the report are included here. The neural network density models consistently outperform simple ones like multivariate Laplace and a non-spherical distributions (from Te-Won Lee's group), generalized Gaussian and Student's t- distribution (from Jonathon Chambers' group).

#### Convergence speed comparison, two sources
![alt text](https://github.com/lixilinx/IVA4Cocktail/blob/master/sir_vs_time.png)

#### Efficiency comparison 1, 10 s speech length
![alt text](https://github.com/lixilinx/IVA4Cocktail/blob/master/sir_vs_N.png)

#### Efficiency comparison 2, three sources 
Need to halve the step size in iva_batch, otherwise, the t-distribution prior may cause divergence.
![alt text](https://github.com/lixilinx/IVA4Cocktail/blob/master/sir_vs_length.png)

#### Sample separation results for subjective comparison

[These](https://drive.google.com/file/d/18xrjgKNbWOnl0t_w0XnsB0zOr4EOWpX-/view?usp=sharing) are some typical mixtures and separation results of 10 sources with length 10 second. The neural network density models always have better subjective separation performance, even for the first set of sample results, where the signal to interference ratio (SIR) of multivariate Laplace model is 0.3 dB higher than that of neural network one. One reason is that SIR is not very sensitive to errors like the low pass and high pass bands permutations since most speech energy locates in low frequency band, while human ears are picky.
