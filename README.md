# IVA4Cocktail

Speech density estimation for multichannel convolutive speech separation. I use independent vector analysis (IVA) as the separation framework. The code can also be used for music or other source separations. Please check the report for details.  

## Speech Density Estimation

You need to correctly set the path parameter for WavLoader in 'main.py' to train the density model. 

Set dim_h = 0 to get a feedforward neural network (FNN) density model, and 1 <= dim_h <= dim_f to get a recurrent neural network (RNN) density model.

Even with a memoryless density model, the whole signal process graph still defines a deep recurrent network due to the way to update the separation matrices. Momentum and first order optimization methods seem perform not very well (might not with the best hyperparameters in my trials). A [second order method](https://ieeexplore.ieee.org/document/7875097) saves the hyperparameter tuning effort. 

## Separation Performance Testing

I use [this code](https://www.mathworks.com/matlabcentral/fileexchange/5116-room-impulse-response-generator) to generate room impulse response (RIR). It is light and fast. But, as the original image source method, it cannot simulate fractional delays. So, I generate the RIR with higher sampling rate, and then decimate to the correct sampling rate. 

You need to correctly set the path in 'generate_mixtures.m' to run the performance comparison simulations. Sample FNN and RNN density models prepared as in the report are included here. The neural network density models consistently outperform simple ones like multivariate Laplace or generalized Gaussian distribution.

### Sample separation results

[These](https://drive.google.com/file/d/18xrjgKNbWOnl0t_w0XnsB0zOr4EOWpX-/view?usp=sharing) are some typical mixtures and separation results of 10 sources with length 10 second. The neural network density models always have better subjective separation performance, even for the first set of sample results, where the signal to interference ratio (SIR) of multivariate Laplace model is 0.3 dB higher than that of neural network one. One reason is that SIR is not very sensitive to errors like the low pass and high pass bands permutations since most speech energy locates in low frequency band, while human ears are picky.
