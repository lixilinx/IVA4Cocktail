# IVA4Cocktail

Speech density estimation for multichannel convolutive speech separation. I use independent vector analysis (IVA) as the separation framework. The code can also be used for music or other source separations. Please check the report for details.  

## Speech Density Estimation

You need to correctly set the path parameter for WavLoader in 'main.py' to train the density model. 

Set dim_h = 0 to get an memoryless density model, and 1 <= dim_h <= dim_f to get a recurrent density model.

Even with a memoryless density model, the whole signal process graph still defines a deep recurrent network due to the way to update the separation matrices. Momentum and first order optimization methods seem perform not very good (might not with the best hyperparameters in my trials). A [second order method](https://ieeexplore.ieee.org/document/7875097) saves the hyperparameter tuning effort. 

## Separation Performance Testing

I use [this code](https://www.mathworks.com/matlabcentral/fileexchange/5116-room-impulse-response-generator) to generate room impulse response (RIR). It is light and fast. But, as the original image source model, it cannot simulate fractional delays. So, I generate the RIR with higher sampling rate, and then decimate to the correct sampling rate. 

You need to correctly set the path in 'generate_mixtures.m' to run the performance comparison simulations. The neural network density models consistently outperform simple ones like multivariate Laplace or generalized Gaussian distribution.
