# [IVA4Cocktail](https://arxiv.org/abs/2008.11273)

Speech density estimation for multichannel convolutive speech/music separation. I use independent vector analysis (IVA) as the separation framework. Please check the report for details.

Please use the [archived old code](https://github.com/lixilinx/IVA4Cocktail/releases/tag/v1) to reproduce the results reported [here](https://arxiv.org/abs/2008.11273v2). I rewrote the code to make it more organized.

Unlike the popular end-to-end supervised speech separation methods, the target here is to learn a neural network density model for unsupervised separation. The resultant density model can be used for, e.g., online or batch separation, separation of different number of sources, separation of artificial or realistic mixtures, without the need to retrain any different specific supervised separation model.           

### On the Pytorch training code

artificial_mixture_generator.py: the actual mixing matrix is inv(a_FIR_system) * (another_FIR_system) since we change the mixing matrix constantly. 

dnn_source_priors.py: a memoryless circular source model is defined there. If one wants to recover the phase of each bin as well, noncircular density model must be used. Recovering of phase (up to certain global rotation ambiguity) is nontrivial since this will deconvole/dereverberate the speech. This is achieved by forcing the reconstructed speech using the estimated phases to be coherent with the original source as well. 

losses.py: except for the standard coherence loss, a symmetric Itakura–Saito distance loss can be used to recover the amplitude of speech as well (of course, up to a certain global scaling ambiguity). Pre-emphasizing the high frequencies can make the amplitude modeling easier (set the LPC in artificial_mixture_generator.py properly).

short_time_Fourier_transform.py: this should work with Pytorch's old (torch version 1.7) and new (version 1.8) FFT APIs. I still use the old view_as_real format for complex numbers.  

preconditioned_stochastic_gradient_descent.py: this is a [second order optimizer](https://ieeexplore.ieee.org/document/7875097). I use it mainly to save the hyperparameter tuning efforts.  

Lastly, demo.m is a Matlab/Octave file showing the usage of a trained circular density model with the default settings in config.py. There also are some pre-designed window functions by [this method](https://ieeexplore.ieee.org/document/8304771).

### Some sample separation results for subjective comparison (using the density model in the archived old code)

[These](https://drive.google.com/file/d/18xrjgKNbWOnl0t_w0XnsB0zOr4EOWpX-/view?usp=sharing) are some typical mixtures with simulated RIRs and separation results of 10 sources with length 10 second. The neural network density models always have better subjective separation performance, even for the first set of sample results, where the signal to interference ratio (SIR) of multivariate Laplace model is 0.3 dB higher than that of neural network one. One reason is that SIR is not very sensitive to errors like the low pass and high pass bands permutations since most speech energy locates in low frequency band, while human ears are picky.
