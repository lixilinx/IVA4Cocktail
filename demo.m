[x, fs] = audioread('a_multichannel_wav_file.wav');
assert(fs==16000)
x = x'; num_mic = size(x, 1);

fft_size = 1024; hop_size = 320;
load(['win_', int2str(fft_size), '_', int2str(hop_size), '.mat'])
load('circ_src_prior.mat')
W_iva = 100*repmat(eye(num_mic), 1, 1, fft_size/2-1);
lr_iva = 0.05;

t=1;
y = zeros(num_mic, size(x,2));
Y = zeros(num_mic, fft_size/2-1);
while t+fft_size-1 <= size(x,2)
    X = fft(win .* x(:, t:t+fft_size-1), [], 2);
    X = X(:, 2:fft_size/2);
    for k = 1 : fft_size/2-1
        Y(:,k) = W_iva(:,:,k)*X(:,k);
    end
    avgP = mean(Y.*conj(Y), 2) + 1e-10;
    normY = Y./sqrt(avgP);
    
    %% the dnn source prior part
    f = tanh( [log(normY.*conj(normY) + 1e-10), log(avgP), ones(num_mic, 1)] * W0 );
    f = tanh( [f, ones(num_mic, 1)] * W1 );
    f = [f, ones(num_mic, 1)] * W2;
    f = log(1 + exp(f));
    f = f.*normY;
    
    for k = 1 : fft_size/2-1
        lr = lr_iva/sqrt( 2 - 2*real(f(:,k)'*Y(:,k)) + (f(:,k)'*f(:,k))*(Y(:,k)'*Y(:,k)) );
        W_iva(:,:,k) = (1 + lr)*W_iva(:,:,k) - lr*f(:,k)*Y(:,k)'*W_iva(:,:,k);
    end
    
    for k = 1 : fft_size/2-1
        Y(:,k) = diag(diag(inv(W_iva(:,:,k))))*Y(:,k);
    end

    y(:, t:t+fft_size-1) = y(:, t:t+fft_size-1) + win.*ifft([zeros(num_mic,1), Y, zeros(num_mic,1), conj(Y(:,end:-1:1))], [], 2);
    
    t=t+hop_size;
end
audiowrite('out.wav', y', fs);