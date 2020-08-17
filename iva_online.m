function [y, y_gt] = iva_online( x, method, mxts_gt, lr )
% inputs:
% x: mixtures
% method: tested method
% mxts_gt: ground truth mixtures, only for SIR performance testing purpose
% lr: learning rate
%
% outputs:
% y: separated outputs
% y_gt: separated outputs for the ground truth mixtures, only for SIR performance testing purpose
num_mic = size(x, 1);
fft_size = 512; hop_size = 160; load fb_512_512_160;
switch method
    case 0 % the iva base line
        iva_baseline = 1; 
    case 1 % learned FNN model
        iva_baseline = 0;
        h = zeros(num_mic, 0);
        load grad_fnn_mdl
    case 2 % learned RNN model
        iva_baseline = 0;
        h = zeros(num_mic, 128);
        load grad_rnn_mdl
    otherwise
        error('undefined')
end

W_iva = 1e3*repmat(eye(num_mic), 1, 1, 255);
y = zeros(size(x));
y_gt = zeros(size(mxts_gt));
t = 1;
while t+fft_size-1 <= size(x,2)
    X = fft(win_a .* x(:, t : t+fft_size-1), [], 2);
    X = X(:, 2:fft_size/2);
    
    Y = zeros(size(X));
    for k = 1 : 255
        Y(:,k) = W_iva(:,:, k)*X(:,k);
    end
    
    Z = Y;
    for k = 1 : 255
        A = inv(W_iva(:,:,k));
        Z(:,k) = diag(A).*Y(:,k);
    end
    y(:, t : t+fft_size-1) = y(:, t : t+fft_size-1) + win_s .* ifft([zeros(num_mic,1), Z, zeros(num_mic,1), conj(Z(:,end:-1:1))], [], 2); % separated outputs
    
    % these results are only for SIR performance testing purpose
    for n = 1 : num_mic
        x_gt = squeeze(mxts_gt(:, n, t : t+fft_size-1));
        X_gt = fft(win_a .* x_gt, [], 2);
        X_gt = X_gt(:, 2:fft_size/2);
        
        Z_gt = zeros(size(X_gt));
        for k = 1 : 255
            A = inv(W_iva(:,:,k));
            Z_gt(:,k) = diag(A).*( W_iva(:,:, k)*X_gt(:,k) );
        end
        y_gt(:, n, t : t+fft_size-1) = y_gt(:, n, t : t+fft_size-1) + reshape(win_s .* ifft([zeros(num_mic,1), Z_gt, zeros(num_mic,1), conj(Z_gt(:,end:-1:1))], [], 2), num_mic, 1, fft_size);
    end
    
    % update the filter coefficients
    F = zeros(num_mic, 255*2);
    F(:, 1:2:end) = real(Y);
    F(:, 2:2:end) = imag(Y);
    G = F./sqrt(sum(F.*F, 2)/255 + 1e-7); % the gradiet in IVA base line
    if iva_baseline == 0
        logE = log( sum(F.*F, 2)/255 + 1e-7 );
        F = F./sqrt( sum(F.*F, 2)/255 + 1e-7 );
        F = log(F(:, 1:2:end).^2 + F(:, 2:2:end).^2 + 1e-7);
        F = tanh([F, logE, h, ones(num_mic, 1)] * W1);
        h = F(:, 1:size(h,2));
        F = tanh([F, ones(num_mic, 1)] * W2);
        F = [F, ones(num_mic, 1)] * W3;
        F = log(1 + exp(F));
        G = kron(F, [1, 1]) .* G;
    end
    G = G(:, 1:2:end) + sqrt(-1)*G(:, 2:2:end);
    for k = 1 : 255
        grad = G(:,k)*Y(:,k)' - eye(num_mic);
        W_iva(:,:,k) = W_iva(:,:,k) - lr*grad*W_iva(:,:,k)/sqrt(trace(grad*grad') + (num_mic - 2));
    end  
    
    t = t + hop_size;
end