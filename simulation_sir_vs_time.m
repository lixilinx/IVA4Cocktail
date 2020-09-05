clear all; close all; clc
fs = 16000; % sampling rate
wav_len = 30*fs;
N = 2; lr = 0.03; 
num_iter = 50;
SIRs = zeros(wav_len, 6);
for iter = 1 : num_iter
    [mxts, mxts_gt] = generate_mixtures( N, wav_len, fs );
    
    %% test IVA with different source priors
    for test_case = 0 : 5
        [y, y_gt] = iva_online( mxts, test_case, mxts_gt, lr );
        
        energy12 = squeeze(y_gt(1, 1, :)).^2 + squeeze(y_gt(2, 2, :)).^2 + eps;
        energy21 = squeeze(y_gt(1, 2, :)).^2 + squeeze(y_gt(2, 1, :)).^2 + eps;
        SIR = filter(1, [1, 1/fs - 1], energy12) ./ filter(1, [1, 1/fs - 1], energy21);
        if sum(energy12) < sum(energy21)
            SIR = 1./SIR;
        end
        SIRs(:, test_case+1) = SIRs(:, test_case+1) + SIR;
    end
end
plot([1:wav_len]/fs, 10*log10(SIRs(:,1)/num_iter), 'r');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,2)/num_iter), 'm');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,3)/num_iter), 'c');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,4)/num_iter), 'g');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,5)/num_iter), 'b');
hold on; plot([1:wav_len]/fs, 10*log10(SIRs(:,6)/num_iter), 'k');
xlabel('Time (s)'); ylabel('SIR (dB)');
legend('Laplace', 'GGD', 'Student', 'Non-Spherical', 'FNN', 'RNN');